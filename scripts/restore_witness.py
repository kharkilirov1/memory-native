"""Hardened restore witness: streamed counter state -> live model, WITHOUT the donor ever
being fully resident. Closes the open gate of the gemma-4-12B conversion (CLAUDE.md item
15: restore + PPL/KL vs fp NOT yet run; the first attempt died silently).

Why the naive path dies: ``from_pretrained(dtype=fp32)`` holds the WHOLE fp donor (44.6 GiB
for gemma-4-12B) before the swap even starts. This script never does that:

  phase 1 (optional, EVAL_TOKENS>0): fp reference NLL per token + fp logits at KL_POSITIONS
     sampled positions, computed BLOCK-STREAMED from the safetensors shards (one block
     resident at a time, the conversion's own machinery) and cached to disk. Peak: one
     block + activations.
  phase 2: meta skeleton -> counter modules rebuilt from the streamed state
     (``restore_counter_structure``) -> every remaining meta tensor materialized straight
     from the shards (embeddings, norms, biases, lm_head; rotary rebuilt from a config
     probe, NEVER from the checkpoint -- to_empty() garbage is a known trap). Peak: the
     restored model itself (~11-12 GiB for 12B).
  phase 3: finiteness probe, top-5 continuation, then PPL of the restored model and KL vs
     the phase-1 cache, logits computed in CHUNKS so [tokens, vocab] never materializes.

faulthandler + per-phase RSS logging make a silent death impossible to miss.

Run (12B box):
  MODEL=<donor snapshot dir> STATE_DIR=<streamed out_dir> EVAL_TOKENS=8192 \
      PYTHONPATH=src python scripts/restore_witness.py
env: MODEL, STATE_DIR (streamed conversion output) or CKPT (recovery .pt), KIND, GROUP, C,
     SEQ (default 1024), EVAL_TOKENS (0 = skip PPL/KL), KL_POSITIONS (default 256),
     PROMPT, CACHE (phase-1 cache path, default <STATE_DIR>/fp_eval_cache.pt), DTYPE.
"""
from __future__ import annotations

import faulthandler
import gc
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

faulthandler.enable()

MODEL = os.environ["MODEL"]
STATE_DIR = os.environ.get("STATE_DIR", "")
CKPT = os.environ.get("CKPT", "")
KIND = os.environ.get("KIND", "counter_packed")
GROUP = int(os.environ.get("GROUP", "128"))
C = int(os.environ.get("C", "11"))
SEQ = int(os.environ.get("SEQ", "1024"))
EVAL_TOKENS = int(os.environ.get("EVAL_TOKENS", "4096"))
KL_POSITIONS = int(os.environ.get("KL_POSITIONS", "256"))
PROMPT = os.environ.get("PROMPT", "The capital of France is")
DTYPE = {"fp32": torch.float32, "bf16": torch.bfloat16}[os.environ.get("DTYPE", "fp32")]
CACHE = os.environ.get("CACHE", "")
LOGIT_CHUNK = int(os.environ.get("LOGIT_CHUNK", "128"))


def rss_gib() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 2**30
    except Exception:
        try:
            with open("/proc/self/status") as handle:
                for line in handle:
                    if line.startswith("VmRSS"):
                        return int(line.split()[1]) / 2**20
        except Exception:
            return float("nan")
    return float("nan")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] rss={rss_gib():.2f} GiB  {msg}", flush=True)


def eval_ids(tokenizer) -> torch.Tensor:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids[0][:EVAL_TOKENS]
    n = (ids.numel() // SEQ) * SEQ
    if n == 0:
        raise SystemExit(f"EVAL_TOKENS={EVAL_TOKENS} < SEQ={SEQ}")
    return ids[:n].view(-1, SEQ)


@torch.no_grad()
def phase1_fp_reference(windows: torch.Tensor, cache_path: str) -> None:
    from memory_native.donor.streaming import (
        _WeightSource, _build_probe, _computed_buffers, _materialize,
        _resolve_decoder, _run_block,
    )
    from transformers import AutoConfig, AutoModelForCausalLM

    log("phase 1: fp reference, block-streamed from shards")
    src = _WeightSource(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    with torch.device("meta"):
        skeleton = AutoModelForCausalLM.from_config(config)
    inner, stack = _resolve_decoder(skeleton)
    probe = _build_probe(config, "cpu")
    probe_inner, _ = _resolve_decoder(probe)
    computed_embed = _computed_buffers(probe_inner.embed_tokens)
    computed_block = _computed_buffers(probe_inner.layers[0])
    rotary = probe_inner.rotary_emb

    _materialize(inner.embed_tokens, f"{stack}.embed_tokens", src, "cpu", DTYPE,
                 computed=computed_embed)
    buffer = [inner.embed_tokens(w.unsqueeze(0)) for w in windows]  # [1, T] each
    log(f"embeddings done, buffer {sum(b.numel() for b in buffer) / 1e6:.1f} M elems")

    n_blocks = len(inner.layers)
    for i in range(n_blocks):
        block = inner.layers[i]
        _materialize(block, f"{stack}.layers.{i}", src, "cpu", DTYPE,
                     computed=computed_block)
        buffer = _run_block(block, buffer, rotary, "cpu", 1, collect=True)
        block.to("meta")
        gc.collect()
        if i % 4 == 0 or i == n_blocks - 1:
            log(f"  fp block {i + 1}/{n_blocks}")

    _materialize(inner.norm, f"{stack}.norm", src, "cpu", DTYPE, computed={})
    hidden = torch.cat([inner.norm(b) for b in buffer], dim=0)  # [W, SEQ, D]
    del buffer

    if src.has("lm_head.weight"):
        w_head = src.get("lm_head.weight", DTYPE)
    else:  # tied
        w_head = src.get(f"{stack}.embed_tokens.weight", DTYPE)
    flat = hidden.view(-1, hidden.shape[-1])
    total = flat.shape[0]
    kl_index = torch.linspace(0, total - 1, min(KL_POSITIONS, total)).long().unique()
    nll = torch.empty(windows.numel(), dtype=torch.float32)
    kl_logits = torch.empty(kl_index.numel(), w_head.shape[0], dtype=torch.float16)
    kl_pos = {int(p): j for j, p in enumerate(kl_index)}
    targets = windows.view(-1)
    seqlen = windows.shape[1]
    for start in range(0, total, LOGIT_CHUNK):
        # _materialize keeps the shard dtype (bf16 donors), the head is DTYPE: cast at use.
        rows = flat[start:start + LOGIT_CHUNK].to(w_head.dtype)
        logits = rows @ w_head.t()
        logp = F.log_softmax(logits.float(), dim=-1)
        for r in range(rows.shape[0]):
            pos = start + r
            in_window = pos % seqlen
            if in_window + 1 < seqlen:  # next-token target inside the window
                nll[pos] = -logp[r, targets[pos + 1]]
            else:
                nll[pos] = float("nan")
            j = kl_pos.get(pos)
            if j is not None:
                kl_logits[j] = logits[r].to(torch.float16)
    torch.save({"nll": nll, "kl_index": kl_index, "kl_logits": kl_logits,
                "windows": windows}, cache_path)
    valid = nll[~nll.isnan()]
    log(f"phase 1 done: fp ppl={valid.mean().exp().item():.3f} over {valid.numel()} tokens"
        f" -> {cache_path}")
    del hidden, flat, w_head
    src.close()
    gc.collect()


@torch.no_grad()
def phase2_restore() -> nn.Module:
    from memory_native.donor.streaming import (
        _WeightSource, _build_probe, _resolve_decoder, load_streamed_state,
    )
    from memory_native.recovery.runtime import restore_counter_structure
    from transformers import AutoConfig, AutoModelForCausalLM

    log("phase 2: restore counter model (donor never fully resident)")
    if STATE_DIR:
        state = load_streamed_state(STATE_DIR)
    else:
        payload = torch.load(CKPT, map_location="cpu", weights_only=False)
        state = payload["student"] if "student" in payload else payload
    log(f"  state loaded: {len(state)} tensors, "
        f"{sum(t.numel() * t.element_size() for t in state.values()) / 2**30:.2f} GiB")

    config = AutoConfig.from_pretrained(MODEL)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    # Partially-converted donors (gemma-4 unified: vision/audio towers are NOT converted
    # by the decoder-only streaming pass): any linear the state does not cover stays fp
    # and is materialized straight from the shards below. Full paths as skip tokens —
    # substring matching is safe while uncovered modules live in distinctly-named towers.
    uncovered = [
        path for path, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
        and f"{path}.state" not in state and f"{path}.counter.state" not in state
        and path != "lm_head"
    ]
    if uncovered:
        log(f"  {len(uncovered)} linears without counter state (multimodal/skipped) stay fp")
    report = restore_counter_structure(
        model, state, kind=KIND, group=GROUP, C=C,
        kernel_mode="torch", strict_update=False, flip_sample_size=0,
        extra_skip=uncovered,
    )
    log(f"  rebuilt {len(report.swapped)} counter linears ({report.coeffs:,} coeffs)")
    missing, unexpected = model.load_state_dict(state, strict=False)
    log(f"  load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        raise SystemExit(f"unexpected keys in state (first 5): {unexpected[:5]}")

    # Materialize every remaining meta tensor straight from the donor shards; anything the
    # shards do not carry is a COMPUTED tensor and comes from the config probe by full
    # path (depth was the only dimension the probe cut, so `layers.N` maps to `layers.0`).
    import re

    src = _WeightSource(MODEL)
    probe = _build_probe(config, "cpu")
    probe_tensors = {n: t for n, t in list(probe.named_buffers()) + list(probe.named_parameters())}
    _, stack = _resolve_decoder(model)

    n_param, n_buf, n_computed, n_tied = 0, 0, 0, 0
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if not tensor.is_meta:
            continue
        leaf = name.rsplit(".", 1)[-1]
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        if src.has(name):
            value = src.get(name, DTYPE if tensor.is_floating_point() else None)
        elif name == "lm_head.weight" and src.has(f"{stack}.embed_tokens.weight"):
            value = src.get(f"{stack}.embed_tokens.weight", DTYPE)
            n_tied += 1
        else:  # computed (rotary inv_freq class): probe by full path, depth remapped to 0
            source = probe_tensors.get(name)
            if source is None:
                source = probe_tensors.get(re.sub(r"\.layers\.\d+\.", ".layers.0.", name))
            if source is None or source.is_meta:
                raise SystemExit(f"no source for meta tensor {name} -- refusing garbage")
            value = source.detach().clone()
            n_computed += 1
        if isinstance(getattr(parent, leaf, None), nn.Parameter):
            setattr(parent, leaf, nn.Parameter(value, requires_grad=False))
            n_param += 1
        else:
            setattr(parent, leaf, value)
            n_buf += 1
    src.close()
    # Tied heads: named_parameters() deduplicates shared tensors, so a tied lm_head is
    # never visited by the loop and still points at the old meta tensor -- re-tie it to
    # the freshly materialized embeddings (a no-op for untied configs).
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
            if t.is_meta]
    if left:
        raise SystemExit(f"still-meta tensors after materialization: {left[:5]}")
    log(f"  materialized from shards: {n_param} params, {n_buf} buffers "
        f"({n_computed} computed, {n_tied} tied)")
    model.eval()
    return model


@torch.no_grad()
def phase3_witness(model: nn.Module, tokenizer, cache_path: str) -> None:
    log("phase 3: finiteness + top-5 + PPL/KL vs fp cache")
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    out = model(ids).logits
    if not torch.isfinite(out).all():
        raise SystemExit("NON-FINITE logits on the probe prompt")
    top = out[0, -1].topk(5).indices.tolist()
    log(f"  finite ok; top-5 after {PROMPT!r}: {[tokenizer.decode([t]) for t in top]}")

    if not (EVAL_TOKENS and os.path.exists(cache_path)):
        log("  no fp cache -- PPL/KL skipped")
        return
    cache = torch.load(cache_path, weights_only=True)
    windows, fp_nll = cache["windows"], cache["nll"]
    kl_index, fp_kl_logits = cache["kl_index"], cache["kl_logits"]
    seqlen = windows.shape[1]
    nll = torch.empty_like(fp_nll)
    kl_sum, kl_n = 0.0, 0
    kl_pos = {int(p): j for j, p in enumerate(kl_index)}
    for w in range(windows.shape[0]):
        logits = model(windows[w:w + 1]).logits[0]
        logp = F.log_softmax(logits.float(), dim=-1)
        base = w * seqlen
        tgt = windows[w]
        nll[base:base + seqlen - 1] = -logp[:-1].gather(
            1, tgt[1:].unsqueeze(1)).squeeze(1)
        nll[base + seqlen - 1] = float("nan")
        for r in range(seqlen):
            j = kl_pos.get(base + r)
            if j is not None:
                p_fp = F.log_softmax(fp_kl_logits[j].float(), dim=-1)
                kl_sum += F.kl_div(logp[r], p_fp, log_target=True,
                                   reduction="sum").item()
                kl_n += 1
        if w % 2 == 0 or w == windows.shape[0] - 1:
            log(f"  window {w + 1}/{windows.shape[0]}")
    valid = ~nll.isnan() & ~fp_nll.isnan()
    ppl = nll[valid].mean().exp().item()
    fp_ppl = fp_nll[valid].mean().exp().item()
    log(f"RESULT: counter ppl={ppl:.3f} vs fp ppl={fp_ppl:.3f} "
        f"(x{ppl / fp_ppl:.2f}); KL(counter||fp)={kl_sum / max(kl_n, 1):.4f} nats "
        f"over {kl_n} positions")


def main() -> None:
    if not STATE_DIR and not CKPT:
        raise SystemExit("set STATE_DIR (streamed conversion) or CKPT (recovery .pt)")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    cache_path = CACHE or os.path.join(STATE_DIR or os.path.dirname(CKPT) or ".",
                                       "fp_eval_cache.pt")
    if EVAL_TOKENS:
        windows = eval_ids(tokenizer)
        if os.path.exists(cache_path):
            log(f"phase 1 cache exists -> {cache_path} (delete to recompute)")
        else:
            phase1_fp_reference(windows, cache_path)
        gc.collect()
    model = phase2_restore()
    phase3_witness(model, tokenizer, cache_path)
    log("witness complete")


if __name__ == "__main__":
    main()
