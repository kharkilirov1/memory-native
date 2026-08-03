"""Cached-KD recovery: train the counter student against a PRECOMPUTED teacher cache.

The 12B-on-T4 recovery loop: no live teacher (see kd_teacher_cache.py), loss =
top-K logit KD (T=KD_T, renormalized over the cached K) + CE_ALPHA * data CE.
Production counter recipe by default: stats_scope=group + decimation=4 + fused kernel,
cosine counter lr, homotopy alpha schedule, strict-alpha=0 selection — mirrors
run_ptq_recovery.py minus feature-KD (uncacheable) and minus the live teacher.

The student is restored WITHOUT the donor ever fully resident (witness machinery), and
its fp tail is cast to bf16 on CUDA (AdamW on the tail; counter layers self-update in
backward through their own kernels).

Run: MODEL=<donor> STATE_DIR=<streamed state> DATA_DIR=<mix bins> CACHE=<cache dir> \
     CKPT_DIR=<out> PYTHONPATH=src python scripts/kd_cached_recovery.py
env: STEPS (from cache), BATCH/SEQ/SEED (must match cache), KD_T (2.0), CE_ALPHA (0.3),
     COUNTER_LR_START (0.002) / COUNTER_LR_END (1e-4), FP_LR (1e-4), GRAD_CLIP (1.0),
     STATS_SCOPE (group), DECIMATION (4), HOMOTOPY_HOLD (0.2) / HOMOTOPY_END (0.9),
     EVAL_EVERY (300), EVAL_MAX_TOKENS (60000), NUM_BLOCKS (0; smoke only), DEVICE,
     GRAD_CKPT (1), FREEZE_EMBED (1), SPLIT_GPUS (1; 2-GPU model-parallel when >=2
     CUDA devices — see split_across_gpus), SPLIT_AT (0 = n_layers//2), CKPT_TMP
     (staging dir for the atomic best.pt write; point it OFF the output volume when
     the output has a size cap — the ckpt is written slim + best-only for the same
     reason).

The two 12B-on-T4 fit levers (both default ON, both no-ops at small scale):
- GRAD_CKPT: REENTRANT activation checkpointing on the decoder blocks. Reentrant is
  the only mode compatible with the eager-only counter layers: its first pass runs
  under no_grad, where the counter forward takes the plain path (no reuse guard, no
  autograd Function), and the backward-time recompute builds the Function graph
  exactly once -- its backward fires the counter update as usual. Non-reentrant
  checkpointing re-invokes the Function with the guard already set and raises.
  Without this the 48-block forward alone carries ~5 GiB of activations on top of a
  ~12 GiB resident student (v3 died at 14.31 GiB allocated, inside the HF forward).
- FREEZE_EMBED: the tied 262k x 3840 embedding is ~1.0B fp params; AdamW would
  lazily allocate ~8.5 GiB of moments at the FIRST step (after the forward already
  fits), plus a ~2 GiB dense lm_head weight grad every step. Frozen embeddings keep
  the fp AdamW tail = norms (+ never-exercised multimodal linears); recovery
  capacity rides the counter layers, which is the point of the method. Documented
  delta vs the 1.5B production recipe (there the tail includes embeddings).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL = os.environ["MODEL"]
STATE_DIR = os.environ["STATE_DIR"]
DATA_DIR = os.environ["DATA_DIR"]
CACHE = os.environ["CACHE"]
CKPT_DIR = os.environ.get("CKPT_DIR", "ckpt_cached_kd")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
KD_T = float(os.environ.get("KD_T", "2.0"))
CE_ALPHA = float(os.environ.get("CE_ALPHA", "0.3"))
COUNTER_LR_START = float(os.environ.get("COUNTER_LR_START", "0.002"))
COUNTER_LR_END = float(os.environ.get("COUNTER_LR_END", "0.0001"))
FP_LR = float(os.environ.get("FP_LR", "0.0001"))
GRAD_CLIP = float(os.environ.get("GRAD_CLIP", "1.0"))
STATS_SCOPE = os.environ.get("STATS_SCOPE", "group")
DECIMATION = int(os.environ.get("DECIMATION", "4"))
HOMOTOPY_HOLD = float(os.environ.get("HOMOTOPY_HOLD", "0.2"))
HOMOTOPY_END = float(os.environ.get("HOMOTOPY_END", "0.9"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "300"))
EVAL_MAX_TOKENS = int(os.environ.get("EVAL_MAX_TOKENS", "60000"))
NUM_BLOCKS = int(os.environ.get("NUM_BLOCKS", "0"))
LOG_EVERY = int(os.environ.get("LOG_EVERY", "25"))
GRAD_CKPT = os.environ.get("GRAD_CKPT", "1") == "1"
FREEZE_EMBED = os.environ.get("FREEZE_EMBED", "1") == "1"
SPLIT_GPUS = os.environ.get("SPLIT_GPUS", "1") == "1"
SPLIT_AT = int(os.environ.get("SPLIT_AT", "0"))
CKPT_TMP = os.environ.get("CKPT_TMP", "")
EVAL_AT_START = os.environ.get("EVAL_AT_START", "1") == "1"


def log(msg: str) -> None:
    print(f"[kd {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cosine(start: float, end: float, progress: float) -> float:
    progress = min(1.0, max(0.0, progress))
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))


def homotopy_alpha(progress: float) -> float:
    if progress <= HOMOTOPY_HOLD:
        return 1.0
    if progress >= HOMOTOPY_END:
        return 0.0
    span = (progress - HOMOTOPY_HOLD) / (HOMOTOPY_END - HOMOTOPY_HOLD)
    return 0.5 * (1.0 + math.cos(math.pi * span))


def _tree_to(obj, dev):
    if torch.is_tensor(obj):
        return obj.to(dev)
    if isinstance(obj, (tuple, list)):
        moved = [_tree_to(o, dev) for o in obj]
        return tuple(moved) if isinstance(obj, tuple) else moved
    return obj


def split_across_gpus(student) -> int:
    """Model-parallel over 2 GPUs: layers[split:] move to cuda:1; embeddings, final
    norm and lm_head STAY on cuda:0 so the embed/head tie survives untouched. Device
    movers ride forward pre-hooks (inside the checkpointed region, so both the
    no-grad pass and the recompute cross the boundary identically, and autograd's
    .to nodes route grads back). The two hidden-state boundaries (block split-1 ->
    split, last block -> final norm) are the only real transfers -- [B, S, d] each.

    The v4 lesson forcing this: the 12B counter buffers alone are ~11.7 GiB resident
    (packed state 8.2 + salient idx/val 1.3 + salient perm 1.7 + scales/v 0.5) plus
    the frozen embedding 2.0 -- ~13.8 GiB before a single activation on a 14.56 GiB
    T4. No activation trick fixes resident state; a second T4 does."""
    from memory_native.donor.streaming import _resolve_decoder

    inner, _ = _resolve_decoder(student)
    n = len(inner.layers)
    split = SPLIT_AT if SPLIT_AT > 0 else n // 2
    dev0, dev1 = torch.device("cuda", 0), torch.device("cuda", 1)

    def mover(dev):
        def hook(module, args, kwargs):
            return _tree_to(args, dev), {k: _tree_to(v, dev) for k, v in kwargs.items()}
        return hook

    for i in range(split, n):
        inner.layers[i].to(dev1)
        inner.layers[i].register_forward_pre_hook(mover(dev1), with_kwargs=True)
    inner.norm.register_forward_pre_hook(mover(dev0), with_kwargs=True)
    log(f"2-GPU split: layers {split}..{n - 1} -> cuda:1; embed/norm/head + "
        f"layers 0..{split - 1} on cuda:0")
    return split


class ShardedCache:
    def __init__(self, cache_dir: str, batch: int):
        self.dir = cache_dir
        self.meta = json.load(open(os.path.join(cache_dir, "cache_manifest.json")))
        self.rows_per_step = batch
        self.shard_steps = int(self.meta["shard_steps"])
        self._shard_id, self._shard = -1, None

    def step(self, step: int, device):
        sid = step // self.shard_steps
        if sid != self._shard_id:
            self._shard = torch.load(os.path.join(self.dir, f"cache_{sid:04d}.pt"),
                                     map_location="cpu", weights_only=True)
            self._shard_id = sid
        r0 = (step - sid * self.shard_steps) * self.rows_per_step
        sl = slice(r0, r0 + self.rows_per_step)
        return self._shard["idx"][sl].to(device), self._shard["val"][sl].to(device)


def kd_and_ce_losses(logits: torch.Tensor, idx: torch.Tensor, val: torch.Tensor,
                     targets: torch.Tensor, T: float):
    """Top-K KD (teacher renormalized over its cached K) + CE, WITHOUT any full-vocab
    fp32 copy: gathered logits minus logsumexp. At 262k vocab the log_softmax(float())
    route costs ~1.6-2 GiB of transient+retained on top of a 12 GiB-resident 12B student
    — the difference between OOM and fitting a T4."""
    lse_T = torch.logsumexp(logits / T, dim=-1, keepdim=True)           # [B, S, 1]
    logq_T = logits.gather(-1, idx.long()) / T - lse_T                  # [B, S, K]
    p = F.softmax(val.float() / T, dim=-1)
    kd = (p * (torch.log(p.clamp_min(1e-9)) - logq_T.float())).sum(-1).mean() * (T * T)
    shifted = logits[:, :-1]
    lse_1 = torch.logsumexp(shifted, dim=-1)                            # [B, S-1]
    tgt = shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    ce = (lse_1.float() - tgt.float()).mean()
    return kd, ce


@torch.no_grad()
def restore_student():
    from memory_native.donor.streaming import (
        _WeightSource, _build_probe, _resolve_decoder, load_streamed_state,
    )
    from memory_native.recovery.runtime import restore_counter_structure
    from transformers import AutoConfig, AutoModelForCausalLM
    import re

    state = (torch.load(STATE_DIR, map_location="cpu", weights_only=True)
             if STATE_DIR.endswith(".pt") else load_streamed_state(STATE_DIR))
    config = AutoConfig.from_pretrained(MODEL)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    if NUM_BLOCKS:
        inner, _ = _resolve_decoder(model)
        inner.layers = nn.ModuleList(list(inner.layers)[:NUM_BLOCKS])
        model.config.num_hidden_layers = NUM_BLOCKS
        # smoke truncation: drop state keys of layers beyond the cut
        keep = re.compile(r"\.layers\.(\d+)\.")
        state = {k: v for k, v in state.items()
                 if not keep.search(k) or int(keep.search(k).group(1)) < NUM_BLOCKS}
    uncovered = [
        path for path, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
        and f"{path}.state" not in state and f"{path}.counter.state" not in state
        and path != "lm_head"
    ]
    report = restore_counter_structure(
        model, state, kind="counter_packed", group=128, C=11,
        kernel_mode="auto", strict_update=True, flip_sample_size=4096,
        lr=COUNTER_LR_START, lr_scale=2e-4, local_grad_clip=1.0, residual_alpha=1.0,
        stats_scope=STATS_SCOPE, decimation=DECIMATION,
        extra_skip=uncovered,
    )
    log(f"restored {len(report.swapped)} counter linears; {len(uncovered)} fp-only linears")
    # The RMS second moment is scope-shaped ([out,1] row vs [out,G] group) and a WARM
    # state carries zeros there anyway: drop mismatching .v keys so the fresh zeros of
    # the restored scope stand (cross-scope v is meaningless, not restorable).
    dropped_v = 0
    for key in [k for k in state if k.endswith(".v")]:
        try:
            buf = model.get_submodule(key.rsplit(".", 1)[0]).v
        except AttributeError:
            continue
        if buf.shape != state[key].shape:
            del state[key]
            dropped_v += 1
    if dropped_v:
        log(f"  dropped {dropped_v} scope-mismatched .v keys (fresh zeros stand)")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise SystemExit(f"unexpected keys: {unexpected[:5]}")

    src = _WeightSource(MODEL)
    probe = _build_probe(config, "cpu")
    probe_tensors = {n: t for n, t in
                     list(probe.named_buffers()) + list(probe.named_parameters())}
    _, stack = _resolve_decoder(model)
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if not tensor.is_meta:
            continue
        leaf = name.rsplit(".", 1)[-1]
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        if src.has(name):
            value = src.get(name, torch.float32 if tensor.is_floating_point() else None)
        elif name == "lm_head.weight" and src.has(f"{stack}.embed_tokens.weight"):
            value = src.get(f"{stack}.embed_tokens.weight", torch.float32)
        else:
            source = probe_tensors.get(name)
            if source is None:
                source = probe_tensors.get(
                    re.sub(r"\.layers\.\d+\.", ".layers.0.", name))
            if source is None or source.is_meta:
                raise SystemExit(f"no source for meta tensor {name}")
            value = source.detach().clone()
        if isinstance(getattr(parent, leaf, None), nn.Parameter):
            setattr(parent, leaf, nn.Parameter(value, requires_grad=True))
        else:
            setattr(parent, leaf, value)
    src.close()
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
            if t.is_meta]
    if left:
        raise SystemExit(f"still-meta after materialization: {left[:5]}")
    return model


def main() -> None:
    from recovery_session import DomainMix
    from memory_native.eval import perplexity
    from memory_native.group_scale_packed import PackedGroupScaleCounterLinear
    from memory_native.recovery.runtime import evaluate_at_alpha, metric_from_ppl

    os.makedirs(CKPT_DIR, exist_ok=True)
    cache_meta = json.load(open(os.path.join(CACHE, "cache_manifest.json")))
    STEPS = int(os.environ.get("STEPS", str(cache_meta["steps"])))
    BATCH, SEQ, SEED = cache_meta["batch"], cache_meta["seq"], cache_meta["seed"]
    assert STEPS <= cache_meta["steps"], "cache is shorter than requested STEPS"
    mix = DomainMix(DATA_DIR, seq=SEQ, batch=BATCH, seed=SEED)
    cache = ShardedCache(CACHE, BATCH)

    student = restore_student()
    student.train()
    if FREEZE_EMBED:
        seen, frozen = set(), 0
        for mod in (student.get_input_embeddings(), student.get_output_embeddings()):
            for p in (mod.parameters() if mod is not None else []):
                if id(p) in seen:
                    continue
                seen.add(id(p))
                p.requires_grad_(False)
                frozen += p.numel()
        log(f"froze embeddings/lm_head ({frozen / 1e6:.0f}M params)")
    if DEVICE == "cuda":
        for p in student.parameters():
            p.data = p.data.to(torch.bfloat16)
        student = student.to(DEVICE)
        if SPLIT_GPUS and torch.cuda.device_count() >= 2:
            split_across_gpus(student)
    student.config.use_cache = False    # never needed: KD forward + labels-CE eval only
    if GRAD_CKPT:
        # reentrant ONLY -- see module docstring for the counter-guard interaction
        student.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": True})
        # frozen embeddings would leave the checkpointed blocks with no grad-requiring
        # input -> no backward would ever reach the counter layers; the hook restores
        # the grad path without unfreezing the weight
        student.enable_input_require_grads()
    counters = [m for m in student.modules()
                if isinstance(m, PackedGroupScaleCounterLinear)]
    fp_params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(fp_params, lr=FP_LR, weight_decay=0.0)
    log(f"student on {DEVICE}: {len(counters)} counter layers, "
        f"{sum(p.numel() for p in fp_params) / 1e6:.0f}M fp params; "
        f"scope={STATS_SCOPE} dec={DECIMATION} steps={STEPS} batch={BATCH}x{SEQ} "
        f"ckpt={int(GRAD_CKPT)} freeze_embed={int(FREEZE_EMBED)}")

    val = mix.val_batches(DEVICE, max_tokens=EVAL_MAX_TOKENS)

    def strict_eval():
        res = {f"ppl_{n}": perplexity(student, b) for n, b in val.items()}
        return res

    # Slim checkpoint: a full 12B student state_dict is ~13-15 GiB; best+latest+tmp
    # would blow Kaggle's ~20 GiB output cap at the FIRST eval. Keep best.pt only and
    # drop everything reconstructible: frozen fp params (donor has them; tied lm_head
    # is caught via remove_duplicate=False) and the immutable salient/perm/v buffers
    # (the conversion state has them; v is an optimizer stat, not model state).
    frozen_fp = {name for name, p in student.named_parameters(remove_duplicate=False)
                 if not p.requires_grad}
    drop_suffix = (".salient_idx", ".salient_val", ".perm", ".v")

    def slim_payload(step: int, metric: float) -> dict:
        keep = {k: v.cpu() for k, v in student.state_dict().items()
                if k not in frozen_fp and not k.endswith(drop_suffix)}
        return {"step": step, "student": keep, "strict_metric": metric,
                "format": {"group": 128, "C": 11, "stats_scope": STATS_SCOPE,
                           "decimation": DECIMATION},
                "partial_state": "frozen fp + salient/perm/v dropped; rebase on the "
                                 "conversion state + donor to reload"}

    history = []
    best = float("inf")
    if EVAL_AT_START:
        # the WARM baseline: without it a homotopy-turbulent curve is unreadable
        # (recovery-below-warm vs degradation look identical mid-run)
        res = evaluate_at_alpha(student, 0.0, strict_eval)
        metric = metric_from_ppl(res)
        log(f"strict alpha=0 WARM {res} metric={metric:.4f}")
        history.append({"step": 0, "metric": metric,
                        **{k: float(v) for k, v in res.items()}})
        json.dump(history, open(os.path.join(CKPT_DIR, "metrics.json"), "w"), indent=1)
    t0 = time.time()
    for step in range(STEPS):
        progress = step / max(STEPS - 1, 1)
        clr = cosine(COUNTER_LR_START, COUNTER_LR_END, progress)
        alpha = homotopy_alpha(progress)
        for m in counters:
            m.set_lr(clr)
            m.set_residual_alpha(alpha)
        for g in opt.param_groups:
            g["lr"] = cosine(FP_LR, FP_LR * 0.1, progress)

        ids = mix.batch_at(step, DEVICE)
        idx, valp = cache.step(step, DEVICE)
        out = student(ids).logits
        loss_kd, loss_ce = kd_and_ce_losses(out, idx, valp, ids[:, 1:], KD_T)
        loss = loss_kd + CE_ALPHA * loss_ce
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fp_params, GRAD_CLIP)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if (step + 1) % LOG_EVERY == 0:
            log(f"step {step + 1}/{STEPS} kd={loss_kd.item():.4f} ce={loss_ce.item():.4f} "
                f"clr={clr:.5f} alpha={alpha:.2f} {(time.time() - t0) / (step + 1):.2f}s/step")
        if (step + 1) % EVAL_EVERY == 0 or step + 1 == STEPS:
            res = evaluate_at_alpha(student, 0.0, strict_eval)
            metric = metric_from_ppl(res)
            log(f"strict alpha=0 {res} metric={metric:.4f}")
            history.append({"step": step + 1, "metric": metric,
                            **{k: float(v) for k, v in res.items()}})
            json.dump(history, open(os.path.join(CKPT_DIR, "metrics.json"), "w"),
                      indent=1)
            if metric < best:
                best = metric
                stage = CKPT_TMP if CKPT_TMP else CKPT_DIR
                os.makedirs(stage, exist_ok=True)
                tmp = os.path.join(stage, "best.pt.tmp")
                torch.save(slim_payload(step + 1, metric), tmp)
                shutil.move(tmp, os.path.join(CKPT_DIR, "best.pt"))
                log(f"new best metric={metric:.4f} (slim ckpt)")
    log(f"done: best strict metric={best:.4f}")


if __name__ == "__main__":
    main()
