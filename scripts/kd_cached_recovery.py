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
     EVAL_EVERY (300), EVAL_MAX_TOKENS (60000), NUM_BLOCKS (0; smoke only), DEVICE.
"""
from __future__ import annotations

import json
import math
import os
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


def kd_topk_loss(student_logits: torch.Tensor, idx: torch.Tensor, val: torch.Tensor,
                 T: float) -> torch.Tensor:
    """KL(teacher_topK || student) with the teacher renormalized over its cached top-K."""
    p = F.softmax(val.float() / T, dim=-1)                              # [B, S, K]
    logq_full = F.log_softmax(student_logits.float() / T, dim=-1)
    logq = logq_full.gather(-1, idx.long())
    return (p * (torch.log(p.clamp_min(1e-9)) - logq)).sum(-1).mean() * (T * T)


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
    if DEVICE == "cuda":
        for p in student.parameters():
            p.data = p.data.to(torch.bfloat16)
        student = student.to(DEVICE)
    student.train()
    counters = [m for m in student.modules()
                if isinstance(m, PackedGroupScaleCounterLinear)]
    fp_params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(fp_params, lr=FP_LR, weight_decay=0.0)
    log(f"student on {DEVICE}: {len(counters)} counter layers, "
        f"{sum(p.numel() for p in fp_params) / 1e6:.0f}M fp params; "
        f"scope={STATS_SCOPE} dec={DECIMATION} steps={STEPS} batch={BATCH}x{SEQ}")

    val = mix.val_batches(DEVICE, max_tokens=EVAL_MAX_TOKENS)

    def strict_eval():
        res = {f"ppl_{n}": perplexity(student, b) for n, b in val.items()}
        return res

    best = float("inf")
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
        loss_kd = kd_topk_loss(out, idx, valp, KD_T)
        loss_ce = F.cross_entropy(out[:, :-1].reshape(-1, out.shape[-1]).float(),
                                  ids[:, 1:].reshape(-1))
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
            payload = {"step": step + 1, "student": student.state_dict(),
                       "strict_metric": metric,
                       "format": {"group": 128, "C": 11, "stats_scope": STATS_SCOPE,
                                  "decimation": DECIMATION}}
            tmp = os.path.join(CKPT_DIR, "latest.pt.tmp")
            torch.save(payload, tmp)
            os.replace(tmp, os.path.join(CKPT_DIR, "latest.pt"))
            if metric < best:
                best = metric
                torch.save(payload, os.path.join(CKPT_DIR, "best.pt.tmp"))
                os.replace(os.path.join(CKPT_DIR, "best.pt.tmp"),
                           os.path.join(CKPT_DIR, "best.pt"))
                log(f"new best metric={metric:.4f}")
    log(f"done: best strict metric={best:.4f}")


if __name__ == "__main__":
    main()
