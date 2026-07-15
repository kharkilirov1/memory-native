"""Solver-v3 recovery: group-PTQ -> packed group counters -> low-LR distillation.

Defaults deliberately differ from the old per-row witness:
  * PTQ_MODE=gptq_group keeps v3 group-128 scales and act-order metadata.
  * COUNTER_KIND=counter_packed selects PackedGroupScaleCounterLinear for the group path.
  * GROUP_KERNEL_MODE=auto + STRICT_UPDATE=1 use group-aware Triton forward, grad_x, and
    update-from-IO on CUDA; no dense W or grad_w is materialized.
  * C=11 uses all 63 reachable 6-bit states.
  * counter LR is cosine 2e-3 -> 1e-4; there is no destructive lr=0.008 hot phase.
  * residual homotopy alpha is held, then cosine-decayed to zero.
  * optional hidden-state KD repairs internal representations, not only final logits.

Environment: MODEL, DATA_DIR, CKPT_DIR, STEPS, CALIB_BATCHES, SEQ, BATCH, GROUP, C,
COUNTER_KIND, GROUP_KERNEL_MODE, STRICT_UPDATE, COUNTER_LR_START/END, HOMOTOPY_HOLD,
FEATURE_KD_ALPHA/STRIDE.
"""
from __future__ import annotations

import json
import math
import os
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from memory_native.donor.ptq import ptq_warm_start
from memory_native.group_scale_packed import PackedGroupScaleCounterLinear
from memory_native.recovery.distill import kd_divergence
from recovery_session import DomainMix, eval_all


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B")
DATA_DIR = os.environ.get("DATA_DIR", "/content/data/mix_full")
CKPT_DIR = os.environ.get("CKPT_DIR", "/content/drive/MyDrive/mn_recovery_15b_v3")
SEQ = int(os.environ.get("SEQ", "512"))
BATCH = int(os.environ.get("BATCH", "8"))
STEPS = int(os.environ.get("STEPS", "6000"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "400"))
CALIB_BATCHES = int(os.environ.get("CALIB_BATCHES", "128"))
GROUP = int(os.environ.get("GROUP", "128"))
C = int(os.environ.get("C", "11"))
PTQ_MODE = os.environ.get("PTQ_MODE", "gptq_group")
COUNTER_KIND = os.environ.get("COUNTER_KIND", "counter_packed")
GROUP_KERNEL_MODE = os.environ.get("GROUP_KERNEL_MODE", "auto")
STRICT_UPDATE = env_bool("STRICT_UPDATE", True)
REFINE_ITERS = int(os.environ.get("REFINE_ITERS", "2"))
SCALE_REFIT = os.environ.get("SCALE_REFIT", "hdiag")
COUNTER_LR_START = float(os.environ.get("COUNTER_LR_START", "0.002"))
COUNTER_LR_END = float(os.environ.get("COUNTER_LR_END", "0.0001"))
FP_LR_START = float(os.environ.get("FP_LR_START", "0.0001"))
FP_LR_END = float(os.environ.get("FP_LR_END", "0.00001"))
HOMOTOPY_ALPHA_START = float(os.environ.get("HOMOTOPY_ALPHA_START", "1.0"))
HOMOTOPY_HOLD = float(os.environ.get("HOMOTOPY_HOLD", "0.20"))
HOMOTOPY_END = float(os.environ.get("HOMOTOPY_END", "0.90"))
FEATURE_KD_ALPHA = float(os.environ.get("FEATURE_KD_ALPHA", "0.05"))
FEATURE_KD_STRIDE = int(os.environ.get("FEATURE_KD_STRIDE", "4"))
CE_ALPHA = float(os.environ.get("CE_ALPHA", "0.3"))
KD_T = float(os.environ.get("KD_T", "2.0"))
GRAD_CLIP = float(os.environ.get("GRAD_CLIP", "1.0"))
SEED = int(os.environ.get("SEED", "0"))

os.makedirs(CKPT_DIR, exist_ok=True)
LOG = os.path.join(CKPT_DIR, "ptq_recovery_v3.jsonl")
CKPT = os.path.join(CKPT_DIR, "ptq_rec_v3_latest.pt")
BEST = os.path.join(CKPT_DIR, "ptq_rec_v3_best.pt")


def cosine(start: float, end: float, p: float) -> float:
    p = min(1.0, max(0.0, p))
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * p))


def homotopy_alpha(progress: float) -> float:
    if progress <= HOMOTOPY_HOLD:
        return HOMOTOPY_ALPHA_START
    if progress >= HOMOTOPY_END:
        return 0.0
    q = (progress - HOMOTOPY_HOLD) / max(HOMOTOPY_END - HOMOTOPY_HOLD, 1e-12)
    return cosine(HOMOTOPY_ALPHA_START, 0.0, q)


def set_counter_controls(model, lr: float, alpha: float) -> int:
    count = 0
    for module in model.modules():
        if hasattr(module, "set_lr"):
            module.set_lr(lr)
            count += 1
        elif hasattr(module, "lr") and hasattr(module, "state"):
            module.lr = float(lr)
            count += 1
        if hasattr(module, "set_residual_alpha"):
            module.set_residual_alpha(alpha)
    return count


def feature_distill(student_hidden, teacher_hidden, stride: int) -> torch.Tensor:
    indices = list(range(stride, min(len(student_hidden), len(teacher_hidden)), stride))
    last = min(len(student_hidden), len(teacher_hidden)) - 1
    if last not in indices:
        indices.append(last)
    terms = []
    for i in indices:
        s = student_hidden[i].float()
        t = teacher_hidden[i].float()
        terms.append((1.0 - F.cosine_similarity(s, t, dim=-1)).mean())
    return torch.stack(terms).mean()


def eval_metric(result: dict) -> float:
    ppls = [float(v) for k, v in result.items() if k.startswith("ppl") and float(v) > 0]
    return sum(math.log(v) for v in ppls) / max(len(ppls), 1)


torch.manual_seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if dev == "cuda" else torch.float32
print(
    f"solver-v3 recovery: device={dev} mode={PTQ_MODE} kind={COUNTER_KIND} "
    f"kernel={GROUP_KERNEL_MODE} strict={STRICT_UPDATE} group={GROUP} C={C} steps={STEPS}",
    flush=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
mix = DomainMix(DATA_DIR, seq=SEQ, batch=BATCH, seed=SEED)
val = mix.val_batches(dev)
calib = [mix.batch_at(100_000 + i, dev) for i in range(CALIB_BATCHES)]

teacher = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=dtype, attn_implementation="sdpa"
).to(dev).eval()
student = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=dtype, attn_implementation="sdpa"
).to(dev)

group_kernel_kw = {}
if PTQ_MODE in {"gptq_group", "group128v3", "group"}:
    group_kernel_kw = {
        "kernel_mode": GROUP_KERNEL_MODE,
        "strict_update": STRICT_UPDATE,
    }
report = ptq_warm_start(
    student,
    calib,
    mode=PTQ_MODE,
    kind=COUNTER_KIND,
    C=C,
    group=GROUP,
    refine_iters=REFINE_ITERS,
    scale_refit=SCALE_REFIT,
    lr=COUNTER_LR_START,
    lr_scale=2e-4,
    residual_alpha=HOMOTOPY_ALPHA_START,
    local_grad_clip=1.0,
    cache_mode="int8",  # used only by legacy row-scale counter kinds
    **group_kernel_kw,
)
print("swap:", report, flush=True)
packed_group_layers = [m for m in student.modules() if isinstance(m, PackedGroupScaleCounterLinear)]
if packed_group_layers:
    max_scratch = max(m.strict_scratch_bytes() for m in packed_group_layers)
    max_dense_grad = max(m.out_features * m.in_features * 4 for m in packed_group_layers)
    print(
        f"packed-group layers={len(packed_group_layers)}; largest strict scratch="
        f"{max_scratch / 2**20:.2f} MiB vs dense grad_w={max_dense_grad / 2**20:.2f} MiB",
        flush=True,
    )

log = open(LOG, "a", encoding="utf-8")


def emit(step, payload):
    log.write(json.dumps({"step": step, "t": time.time(), **payload}, ensure_ascii=False) + "\n")
    log.flush()


student.eval()
warm = eval_all(student, tokenizer, val, dev)
student.train()
print("[v3 warm]", {k: round(v, 2) for k, v in warm.items() if k.startswith("ppl")}, flush=True)
emit(
    0,
    {
        "phase": "warm", "mode": PTQ_MODE, "kind": COUNTER_KIND,
        "kernel_mode": GROUP_KERNEL_MODE, "strict_update": STRICT_UPDATE,
        "group": GROUP, "C": C, **warm,
    },
)

fp = [p for p in student.parameters() if p.requires_grad]
opt = torch.optim.AdamW(fp, lr=FP_LR_START) if fp else None
best_metric = float("inf")
t_last = time.perf_counter()

for step in range(STEPS):
    progress = step / max(STEPS - 1, 1)
    counter_lr = cosine(COUNTER_LR_START, COUNTER_LR_END, progress)
    alpha = homotopy_alpha(progress)
    n_counter = set_counter_controls(student, counter_lr, alpha)
    fp_lr = cosine(FP_LR_START, FP_LR_END, progress)
    if opt is not None:
        for pg in opt.param_groups:
            pg["lr"] = fp_lr

    ids = mix.batch_at(step, dev)
    use_features = FEATURE_KD_ALPHA > 0
    with torch.no_grad():
        tout = teacher(ids, output_hidden_states=use_features)
    sout = student(ids, labels=ids, output_hidden_states=use_features)
    logit_kd = kd_divergence(sout.logits, tout.logits, KD_T)
    feat_kd = (
        feature_distill(sout.hidden_states, tout.hidden_states, FEATURE_KD_STRIDE)
        if use_features else torch.zeros((), device=dev)
    )
    loss = logit_kd + CE_ALPHA * sout.loss + FEATURE_KD_ALPHA * feat_kd

    if opt is not None:
        opt.zero_grad(set_to_none=True)
    loss.backward()
    if opt is not None:
        torch.nn.utils.clip_grad_norm_(fp, GRAD_CLIP)
        opt.step()

    if (step + 1) % 100 == 0:
        dt = (time.perf_counter() - t_last) / 100
        t_last = time.perf_counter()
        print(
            f"step {step+1:5d}/{STEPS} loss={float(loss.detach()):.3f} "
            f"kd={float(logit_kd.detach()):.3f} feat={float(feat_kd.detach()):.3f} "
            f"counter_lr={counter_lr:.6g} alpha={alpha:.3f} {dt:.2f}s/step",
            flush=True,
        )

    if (step + 1) % EVAL_EVERY == 0:
        student.eval()
        result = eval_all(student, tokenizer, val, dev)
        student.train()
        metric = eval_metric(result)
        payload = {
            "phase": "eval",
            "metric": metric,
            "counter_lr": counter_lr,
            "fp_lr": fp_lr,
            "alpha": alpha,
            "counter_layers": n_counter,
            **result,
        }
        emit(step + 1, payload)
        print(
            "  eval",
            {k: round(v, 2) for k, v in result.items() if k.startswith("ppl")},
            f"metric={metric:.4f}",
            flush=True,
        )
        state = {
            "step": step + 1,
            "student": student.state_dict(),
            "metric": metric,
            "counter_lr": counter_lr,
            "fp_lr": fp_lr,
            "alpha": alpha,
        }
        torch.save(state, CKPT + ".tmp")
        os.replace(CKPT + ".tmp", CKPT)
        if metric < best_metric:
            best_metric = metric
            torch.save(state, BEST + ".tmp")
            os.replace(BEST + ".tmp", BEST)
            print(f"  new best metric={best_metric:.4f}", flush=True)

log.close()
