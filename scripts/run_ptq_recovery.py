"""Solver-v3 recovery with honest strict-ternary evaluation and resumable state.

Training may use residual homotopy ``t + alpha*c/C``. Every reported selection metric and every
best-checkpoint decision is evaluated at ``alpha=0``; homotopy metrics are logged separately.
Resume reconstructs counter modules directly from the checkpoint and skips PTQ/Hessian collection.
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
from memory_native.recovery.runtime import (
    atomic_torch_save,
    build_ptq_counter_kwargs,
    capture_rng_state,
    evaluate_at_alpha,
    metric_from_ppl,
    observe_counter_telemetry,
    prefix_metrics,
    restore_counter_structure,
    restore_rng_state,
)
from recovery_session import DomainMix, eval_all


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() not in {"0", "false", "no", "off"}


MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B")
DATA_DIR = os.environ.get("DATA_DIR", "/content/data/mix_full")
CKPT_DIR = os.environ.get("CKPT_DIR", "/content/drive/MyDrive/mn_recovery_15b_v3")
SEQ = int(os.environ.get("SEQ", "512"))
BATCH = int(os.environ.get("BATCH", "8"))
STEPS = int(os.environ.get("STEPS", "6000"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "400"))
LOG_EVERY = int(os.environ.get("LOG_EVERY", "100"))
CALIB_BATCHES = int(os.environ.get("CALIB_BATCHES", "128"))
GROUP = int(os.environ.get("GROUP", "128"))
C = int(os.environ.get("C", "11"))
PTQ_MODE = os.environ.get("PTQ_MODE", "gptq_group")
COUNTER_KIND = os.environ.get("COUNTER_KIND", "counter_packed")
GROUP_KERNEL_MODE = os.environ.get("GROUP_KERNEL_MODE", "auto")
STRICT_UPDATE = env_bool("STRICT_UPDATE", True)
FLIP_SAMPLE_SIZE = int(os.environ.get("FLIP_SAMPLE_SIZE", "4096"))
REFINE_ITERS = int(os.environ.get("REFINE_ITERS", "2"))
SCALE_REFIT = os.environ.get("SCALE_REFIT", "hdiag")
# Consolidated solver ingredients (defaults unchanged: sym grid, no salient split).
GRID = os.environ.get("GRID", "sym")
ITF_ITERS = int(os.environ.get("ITF_ITERS", "3"))
SALIENT_FIRST = float(os.environ.get("SALIENT_FIRST", "0.0"))
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
EVAL_HOMOTOPY = env_bool("EVAL_HOMOTOPY", True)
RESUME = env_bool("RESUME", True)

os.makedirs(CKPT_DIR, exist_ok=True)
LOG = os.path.join(CKPT_DIR, "ptq_recovery_v3.jsonl")
CKPT = os.path.join(CKPT_DIR, "ptq_rec_v3_latest.pt")
BEST = os.path.join(CKPT_DIR, "ptq_rec_v3_best.pt")
RESUME_PATH = os.environ.get("RESUME_PATH", CKPT)


def cosine(start: float, end: float, progress: float) -> float:
    progress = min(1.0, max(0.0, progress))
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))


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
    return torch.stack([
        (1.0 - F.cosine_similarity(student_hidden[i].float(), teacher_hidden[i].float(), dim=-1)).mean()
        for i in indices
    ]).mean()


def evaluate_pair(student, tokenizer, val, device, train_alpha: float):
    strict = evaluate_at_alpha(student, 0.0, lambda: eval_all(student, tokenizer, val, device))
    has_homotopy = any(hasattr(module, "set_residual_alpha") for module in student.modules())
    if not (EVAL_HOMOTOPY and has_homotopy and train_alpha != 0.0):
        return strict, None
    homotopy = evaluate_at_alpha(
        student, train_alpha, lambda: eval_all(student, tokenizer, val, device)
    )
    return strict, homotopy


def checkpoint_payload(
    *, step: int, student, optimizer, strict_metric: float, best_metric: float,
    counter_lr: float, fp_lr: float, train_alpha: float,
) -> dict:
    payload = {
        "version": 2,
        "step": int(step),
        "student": student.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "strict_metric": float(strict_metric),
        "best_metric": float(best_metric),
        "counter_lr": float(counter_lr),
        "fp_lr": float(fp_lr),
        "train_alpha": float(train_alpha),
        "inference_alpha": 0.0,
        "format": {
            "model": MODEL, "ptq_mode": PTQ_MODE, "counter_kind": COUNTER_KIND,
            "group": GROUP, "C": C, "kernel_mode": GROUP_KERNEL_MODE,
            "strict_update": STRICT_UPDATE, "flip_sample_size": FLIP_SAMPLE_SIZE,
            "grid": GRID, "salient_first": SALIENT_FIRST,
        },
    }
    payload.update(capture_rng_state())
    return payload


def load_checkpoint(path: str) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


torch.manual_seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if dev == "cuda" else torch.float32
resume_payload = load_checkpoint(RESUME_PATH) if RESUME and os.path.exists(RESUME_PATH) else None
print(
    f"solver-v3 recovery: device={dev} mode={PTQ_MODE} kind={COUNTER_KIND} "
    f"kernel={GROUP_KERNEL_MODE} strict={STRICT_UPDATE} group={GROUP} C={C} "
    f"steps={STEPS} resume={bool(resume_payload)}",
    flush=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
mix = DomainMix(DATA_DIR, seq=SEQ, batch=BATCH, seed=SEED)
val = mix.val_batches(dev)
teacher = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=dtype, attn_implementation="sdpa"
).to(dev).eval()
student = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=dtype, attn_implementation="sdpa"
).to(dev)

counter_kwargs = build_ptq_counter_kwargs(
    PTQ_MODE,
    lr=COUNTER_LR_START,
    lr_scale=2e-4,
    local_grad_clip=1.0,
    residual_alpha=HOMOTOPY_ALPHA_START,
    cache_mode="int8",
    kernel_mode=GROUP_KERNEL_MODE,
    strict_update=STRICT_UPDATE,
    flip_sample_size=FLIP_SAMPLE_SIZE,
)

start_step = 0
if resume_payload is None:
    calib = [mix.batch_at(100_000 + i, dev) for i in range(CALIB_BATCHES)]
    report = ptq_warm_start(
        student, calib, mode=PTQ_MODE, kind=COUNTER_KIND, C=C, group=GROUP,
        refine_iters=REFINE_ITERS, scale_refit=SCALE_REFIT,
        grid=GRID, itf_iters=ITF_ITERS, salient_first=SALIENT_FIRST, **counter_kwargs,
    )
    print("swap:", report, flush=True)
else:
    fmt = resume_payload.get("format", {})
    for key, current in (("group", GROUP), ("C", C)):
        saved = fmt.get(key)
        if saved is not None and int(saved) != int(current):
            raise ValueError(f"resume {key}={saved} does not match requested {current}")
    report = restore_counter_structure(
        student, resume_payload["student"], kind=COUNTER_KIND, group=GROUP, C=C,
        **counter_kwargs,
    )
    student.to(dev)
    incompatible = student.load_state_dict(resume_payload["student"], strict=False)
    old_sr_missing = [key for key in incompatible.missing_keys if key.endswith(".sr_step")]
    other_missing = [key for key in incompatible.missing_keys if key not in old_sr_missing]
    if other_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint structure mismatch: missing={other_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    start_step = int(resume_payload.get("step", 0))
    if old_sr_missing:
        for module in student.modules():
            if isinstance(module, PackedGroupScaleCounterLinear):
                module._sr_step = start_step
                module.sr_step.fill_(start_step)
    print(f"restored {report}; start_step={start_step}; PTQ solver skipped", flush=True)

packed_group_layers = [m for m in student.modules() if isinstance(m, PackedGroupScaleCounterLinear)]
for layer in packed_group_layers:
    layer.observe_flip_sample(reset=True)
if packed_group_layers:
    max_scratch = max(m.strict_scratch_bytes() for m in packed_group_layers)
    max_dense_grad = max(m.out_features * m.in_features * 4 for m in packed_group_layers)
    print(
        f"packed-group layers={len(packed_group_layers)}; largest strict scratch="
        f"{max_scratch / 2**20:.2f} MiB vs dense grad_w={max_dense_grad / 2**20:.2f} MiB",
        flush=True,
    )

fp = [p for p in student.parameters() if p.requires_grad]
opt = torch.optim.AdamW(fp, lr=FP_LR_START) if fp else None
if resume_payload is not None and opt is not None and resume_payload.get("optimizer") is not None:
    opt.load_state_dict(resume_payload["optimizer"])
    for state in opt.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(dev)
if resume_payload is not None:
    restore_rng_state(resume_payload)

log = open(LOG, "a", encoding="utf-8")


def emit(step, payload):
    log.write(json.dumps({"step": step, "t": time.time(), **payload}, ensure_ascii=False) + "\n")
    log.flush()


progress0 = start_step / max(STEPS - 1, 1)
initial_alpha = homotopy_alpha(progress0)
initial_counter_lr = cosine(COUNTER_LR_START, COUNTER_LR_END, progress0)
initial_fp_lr = cosine(FP_LR_START, FP_LR_END, progress0)
set_counter_controls(student, initial_counter_lr, initial_alpha)
strict_warm, homotopy_warm = evaluate_pair(student, tokenizer, val, dev, initial_alpha)
strict_warm_metric = metric_from_ppl(strict_warm)
print(
    "[strict ternary warm/resume alpha=0]",
    {k: round(v, 2) for k, v in strict_warm.items() if k.startswith("ppl")},
    f"metric={strict_warm_metric:.4f}", flush=True,
)
if homotopy_warm is not None:
    print(
        f"[homotopy diagnostic alpha={initial_alpha:.3f}]",
        {k: round(v, 2) for k, v in homotopy_warm.items() if k.startswith("ppl")}, flush=True,
    )
emit(
    start_step,
    {
        "phase": "resume" if resume_payload is not None else "warm",
        "strict_metric": strict_warm_metric,
        "train_alpha": initial_alpha,
        "inference_alpha": 0.0,
        **strict_warm,
        **prefix_metrics("strict", strict_warm),
        **(prefix_metrics("homotopy", homotopy_warm) if homotopy_warm is not None else {}),
    },
)

best_metric = (
    float(resume_payload.get(
        "best_metric", resume_payload.get("strict_metric", resume_payload.get("metric", strict_warm_metric))
    )) if resume_payload is not None else strict_warm_metric
)
if resume_payload is None:
    initial_state = checkpoint_payload(
        step=0, student=student, optimizer=opt, strict_metric=strict_warm_metric,
        best_metric=best_metric, counter_lr=initial_counter_lr, fp_lr=initial_fp_lr,
        train_alpha=initial_alpha,
    )
    atomic_torch_save(initial_state, CKPT)
    atomic_torch_save(initial_state, BEST)

if start_step >= STEPS:
    print(f"checkpoint already completed {start_step} >= requested STEPS={STEPS}", flush=True)
    log.close()
    raise SystemExit(0)

t_last = time.perf_counter()
for step in range(start_step, STEPS):
    progress = step / max(STEPS - 1, 1)
    counter_lr = cosine(COUNTER_LR_START, COUNTER_LR_END, progress)
    alpha = homotopy_alpha(progress)
    n_counter = set_counter_controls(student, counter_lr, alpha)
    fp_lr = cosine(FP_LR_START, FP_LR_END, progress)
    if opt is not None:
        for group in opt.param_groups:
            group["lr"] = fp_lr

    ids = mix.batch_at(step, dev)
    use_features = FEATURE_KD_ALPHA > 0
    with torch.no_grad():
        teacher_out = teacher(ids, output_hidden_states=use_features)
    student_out = student(ids, labels=ids, output_hidden_states=use_features)
    logit_kd = kd_divergence(student_out.logits, teacher_out.logits, KD_T)
    feat_kd = (
        feature_distill(student_out.hidden_states, teacher_out.hidden_states, FEATURE_KD_STRIDE)
        if use_features else torch.zeros((), device=dev)
    )
    loss = logit_kd + CE_ALPHA * student_out.loss + FEATURE_KD_ALPHA * feat_kd

    if opt is not None:
        opt.zero_grad(set_to_none=True)
    loss.backward()
    if opt is not None:
        torch.nn.utils.clip_grad_norm_(fp, GRAD_CLIP)
        opt.step()

    telemetry = None
    if (step + 1) % LOG_EVERY == 0:
        dt = (time.perf_counter() - t_last) / LOG_EVERY
        t_last = time.perf_counter()
        telemetry = observe_counter_telemetry(packed_group_layers)
        print(
            f"step {step+1:5d}/{STEPS} loss={float(loss.detach()):.3f} "
            f"kd={float(logit_kd.detach()):.3f} feat={float(feat_kd.detach()):.3f} "
            f"counter_lr={counter_lr:.6g} alpha={alpha:.3f} "
            f"flip_alt={telemetry['flip_rate_alt']:.4f} "
            f"edge={telemetry['counter_edge_sample']:.4f} {dt:.2f}s/step",
            flush=True,
        )

    if (step + 1) % EVAL_EVERY == 0 or step + 1 == STEPS:
        strict_result, homotopy_result = evaluate_pair(student, tokenizer, val, dev, alpha)
        strict_metric = metric_from_ppl(strict_result)
        if telemetry is None:
            telemetry = observe_counter_telemetry(packed_group_layers)
        emit(
            step + 1,
            {
                "phase": "eval", "strict_metric": strict_metric,
                "counter_lr": counter_lr, "fp_lr": fp_lr,
                "train_alpha": alpha, "inference_alpha": 0.0,
                "counter_layers": n_counter, **telemetry,
                **strict_result,
                **prefix_metrics("strict", strict_result),
                **(prefix_metrics("homotopy", homotopy_result) if homotopy_result is not None else {}),
            },
        )
        print(
            "  strict alpha=0",
            {k: round(v, 2) for k, v in strict_result.items() if k.startswith("ppl")},
            f"metric={strict_metric:.4f}", flush=True,
        )
        if homotopy_result is not None:
            print(
                f"  homotopy alpha={alpha:.3f}",
                {k: round(v, 2) for k, v in homotopy_result.items() if k.startswith("ppl")},
                flush=True,
            )

        improved = strict_metric < best_metric
        if improved:
            best_metric = strict_metric
        state = checkpoint_payload(
            step=step + 1, student=student, optimizer=opt, strict_metric=strict_metric,
            best_metric=best_metric, counter_lr=counter_lr, fp_lr=fp_lr, train_alpha=alpha,
        )
        atomic_torch_save(state, CKPT)
        if improved:
            atomic_torch_save(state, BEST)
            print(f"  new strict-ternary best metric={best_metric:.4f}", flush=True)

log.close()
