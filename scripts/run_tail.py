"""Tail experiment: resume from a plateaued checkpoint and decay the LR, with the telemetry
that separates the three plateau regimes.

The main run plateaued at ~EN 87 / RU 46 / code 19 / math 65 on a CONSTANT counter lr (0.008).
Two suspects:
  1. LR noise-ball floor (Prop 1): constant-step finite-state updates don't converge to a
     point, they orbit the optimum in a ball of radius ~ lr (+ SR noise). Decaying lr shrinks
     the ball -> PPL should drop.
  2. Accumulator ceiling (OPEN 1): the bounded counter saturates; no lr helps.

Naive "decay lr, watch PPL" is CONFOUNDED on finite-state weights: too small an lr makes the
per-step tick (-lr*grad*C/s) fall below one quantum, so stochastic rounding stops firing and
flips freeze -- looks identical to a ceiling but is just a sub-quantum step. `state_statistics`
disambiguates:
  * PPL falls + flips continue                       -> LR noise-ball (suspect 1, boring/good)
  * PPL flat + flips FROZEN + saturation low         -> sub-quantum lr (schedule artifact)
  * PPL flat + flips continue + saturation HIGH      -> accumulator ceiling (OPEN 1 in the wild)

Env: CKPT_DIR (Drive, holds ckpt.pt), DATA_DIR, MODEL, TAIL_STEPS (default 3000),
LR_RUNGS ("0.008,0.002,0.0005"). Logs {ppl, flip_rate, counter_edge, scale_mean, d_scale}
to CKPT_DIR/tail_log.jsonl.
"""
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memory_native.donor.qwen import qwen_to_counter
from memory_native.recovery.distill import kd_divergence
from recovery_session import DomainMix, eval_all

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B")
DATA_DIR = os.environ.get("DATA_DIR", "/content/data/mix_full")
CKPT_DIR = os.environ.get("CKPT_DIR", "/content/drive/MyDrive/mn_recovery_15b")
SRC_CKPT = os.environ.get("SRC_CKPT", os.path.join(CKPT_DIR, "ckpt.pt"))
SEQ = int(os.environ.get("SEQ", "512"))
BATCH = int(os.environ.get("BATCH", "8"))
TAIL_STEPS = int(os.environ.get("TAIL_STEPS", "3000"))
FP_LR0 = 3e-4
SCHEDULE = os.environ.get("SCHEDULE", "rungs")          # "cosine" or "rungs"
RUNGS = [float(x) for x in os.environ.get("LR_RUNGS", "0.008,0.002,0.0005").split(",")]
LR_MAX = float(os.environ.get("LR_MAX", str(RUNGS[0])))  # cosine start = constant-run lr
LR_MIN = float(os.environ.get("LR_MIN", "1e-4"))         # cosine floor
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "200"))
SEED = int(os.environ.get("SEED", "0"))

TAIL_CKPT = os.path.join(CKPT_DIR, "tail_ckpt.pt")
TAIL_LOG = os.path.join(CKPT_DIR, "tail_log.jsonl")

torch.manual_seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if dev == "cuda" else torch.float32
kind = "counter_packed" if dev == "cuda" else "counter_rms"
print(f"tail: device={dev} kind={kind} steps={TAIL_STEPS} rungs={RUNGS} src={SRC_CKPT}",
      flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
teacher = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=dtype, attn_implementation="sdpa").to(dev).eval()
student = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=dtype, attn_implementation="sdpa").to(dev)
qwen_to_counter(student, kind=kind, threshold_ratio=0.5, lr=RUNGS[0],
                local_grad_clip=1.0, cache_mode="int8")
ck = torch.load(SRC_CKPT, map_location=dev, weights_only=False)
student.load_state_dict(ck["student"])
print(f"resumed student from step {ck.get('step')}", flush=True)

# counter layers (inner layers are reached through CounterLinearWithBias by .modules())
counter_layers = [m for m in student.modules()
                  if hasattr(m, "state") and hasattr(m, "lr") and hasattr(m, "state_statistics")]
total_coeffs = sum(m.out_features * m.in_features for m in counter_layers)
print(f"{len(counter_layers)} counter layers, {total_coeffs:,} coeffs", flush=True)


def set_counter_lr(lr):
    for m in counter_layers:
        m.lr = float(lr)


import math


def lr_for(step):
    if SCHEDULE == "cosine":                            # smooth LR_MAX -> LR_MIN over the run
        return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * step / max(TAIL_STEPS, 1)))
    frac = step / max(TAIL_STEPS, 1)                    # stepped rungs
    return RUNGS[min(int(frac * len(RUNGS)), len(RUNGS) - 1)]


# weight_flips is DEAD on the CUDA fused path: the Triton kernel mutates packed state directly
# and never increments the buffer (CLAUDE.md: "kernel doesn't report flip-rate; torch path
# does"). So measure flips the only reliable way -- diff the decoded ternary state t of a
# sampled set of layers across the interval. Sampled (not all 196) to keep the decode cheap.
_FLIP_SAMPLE = counter_layers[::max(len(counter_layers) // 12, 1)]


@torch.no_grad()
def _decode_t(m):
    t, _ = m._decode_rows(0, m.out_features)   # int ternary {-1,0,1} [out,in]
    return t.clone()


_prev_t = {id(m): _decode_t(m) for m in _FLIP_SAMPLE}


@torch.no_grad()
def telemetry():
    edge = zero = cabs = smean = 0.0
    for m in counter_layers:
        s = m.state_statistics()
        edge += s.get("counter_edge", 0.0)
        zero += s.get("zero", 0.0)
        cabs += s.get("counter_abs_mean", 0.0)
        smean += s.get("scale_mean", 0.0)
    n = len(counter_layers)
    # true flip fraction over the interval: sign-of-t changes on the sampled layers
    changed = total = 0
    for m in _FLIP_SAMPLE:
        t = _decode_t(m)
        changed += int((t != _prev_t[id(m)]).sum())
        total += t.numel()
        _prev_t[id(m)] = t
    return {"counter_edge": edge / n, "zero_frac": zero / n,
            "counter_abs_mean": cabs / n, "scale_mean": smean / n,
            "flip_frac_interval": changed / max(total, 1)}


mix = DomainMix(DATA_DIR, seq=SEQ, batch=BATCH, seed=SEED + 777)  # fresh stream offset
val = mix.val_batches(dev)
student.train()
fp = [p for p in student.parameters() if p.requires_grad]
opt = torch.optim.AdamW(fp, lr=FP_LR0)

log = open(TAIL_LOG, "a", encoding="utf-8")
def emit(step, payload):
    log.write(json.dumps({"step": step, "t": time.time(), **payload}, ensure_ascii=False) + "\n")
    log.flush()

base = eval_all(student, tokenizer, val, dev)
tel0 = telemetry()
print("[tail start]", {k: round(v, 1) for k, v in base.items() if k.startswith("ppl")},
      "edge", round(tel0["counter_edge"], 3), flush=True)
emit(0, {"phase": "tail_start", **base, **tel0})

prev_scale = tel0["scale_mean"]
print(f"[schedule] {SCHEDULE} lr {LR_MAX} -> {LR_MIN if SCHEDULE=='cosine' else RUNGS[-1]} "
      f"over {TAIL_STEPS} steps", flush=True)
t_last = time.perf_counter()
for step in range(TAIL_STEPS):
    lr = lr_for(step)                                   # set every step (cheap; cosine changes each step)
    set_counter_lr(lr)
    for g in opt.param_groups:
        g["lr"] = FP_LR0 * (lr / LR_MAX)

    ids = mix.batch_at(step, dev)
    with torch.no_grad():
        t_logits = teacher(ids).logits
    out = student(ids, labels=ids)
    loss = kd_divergence(out.logits, t_logits, 2.0) + 0.3 * out.loss
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(fp, 1.0)
    opt.step()

    if (step + 1) % EVAL_EVERY == 0:
        r = eval_all(student, tokenizer, val, dev)
        tel = telemetry()
        flip_rate = tel["flip_frac_interval"]   # true t-diff over the interval (fused-safe)
        d_scale = abs(tel["scale_mean"] - prev_scale)
        prev_scale = tel["scale_mean"]
        dt = (time.perf_counter() - t_last) / EVAL_EVERY
        t_last = time.perf_counter()
        print(f"tail {step+1:5d}/{TAIL_STEPS} lr={lr:<6} "
              f"ppl_en {r['ppl_en']:.1f} ru {r['ppl_ru']:.1f} code {r['ppl_code']:.1f} "
              f"math {r['ppl_math']:.1f} | flip_rate {flip_rate:.4f} edge {tel['counter_edge']:.3f} "
              f"d_scale {d_scale:.2e} {dt:.2f}s/st", flush=True)
        emit(step + 1, {"phase": "tail_eval", "lr": lr, "flip_rate": flip_rate,
                        "d_scale": d_scale, **{k: v for k, v in r.items() if k != "gen"},
                        "gen": r["gen"], **tel})
        torch.save({"step": step + 1, "student": student.state_dict()}, TAIL_CKPT + ".tmp")
        os.replace(TAIL_CKPT + ".tmp", TAIL_CKPT)

log.close()
print("tail done ->", TAIL_LOG, flush=True)
