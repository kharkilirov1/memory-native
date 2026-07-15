"""Stage B witness: does the calibrated PTQ start COMPOUND with recovery training?

Baseline for comparison is recorded in results/recovery_15b_main.md: recovery from the
NAIVE warm-start (575k) reached EN 109.0 / RU 76.7 / code 41.8 / math 91.4 at step 2000
(B8xT512, stable recipe). Here: identical recipe, identical data stream, but the student
starts from ptq_warm_start(mode="gptq") -- per-row scales, counter-format compatible --
i.e. EN ~17.5k instead of 575k. Frozen ordering forecast: the PTQ-start run sits BELOW the
naive curve at every eval; absolutes not predicted (retired).

Env: MODEL, DATA_DIR, CKPT_DIR, STEPS (default 2000), CALIB_BATCHES (64), SEQ, BATCH.
Logs to CKPT_DIR/ptq_recovery_log.jsonl; checkpoint ptq_rec_ckpt.pt every EVAL_EVERY.
"""
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memory_native.donor.ptq import ptq_warm_start
from memory_native.recovery.distill import kd_divergence
from recovery_session import DomainMix, eval_all

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B")
DATA_DIR = os.environ.get("DATA_DIR", "/content/data/mix_full")
CKPT_DIR = os.environ.get("CKPT_DIR", "/content/drive/MyDrive/mn_recovery_15b")
SEQ = int(os.environ.get("SEQ", "512"))
BATCH = int(os.environ.get("BATCH", "8"))
STEPS = int(os.environ.get("STEPS", "2000"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "400"))
CALIB_BATCHES = int(os.environ.get("CALIB_BATCHES", "64"))
FP_LR, COUNTER_LR, GRAD_CLIP = 3e-4, 0.008, 1.0
SEED = 0

LOG = os.path.join(CKPT_DIR, "ptq_recovery_log.jsonl")
CKPT = os.path.join(CKPT_DIR, "ptq_rec_ckpt.pt")

torch.manual_seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if dev == "cuda" else torch.float32
kind = "counter_packed" if dev == "cuda" else "counter_rms"
print(f"ptq-recovery: device={dev} kind={kind} steps={STEPS}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
mix = DomainMix(DATA_DIR, seq=SEQ, batch=BATCH, seed=SEED)   # SAME stream as the main run
val = mix.val_batches(dev)
calib = [mix.batch_at(100_000 + i, dev) for i in range(CALIB_BATCHES)]  # off-stream calib

teacher = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=dtype, attn_implementation="sdpa").to(dev).eval()
student = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=dtype, attn_implementation="sdpa").to(dev)
report = ptq_warm_start(student, calib, mode="gptq", kind=kind,
                        lr=COUNTER_LR, local_grad_clip=1.0, cache_mode="int8")
print("swap:", report, flush=True)

log = open(LOG, "a", encoding="utf-8")
def emit(step, payload):
    log.write(json.dumps({"step": step, "t": time.time(), **payload}, ensure_ascii=False) + "\n")
    log.flush()

student.eval()
warm = eval_all(student, tokenizer, val, dev)
student.train()
print("[gptq warm]", {k: round(v, 1) for k, v in warm.items() if k.startswith("ppl")}, flush=True)
emit(0, {"phase": "gptq_warm", **warm})

fp = [p for p in student.parameters() if p.requires_grad]
opt = torch.optim.AdamW(fp, lr=FP_LR)
t_last = time.perf_counter()
for step in range(STEPS):
    ids = mix.batch_at(step, dev)
    with torch.no_grad():
        t_logits = teacher(ids).logits
    out = student(ids, labels=ids)
    loss = kd_divergence(out.logits, t_logits, 2.0) + 0.3 * out.loss
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(fp, GRAD_CLIP)
    opt.step()
    if (step + 1) % 100 == 0:
        dt = (time.perf_counter() - t_last) / 100
        t_last = time.perf_counter()
        print(f"step {step+1:5d}/{STEPS}  loss {float(loss.detach()):.3f}  {dt:.2f}s/step",
              flush=True)
    if (step + 1) % EVAL_EVERY == 0:
        r = eval_all(student, tokenizer, val, dev)
        print("  eval", {k: round(v, 1) for k, v in r.items() if k.startswith("ppl")}, flush=True)
        emit(step + 1, {"phase": "eval", **r})
        torch.save({"step": step + 1, "student": student.state_dict()}, CKPT + ".tmp")
        os.replace(CKPT + ".tmp", CKPT)

# reference: the naive-start curve at the same step count (recorded main-run numbers)
print("\nnaive-start reference @2000: EN 109.0 RU 76.7 code 41.8 math 91.4", flush=True)
log.close()
