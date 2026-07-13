"""One-shot Colab entrypoint: warm-start Qwen2.5-1.5B into counters and run a recovery session.

All orchestration lives here (not in notebook cells) so the Colab notebook is a single flat
bootstrap cell -- nothing indented to fight the editor. Config comes from env vars with pilot
defaults; the heavy code (memory_native + recovery_session) ships in the Kaggle dataset.

Env knobs: MAX_HOURS, SEQ, BATCH, CKPT_DIR, DATA_DIR, MODEL, plus the stable-recipe constants.
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memory_native.donor.qwen import qwen_to_counter
from memory_native.eval import perplexity
from recovery_session import DomainMix, run_session

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B")
DATA_DIR = os.environ.get("DATA_DIR", "/content/data")
CKPT_DIR = os.environ.get("CKPT_DIR", "/content/ckpt")   # Drive path for the main run
SEQ = int(os.environ.get("SEQ", "1024"))
BATCH = int(os.environ.get("BATCH", "8"))
MAX_HOURS = float(os.environ.get("MAX_HOURS", "0.8"))
SEED = int(os.environ.get("SEED", "0"))
os.makedirs(CKPT_DIR, exist_ok=True)
CKPT, LOG = os.path.join(CKPT_DIR, "ckpt.pt"), os.path.join(CKPT_DIR, "log.jsonl")

torch.manual_seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if dev == "cuda" else torch.float32
print(f"device={dev} dtype={dtype} seq={SEQ} batch={BATCH} max_hours={MAX_HOURS}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
teacher = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=dtype, attn_implementation="sdpa").to(dev).eval()
student = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=dtype, attn_implementation="sdpa").to(dev)
kind = "counter_packed" if dev == "cuda" else "counter_rms"
report = qwen_to_counter(student, kind=kind, threshold_ratio=0.5,
                         lr=0.008, local_grad_clip=1.0, cache_mode="int8")
print("swap:", report, "kind:", kind, flush=True)
if dev == "cuda":
    print(f"GPU mem after build: {torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)

mix = DomainMix(DATA_DIR, seq=SEQ, batch=BATCH, seed=SEED)
print(f"epoch steps: {mix.epoch_steps()}", flush=True)

base_path = os.path.join(CKPT_DIR, "teacher_baseline.json")
if not os.path.exists(base_path):
    val = mix.val_batches(dev)
    base = {f"ppl_{n}": perplexity(teacher, b) for n, b in val.items()}
    json.dump(base, open(base_path, "w"))
    print("teacher baseline:", {k: round(v, 1) for k, v in base.items()}, flush=True)

final = run_session(
    student=student, teacher=teacher, tokenizer=tokenizer, mix=mix, device=dev,
    steps=mix.epoch_steps(), ckpt_path=CKPT, log_path=LOG,
    kd_alpha=1.0, ce_alpha=0.3, temperature=2.0, lr=3e-4, grad_clip=1.0,
    eval_every=400, ckpt_every=400, max_hours=MAX_HOURS)

base = json.load(open(base_path))
warm = next(json.loads(l) for l in open(LOG, encoding="utf-8")
            if json.loads(l).get("phase") == "warm")
print("\n=== recovery report (per-domain PPL) ===", flush=True)
print(f"{'domain':8s} {'fp teacher':>12s} {'warm-start':>12s} {'recovered':>12s}")
for d in ("en", "ru", "code", "math"):
    print(f"{d:8s} {base['ppl_'+d]:12.1f} {warm['ppl_'+d]:12.1f} {final['ppl_'+d]:12.1f}",
          flush=True)
print("\n=== generations (recovered student) ===", flush=True)
for k, v in final["gen"].items():
    print(f"--- {k}\n{v}\n", flush=True)
