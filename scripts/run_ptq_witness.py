"""Three-way warm-start witness on the real donor: naive TWN vs optimal ternary vs GPTQ.

No training anywhere -- this measures ONLY where the counter format STARTS before recovery.
Motivated by Bonsai-27B (post-training quantization, no retrain, ~90% retention at ~1.1 bpw):
if a calibrated quantizer starts the 1.5B at PPL ~tens instead of ~5e5, the recovery finetune
becomes a last-mile polish instead of a resurrection.

Env: MODEL, DATA_DIR, CKPT_DIR (json output), CALIB_BATCHES (default 64), SEQ, BATCH,
MODES (default "naive,optimal,gptq"). Output: per-domain warm PPL table + ptq_witness.json.
"""
import gc
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memory_native.donor.ptq import ptq_warm_start
from memory_native.donor.qwen import qwen_to_counter
from memory_native.eval import perplexity
from recovery_session import DomainMix

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B")
DATA_DIR = os.environ.get("DATA_DIR", "/content/data/mix_full")
CKPT_DIR = os.environ.get("CKPT_DIR", "/content/drive/MyDrive/mn_recovery_15b")
SEQ = int(os.environ.get("SEQ", "512"))
BATCH = int(os.environ.get("BATCH", "8"))
CALIB_BATCHES = int(os.environ.get("CALIB_BATCHES", "64"))
MODES = os.environ.get("MODES", "naive,optimal,gptq").split(",")
KIND_DEFAULT = "counter_packed" if torch.cuda.is_available() else "counter_rms"
KIND = os.environ.get("KIND", KIND_DEFAULT)

dev = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if dev == "cuda" else torch.float32
print(f"ptq witness: device={dev} kind={KIND} calib={CALIB_BATCHES}x{BATCH}x{SEQ}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
mix = DomainMix(DATA_DIR, seq=SEQ, batch=BATCH, seed=123)
val = mix.val_batches(dev)
calib = [mix.batch_at(i, dev) for i in range(CALIB_BATCHES)]

results = {}
fp = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=dtype, attn_implementation="sdpa").to(dev).eval()
results["fp"] = {f"ppl_{n}": perplexity(fp, b) for n, b in val.items()}
print("fp teacher:", {k: round(v, 1) for k, v in results["fp"].items()}, flush=True)
del fp
gc.collect()
torch.cuda.empty_cache() if dev == "cuda" else None

for mode in MODES:
    t0 = time.perf_counter()
    student = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=dtype, attn_implementation="sdpa").to(dev)
    if mode == "naive":
        qwen_to_counter(student, kind=KIND, threshold_ratio=0.5,
                        lr=0.008, local_grad_clip=1.0, cache_mode="int8")
    else:
        ptq_warm_start(student, calib if mode == "gptq" else [], mode=mode, kind=KIND,
                       lr=0.008, local_grad_clip=1.0, cache_mode="int8")
    student.eval()
    ppl = {f"ppl_{n}": perplexity(student, b) for n, b in val.items()}
    results[mode] = ppl
    print(f"[{mode:8s}] " + " ".join(f"{k}={v:.1f}" for k, v in ppl.items()) +
          f"   ({time.perf_counter()-t0:.0f}s)", flush=True)
    del student
    gc.collect()
    torch.cuda.empty_cache() if dev == "cuda" else None

os.makedirs(CKPT_DIR, exist_ok=True)
with open(os.path.join(CKPT_DIR, "ptq_witness.json"), "w") as f:
    json.dump(results, f, indent=1)

print("\n=== warm-start PPL (NO recovery training) ===", flush=True)
doms = ["en", "ru", "code", "math"]
print(f"{'variant':10s} " + " ".join(f"{d:>12s}" for d in doms))
for name, ppl in results.items():
    print(f"{name:10s} " + " ".join(f"{ppl['ppl_'+d]:12.1f}" for d in doms), flush=True)
