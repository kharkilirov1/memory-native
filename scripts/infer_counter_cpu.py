"""CPU inference from a counter checkpoint (strict ternary, alpha=0).

Loads the donor skeleton, rebuilds the saved counter modules on top of it
(no PTQ solve, no Hessians), restores the trained state and generates text on
the CPU. This is the deployable path: the ternary counter state IS the model.

env: MODEL (donor id), CKPT (checkpoint .pt), PROMPT, MAX_NEW, GEN_TEMP, TOP_K,
     GROUP, C, ALPHA (0 = strict ternary, the only deployable setting).
"""
from __future__ import annotations

import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memory_native.recovery.runtime import restore_counter_structure, temporary_residual_alpha

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B")
CKPT = os.environ["CKPT"]
PROMPT = os.environ.get("PROMPT", "The capital of France is")
MAX_NEW = int(os.environ.get("MAX_NEW", "40"))
TEMP = float(os.environ.get("GEN_TEMP", "0.0"))
TOP_K = int(os.environ.get("TOP_K", "40"))
GROUP = int(os.environ.get("GROUP", "128"))
C = int(os.environ.get("C", "11"))
ALPHA = float(os.environ.get("ALPHA", "0.0"))


def main() -> None:
    t0 = time.time()
    print(f"loading donor skeleton {MODEL} (cpu, fp32)", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()

    print(f"loading checkpoint {CKPT}", flush=True)
    payload = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = payload["student"] if "student" in payload else payload
    fmt = payload.get("format", {}) if isinstance(payload, dict) else {}
    print(f"  step={payload.get('step')} metric={payload.get('strict_metric')} format={fmt}",
          flush=True)

    # kernel_mode="torch": the gemm/triton paths are CUDA-only.
    report = restore_counter_structure(
        model, state, kind="counter_packed", group=GROUP, C=C,
        kernel_mode="torch", strict_update=False, flip_sample_size=0,
        residual_alpha=ALPHA,
    )
    print(f"  rebuilt {len(report.swapped)} counter linears "
          f"({report.coeffs:,} coeffs)", flush=True)

    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if "rotary" not in k]
    print(f"  load_state_dict: missing={len(real_missing)} unexpected={len(unexpected)}",
          flush=True)
    if real_missing[:3]:
        print(f"  first missing: {real_missing[:3]}", flush=True)

    ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    print(f"\nprompt: {PROMPT!r}\ngenerating {MAX_NEW} tokens at alpha={ALPHA} "
          f"(strict ternary)...\n", flush=True)

    gen_start = time.time()
    out = ids
    with torch.no_grad(), temporary_residual_alpha(model, ALPHA):
        for i in range(MAX_NEW):
            logits = model(out).logits[:, -1, :]
            if TEMP <= 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / TEMP, dim=-1)
                topv, topi = probs.topk(min(TOP_K, probs.shape[-1]), dim=-1)
                nxt = topi.gather(-1, torch.multinomial(topv, 1))
            out = torch.cat([out, nxt], dim=-1)
            if i == 0:
                print(f"  [first token in {time.time() - gen_start:.1f}s]", flush=True)
    dt = time.time() - gen_start

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print("=" * 60)
    print(text)
    print("=" * 60)
    print(f"{MAX_NEW} tokens in {dt:.1f}s = {MAX_NEW / dt:.2f} tok/s "
          f"(full-context recompute, no KV cache)")
    print(f"total wall time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
