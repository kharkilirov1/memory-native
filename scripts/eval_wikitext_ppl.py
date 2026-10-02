"""Standardized WikiText-2 perplexity — the external-comparability protocol.

The project's PPL numbers so far live on custom val slices, which gates internal ladders
perfectly but cannot be placed next to published GPTQ/AWQ/AQLM/QuIP# tables. This harness
is the standard protocol those tables use (GPTQ-paper convention):

  * wikitext-2-raw-v1 TEST split, documents joined with "\n\n", tokenized as ONE stream;
  * non-overlapping windows of SEQ tokens (published tables use the model's 2048);
  * NLL over all next-token predictions inside each window (first token of each window
    has no target and is excluded); PPL = exp(mean NLL) over the whole stream.

Modes:
  fp donor:            MODEL=Qwen/Qwen2.5-0.5B python scripts/eval_wikitext_ppl.py
  streamed counter:    MODEL=... STATE_DIR=<streamed out_dir> ...   (restore, then eval)
  recovery checkpoint: MODEL=... CKPT=<ckpt.pt> ...

MAX_WINDOWS bounds CPU budgets (0 = the full test set, ~280k tokens; published numbers
use the full set — report the window count alongside any bounded run).

env: MODEL, STATE_DIR, CKPT, KIND, GROUP, C, SEQ (2048), MAX_WINDOWS (0), DTYPE (fp32),
     ALPHA (0 = strict ternary, the only deployable setting).
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
import torch.nn.functional as F

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B")
STATE_DIR = os.environ.get("STATE_DIR", "")
CKPT = os.environ.get("CKPT", "")
KIND = os.environ.get("KIND", "counter_packed")
GROUP = int(os.environ.get("GROUP", "128"))
C = int(os.environ.get("C", "11"))
SEQ = int(os.environ.get("SEQ", "2048"))
MAX_WINDOWS = int(os.environ.get("MAX_WINDOWS", "0"))
DTYPE = {"fp32": torch.float32, "bf16": torch.bfloat16}[os.environ.get("DTYPE", "fp32")]
ALPHA = float(os.environ.get("ALPHA", "0.0"))


def build_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE)
    model.eval()
    label = "fp"
    if STATE_DIR or CKPT:
        from memory_native.donor.streaming import load_streamed_state
        from memory_native.recovery.runtime import restore_counter_structure

        if STATE_DIR:
            state = load_streamed_state(STATE_DIR)
            label = f"counter[{os.path.basename(os.path.normpath(STATE_DIR))}]"
        else:
            payload = torch.load(CKPT, map_location="cpu", weights_only=False)
            state = payload["student"] if "student" in payload else payload
            label = f"counter[{os.path.basename(CKPT)}]"
        report = restore_counter_structure(
            model, state, kind=KIND, group=GROUP, C=C,
            kernel_mode="torch", strict_update=False, flip_sample_size=0,
            residual_alpha=ALPHA,
        )
        missing, unexpected = model.load_state_dict(state, strict=False)
        real_missing = [k for k in missing if "rotary" not in k and "inv_freq" not in k]
        print(f"restored {len(report.swapped)} counter linears "
              f"({report.coeffs:,} coeffs); missing={len(real_missing)} "
              f"unexpected={len(unexpected)}", flush=True)
        if unexpected or real_missing:
            raise SystemExit(f"state mismatch: missing={real_missing[:5]} "
                             f"unexpected={unexpected[:5]}")
        model.eval()
    return tok, model, label


@torch.no_grad()
def main() -> None:
    tok, model, label = build_model()
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    n_windows = ids.numel() // SEQ
    if MAX_WINDOWS:
        n_windows = min(n_windows, MAX_WINDOWS)
    print(f"model={MODEL} arm={label} seq={SEQ} windows={n_windows} "
          f"({n_windows * SEQ} of {ids.numel()} test tokens)", flush=True)

    total_nll, total_tokens = 0.0, 0
    t0 = time.time()
    for w in range(n_windows):
        window = ids[w * SEQ:(w + 1) * SEQ].unsqueeze(0)
        logits = model(window).logits[0].float()
        logp = F.log_softmax(logits[:-1], dim=-1)
        nll = -logp.gather(1, window[0, 1:].unsqueeze(1)).sum().item()
        total_nll += nll
        total_tokens += SEQ - 1
        ppl = torch.tensor(total_nll / total_tokens).exp().item()
        print(f"  window {w + 1}/{n_windows}  running ppl={ppl:.4f} "
              f"({(time.time() - t0) / (w + 1):.1f}s/window)", flush=True)
    print(f"\nRESULT wikitext2 ppl={torch.tensor(total_nll / total_tokens).exp().item():.4f}"
          f"  ({label}, seq={SEQ}, {total_tokens} scored tokens)")


if __name__ == "__main__":
    main()
