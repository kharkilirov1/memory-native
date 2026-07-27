"""CLI for the block-sequential streaming conversion.

Converts a donor that does not fit in memory: one transformer block is resident
at a time, so peak memory is set by the block width rather than the model size
(measured on Qwen2.5-1.5B: 16.23 GiB in-memory vs 3.66 GiB streaming). Writes
counter state per block plus a manifest, so an interrupted run resumes at the
first unfinished block.

    MODEL=<path or hf id>  OUT_DIR=<dir>  python scripts/convert_streaming.py

env: MODEL (a LOCAL snapshot directory, or an id already in the HF cache),
     OUT_DIR, DATA_DIR (mix corpus; falls back to random ids when unset),
     CALIB_BATCHES, SEQ, MICRO_BATCH, GROUP, C, GRID, SALIENT_FIRST,
     SALIENT_SCOPE, IN_SWEEP_REFIT, CASCADE (1 = calibrate each block on the
     already-converted previous one, the default), DEVICE, DTYPE, RESUME.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch

from memory_native.donor.streaming import convert_streaming, load_streamed_state


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else raw not in {"0", "false", "False"}


MODEL = os.environ.get("MODEL", "")
OUT_DIR = os.environ.get("OUT_DIR", "")
DATA_DIR = os.environ.get("DATA_DIR", "")
CALIB_BATCHES = int(os.environ.get("CALIB_BATCHES", "16"))
SEQ = int(os.environ.get("SEQ", "512"))
MICRO_BATCH = int(os.environ.get("MICRO_BATCH", "1"))
GROUP = int(os.environ.get("GROUP", "128"))
C = int(os.environ.get("C", "11"))
GRID = os.environ.get("GRID", "itf")
SALIENT_FIRST = float(os.environ.get("SALIENT_FIRST", "0.02"))
SALIENT_SCOPE = os.environ.get("SALIENT_SCOPE", "layer")
IN_SWEEP_REFIT = _env_bool("IN_SWEEP_REFIT", True)
CASCADE = _env_bool("CASCADE", True)
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = {"fp32": torch.float32, "fp16": torch.float16,
         "bf16": torch.bfloat16}[os.environ.get("DTYPE", "fp32")]
RESUME = _env_bool("RESUME", True)


def calibration_batches():
    """Mix-corpus batches when DATA_DIR is given, random ids otherwise (the
    latter is only meaningful for a plumbing smoke, not for quality)."""
    if DATA_DIR:
        from memory_native.recovery.runtime import MixCorpus  # type: ignore

        mix = MixCorpus(DATA_DIR, seq=SEQ, batch=1)
        return [mix.batch_at(100_000 + i, "cpu") for i in range(CALIB_BATCHES)]
    from transformers import AutoConfig

    vocab = AutoConfig.from_pretrained(MODEL).vocab_size
    torch.manual_seed(1234)
    print("DATA_DIR unset: calibrating on RANDOM ids — plumbing only, not quality",
          flush=True)
    return [torch.randint(0, vocab, (1, SEQ)) for _ in range(CALIB_BATCHES)]


def main() -> int:
    if not MODEL or not OUT_DIR:
        print(__doc__)
        print("error: MODEL and OUT_DIR are required", file=sys.stderr)
        return 2

    started = time.time()
    report = convert_streaming(
        MODEL, calibration_batches(), OUT_DIR, kind="counter_packed", C=C, group=GROUP,
        dtype=DTYPE, device=DEVICE, micro_batch=MICRO_BATCH, cascade=CASCADE,
        resume=RESUME, grid=GRID, salient_first=SALIENT_FIRST,
        salient_scope=SALIENT_SCOPE, in_sweep_refit=IN_SWEEP_REFIT,
    )
    state = load_streamed_state(OUT_DIR)
    ternary = sum(v.numel() * v.element_size()
                  for k, v in state.items() if k.endswith("counter.state"))
    print(f"\nblocks {report.blocks_converted} converted "
          f"({report.blocks_resumed} already done), targets {len(report.targets)}, "
          f"coeffs {report.coeffs:,}")
    print(f"peak resident {report.peak_gib:.2f} GiB, "
          f"ternary body {ternary / 2**20:.1f} MiB, "
          f"wall {(time.time() - started) / 60:.1f} min")
    print(f"state in {OUT_DIR} — load with "
          f"memory_native.donor.streaming.load_streamed_state()")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
