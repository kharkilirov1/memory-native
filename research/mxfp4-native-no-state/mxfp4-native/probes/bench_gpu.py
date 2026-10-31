#!/usr/bin/env python
"""GPU throughput/memory bench: packed-NA layers vs dense baselines.

Arms:
  dense_fp32      nn.Linear fp32 + AdamW                    (classic reference)
  dense_bf16      fp32 master + bf16 autocast + AdamW       (industry standard)
  dense_bf16_c    same, torch.compile'd                     (industry optimized)
  na5 / na6       MXNLinear (PackedNA5 / PackedNA6), eager
  na6c            NA6 after enable_compile()
  mx6             MX6Linear (MXFP4 9-state)                 (project control)

One "step" = forward + backward + state update (the counter arms update inside
backward; the dense arms additionally call optimizer.step()).

Usage:
    python probes/bench_gpu.py --d 1024 --hidden 4096 --batch 64 --steps 50
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mx6.layer import MX6Linear, MXNLinear              # noqa: E402
from mx6.packed_na import PackedNA5, PackedNA6          # noqa: E402
from mx6.packed_na import enable_compile, disable_compile  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class DenseStack(nn.Module):
    """Two linears; optional bf16 autocast; fp32 master weights either way."""

    def __init__(self, d: int, hidden: int, bf16: bool):
        super().__init__()
        self.l1 = nn.Linear(d, hidden)
        self.l2 = nn.Linear(hidden, d)
        self.bf16 = bf16

    def forward(self, x):
        if self.bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self.l2(self.l1(x))
        return self.l2(self.l1(x))


def build(kind: str, d: int, hidden: int):
    if kind.startswith("dense"):
        bf16 = "bf16" in kind
        model = DenseStack(d, hidden, bf16).to(DEV)
        if kind.endswith("_c"):
            model = torch.compile(model)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        return model, opt
    if kind == "na5":
        return nn.Sequential(MXNLinear(d, hidden, PackedNA5), MXNLinear(hidden, d, PackedNA5)).to(DEV), None
    if kind in ("na6", "na6c", "na6f", "na6d"):
        fused = {"na6f": "on", "na6d": "off"}.get(kind, "auto")
        return nn.Sequential(
            MXNLinear(d, hidden, PackedNA6, fused=fused),
            MXNLinear(hidden, d, PackedNA6, fused=fused)).to(DEV), None
    if kind == "mx6":
        return nn.Sequential(MX6Linear(d, hidden, lr=0.01), MX6Linear(hidden, d, lr=0.01)).to(DEV), None
    raise ValueError(kind)


def state_bits_per_param(model: nn.Module, kind: str) -> float:
    """Bits each parameter needs during training (weights + optimizer state).

    Everything is in bits: mixing bytes for dense and bits for the counter arms
    understates the dense footprint by 8x.
    """
    if kind.startswith("dense"):
        return 96.0           # fp32 master (4 B) + two Adam moments (8 B)
    if kind == "mx6":
        return 8.26           # MXFP4 E2M1 codes stored dense (uint8) + E8M0 scale
    total = sum(m.persistent_bytes() for m in model if hasattr(m, "persistent_bytes"))
    params = sum(m.in_features * m.out_features for m in model
                 if hasattr(m, "in_features") and hasattr(m, "out_features"))
    return total * 8 / params if params else 0.0


def bench(kind: str, d: int, hidden: int, batch: int, steps: int, warmup: int) -> dict:
    torch.manual_seed(0)
    model, opt = build(kind, d, hidden)
    model.train()
    x = torch.randn(batch, d, device=DEV)

    def one_step():
        y = model(x)
        loss = y.float().square().mean()
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        if opt is not None:
            opt.step()

    for _ in range(warmup):
        one_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(steps):
        one_step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak_mb = torch.cuda.max_memory_allocated() / 2**20
    return {
        "arm": kind,
        "ms": elapsed / steps * 1000,
        "tok_s": batch * d * steps / elapsed,
        "peak_mb": peak_mb,
        "bits_per_param": state_bits_per_param(model, kind),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--arms", default="dense_fp32,dense_bf16,dense_bf16_c,na5,na6,na6c,na6f,mx6")
    args = ap.parse_args(argv)

    print(f"device={torch.cuda.get_device_name(0)} d={args.d} hidden={args.hidden} "
          f"batch={args.batch} steps={args.steps} torch={torch.__version__}")

    rows, skipped = [], []
    for kind in args.arms.split(","):
        kind = kind.strip()
        if kind == "na6c":
            enable_compile()
        try:
            rows.append(bench(kind, args.d, args.hidden, args.batch, args.steps, args.warmup))
        except Exception as exc:
            skipped.append((kind, f"{type(exc).__name__}: {exc}".splitlines()[0][:90]))
        finally:
            if kind == "na6c":
                disable_compile()

    base = next(r["ms"] for r in rows if r["arm"] == "dense_bf16")
    print(f"\n{'arm':14} {'ms/step':>9} {'tok/s':>11} {'vs bf16':>8} "
          f"{'peak MB':>9} {'bits/param':>11}")
    for r in rows:
        print(f"{r['arm']:14} {r['ms']:9.2f} {r['tok_s']:11.0f} {base / r['ms']:7.2f}x "
              f"{r['peak_mb']:9.1f} {r['bits_per_param']:11.2f}")
    for kind, why in skipped:
        print(f"{kind:14} {'n/a':>9} {'n/a':>11} {'n/a':>8} {'n/a':>9} {'n/a':>8}   ({why})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
