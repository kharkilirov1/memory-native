#!/usr/bin/env python
"""Throughput bench for the packed-NA layers against a dense baseline.

Measures one training step (forward + backward + in-weight update) for:
  dense   nn.Linear + AdamW            (fp32 baseline)
  na5     MXNLinear / PackedNA5        (5-bit, eager)
  na6     MXNLinear / PackedNA6        (6-bit, eager)
  na6c    same, after enable_compile()
  mx6     MX6Linear (MXFP4 9-state)    (format control)

The counter layers update inside backward, so their "step" is fwd+bwd only;
the dense arm additionally calls optimizer.step().

Usage:
    python probes/bench_speed.py --d 512 --batch 32 --steps 30
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mx6.layer import MX6Linear, MXNLinear          # noqa: E402
from mx6.packed_na import PackedNA5, PackedNA6      # noqa: E402
from mx6.packed_na import enable_compile, disable_compile  # noqa: E402


def build_stack(kind: str, d: int, hidden: int):
    """Two linears: d -> hidden -> d, matching the char-LM body."""
    if kind == "dense":
        layers = [nn.Linear(d, hidden), nn.Linear(hidden, d)]
        opt = torch.optim.AdamW([p for l in layers for p in l.parameters()], lr=3e-3)
        return nn.Sequential(*layers), opt
    if kind == "na5":
        return nn.Sequential(MXNLinear(d, hidden, PackedNA5), MXNLinear(hidden, d, PackedNA5)), None
    if kind in ("na6", "na6c"):
        return nn.Sequential(MXNLinear(d, hidden, PackedNA6), MXNLinear(hidden, d, PackedNA6)), None
    if kind == "mx6":
        return nn.Sequential(MX6Linear(d, hidden, lr=0.01), MX6Linear(hidden, d, lr=0.01)), None
    raise ValueError(kind)


def bench(kind: str, d: int, hidden: int, batch: int, steps: int, warmup: int) -> dict:
    torch.manual_seed(0)
    model, opt = build_stack(kind, d, hidden)
    model.train()
    x = torch.randn(batch, d)

    def one_step():
        y = model(x)
        loss = y.square().mean()
        if opt is not None:
            opt.zero_grad()
        loss.backward()
        if opt is not None:
            opt.step()

    for _ in range(warmup):
        one_step()

    t0 = time.perf_counter()
    for _ in range(steps):
        one_step()
    elapsed = time.perf_counter() - t0

    per_step_ms = elapsed / steps * 1000
    tokens = batch * d
    return {
        "arm": kind,
        "ms_per_step": per_step_ms,
        "tok_per_s": tokens * steps / elapsed,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args(argv)

    print(f"d={args.d} hidden={args.hidden} batch={args.batch} "
          f"steps={args.steps} threads={torch.get_num_threads()}")

    results = []
    unavailable = []
    for kind in ("dense", "na5", "na6", "na6c", "mx6"):
        if kind == "na6c":
            enable_compile()
        try:
            results.append(bench(kind, args.d, args.hidden, args.batch,
                                 args.steps, args.warmup))
        except Exception as exc:  # e.g. torch.compile needs a C++ toolchain on CPU
            unavailable.append((kind, f"{type(exc).__name__}: {exc}".splitlines()[0]))
        finally:
            if kind == "na6c":
                disable_compile()

    base = next(r["ms_per_step"] for r in results if r["arm"] == "dense")
    print(f"\n{'arm':8} {'ms/step':>10} {'tok/s':>12} {'vs dense':>10}")
    for r in results:
        print(f"{r['arm']:8} {r['ms_per_step']:10.2f} {r['tok_per_s']:12.0f} "
              f"{base / r['ms_per_step']:9.2f}x")
    for kind, why in unavailable:
        print(f"{kind:8} {'n/a':>10} {'n/a':>12} {'n/a':>10}   ({why})")
    print("\n(vs dense > 1 means slower than nn.Linear+AdamW)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
