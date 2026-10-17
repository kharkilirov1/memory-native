#!/usr/bin/env python
"""Timing witness for the pure-PyTorch counter step (CPU box, no GPU).

Measures the deployed packed group-scale layer against a dense linear + AdamW
baseline on the same shape, then splits the counter step into its phases so a
change can be attributed:

    forward  -> visible-weight decode + GEMM
    grad_x   -> decode again + GEMM   (the second decode of the step)
    update   -> correlation GEMM + the elementwise automaton chain

Timing uses `torch.utils.benchmark.blocked_autorange` (adaptive warmup + repeats,
median and IQR) instead of a hand-rolled loop: the step is 10s-100s of ms, and a
median over a fixed small repeat count is too noisy on a shared box.

    python scripts/bench_counter_step.py                      # 1536x1536, batch 64, 10 threads
    python scripts/bench_counter_step.py --in 2048 --batch 512
    python scripts/bench_counter_step.py --threads 1          # single-thread A/B

Compare two checkouts by running this same file under each one (the script puts
its own ``../src`` first on sys.path, so a worktree copy tests that worktree).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.benchmark import Timer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memory_native import PackedGroupScaleCounterLinear  # noqa: E402


def _bench(fn, threads: int, run_time: float) -> tuple[float, float]:
    measurement = Timer(stmt="fn()", globals={"fn": fn}, num_threads=threads).blocked_autorange(
        min_run_time=run_time)
    return measurement.median * 1000.0, measurement.iqr * 1000.0


def _dense_step(in_features: int, out_features: int, batch: int,
                threads: int, run_time: float) -> float:
    torch.manual_seed(0)
    dense = torch.nn.Linear(in_features, out_features, bias=False)
    opt = torch.optim.AdamW(dense.parameters(), lr=1e-3)
    x = torch.randn(batch, in_features)

    def step():
        opt.zero_grad(set_to_none=True)
        dense(x).square().mean().backward()
        opt.step()

    return _bench(step, threads, run_time)[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="in_features", type=int, default=1536)
    ap.add_argument("--out", dest="out_features", type=int, default=1536)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--C", type=int, default=11)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=10,
                    help="torch CPU threads (10 = this box's realistic default)")
    ap.add_argument("--run-time", type=float, default=3.0,
                    help="target seconds per blocked_autorange measurement")
    args = ap.parse_args(argv)

    torch.set_num_threads(max(1, args.threads))
    I, O, B = args.in_features, args.out_features, args.batch
    torch.manual_seed(0)
    layer = PackedGroupScaleCounterLinear(
        I, O, group=args.group, C=args.C, kernel_mode="gemm", lr=1e-3)
    x = torch.randn(B, I)
    go = torch.randn(B, O)

    phases = {
        "forward (decode+gemm)": lambda: layer._forward_2d(x),
        "grad_x (decode+gemm)": lambda: layer._grad_x_2d(go),
        "update (chain)": lambda: layer._update_from_io(x, go),
        "full fwd+bwd step": lambda: layer(x).square().mean().backward(),
    }
    print(f"layer {O}x{I} group={args.group} C={args.C} batch={B} threads={torch.get_num_threads()}")
    print(f"{'phase':26s} {'median ms':>10s} {'IQR ms':>8s}")
    results = {}
    for name, fn in phases.items():
        median, iqr = _bench(fn, args.threads, args.run_time)
        results[name] = median
        print(f"  {name:24s} {median:10.1f} {iqr:8.1f}")
    dense = _dense_step(I, O, B, args.threads, args.run_time)
    print(f"  {'dense+AdamW step':24s} {dense:10.1f}")
    print(f"counter/dense = {results['full fwd+bwd step'] / dense:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
