#!/usr/bin/env python
"""Fair GPU A/B: counter training step vs dense Linear + AdamW, phase by phase.

The project's speed target is "counter step <= dense AdamW step". This harness measures
both sides at the same shape/batch and splits the gap so a change is aimed at the right
term:

    dense   : fwd | grad_x | grad_w | AdamW step      + full step
    counter : fwd | grad_x | update (correlation+automaton) + full step

Both full steps include their state update (dense `opt.step()`, counter
`_update_from_io`), which the older CPU witness did not do for the counter side.

    python scripts/bench_vs_adam.py --shape 1536 --batch 64 --mode all
    python scripts/bench_vs_adam.py --shape 4096 --batch 512 --mode gemm
    python scripts/bench_vs_adam.py --shape 1536 --batch 64 --mode hybrid --stats-scope group
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.benchmark import Timer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memory_native import PackedGroupScaleCounterLinear  # noqa: E402


def bench(fn, run_time: float) -> float:
    return Timer(stmt="fn()", globals={"fn": fn}).blocked_autorange(
        min_run_time=run_time).median * 1e3


def _dense(in_features: int, out_features: int, batch: int, dev: str, run_time: float) -> dict:
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(out_features, in_features, device=dev) * 0.02)
    opt = torch.optim.AdamW([w], lr=1e-3)
    x = torch.randn(batch, in_features, device=dev)
    go = torch.randn(batch, out_features, device=dev)

    def step():
        opt.zero_grad(set_to_none=True)
        (x @ w.t()).backward(go)
        opt.step()

    out = {
        "fwd": bench(lambda: x @ w.t(), run_time),
        "grad_x": bench(lambda: go @ w, run_time),
        "grad_w": bench(lambda: go.t() @ x, run_time),
        "state update": bench(lambda: opt.step(), run_time),
        "full step": bench(step, run_time),
    }
    out["_bytes"] = w.numel() * 4
    return out


def _counter(in_features: int, out_features: int, batch: int, dev: str,
             mode: str, group: int, C: int, run_time: float,
             stats_scope: str = "row", cache_bytes: int = 0,
             decimation: int = 1) -> dict:
    torch.manual_seed(0)
    layer = PackedGroupScaleCounterLinear(
        in_features, out_features, group=group, C=C,
        kernel_mode=mode, lr=1e-3, stats_scope=stats_scope,
        weight_cache_bytes=cache_bytes, decimation=decimation).to(dev)
    x = torch.randn(batch, in_features, device=dev)
    go = torch.randn(batch, out_features, device=dev)

    def full():
        y = layer(x)
        y.backward(go)
        layer._update_from_io(x, go)

    out = {
        "fwd": bench(lambda: layer._forward_2d(x), run_time),
        "grad_x": bench(lambda: layer._grad_x_2d(go), run_time),
        "state update": bench(lambda: layer._update_from_io(x, go), run_time),
        "full step": bench(full, run_time),
    }
    out["_bytes"] = layer.state.numel()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shape", type=int, default=1536)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--C", type=int, default=11)
    ap.add_argument("--mode", choices=["gemm", "triton", "hybrid", "both", "all"], default="all")
    ap.add_argument("--stats-scope", choices=["row", "group"], default="row")
    ap.add_argument("--decimation", type=int, default=1,
                    help="update every group every k steps (group scope only)")
    ap.add_argument("--cache-bytes", type=int, default=0,
                    help="per-layer decoded-weight cache cap (0 = off, e.g. 64<<20)")
    ap.add_argument("--run-time", type=float, default=2.0)
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        print("needs CUDA")
        return 0
    dev = "cuda"
    I = O = args.shape
    B = args.batch
    print(f"layer {O}x{I}  batch {B}  group {args.group}  C={args.C}  "
          f"stats={args.stats_scope}  dec={args.decimation}  "
          f"({torch.cuda.get_device_name(0)})")

    dense = _dense(I, O, B, dev, args.run_time)
    modes = {"both": ["gemm", "triton"], "all": ["gemm", "triton", "hybrid"]}.get(
        args.mode, [args.mode])
    counter = {
        m: _counter(I, O, B, dev, m, args.group, args.C, args.run_time,
                    args.stats_scope, args.cache_bytes, args.decimation)
        for m in modes
    }

    phases = ["fwd", "grad_x", "state update", "full step"]
    print(f"{'phase':24s} {'dense':>13s} " + " ".join(f"{m:>16s}" for m in modes))
    for phase in phases:
        line = f"  {phase:22s} {dense[phase]:10.3f} ms"
        for m in modes:
            line += f" {counter[m][phase]:9.3f} ms {counter[m][phase] / dense[phase]:5.2f}x"
        print(line)

    for m in modes:
        total, base = counter[m]["full step"], dense["full step"]
        print(f"\n  [{m}] overhead vs dense {total - base:+.3f} ms ({total / base:.2f}x)")
        if m in {"gemm", "hybrid"}:
            decode = (counter[m]["fwd"] - dense["fwd"]) + (counter[m]["grad_x"] - dense["grad_x"])
            autom = counter[m]["state update"] - dense["state update"]
            print(f"     decode-in-fwd/grad_x {decode:+.3f} ms | update-vs-AdamW {autom:+.3f} ms")
    print(f"\n  packed state {counter[modes[0]]['_bytes']/2**20:.2f} MiB vs dense weights "
          f"{dense['_bytes']/2**20:.2f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
