#!/usr/bin/env python
"""CUDA-graph replay timing for the LUT-GEMM scaling story (research harness).

The eager harness (``bench_lut_scaling.py``) charges every per-call cost: Python
dispatch, Triton launch argument marshalling, allocator hits. In this environment a
single small Triton launch costs tens of microseconds, which dominates every path at
M <= 32 and hides the actual GPU work. A CUDA graph captures the whole pipeline once and
replays it as one launch, so the replay number is the GPU time plus one launch.

Both numbers are honest and answer different questions:

  eager  = what a training/eval loop pays per call today;
  graph  = the kernel-side ceiling (what an inference engine with graphs would pay).

Only paths with fixed input/output tensors can be captured (inputs are allocated before
capture, as a serving engine would). Strict ternary semantics (alpha=0, no salient).
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory_native.group_scale_kernels import triton_group_decode_matmul  # noqa: E402
from memory_native.group_scale_packed import PackedGroupScaleCounterLinear  # noqa: E402
from memory_native.lut_scaling import (  # noqa: E402
    HAS_TRITON, lut_build, lut_gather_multi, ternary_index_build,
)


def graph_ms(fn, *, warmup: int = 15, iters: int = 50, reps: int = 7) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        fn()
    per_rep = []
    for _ in range(reps):
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        for _ in range(iters):
            graph.replay()
        ev1.record()
        torch.cuda.synchronize()
        per_rep.append(ev0.elapsed_time(ev1) / iters)
    per_rep.sort()
    return {"graph_ms": statistics.median(per_rep),
            "graph_iqr_ms": per_rep[(3 * len(per_rep)) // 4] - per_rep[len(per_rep) // 4]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shapes", default="1024,2048,4096")
    ap.add_argument("--batches", default="1,2,4,8,16,32,64,128,256,512")
    ap.add_argument("--block-n", type=int, default=128)
    ap.add_argument("--out-dir", default=str(ROOT / "research/results/lut_scaling_2026_10_17"))
    args = ap.parse_args(argv)

    if not (torch.cuda.is_available() and HAS_TRITON):
        print("needs CUDA + Triton")
        return 0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for shape in [int(s) for s in args.shapes.split(",") if s]:
        n = k = shape
        torch.manual_seed(0)
        layer = PackedGroupScaleCounterLinear(k, n, group=128, C=11,
                                              kernel_mode="triton", lr=1e-3).cuda()
        from memory_native.packed import pack_codes
        layer.state.copy_(pack_codes(torch.randint(0, 63, (n, k), dtype=torch.uint8,
                                                   device="cuda")))
        layer.scale.copy_(torch.rand(n, k // 128, device="cuda") * 0.02 + 0.005)
        w = layer.visible_weight().contiguous()
        w_bf16 = w.bfloat16()
        idx = ternary_index_build(layer.state, k, C=11)

        for m in [int(b) for b in args.batches.split(",") if b]:
            torch.manual_seed(99 + m)
            x = torch.randn(m, k, device="cuda")
            xp = x[:, layer.perm.long()].contiguous()
            lut = lut_build(xp)

            def dense_fp32():
                return x @ w.t()

            def dense_bf16():
                return (x.bfloat16() @ w_bf16.t()).float()

            def fused_decode():
                return triton_group_decode_matmul(x, layer.state, layer.scale, layer.perm,
                                                  C=11, group=128, residual_alpha=0.0)

            def lut_gather():
                return lut_gather_multi(lut, idx, layer.scale, group=128,
                                        block_n=args.block_n)

            def lut_full():
                ii = ternary_index_build(layer.state, k, C=11)
                ll = lut_build(xp)
                return lut_gather_multi(ll, ii, layer.scale, group=128,
                                        block_n=args.block_n)

            for name, fn in (("dense_fp32", dense_fp32), ("dense_bf16", dense_bf16),
                             ("fused_decode", fused_decode), ("lut_gather", lut_gather),
                             ("lut_full", lut_full)):
                st = graph_ms(fn)
                rows.append({"shape": shape, "M": m, "path": name, **st})
                print(f"  {shape:5d} M={m:4d} {name:13s} graph {st['graph_ms']:8.4f} ms "
                      f"(iqr {st['graph_iqr_ms']:7.4f})")

    with open(out_dir / "graph_results.json", "w") as fh:
        json.dump({"device": torch.cuda.get_device_name(0),
                   "torch": torch.__version__, "cuda": torch.version.cuda,
                   "rows": rows}, fh, indent=1)
    with open(out_dir / "graph_results.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nartifacts: {out_dir/'graph_results.json'} {out_dir/'graph_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
