#!/usr/bin/env python
"""Does LUT-GEMM beat the existing decode paths? Run on a CUDA device.

Compares three ways to compute ``x @ W.T`` for a group-ternary packed layer:

  gemm    decode the whole weight to fp32, then cuBLAS  (memory: 4 B/param transient)
  decode  the in-repo fused decode-in-GEMM Triton kernel (no materialization)
  lut     Method-of-Four-Russians: 81 partial sums per (token, 4-block) + gather-add

Reports ms and the effective packed-state bandwidth, so the number can be read against the
roofline instead of in a vacuum: for a 4096x4096 layer the packed state is 12.6 MiB, which
is ~7 us at 1.8 TB/s. A path that lands far above that is ALU/occupancy-bound, which is the
only regime where LUT can win.

    python scripts/bench_lut_gemm.py                 # 4096x4096, M=1
    python scripts/bench_lut_gemm.py --shape 2048 --block-n 128
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memory_native.group_scale_packed import PackedGroupScaleCounterLinear  # noqa: E402
from memory_native.lut_gemm import HAS_TRITON, triton_lut_group_matmul  # noqa: E402


def _bench(fn, threads: int, run_time: float) -> float:
    from torch.utils.benchmark import Timer

    return Timer(stmt="fn()", globals={"fn": fn}, num_threads=threads).blocked_autorange(
        min_run_time=run_time).median * 1000.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shape", type=int, default=4096, help="square layer size N=K")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--C", type=int, default=11)
    ap.add_argument("--block-n", type=int, default=64)
    ap.add_argument("--run-time", type=float, default=2.0)
    args = ap.parse_args(argv)

    if not torch.cuda.is_available() or not HAS_TRITON:
        print("needs CUDA + Triton; nothing to measure here")
        return 0

    n = k = args.shape
    torch.manual_seed(0)
    layer = PackedGroupScaleCounterLinear(k, n, group=args.group, C=args.C,
                                          kernel_mode="triton", lr=1e-3).cuda()
    x = torch.randn(1, k, device="cuda")
    packed_bytes = layer.state.numel()
    gbps = lambda ms: packed_bytes / (ms / 1000) / 1e9

    # reference for the LUT path (exact fp32 weight, original order)
    w = layer.visible_weight()
    ref = (x @ w.t())[0]

    rows = {}

    def gemm():
        layer.visible_weight(dtype=torch.float32)
    rows["gemm decode-once"] = _bench(lambda: x @ layer.visible_weight().t(), 10, args.run_time)

    from memory_native.group_scale_kernels import triton_group_decode_matmul
    rows["decode-in-GEMM"] = _bench(
        lambda: triton_group_decode_matmul(x.contiguous(), layer.state, layer.scale,
                                           layer.perm, C=args.C, group=args.group,
                                           residual_alpha=0.0), 10, args.run_time)

    def lut():
        return triton_lut_group_matmul(x, layer.state, layer.scale, layer.perm,
                                       group=args.group, C=args.C, block_n=args.block_n)
    got = lut()
    err = float((got - ref).abs().max() / ref.abs().max())
    rows["LUT (build+gather)"] = _bench(lut, 10, args.run_time)

    print(f"layer {n}x{k} group={args.group} C={args.C} M=1  packed state {packed_bytes/2**20:.2f} MiB")
    print(f"LUT vs exact weight: max rel err {err:.2e}")
    for name, ms in rows.items():
        print(f"  {name:20s} {ms:8.3f} ms   {gbps(ms):6.1f} GB/s of packed state")
    floor_us = packed_bytes / 1.8e12 * 1e6
    print(f"  {'bandwidth floor @1.8TB/s':20s} {floor_us/1000:8.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
