#!/usr/bin/env python
"""Multi-token LUT-GEMM scaling sweep (research harness, CUDA required).

Compares, for ``x[M, K] @ W.T`` on a strict ternary packed layer (``residual_alpha=0``,
no salient):

  A  dense fp32        ``x @ W.t()`` with the materialized fp32 weight (cuBLAS)
  A' dense bf16        same with bf16 operands (tensor cores; separate accuracy class)
  B  fused decode      in-repo decode-in-GEMM Triton kernel (no materialization)
  C  LUT total         index build + LUT build + gather (what a loop pays per step)
  C' LUT amortized     gather only, with the weight-side index table prebuilt
  C'' LUT build only   the preprocessing half, reported separately

Every timed path uses CUDA events with warmup, ``iters`` per rep and ``reps`` reps; the
reported number is the median of the per-rep means and the IQR is reported too. Peak
allocated memory is measured per path with reset/reset-peak. First-call (JIT) cost is
measured once and reported separately, never inside the steady-state number.

Writes ``results.json`` and ``results.csv`` (one row per shape/M/path) and prints a compact
summary for each (shape, M).
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory_native.group_scale_kernels import triton_group_decode_matmul  # noqa: E402
from memory_native.group_scale_packed import PackedGroupScaleCounterLinear  # noqa: E402
from memory_native.lut_scaling import (  # noqa: E402
    HAS_TRITON, lut_build, lut_gather_multi, lut_matmul_multi, ternary_index_build,
)


def bench_events(fn, *, warmup: int = 10, iters: int = 50, reps: int = 5) -> dict:
    """CUDA-event timing: warmup, then reps x iters, median of per-rep means."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    per_rep = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        per_rep.append(start.elapsed_time(end) / iters)
    per_rep.sort()
    q1 = per_rep[len(per_rep) // 4] if len(per_rep) > 1 else per_rep[0]
    q3 = per_rep[(3 * len(per_rep)) // 4] if len(per_rep) > 1 else per_rep[0]
    return {"ms": statistics.median(per_rep), "iqr_ms": q3 - q1,
            "ms_reps": [round(v, 6) for v in per_rep]}


def first_call_ms(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


def peak_mem_mb(fn, *, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return max(peak - base, 0) / 2 ** 20


def make_layer(shape: int, group: int, cbits: int):
    from memory_native.packed import pack_codes

    n = k = shape
    torch.manual_seed(0)
    layer = PackedGroupScaleCounterLinear(k, n, group=group, C=cbits,
                                          kernel_mode="triton", lr=1e-3).cuda()
    levels = 3 * (2 * cbits - 1)
    layer.state.copy_(pack_codes(torch.randint(0, levels, (n, k), dtype=torch.uint8,
                                               device="cuda")))
    layer.scale.copy_(torch.rand(n, k // group, device="cuda") * 0.02 + 0.005)
    return layer


def rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    denom = max(float(ref.abs().max()), 1e-12)
    return float((got - ref).abs().max() / denom)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shapes", default="1024,2048,4096")
    ap.add_argument("--batches", default="1,2,4,8,16,32,64,128,256,512")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--C", type=int, default=11)
    ap.add_argument("--block-n", type=int, default=128, help="LUT gather row block")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out-dir", default=str(ROOT / "research/results/lut_scaling_2026_10_17"))
    args = ap.parse_args(argv)

    if not torch.cuda.is_available() or not HAS_TRITON:
        print("needs CUDA + Triton; nothing to measure here")
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    meta = {
        "device": device,
        "vram_gib": round(props.total_memory / 2 ** 30, 2),
        "sm_count": props.multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": __import__("triton").__version__,
        "commit": __import__("subprocess").run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "group": args.group, "C": args.C, "block_n": args.block_n,
        "iters": args.iters, "reps": args.reps,
        "semantics": "strict ternary (residual_alpha=0), no salient channel",
    }
    rows = []
    shaped = [int(s) for s in args.shapes.split(",") if s]
    batches = [int(b) for b in args.batches.split(",") if b]

    for shape in shaped:
        layer = make_layer(shape, args.group, args.C)
        w_fp32 = layer.visible_weight().contiguous()
        w_bf16 = w_fp32.bfloat16()
        packed_bytes = layer.state.numel()
        dense_fp32_bytes = w_fp32.numel() * 4
        dense_bf16_bytes = w_bf16.numel() * 2
        idx = ternary_index_build(layer.state, layer.in_features, C=args.C)

        for m in batches:
            torch.manual_seed(1234 + m)
            x = torch.randn(m, shape, device="cuda", dtype=torch.float32)
            ref = x @ w_fp32.t()
            ref_max = max(float(ref.abs().max()), 1e-12)
            x_perm = x[:, layer.perm.long()].contiguous()

            lut = lut_build(x_perm)
            lut_bytes = lut.numel() * 4

            def dense_fp32():
                return x @ w_fp32.t()

            def dense_bf16():
                return (x.bfloat16() @ w_bf16.t()).float()

            def fused_decode():
                return triton_group_decode_matmul(
                    x.contiguous(), layer.state, layer.scale, layer.perm,
                    C=args.C, group=args.group, residual_alpha=0.0)

            def lut_total():
                return lut_matmul_multi(x, layer.state, layer.scale, layer.perm,
                                        group=args.group, C=args.C, block_n=args.block_n)

            def lut_amortized():
                return lut_gather_multi(lut, idx, layer.scale, group=args.group,
                                        block_n=args.block_n)

            def lut_cached_idx():
                # realistic serving: weight-side index table cached once, LUT rebuilt for
                # every new input token batch, then the gather
                fresh_lut = lut_build(x_perm)
                return lut_gather_multi(fresh_lut, idx, layer.scale, group=args.group,
                                        block_n=args.block_n)

            def lut_prep_only():
                ternary_index_build(layer.state, layer.in_features, C=args.C)
                return lut_build(x_perm)

            got_fp32 = dense_fp32()
            got_bf16 = dense_bf16()
            got_fused = fused_decode()
            got_lut = lut_total()

            paths = [
                ("dense_fp32", dense_fp32, dense_fp32_bytes, rel_err(got_fp32, ref)),
                ("dense_bf16", dense_bf16, dense_bf16_bytes, rel_err(got_bf16, ref)),
                ("fused_decode", fused_decode, packed_bytes, rel_err(got_fused, ref)),
                ("lut_total", lut_total, packed_bytes, rel_err(got_lut, ref)),
                ("lut_amortized", lut_amortized, packed_bytes, rel_err(got_lut, ref)),
                ("lut_cached_idx", lut_cached_idx, packed_bytes, rel_err(got_lut, ref)),
                ("lut_prep_only", lut_prep_only, packed_bytes + lut_bytes, None),
            ]
            for name, fn, nbytes, err in paths:
                first = first_call_ms(fn)
                stats = bench_events(fn, iters=args.iters, reps=args.reps)
                mem = peak_mem_mb(fn)
                gbps = nbytes / (stats["ms"] / 1e3) / 1e9
                row = {
                    "shape": shape, "M": m, "path": name,
                    "ms": stats["ms"], "iqr_ms": stats["iqr_ms"],
                    "first_call_ms": first,
                    "peak_mem_mb": mem,
                    "bytes_streamed": nbytes,
                    "gbps": gbps,
                    "max_rel_err": err,
                    "lut_buf_bytes": lut_bytes,
                    "idx_buf_bytes": idx.numel() * idx.element_size(),
                    "out_buf_bytes": m * shape * 4,
                }
                rows.append(row)
                err_s = "n/a" if err is None else f"{err:.2e}"
                print(f"  {shape:5d} M={m:4d} {name:14s} {stats['ms']:9.4f} ms "
                      f"(iqr {stats['iqr_ms']:7.4f})  {gbps:8.1f} GB/s  "
                      f"peak {mem:7.1f} MiB  err {err_s}")
            with open(out_dir / "results.json", "w") as fh:
                json.dump({"meta": meta, "rows": rows}, fh, indent=1)
            with open(out_dir / "results.csv", "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    # crossover summary: LUT amortized vs the best dense/fused alternative per M
    by_key = {}
    for row in rows:
        by_key.setdefault((row["shape"], row["M"]), {})[row["path"]] = row["ms"]
    print("\nCROSSOVER (lut_amortized / min(dense_fp32, fused_decode))")
    for (shape, m), vals in sorted(by_key.items()):
        base = min(vals["dense_fp32"], vals["fused_decode"])
        print(f"  {shape}x{shape} M={m:4d}: lut {vals['lut_amortized']:9.4f} ms | "
              f"dense_fp32 {vals['dense_fp32']:9.4f} | fused_decode {vals['fused_decode']:9.4f} "
              f"| ratio {vals['lut_amortized'] / base:6.2f}x")
    print(f"\nartifacts: {out_dir/'results.json'} {out_dir/'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
