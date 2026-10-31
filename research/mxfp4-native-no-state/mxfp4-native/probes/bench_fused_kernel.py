#!/usr/bin/env python
"""GPU microbench for the fused decode+matmul kernel (fail-closed on parity).

For layers 1024->4096 and 4096->1024 and M in {1, 32, 128, 512} times (median of
CUDA-event-timed iterations after warmup):

  dense_bf16     torch.matmul(x_bf16, W_bf16.T)           (weights already dense)
  unfused_bf16   decode -> bf16 -> torch.matmul           (decode counted)
  fused_fp32     packed_matmul(compute_dtype="fp32-ieee") (old-kernel arithmetic;
                 tiles are now autotuned, so not the literal old 64/64/64 launch)
  fused_tf32     packed_matmul(compute_dtype="tf32")
  fused_bf16     packed_matmul(compute_dtype="bf16")      (new default)

Before timing, every fused mode is checked against the torch reference (same
bounds as tests/test_triton_kernel.py); any failure aborts with exit code 1.

Usage (Colab / any CUDA box):
    python probes/bench_fused_kernel.py [--fmt NA6|NA5] [--iters 100] [--warmup 20]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mx6 import triton_kernel as tk                              # noqa: E402
from mx6.packed_na import PackedNA5, PackedNA6                   # noqa: E402

MODES = ["fp32-ieee", "tf32", "bf16"]


def _bf16r(t):
    return t.to(torch.bfloat16).to(torch.float32)


def parity(out, a, W, mode, tag):
    """W: reference so ref = a @ W. Fail-closed."""
    absb = a.abs() @ W.abs()
    if mode == "fp32-ieee":
        ok = torch.allclose(out, a @ W, atol=1e-4, rtol=1e-4)
    elif mode == "bf16":
        ok = (torch.allclose(out, _bf16r(a) @ W, atol=1e-4, rtol=1e-4)
              and bool(((out - a @ W).abs() <= 2.0 ** -8 * absb + 1e-4).all()))
    else:
        ok = bool(((out - a @ W).abs() <= 2.0 ** -10 * absb + 1e-4).all())
    if not ok:
        err = (out - a @ W).abs().max().item()
        print(f"PARITY FAIL {tag} mode={mode} max|err|={err:.3e}", file=sys.stderr)
        sys.exit(1)


def timeit(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmt", default="NA6", choices=["NA5", "NA6"])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--ms", type=int, nargs="+", default=[1, 32, 128, 512])
    args = ap.parse_args()

    if not torch.cuda.is_available() or not tk.HAS_TRITON:
        print("needs CUDA + Triton", file=sys.stderr)
        sys.exit(2)
    import triton
    fmt = PackedNA6 if args.fmt == "NA6" else PackedNA5
    dev = "cuda"
    print(f"device: {torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)}")
    print(f"torch {torch.__version__} | triton {triton.__version__} | fmt {args.fmt}")
    lut = fmt.code_to_value.to(dev)

    rows = []
    for (K, N) in [(1024, 4096), (4096, 1024)]:
        torch.manual_seed(0)
        w = torch.randn(N, K, device=dev) * 0.05
        codes, sb = fmt.quantize(w, headroom=2.0)
        W32 = fmt.dequantize(codes, sb)                  # [N, K] fp32
        Wbf = W32.to(torch.bfloat16)
        assert torch.equal(Wbf.float(), W32), "decoded weight not exact in bf16"
        Wt = W32.t().contiguous()
        for M in args.ms:
            x = torch.randn(M, K, device=dev)
            xbf = x.to(torch.bfloat16)
            tag = f"{K}->{N} M={M}"
            res = {}
            for mode in MODES:
                out = tk.packed_matmul(x, codes, sb, lut, compute_dtype=mode)
                parity(out, x, Wt, mode, tag)
                res[mode] = out
            cfg = tk.best_config("fwd")
            cfg_s = (f"BM{cfg.kwargs['BLOCK_M']} BN{cfg.kwargs['BLOCK_N']} "
                     f"BK{cfg.kwargs['BLOCK_K']} w{cfg.num_warps} s{cfg.num_stages}"
                     if cfg is not None else "n/a")  # last launched = bf16 mode
            t_dense = timeit(lambda: torch.matmul(xbf, Wbf.t()), args.warmup, args.iters)
            t_unf = timeit(lambda: torch.matmul(
                xbf, fmt.dequantize(codes, sb).to(torch.bfloat16).t()), args.warmup, args.iters)
            t = {m: timeit(lambda m=m: tk.packed_matmul(x, codes, sb, lut, compute_dtype=m),
                           args.warmup, args.iters) for m in MODES}
            rows.append((f"{K}->{N}", M, t_dense, t_unf, t["fp32-ieee"], t["tf32"],
                         t["bf16"], cfg_s))

    print("\nparity: all fused modes passed before timing (fail-closed)\n")
    print("| layer | M | dense bf16 ms | unfused ms | fused fp32 ms | fused tf32 ms "
          "| fused bf16 ms | bf16 vs dense | bf16 vs unfused | bf16 autotune cfg |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for layer, M, td, tu, tf, tt, tb, cfg_s in rows:
        print(f"| {layer} | {M} | {td:.4f} | {tu:.4f} | {tf:.4f} | {tt:.4f} | {tb:.4f} "
              f"| {td / tb:.2f}x | {tu / tb:.2f}x | {cfg_s} |")
    print("\n('x' columns = speed of the fused bf16 kernel relative to that baseline; "
          ">1 means fused is faster.)")


if __name__ == "__main__":
    main()
