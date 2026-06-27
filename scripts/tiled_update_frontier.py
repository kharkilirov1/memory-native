"""Tiled update frontier: cuBLAS gradient tiles + fused counter transition.

This is the practical bridge between the two extremes documented in results/ACCELERATION.md:

  * full grad_w + fused update: fastest, but a full [out,in] gradient is transiently live;
  * strict update-from-IO: no grad_w, but a hand-written GEMM is extremely slow on current Triton.

For a tile size R, this benchmark materializes only [R, in] gradient tiles:

    grad_w_tile = grad_out[:, lo:hi].T @ x
    triton_counter_update(state[lo:hi], scale[lo:hi], v[lo:hi], grad_w_tile)

The repository patch makes PackedRMSCounterLinear._fused_update row-slice aware, so normal
training with tile_rows=R can use the same route.  On CUDA this should find the Pareto frontier:
extra peak memory ~= R * in_features * 4 bytes, while speed approaches the full cuBLAS path for
large enough R.

Run:
    PYTHONPATH=src python scripts/tiled_update_frontier.py --d 2048 --M 4096 --device cuda
"""
from __future__ import annotations

import argparse
import time

import torch

from memory_native.packed import PackedRMSCounterLinear
from memory_native.fused_update import HAS_TRITON, triton_counter_update
from memory_native.update_from_io import HAS_TRITON as HAS_TRITON_IO

try:
    from memory_native.update_from_io import triton_counter_update_from_io
except Exception:  # pragma: no cover
    triton_counter_update_from_io = None


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _bench(fn, device: torch.device, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / iters * 1e3


def _peak_delta(fn, device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    fn()
    _sync(device)
    peak = torch.cuda.max_memory_allocated()
    return max(0, int(peak - before))


def _fmt(n: int) -> str:
    if n <= 0:
        return "n/a"
    units = ["B", "KiB", "MiB", "GiB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024
    return f"{x:.2f} GiB"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=2048, help="in=out feature size")
    ap.add_argument("--M", type=int, default=4096, help="batch*seq rows")
    ap.add_argument("--C", type=int, default=11)
    ap.add_argument("--tiles", default="64,128,256,512,1024,2048")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    dev = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    if dev.type != "cuda":
        print("WARNING: CUDA not available; this script is meant for GPU frontier numbers.")
    if not HAS_TRITON:
        print("WARNING: Triton fused update unavailable; only grad tile GEMM timing is meaningful.")

    d, M, C = args.d, args.M, args.C
    x = torch.randn(M, d, device=dev)
    go = torch.randn(M, d, device=dev)
    lay = PackedRMSCounterLinear(d, d, C=C).to(dev)

    def full_grad_plus_fused():
        gw = go.t() @ x
        if HAS_TRITON and dev.type == "cuda":
            triton_counter_update(lay.state, lay.scale, lay.v, gw.float(), C=C, lr=lay.lr,
                                  lr_scale=lay.lr_scale, rms_beta=lay.rms_beta,
                                  rms_eps=lay.rms_eps, seed=123)

    print(f"\ntiled-update frontier  d={d} M={M} C={C} device={dev}")
    print("shape gradient bytes: full =", _fmt(d * d * 4))
    print("-" * 86)
    print(f"{'path':>18s} {'tile_rows':>10s} {'grad_tile_mem':>14s} {'time_ms':>12s} {'peak_delta':>14s}")
    print("-" * 86)
    full_t = _bench(full_grad_plus_fused, dev, args.iters, args.warmup)
    full_p = _peak_delta(full_grad_plus_fused, dev)
    print(f"{'full-gw+fused':>18s} {d:10d} {_fmt(d*d*4):>14s} {full_t:12.3f} {_fmt(full_p):>14s}")

    for R in [int(x) for x in args.tiles.split(',') if x.strip()]:
        if R <= 0 or R > d:
            continue
        def tiled():
            seed = 1000
            for lo in range(0, d, R):
                hi = min(lo + R, d)
                gw = go[:, lo:hi].t().contiguous() @ x
                if HAS_TRITON and dev.type == "cuda":
                    triton_counter_update(lay.state[lo:hi], lay.scale[lo:hi], lay.v[lo:hi], gw.float(),
                                          C=C, lr=lay.lr, lr_scale=lay.lr_scale,
                                          rms_beta=lay.rms_beta, rms_eps=lay.rms_eps, seed=seed)
                    seed += 1
        t = _bench(tiled, dev, args.iters, args.warmup)
        p = _peak_delta(tiled, dev)
        print(f"{'tile-gw+fused':>18s} {R:10d} {_fmt(R*d*4):>14s} {t:12.3f} {_fmt(p):>14s}")

    if triton_counter_update_from_io is not None and HAS_TRITON_IO and dev.type == "cuda":
        def strict_io():
            triton_counter_update_from_io(lay.state, lay.scale, lay.v, x, go, C=C, lr=lay.lr,
                                          lr_scale=lay.lr_scale, rms_beta=lay.rms_beta,
                                          rms_eps=lay.rms_eps, seed=777)
        t = _bench(strict_io, dev, max(1, args.iters // 10), max(1, args.warmup // 2))
        p = _peak_delta(strict_io, dev)
        print(f"{'strict-from-IO':>18s} {0:10d} {_fmt(0):>14s} {t:12.3f} {_fmt(p):>14s}")
    print("-" * 86)
    print("Use the largest tile_rows that fits the peak-memory budget; it preserves cuBLAS GEMM")
    print("throughput while bounding the transient gradient to tile_rows*in_features*4 bytes.")


if __name__ == "__main__":
    main()
