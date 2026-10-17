#!/usr/bin/env python
"""Full-step cost and memory decomposition for dense AdamW vs Memory Native paths.

Answers "where does the step time actually go" instead of quoting one number:

  dense_bf16 / dense_fp32        nn.Linear + AdamW (real grads, real update)
  packed_fused                   kernel_mode="triton", no weight cache: decode-in-GEMM
                                 forward and grad_x, fused counter update (no [out,in] fp32)
  packed_decode_once             kernel_mode="gemm", no weight cache: one transient fp32
                                 decode per step, then cuBLAS matmuls
  packed_cache_g               kernel_mode="gemm", group-scope update, persistent fp32
                                 weight cache (the fast-but-memory-hungry path)
  packed_cache_g_dec4/8          same with decimated updates (different algorithm!)

Per-phase CUDA-event timing (warmup, iters, reps, median + IQR), plus memory split:

  persistent    resident tensors of the layer/model while stepping (params, buffers,
                packed state, scale/v, and the fp32 weight cache when enabled)
  live_temp     peak_allocated - persistent (activations, grads, transient decode buffer)
  peak_alloc    torch.cuda.max_memory_allocated() delta over the step
  peak_reserved torch.cuda.max_memory_reserved() delta (allocator footprint)

Shapes are (out_features, in_features); tokens are rows per step. Strict ternary
(alpha=0), no salient. Every variant is measured on identical inputs and fp32 activations;
bf16 is reported as its own precision class, never mixed into the fp32 comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory_native.group_scale_packed import PackedGroupScaleCounterLinear  # noqa: E402
from memory_native.packed import pack_codes  # noqa: E402

SHAPES = [(896, 896), (4864, 896), (896, 4864), (1024, 1024), (2048, 2048), (4096, 4096)]
TOKENS = [64, 128, 256, 512, 1024]
CACHE = 1 << 30


def bench_events(fn, *, warmup: int = 8, iters: int = 20, reps: int = 5) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    per_rep = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        per_rep.append(s.elapsed_time(e) / iters)
    per_rep.sort()
    return {"ms": statistics.median(per_rep),
            "iqr_ms": per_rep[(3 * len(per_rep)) // 4] - per_rep[len(per_rep) // 4]}


def tensor_bytes(t) -> int:
    return t.numel() * t.element_size() if t is not None else 0


def persistent_bytes(layer=None, lin=None, opt=None) -> int:
    total = 0
    if lin is not None:
        total += sum(tensor_bytes(p) for p in lin.parameters())
        total += sum(tensor_bytes(b) for b in lin.buffers())
        if opt is not None:
            for st in opt.state.values():
                total += sum(tensor_bytes(v) for v in st.values() if torch.is_tensor(v))
    if layer is not None:
        total += sum(tensor_bytes(p) for p in layer.parameters())
        total += sum(tensor_bytes(b) for b in layer.buffers())
        if getattr(layer, "_dense_cache", None) is not None:
            total += tensor_bytes(layer._dense_cache)
        if getattr(layer, "_weight_cache", None) is not None:
            total += tensor_bytes(layer._weight_cache[1])
    return total


def memory_profile(fn, persistent: int, *, warmup: int = 3) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base_alloc = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated() - base_alloc
    peak_reserved = torch.cuda.max_memory_reserved() - base_alloc
    live = max(peak_alloc - persistent, 0)
    return {"persistent_mb": persistent / 2 ** 20,
            "live_temp_mb": live / 2 ** 20,
            "peak_alloc_mb": peak_alloc / 2 ** 20,
            "peak_reserved_mb": peak_reserved / 2 ** 20}


def make_dense(out_f: int, in_f: int, dtype=torch.float32):
    torch.manual_seed(0)
    lin = nn.Linear(in_f, out_f, bias=False, dtype=dtype).cuda()
    with torch.no_grad():
        lin.weight.mul_(0.02)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-3)
    return lin, opt


def make_counter(out_f: int, in_f: int, *, kernel_mode: str, stats_scope: str,
                 decimation: int, cache_bytes: int):
    torch.manual_seed(0)
    layer = PackedGroupScaleCounterLinear(
        in_f, out_f, group=128, C=11, kernel_mode=kernel_mode, lr=1e-3,
        stats_scope=stats_scope, decimation=decimation,
        weight_cache_bytes=cache_bytes).cuda()
    levels = 3 * (2 * 11 - 1)
    layer.state.copy_(pack_codes(torch.randint(0, levels, (out_f, in_f),
                                               dtype=torch.uint8, device="cuda")))
    layer.scale.copy_(torch.rand(out_f, in_f // 128, device="cuda") * 0.02 + 0.005)
    return layer


def measure_dense(out_f: int, in_f: int, tokens: int, flags: str) -> list[dict]:
    for dtype, tag in ((torch.float32, "dense_fp32"), (torch.bfloat16, "dense_bf16")):
        lin, opt = make_dense(out_f, in_f, dtype)
        torch.manual_seed(99 + tokens)
        x = torch.randn(tokens, in_f, device="cuda", dtype=dtype)
        go = torch.randn(tokens, out_f, device="cuda", dtype=dtype)
        rows = []

        def fwd():
            with torch.no_grad():
                lin(x)

        def bwd():
            lin(x).backward(go)

        def step():
            opt.zero_grad(set_to_none=True)
            lin(x).backward(go)
            opt.step()

        w = lin.weight
        w.grad = torch.randn_like(w)
        opt_step = lambda: opt.step()
        phases = {"fwd": bench_events(fwd), "bwd": bench_events(bwd),
                  "opt_step": bench_events(opt_step), "full_step": bench_events(step)}
        del w.grad
        persistent = persistent_bytes(lin=lin, opt=opt)
        mem = memory_profile(step, persistent)
        for phase, st in phases.items():
            rows.append({"variant": tag, "out_f": out_f, "in_f": in_f, "tokens": tokens,
                         "phase": phase, **st, **mem, "flags": flags})
        return rows


def measure_counter(out_f: int, in_f: int, tokens: int, variant: str,
                    kernel_mode: str, stats_scope: str, decimation: int,
                    cache_bytes: int) -> list[dict]:
    layer = make_counter(out_f, in_f, kernel_mode=kernel_mode, stats_scope=stats_scope,
                         decimation=decimation, cache_bytes=cache_bytes)
    torch.manual_seed(99 + tokens)
    x = torch.randn(tokens, in_f, device="cuda", dtype=torch.float32)
    go = torch.randn(tokens, out_f, device="cuda", dtype=torch.float32)
    rows = []

    def fwd():
        with torch.no_grad():
            layer(x)

    def step():
        y = layer(x)
        y.backward(go)

    # attribution: forward matrix path, grad_x path, updater and any transient decode
    def fwd_inner():
        with torch.no_grad():
            layer._forward_2d(x)

    def gradx_inner():
        with torch.no_grad():
            layer._grad_x_2d(go)

    def update_inner():
        with torch.no_grad():
            layer._update_from_io(x, go)

    def decode_inner():
        with torch.no_grad():
            layer.visible_weight(torch.float32)

    phases = {"fwd": bench_events(fwd), "full_step": bench_events(step),
              "fwd_2d": bench_events(fwd_inner), "grad_x_2d": bench_events(gradx_inner),
              "update_only": bench_events(update_inner)}
    if kernel_mode in ("gemm", "auto") or kernel_mode == "hybrid":
        phases["decode"] = bench_events(decode_inner)
    persistent = persistent_bytes(layer=layer)
    mem = memory_profile(step, persistent)
    for phase, st in phases.items():
        rows.append({"variant": variant, "out_f": out_f, "in_f": in_f, "tokens": tokens,
                     "phase": phase, **st, **mem, "flags": f"km={kernel_mode},scope={stats_scope},dec={decimation},cache={cache_bytes}"})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shapes", default="896x896,4864x896,896x4864,1024x1024,2048x2048,4096x4096")
    ap.add_argument("--tokens", default="64,128,256,512,1024")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out-dir", default=str(ROOT / "research/results/compute_bottleneck_2026_10_17"))
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        print("needs CUDA")
        return 0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shapes = []
    for spec in args.shapes.split(","):
        out_s, in_s = spec.lower().split("x")
        shapes.append((int(out_s), int(in_s)))
    tokens = [int(t) for t in args.tokens.split(",") if t]

    rows = []
    for out_f, in_f in shapes:
        for ntok in tokens:
            print(f"== {out_f}x{in_f} tokens={ntok}")
            rows += measure_dense(out_f, in_f, ntok, "fp32/bf16, real grads+AdamW")
            rows += measure_counter(out_f, in_f, ntok, "packed_fused", "triton", "row", 1, 0)
            rows += measure_counter(out_f, in_f, ntok, "packed_decode_once", "gemm", "row", 1, 0)
            rows += measure_counter(out_f, in_f, ntok, "packed_cache_g", "gemm", "group", 1, CACHE)
            rows += measure_counter(out_f, in_f, ntok, "packed_cache_g_dec4", "gemm", "group", 4, CACHE)
            rows += measure_counter(out_f, in_f, ntok, "packed_cache_g_dec8", "gemm", "group", 8, CACHE)
            # flush the artifacts after every shape/token block
            with open(out_dir / "step_breakdown.json", "w") as fh:
                json.dump({"device": torch.cuda.get_device_name(0),
                           "torch": torch.__version__, "cuda": torch.version.cuda,
                           "commit": __import__("subprocess").run(
                               ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                               capture_output=True, text=True).stdout.strip(),
                           "iters": args.iters, "reps": args.reps,
                           "semantics": "strict ternary alpha=0, no salient",
                           "rows": rows}, fh, indent=1)
            with open(out_dir / "step_breakdown.csv", "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"   rows so far: {len(rows)}")
    print(f"artifacts: {out_dir/'step_breakdown.json'} {out_dir/'step_breakdown.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
