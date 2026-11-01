#!/usr/bin/env python
"""GPU benchmark: matmuls with deployed NVFP4 weights vs dense bf16 cuBLAS (fail-closed on parity).

For 1B-class shapes, M tokens in {4096, 16384}, (N, K) in {(2048,2048), (8192,2048), (2048,8192), (4096,4096),
(16384,4096)}: times (median of CUDA-event iterations, output bf16)
    fwd  y  = x @ W^T          dx  dx = dy @ W
  dense_bf16        torch.matmul on the deployed weight in dense bf16 (cuBLAS; the bar to beat)
  triton_bf16       nibbles -> bf16 in registers, bf16 tensor cores         (exact x)
  dot_scaled        tl.dot_scaled(bf16 x, e2m1 W)                            (exact x; dx: re-quantized W^T)
  dot_scaled_fp4    tl.dot_scaled(e2m1 x, e2m1 W)                            (x quantized: APPROXIMATION)
  torch_scaled_mm   torch._scaled_mm NVFP4 (cuBLASLt)                        (x quantized: APPROXIMATION)
fp4-activation backends are timed three times: "gemm" (x pre-quantized), "+quant(eager)" (the eager-torch quantizer
`quantize_nvfp4` included, ~15 elementwise kernels) and "+quant(fused)" (the Triton one-pass `quantize_nvfp4_fused`
included: nibbles + block scales + swizzled scales, bit-identical to the eager one, checked before timing). Extra rows:
op "quant" = the quantizer alone on x [M, K] (ms, GB/s of bf16 read), op "Wt" = building the W^T operand of the dx GEMM
(dequantize + transpose + re-quantize; weights change every step, so this is a per-step cost of the nvfp4 dx path),
eager vs fused. Activation-quantization error (relative RMS of the NVFP4 round trip of x and dy) is printed per shape.
TFLOPS = 2*M*N*K / time.  Before timing, each backend is checked on 256 rows against the float64 expected result
(same tolerances as tests/test_fp4_gemm.py); a usable backend that fails parity aborts with exit code 1.

Usage (Colab / any CUDA box):   (add --no-e2e to skip the layer table; --shapes "2048x2048,8192x2048" for fewer shapes)
    python probes/bench_fp4_gemm.py [--iters 50] [--warmup 10] [--Ms 4096 16384] [--backends ...] [--no-dx]
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mx6 import fp4_gemm as fg                                    # noqa: E402
from mx6.latent_grid import LatentGridLinear                      # noqa: E402

SHAPES = [(2048, 2048), (8192, 2048), (2048, 8192), (4096, 4096), (16384, 4096)]     # (N, K)
TOL = {"triton_bf16": 1e-4, "dot_scaled": 1e-4, "dot_scaled_fp4": 1e-2, "torch_scaled_mm": 1e-2}
PARITY_ROWS = 256


def timeit(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def rel(a, b):
    return ((a.double() - b.double()).abs().max() / b.double().abs().max()).item()


def parity(backend, q, n, k, dx):
    g = torch.Generator(device="cpu").manual_seed(1)
    x = torch.randn(PARITY_ROWS, k, generator=g).cuda()
    r = rel(fg.fp4_matmul(x, q, backend), fg.expected_matmul(x, q, backend))
    out = [("fwd", r)]
    if dx:
        d = torch.randn(PARITY_ROWS, n, generator=g).cuda()
        out.append(("dx", rel(fg.fp4_matmul_dx(d, q, backend), fg.expected_dx(d, q, backend))))
    return out


E2E_SHAPES = [(2048, 2048), (8192, 2048), (2048, 8192)]          # (N, K)
E2E_M = 16384


def e2e(n, k, m, iters, warmup, gemms=("cublas_bf16", "nvfp4_scaled_mm")):
    """End-to-end ONE layer step: LatentGridLinear (native bf16: decode + fwd GEMM + bwd GEMMs + fused update)
    vs nn.Linear bf16 (fwd + bwd + fused AdamW). Rows: ms for the whole step and for fwd only. Returns list of rows."""
    out = []
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    w0 = torch.randn(n, k) * 0.3

    def run_dense(master_fp32):
        lin = torch.nn.Linear(k, n, bias=False, device="cuda", dtype=torch.float32 if master_fp32 else torch.bfloat16)
        with torch.no_grad():
            lin.weight.copy_(w0)
        opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, fused=True)

        def fwd():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=master_fp32):
                return lin(x)

        def step():
            y = fwd()
            y.backward(dy)
            x.grad = None
            opt.step()
            opt.zero_grad(set_to_none=True)
        return step, fwd

    for name, master in (("nn.Linear autocast(fp32 master)+fusedAdamW", True), ("nn.Linear bf16 weights+fusedAdamW", False)):
        step, fwd = run_dense(master)
        with torch.no_grad():
            tf = timeit(fwd, iters, warmup)
        out.append((name, timeit(step, iters, warmup), tf))
        del step, fwd
    base = out[0][1]
    for g in gemms:
        lg = LatentGridLinear.from_dense(w0, k=8, lr=1e-3, backend="triton").cuda().train()
        lg.gemm = g
        if g != "cublas_bf16" and not fg.backend_status("torch_scaled_mm")[0]:
            print(f"  [{n}x{k}] lg gemm={g}: torch_scaled_mm unavailable, skipped")
            continue

        def step():
            y = lg(x)
            y.backward(dy)
            x.grad = None

        def fwd_only():
            with torch.no_grad():
                return lg(x)
        try:
            ts = timeit(step, iters, warmup)
            tf = timeit(fwd_only, iters, warmup)
        except Exception as e:                                    # noqa: BLE001
            print(f"  [{n}x{k}] lg gemm={g}: {type(e).__name__}: {str(e)[:200]}")
            continue
        gw = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        tu = timeit(lambda: lg.apply_update_(gw), iters, warmup)
        td = timeit(lambda: lg.decode_weight(torch.bfloat16), iters, warmup)
        out.append((f"LatentGridLinear native gemm={g}" + (" (APPROX)" if g != "cublas_bf16" else ""), ts, tf))
        out.append((f"   its update kernel alone / decode alone", tu, td))
        del lg
    return out, base


def run_e2e(args):
    print("\n=== end-to-end layer step (fwd + bwd + weight update), M=%d ===" % E2E_M)
    print(f"{'N':>6} {'K':>6} {'variant':<48} {'step ms':>9} {'fwd ms':>9} {'vs nn.Linear fp32-master':>25}")
    for (n, k) in E2E_SHAPES:
        rows, base = e2e(n, k, E2E_M, max(10, args.iters // 2), args.warmup)
        for name, ts, tf in rows:
            tag = f"{ts / base:.2f}x" if not name.startswith("   ") else "(update ms, decode ms)"
            print(f"{n:6d} {k:6d} {name:<48} {ts:9.3f} {tf:9.3f} {tag:>25}")
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-e2e", action="store_true", help="skip the end-to-end layer table")
    ap.add_argument("--e2e-only", action="store_true", help="only the end-to-end layer table")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--Ms", type=int, nargs="+", default=[4096, 16384])
    ap.add_argument("--backends", nargs="+", default=list(fg.BACKENDS[1:]))
    ap.add_argument("--no-dx", action="store_true")
    ap.add_argument("--shapes", type=str, default=None, help='e.g. "2048x2048,4096x4096" as NxK')
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("no CUDA device: nothing to benchmark")
        return 2
    shapes = SHAPES if not args.shapes else [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",")]
    try:
        import triton
        tv = triton.__version__
    except Exception:
        tv = "none"
    print(f"torch {torch.__version__}  triton {tv}  {torch.cuda.get_device_name()}  "
          f"sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}")
    if args.e2e_only:
        run_e2e(args)
        return 0
    use = []
    for b in args.backends:
        ok, why = fg.backend_status(b)
        print(f"  backend {b:16s} {'AVAILABLE' if ok else 'UNSUPPORTED: ' + why}")
        if ok:
            use.append(b)
    failed = False
    rows = []
    dx_on = not args.no_dx
    for (n, k) in shapes:
        torch.manual_seed(0)
        # build on CPU then move: from_dense copies into the layer's buffers, which live on the layer's device (the old
        # code gave a CUDA weight to a CPU layer -> packed weights on CPU -> "Pointer argument cannot be accessed")
        layer = LatentGridLinear.from_dense(torch.randn(n, k) * 0.3, k=8).cuda()
        q = fg.pack_fp4(layer).to("cuda")
        assert q.device.type == "cuda" and q.nibbles.is_cuda and q.scales.is_cuda and q.tensor_scale.is_cuda
        del layer
        wd = q.dense_bf16()
        ok_b = []
        for b in use:
            try:
                res = parity(b, q, n, k, dx_on)
            except Exception as e:                                # noqa: BLE001
                print(f"[N={n} K={k}] {b}: runtime failure at this shape: {type(e).__name__}: {str(e)[:200]}")
                continue
            bad = [(nm, r) for nm, r in res if not r < TOL[b]]
            print(f"[N={n} K={k}] parity {b}: " + ", ".join(f"{nm} {r:.2e}" for nm, r in res)
                  + ("   <-- FAIL" if bad else ""))
            if bad:
                failed = True
            else:
                ok_b.append(b)
        for m in args.Ms:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            dy = torch.randn(m, n, device="cuda", dtype=torch.bfloat16) if dx_on else None
            qx = fg.quantize_nvfp4(x) if any(b in fg.FP4_ACT_BACKENDS for b in ok_b) else None
            qd = fg.quantize_nvfp4(dy) if (dx_on and qx is not None) else None
            fused_ok = False
            if qx is not None:
                try:
                    for src, ref in ((x, qx), (dy, qd)):
                        if src is None:
                            continue
                        f = fg.quantize_nvfp4_fused(src)
                        assert torch.equal(f.nibbles, ref.nibbles) and torch.equal(f.scales, ref.scales) \
                            and torch.equal(f.tensor_scale, ref.tensor_scale), "fused != eager quantizer"
                        assert torch.equal(f.swizzled_scales().view(torch.uint8), ref.swizzled_scales().view(torch.uint8))
                    fused_ok = True
                except Exception as e:                            # noqa: BLE001
                    print(f"  [{n}x{k} M={m}] fused quantizer unavailable / mismatch: {type(e).__name__}: {str(e)[:200]}")
                    failed = True
                ex = fg.act_quant_error(x)
                msg = f"  [N={n} K={k} M={m}] activation NVFP4 error x: rel_rms {ex['rel_rms']:.4f} rel_max {ex['rel_max']:.4f}"
                if dy is not None:
                    ed = fg.act_quant_error(dy)
                    msg += f" | dy: rel_rms {ed['rel_rms']:.4f} rel_max {ed['rel_max']:.4f}"
                print(msg + f"   (fused quantizer: {'bit-exact vs eager' if fused_ok else 'NOT USED'})")
            if qx is not None:
                qx.swizzled_scales()
            if qd is not None:
                qd.swizzled_scales()
            flops = 2.0 * m * n * k
            ops = [("fwd", lambda: torch.matmul(x, wd.t()), x, qx,
                    lambda b, xx: (lambda: fg.fp4_matmul(xx, q, b, torch.bfloat16)))]
            if dx_on:
                ops.append(("dx", lambda: torch.matmul(dy, wd), dy, qd,
                            lambda b, xx: (lambda: fg.fp4_matmul_dx(xx, q, b, torch.bfloat16))))
            for op, dense_fn, inp, qinp, mk in ops:
                base = timeit(dense_fn, args.iters, args.warmup)
                rows.append((n, k, m, op, "dense_bf16", "", base, flops / base / 1e9, 1.0))
                for b in ok_b:
                    variants = [("", inp)]
                    if b in fg.FP4_ACT_BACKENDS:
                        variants = [("gemm", qinp), ("+quant(eager)", inp)] + ([("+quant(fused)", inp)] if fused_ok else [])
                    for tag, xx in variants:
                        try:
                            if tag == "+quant(eager)":
                                fn = (lambda b=b, xx=xx, mk=mk: mk(b, fg.quantize_nvfp4(xx))())
                            elif tag == "+quant(fused)":
                                fn = (lambda b=b, xx=xx, mk=mk: mk(b, fg.quantize_nvfp4_fused(xx))())
                            else:
                                fn = mk(b, xx)
                            fn()                                  # compile + autotune outside the timing
                            t = timeit(fn, args.iters, args.warmup)
                        except Exception as e:                    # noqa: BLE001
                            print(f"  [{n}x{k} M={m} {op}] {b} {tag}: {type(e).__name__}: {str(e)[:160]}")
                            continue
                        rows.append((n, k, m, op, b, tag, t, flops / t / 1e9, base / t))
            if qx is not None:
                gb = x.numel() * 2 / 1e9
                te = timeit(lambda: fg.quantize_nvfp4(x), args.iters, args.warmup)
                rows.append((n, k, m, "quant", "eager", "x", te, gb / te * 1e3, 1.0))
                if fused_ok:
                    tf_ = timeit(lambda: fg.quantize_nvfp4_fused(x), args.iters, args.warmup)
                    rows.append((n, k, m, "quant", "fused", "x", tf_, gb / tf_ * 1e3, te / tf_))
                if m == args.Ms[0] and dx_on:
                    wdq = q.dequantize()
                    gbw = n * k * 4 / 1e9
                    tw = timeit(lambda: fg.quantize_nvfp4(q.dequantize().t().contiguous()), args.iters, args.warmup)
                    rows.append((n, k, m, "Wt", "eager", "", tw, gbw / tw * 1e3, 1.0))
                    if fused_ok:
                        twf = timeit(lambda: fg.quantize_nvfp4_fused(q.dequantize().t().contiguous()), args.iters, args.warmup)
                        rows.append((n, k, m, "Wt", "fused", "", twf, gbw / twf * 1e3, tw / twf))
                    del wdq
            del x, dy, qx, qd
            torch.cuda.empty_cache()
        del q, wd
        torch.cuda.empty_cache()
    print(f"\n{'N':>6} {'K':>6} {'M':>6} {'op':>5} {'backend':>16} {'mode':>14} {'ms':>8} {'TFLOPS|GB/s':>12} {'vs ref':>8}"
          "   (ref = cuBLAS bf16 for fwd/dx; eager quantizer for quant/Wt; quant/Wt columns are GB/s)")
    for (n, k, m, op, b, tag, t, tf, sp) in rows:
        print(f"{n:6d} {k:6d} {m:6d} {op:>5} {b:>16} {tag:>14} {t:8.3f} {tf:12.1f} {sp:7.2f}x")
    for nm, kern in (("fwd", getattr(fg, "_fwd_bf16_kernel", None)), ("dx", getattr(fg, "_dx_bf16_kernel", None)),
                     ("dot_scaled", getattr(fg, "_dot_scaled_kernel", None))):
        cfg = getattr(kern, "best_config", None)
        if cfg is not None:
            print(f"last autotune winner {nm}: {cfg}")
    if not args.no_e2e:
        run_e2e(args)
    if failed:
        print("PARITY FAILURE: results above must not be trusted")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
