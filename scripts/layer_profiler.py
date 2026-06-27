"""Layer profiler — the truth table for counter-linear acceleration (acceleration memo M1).

Isolates one PackedRMSCounterLinear and times each phase so speed work targets the real wall,
not a guess: forward (decode+GEMM), the pure GEMMs (fwd / grad_x / grad_w correlation), the
packed decode and pack, the counter update (torch tile vs fused Triton kernel), and the
activation quantize. On CUDA it uses cuda events + synchronize; on CPU it uses perf_counter
(absolute numbers only meaningful on GPU, but the relative breakdown is informative anywhere).

    python scripts/layer_profiler.py --d 2048 --batch 4096 --device cuda
"""
from __future__ import annotations

import argparse
import time

import torch

from memory_native.actquant import stochastic_quantize
from memory_native.packed import PackedRMSCounterLinear, pack_codes, unpack_codes
from memory_native.counter import decode_state


def _bench(fn, device, iters=50, warmup=5):
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=2048, help="in=out feature size")
    ap.add_argument("--batch", type=int, default=4096, help="M = batch*seq (rows)")
    ap.add_argument("--C", type=int, default=11)
    ap.add_argument("--bits", type=int, default=4, help="activation save bits")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    dev = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    M, K, N = args.batch, args.d, args.d
    lay = PackedRMSCounterLinear(K, N, C=args.C).to(dev)
    x = torch.randn(M, K, device=dev)
    go = torch.randn(M, N, device=dev)
    W = lay._dense_weight(x.dtype)                       # precomputed dense visible weight

    torch.set_grad_enabled(False)                       # profile raw compute, not autograd bookkeeping
    rows = []
    rows.append(("forward (decode+GEMM)", _bench(lambda: lay._forward_matmul(x), dev)))
    lay_c = PackedRMSCounterLinear(K, N, C=args.C, cache_mode="int8").to(dev)
    lay_c.state.copy_(lay.state); lay_c.scale.copy_(lay.scale); lay_c._build_t_cache()
    rows.append(("forward (int8 cache, no decode)", _bench(lambda: lay_c._forward_matmul(x), dev)))
    rows.append(("pure GEMM fwd  x@W^T", _bench(lambda: x @ W.t(), dev)))
    rows.append(("pure GEMM grad_x  go@W", _bench(lambda: go @ W, dev)))
    rows.append(("pure GEMM grad_w  go^T@x", _bench(lambda: go.t() @ x, dev)))
    rows.append(("decode unpack+state", _bench(lambda: decode_state(unpack_codes(lay.state, K), args.C), dev)))
    rows.append(("pack codes", _bench(lambda: pack_codes(unpack_codes(lay.state, K)), dev)))
    rows.append((f"act quant int{args.bits}", _bench(lambda: stochastic_quantize(x, args.bits), dev)))

    t_i, c_i = lay._decode_rows(0, N)
    gw = (go.t() @ x).float()
    rows.append(("update torch tile", _bench(lambda: lay._update_tile(0, N, gw, t_i, c_i, lay.scale[0:N]), dev)))

    try:
        from memory_native.fused_update import HAS_TRITON, triton_counter_update
        if HAS_TRITON and dev.type == "cuda":
            kw = dict(C=args.C, lr=lay.lr, lr_scale=lay.lr_scale, rms_beta=lay.rms_beta,
                      rms_eps=lay.rms_eps, seed=1)
            rows.append(("update fused triton", _bench(
                lambda: triton_counter_update(lay.state, lay.scale, lay.v, gw, **kw), dev)))
    except Exception as exc:  # pragma: no cover
        rows.append(("update fused triton", float("nan")))
        print("  (triton update unavailable:", exc, ")")

    print(f"\nlayer profiler  d={args.d} M={args.batch} C={args.C}  device={dev}")
    print("-" * 44)
    for name, ms in rows:
        print(f"  {name:26s} {ms:8.3f} ms")
    print("-" * 44)
    print("Read: if forward ~ pure GEMM fwd, decode is hidden by the GEMM; if the gap is large,")
    print("the decode tax is real -> a derived visible cache (memo M5) removes it. grad_w is the")
    print("correlation the strict update-from-IO kernel must form in registers (no dense grad_w).")


if __name__ == "__main__":
    main()
