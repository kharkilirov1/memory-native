"""GPU gate for solver-v3 packed group kernels.

Runs correctness and timing for forward, grad_x and strict update-from-IO. The strict update
reports its bounded O(out*groups) scratch against the dense fp32 grad_w that is deliberately absent.
"""
from __future__ import annotations

import argparse

import torch

from memory_native.counter import decode_state
from memory_native.group_scale_kernels import HAS_TRITON
from memory_native.group_scale_packed import PackedGroupScaleCounterLinear
from memory_native.packed import pack_codes, unpack_codes


def sync_ms(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-features", type=int, default=1536)
    ap.add_argument("--out-features", type=int, default=1536)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    if not torch.cuda.is_available() or not HAS_TRITON:
        raise SystemExit("CUDA + Triton required")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    device = "cuda"
    torch.manual_seed(0)
    k, n, m, group = args.in_features, args.out_features, args.tokens, args.group
    if k % 4 or group % 4:
        raise SystemExit("in-features and group must be divisible by 4")

    perm = torch.randperm(k, device=device)
    layer = PackedGroupScaleCounterLinear(
        k, n, group=group, C=11, perm=perm, residual_alpha=0.35,
        kernel_mode="triton", strict_update=True, local_grad_clip=1.0,
    ).to(device)
    t = torch.randint(-1, 2, (n, k), dtype=torch.int16, device=device)
    c = torch.randint(-10, 11, (n, k), dtype=torch.int16, device=device)
    scales = torch.rand(n, (k + group - 1) // group, device=device) * 0.1 + 0.02
    layer.load_group_state(scales, t, c, perm)
    x = torch.randn(m, k, device=device, dtype=dtype)
    go = torch.randn(m, n, device=device, dtype=dtype)

    layer.eval()
    with torch.no_grad():
        dense_w = layer.visible_weight(dtype=dtype)
        ref_y = x @ dense_w.t()
        got_y = layer(x)
        ref_gx = go @ dense_w
        got_gx = layer._grad_x_2d(go)
    y_err = (got_y.float() - ref_y.float()).abs().max().item()
    gx_err = (got_gx.float() - ref_gx.float()).abs().max().item()

    forward_ms = sync_ms(lambda: layer(x), iters=args.iters)
    gradx_ms = sync_ms(lambda: layer._grad_x_2d(go), iters=args.iters)

    # Stage-0 references (optimization plan L0): what the same math costs via decode + cuBLAS.
    decode_ms = sync_ms(lambda: layer.visible_weight(dtype=dtype), iters=args.iters)
    cublas_fwd_ms = sync_ms(lambda: x @ dense_w.t(), iters=args.iters)
    cublas_gx_ms = sync_ms(lambda: go @ dense_w, iters=args.iters)

    # Update-path timings: strict 3-launch from-IO kernel vs cuBLAS grad_w + GPU reference chain.
    from memory_native.group_scale_kernels import (
        group_counter_update_from_io_hashsr, triton_group_counter_update_from_io,
    )
    state0, scale0, v0 = layer.state.clone(), layer.scale.clone(), layer.v.clone()
    upd_kw = dict(group=group, C=layer.C, lr=layer.lr, lr_scale=layer.lr_scale,
                  rms_beta=layer.rms_beta, rms_eps=layer.rms_eps, seed=0,
                  residual_alpha=layer.residual_alpha, clip=layer.local_grad_clip)

    def strict_update():
        triton_group_counter_update_from_io(
            layer.state, layer.scale, layer.v, x, go, layer.perm, **upd_kw)

    def semi_update():                       # cuBLAS correlation + bit-exact reference on GPU
        codes = unpack_codes(layer.state, k)
        new_codes = group_counter_update_from_io_hashsr(
            codes, layer.scale, layer.v, x, go, layer.perm, **upd_kw)
        layer.state.copy_(pack_codes(new_codes))

    strict_upd_ms = sync_ms(strict_update, iters=args.iters)
    layer.state.copy_(state0); layer.scale.copy_(scale0); layer.v.copy_(v0)
    torch.cuda.reset_peak_memory_stats()
    semi_upd_ms = sync_ms(semi_update, iters=args.iters)
    semi_peak = torch.cuda.max_memory_allocated()
    layer.state.copy_(state0); layer.scale.copy_(scale0); layer.v.copy_(v0)

    # Stage-1 witness: the "gemm" layer mode end-to-end (decode + cuBLAS each call).
    gemm_layer = PackedGroupScaleCounterLinear(
        k, n, group=group, C=11, perm=perm, residual_alpha=0.35,
        kernel_mode="gemm", local_grad_clip=1.0,
    ).to(device)
    gemm_layer.load_group_state(scales, t, c, perm)
    gemm_layer.eval()
    with torch.no_grad():
        gemm_y_err = (gemm_layer(x).float() - ref_y.float()).abs().max().item()
        gemm_gx_err = (gemm_layer._grad_x_2d(go).float() - ref_gx.float()).abs().max().item()
        gemm_fwd_ms = sync_ms(lambda: gemm_layer(x), iters=args.iters)
        gemm_gx_ms = sync_ms(lambda: gemm_layer._grad_x_2d(go), iters=args.iters)

    # One strict update correctness/peak-memory witness. No dense grad_w is built on this path.
    state_before = layer.state.clone()
    torch.cuda.reset_peak_memory_stats()
    layer.train()
    x_train = x.detach().requires_grad_(True)
    y = layer(x_train)
    y.backward(go)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    changed = (layer.state != state_before).any().item()
    codes = unpack_codes(layer.state, k)
    td, cd = decode_state(codes, layer.C)

    dense_grad_bytes = n * k * 4
    scratch = layer.strict_scratch_bytes()
    print(f"shape M={m} N={n} K={k} group={group} dtype={args.dtype}")
    print(f"forward max_abs={y_err:.6g}  {forward_ms:.3f} ms")
    print(f"grad_x max_abs={gx_err:.6g}  {gradx_ms:.3f} ms")
    print(f"[L0] decode(visible_weight)={decode_ms:.3f} ms  cublas_fwd={cublas_fwd_ms:.3f} ms  "
          f"cublas_grad_x={cublas_gx_ms:.3f} ms")
    print(f"[L0] torch-path fwd(decode+cublas)={decode_ms + cublas_fwd_ms:.3f} ms  "
          f"triton/cublas fwd ratio={forward_ms / max(cublas_fwd_ms, 1e-9):.1f}x")
    print(f"[L0] update strict(3-launch from-IO)={strict_upd_ms:.3f} ms  "
          f"semi(cublas grad_w + reference)={semi_upd_ms:.3f} ms  "
          f"semi peak={semi_peak / 2**20:.1f} MiB")
    print(f"[L1] gemm-mode fwd={gemm_fwd_ms:.3f} ms (max_abs={gemm_y_err:.6g})  "
          f"grad_x={gemm_gx_ms:.3f} ms (max_abs={gemm_gx_err:.6g})  "
          f"speedup vs triton: fwd={forward_ms / max(gemm_fwd_ms, 1e-9):.1f}x "
          f"grad_x={gradx_ms / max(gemm_gx_ms, 1e-9):.1f}x")
    print(f"strict update changed_state={changed} finite={torch.isfinite(layer.scale).all().item()}")
    print(f"strict scratch={scratch / 2**20:.3f} MiB vs dense grad_w={dense_grad_bytes / 2**20:.3f} MiB")
    print(f"measured peak allocated={peak / 2**20:.3f} MiB")
    print(f"state range t=[{int(td.min())},{int(td.max())}] c=[{int(cd.min())},{int(cd.max())}]")


if __name__ == "__main__":
    main()
