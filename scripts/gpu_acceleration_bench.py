"""GPU acceleration benchmark — the whole stack on one device (acceleration memo).

Measures, on CUDA, the levers that need real hardware: the int8 derived-cache forward (decode vs
cache vs Tensor-Core _int_mm), the int8 vs fp32 update correlation, the fused update kernel, fused
QKV, and the reversible-anchor memory/speed frontier. Run on a T4:

    python scripts/gpu_acceleration_bench.py

Results are directional (one GPU). See results/ACCELERATION.md for the recorded T4 numbers.
"""
from __future__ import annotations

import time

import torch

from memory_native import ReversibleGPT, GPTConfig, fmt_bytes, peak_training_memory
from memory_native.packed import PackedRMSCounterLinear
from memory_native.int8_compute import int8_correlation, int8_forward_ternary
from memory_native.fused_update import HAS_TRITON, triton_counter_update


def bench(fn, it=50, wu=5):
    for _ in range(wu):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(it):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / it * 1e3


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu")
    d, M, C = 2048, 4096, 11
    torch.set_grad_enabled(False)
    lay = PackedRMSCounterLinear(d, d, C=C).to(dev)
    layc = PackedRMSCounterLinear(d, d, C=C, cache_mode="int8").to(dev)
    layc.state.copy_(lay.state); layc.scale.copy_(lay.scale); layc._build_t_cache()
    x = torch.randn(M, d, device=dev); go = torch.randn(M, d, device=dev)
    Ti8 = layc._t_cache

    print("\n== M5 forward: decode vs int8-cache ==")
    print(f"  decode + GEMM       {bench(lambda: lay._forward_matmul(x)):.3f} ms")
    print(f"  int8 cache + GEMM   {bench(lambda: layc._forward_matmul(x)):.3f} ms")
    print(f"  int8 forward (_int_mm, row-scale) {bench(lambda: int8_forward_ternary(x, Ti8)):.3f} ms")

    print("\n== M6 update correlation: fp32 vs int8 ==")
    print(f"  fp32  go^T @ x      {bench(lambda: go.t() @ x):.3f} ms")
    print(f"  int8 correlation    {bench(lambda: int8_correlation(go, x)):.3f} ms")

    print("\n== update kernel ==")
    gw = (go.t() @ x).float(); t_i, c_i = lay._decode_rows(0, d)
    print(f"  torch tile          {bench(lambda: lay._update_tile(0, d, gw, t_i, c_i, lay.scale[0:d])):.3f} ms")
    if HAS_TRITON and dev.type == "cuda":
        kw = dict(C=C, lr=lay.lr, lr_scale=lay.lr_scale, rms_beta=lay.rms_beta, rms_eps=lay.rms_eps, seed=1)
        print(f"  fused triton        {bench(lambda: triton_counter_update(lay.state, lay.scale, lay.v, gw, **kw)):.3f} ms")

    print("\n== M7 reversible anchors (d=512, 24L, B*T=2048) ==")
    torch.set_grad_enabled(True)
    cfg = GPTConfig(256, 128, 24, 8, 512)
    idx = torch.randint(0, 256, (16, 128), device=dev); tgt = torch.randint(0, 256, (16, 128), device=dev)
    for A in (0, 8):
        m = ReversibleGPT(cfg, "counter_packed", anchor_every=A, C=C, act_save_bits=4).to(dev).train()
        step = lambda: m(idx, tgt)[1].backward()
        pk = peak_training_memory(step, dev)
        print(f"  {'O(1)' if A == 0 else f'anchor={A}':10s} peak {fmt_bytes(pk):>10s}  {bench(step, it=10, wu=2):.1f} ms/step")


if __name__ == "__main__":
    main()
