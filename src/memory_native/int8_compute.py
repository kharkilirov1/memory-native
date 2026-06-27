"""int8 Tensor-Core compute for the counter method (acceleration memo M6).

The three correlations a counter linear needs -- X T^T (forward), Delta T (grad_x), Delta^T X
(update) -- can run on the integer Tensor Cores instead of fp32 scalar paths. The visible weight
T is already int8 in the derived cache (memo M5); this module supplies the activation side:

  * quantize_int8_cols: per-column symmetric int8, STOCHASTIC so it is unbiased (E[scale*q]=x).
  * int8_mm: int8 @ int8 -> int32, via torch._int_mm on CUDA (the Tensor-Core path), fp32 on CPU.
  * int8_correlation: an UNBIASED low-bit estimate of Delta^T X.

Unbiasedness is the whole point: with conditionally-independent stochastic quantizers,
    E[ Q(Delta)^T Q(X) ] = Delta^T X,
so the int GEMM estimates the exact update correlation -- variance, not bias. The counter optimizer
is already a stochastic error-feedback process, so this noise is in-family, not a foreign approx.
The actual Tensor-Core speedup needs a GPU; correctness (unbiasedness + training parity) is checked
on CPU and the CUDA path drops in unchanged.
"""
from __future__ import annotations

import torch

__all__ = ["quantize_int8_cols", "int8_mm", "int8_correlation"]


def quantize_int8_cols(x: torch.Tensor, stochastic: bool = True):
    """Per-column symmetric int8 quantization of [M, D]. Returns (q int8 [M,D], scale [1,D]).
    Stochastic rounding makes it unbiased: E[scale * q] = x."""
    amax = x.abs().amax(dim=0, keepdim=True).clamp_min(1e-12)
    scale = amax / 127.0
    u = x / scale
    if stochastic:
        fl = torch.floor(u)
        q = fl + (torch.rand_like(u) < (u - fl)).to(u.dtype)
    else:
        q = torch.round(u)
    return q.clamp_(-127, 127).to(torch.int8), scale


def int8_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a[M,K] int8 @ b[K,N] int8 -> int32[M,N]. torch._int_mm on CUDA (Tensor Cores); a plain
    int32 matmul on CPU (correct, just not accelerated -- the GPU path is the point)."""
    if a.is_cuda and hasattr(torch, "_int_mm"):
        try:
            return torch._int_mm(a, b)
        except Exception:  # shape/arch constraints -> fall back, stay correct
            pass
    return a.to(torch.int32) @ b.to(torch.int32)


def int8_correlation(delta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Unbiased low-bit estimate of the update correlation G = delta^T x via int8 GEMM.
    delta [M,N], x [M,K] -> G [N,K]. Both sides are stochastically quantized per output/input
    column, so E[G] = delta^T x (the per-column scales factor out as an outer product epilogue)."""
    dq, da = quantize_int8_cols(delta)            # da [1,N]
    xq, xa = quantize_int8_cols(x)                # xa [1,K]
    acc = int8_mm(dq.t().contiguous(), xq)        # [N,K] int32
    return acc.to(torch.float32) * da.t() * xa    # * [N,1] * [1,K]
