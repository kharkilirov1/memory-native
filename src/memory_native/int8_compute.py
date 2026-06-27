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

__all__ = ["quantize_int8_cols", "quantize_int8_rows", "int8_mm", "int8_correlation",
           "int8_forward_ternary"]


def _srq(u: torch.Tensor, stochastic: bool) -> torch.Tensor:
    if stochastic:
        fl = torch.floor(u)
        q = fl + (torch.rand_like(u) < (u - fl)).to(u.dtype)
    else:
        q = torch.round(u)
    return q.clamp_(-127, 127).to(torch.int8)


def quantize_int8_cols(x: torch.Tensor, stochastic: bool = True):
    """Per-COLUMN symmetric int8 of [M, D]. Returns (q int8 [M,D], scale [1,D]). Unbiased when
    stochastic: E[scale * q] = x. This is the right scaling for the UPDATE correlation Delta^T X,
    where the per-output-row and per-input-column scales factor out as an outer product."""
    amax = x.abs().amax(dim=0, keepdim=True).clamp_min(1e-12)
    scale = amax / 127.0
    return _srq(x / scale, stochastic), scale


def quantize_int8_rows(x: torch.Tensor, stochastic: bool = True):
    """Per-ROW (per-token) symmetric int8 of [M, D]. Returns (q int8 [M,D], scale [M,1]). This is
    the right scaling for the FORWARD Y = X T^T: the per-token scale a_m factors OUT of the sum
    over k (Y_mo = a_m * sum_k q_mk T_ok), which a per-column scale would not."""
    amax = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    scale = amax / 127.0
    return _srq(x / scale, stochastic), scale


def int8_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a[M,K] int8 @ b[K,N] int8 -> int32[M,N]. torch._int_mm on CUDA (Tensor Cores); a plain
    int32 matmul on CPU (correct, just not accelerated -- the GPU path is the point)."""
    if a.is_cuda and hasattr(torch, "_int_mm"):
        try:
            return torch._int_mm(a, b)
        except Exception:  # shape/arch constraints -> fall back, stay correct
            pass
    return a.to(torch.int32) @ b.to(torch.int32)


def int8_forward_ternary(x: torch.Tensor, t_int8: torch.Tensor, stochastic: bool = True) -> torch.Tensor:
    """Y_unscaled = (a_x * q_x) T^T for the int8 forward, with the CORRECT per-token (row) scale.
    x [M,K] fp, t_int8 [N,K] int8 (the visible ternary cache). Returns [M,N] fp; multiply by the
    per-output row scale s_o outside (the GEMM epilogue) to get Y = X (diag(s) T)^T. Using a
    per-column X scale here would be wrong -- it cannot be pulled out of the sum over k.

    stochastic=False uses round-to-nearest: deterministic (so F/G stay reversible-safe and the
    eager contract holds) at the cost of a small quantization bias -- the right tradeoff for the
    forward. stochastic=True is unbiased but non-deterministic; use it for the update correlation."""
    xq, ax = quantize_int8_rows(x, stochastic=stochastic)   # ax [M,1]
    acc = int8_mm(xq, t_int8.t().contiguous())              # [M,N] int32
    return acc.to(torch.float32) * ax                       # per-token scale factors out


def int8_correlation(delta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Unbiased low-bit estimate of the update correlation G = delta^T x via int8 GEMM.
    delta [M,N], x [M,K] -> G [N,K]. Both sides are stochastically quantized per output/input
    column, so E[G] = delta^T x (the per-column scales factor out as an outer product epilogue)."""
    dq, da = quantize_int8_cols(delta)            # da [1,N]
    xq, xa = quantize_int8_cols(x)                # xa [1,K]
    acc = int8_mm(dq.t().contiguous(), xq)        # [N,K] int32
    return acc.to(torch.float32) * da.t() * xa    # * [N,1] * [1,K]
