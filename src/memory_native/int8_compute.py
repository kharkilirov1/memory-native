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
           "int8_correlation_presaved", "int8_forward_ternary", "fp8_correlation"]


def fp8_correlation(delta: torch.Tensor, x: torch.Tensor,
                    dtype: torch.dtype = torch.float8_e4m3fn) -> torch.Tensor:
    """Lower-precision grad_w correlation G = delta^T x with fp8 (e4m3) operands, fp32 accumulate
    (fusion-plan lever #5, the "fp8 on grad_w" half). delta [M,N], x [M,K] -> G [N,K].

    NOT bit-exact and NOT stochastic -- this is deterministic round-to-nearest in fp8, so it is a
    slightly BIASED low-precision estimate, parity-gated (a loss/accuracy witness is required before
    adoption; see FUSION_PLAN.md). The counter's error-feedback absorbs the small per-step bias.
    Unlike int8 (no exponent -> needs per-column amax), fp8's exponent carries the range; a single
    per-tensor scale maps amax near the fp8 max so the 3-bit mantissa is used well. On CUDA the same
    scaled fp8 operands map to torch._scaled_mm (the Tensor-Core fp8 GEMM, ~2x fp16 / in-family with
    int8); on CPU there is no fp8 GEMM so we cast fp8->fp32 and matmul (captures the fp8 rounding)."""
    fp8_max = 448.0 if dtype == torch.float8_e4m3fn else 57344.0    # e4m3 vs e5m2
    da = delta.abs().amax().clamp_min(1e-12) / fp8_max
    xa = x.abs().amax().clamp_min(1e-12) / fp8_max
    dq = (delta / da).to(dtype)
    xq = (x / xa).to(dtype)
    if delta.is_cuda and hasattr(torch, "_scaled_mm"):
        try:
            acc = torch._scaled_mm(dq.t().contiguous(), xq,
                                   scale_a=da.reshape(1), scale_b=xa.reshape(1),
                                   out_dtype=torch.float32)
            return acc
        except Exception:  # arch/shape constraints -> fall back, stay correct
            pass
    acc = dq.to(torch.float32).t() @ xq.to(torch.float32)          # fp32 accumulate
    return acc * (da * xa)


def _srq(u: torch.Tensor, stochastic: bool, levels: int = 127) -> torch.Tensor:
    if stochastic:
        fl = torch.floor(u)
        q = fl + (torch.rand_like(u) < (u - fl)).to(u.dtype)
    else:
        q = torch.round(u)
    return q.clamp_(-levels, levels).to(torch.int8)


def quantize_int4_cols(x: torch.Tensor, stochastic: bool = True):
    """Per-COLUMN symmetric int4 of [M, D] (values in [-7,7], stored in int8). The update flip only
    needs the SIGN and in-row RANK of G=Delta^T X, not fp32 precision -- so the correlation can run
    in int4 (INT4 IMMA on Turing+, ~4x fp16 / ~2x int8). Unbiased when stochastic."""
    amax = x.abs().amax(dim=0, keepdim=True).clamp_min(1e-12)
    scale = amax / 7.0
    return _srq(x / scale, stochastic, levels=7), scale


def int4_correlation(delta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Unbiased int4 estimate of the update correlation G = delta^T x. On CPU the int4 GEMM is
    emulated with int32 matmul (correctness); on a GPU it maps to the INT4 Tensor Cores."""
    dq, da = quantize_int4_cols(delta)
    xq, xa = quantize_int4_cols(x)
    acc = (dq.t().to(torch.int32) @ xq.to(torch.int32)).to(torch.float32)
    return acc * da.t() * xa


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


def int8_correlation_presaved(delta: torch.Tensor, qx_int8: torch.Tensor,
                              ax_row: torch.Tensor) -> torch.Tensor:
    """G = delta^T X reusing the activation already saved as int8 -- quantize ONLY delta.

    The forward saved X as int8 codes qx (per-TOKEN row scale ax_row, i.e. X_mk ~ ax_m * qx_mk).
    Since m is summed in delta^T X, that per-row scale folds into delta:
        G_ok = sum_m delta_mo (ax_m qx_mk) = sum_m (delta_mo ax_m) qx_mk.
    So scale delta's rows by ax, quantize that per output column, and int8-GEMM against the saved
    qx -- no re-quantization of X (the part that made int8_correlation lose to cuBLAS).
    delta [M,N], qx_int8 [M,K], ax_row [M,1] -> G [N,K].

    Bias note: this is unbiased w.r.t. the SAVED activation X_hat = ax*qx, not the original X --
    only delta is re-randomized per call, qx is frozen. So a SINGLE step's estimate carries the
    forward's activation-quant error (a per-step bias). It is unbiased OVER TRAINING because the
    forward re-saves X stochastically each step (a fresh qx), and the counter's error-feedback
    accumulates the residual. 'Unbiased' here means in-expectation-over-steps, not per-step."""
    dq, db = quantize_int8_cols(delta * ax_row)   # fold row scale into delta, then per-N-col quant
    acc = int8_mm(dq.t().contiguous(), qx_int8)   # [N,K] int32 -- qx is the SAVED int8 activation
    return acc.to(torch.float32) * db.t()         # [N,K] * [N,1]
