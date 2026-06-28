"""Triton packed kernel — the bandwidth/peak piece of "real sub-byte training".

PackedRMSCounterLinear already stores state at 0.75 byte/weight, but its forward still decodes
to a dense fp32 weight before the matmul, so a dense [out,in] weight is transiently resident.
This module fuses the decode INTO the GEMM: a Triton kernel reads the packed 6-bit state and
the per-row scale, reconstructs each weight tile in registers, and accumulates y = x @ W^T
without ever materializing the dense weight. That removes the transient dense weight from the
forward and reads 0.75 byte/weight from HBM instead of 4.

Scope / status (verified on a Tesla T4 -- see results/SUMMARY.md, results/KERNEL.md):
  * Two kernels run in-GEMM with no dense weight: the FORWARD (y = x W^T) and the backward
    grad_x (grad_x = grad_out W), both decoding packed state in registers. T4-VERIFIED for
    correctness (forward err <= 3e-6, grad_x err <= 1e-5 vs the dense decode). HOWEVER they are
    net-negative in practice: a hand-written Triton matmul loses to torch-decode + cuBLAS and the
    peak is activation-bound, not weight-bound, so these two kernels buy no real memory/speed win.
    They are kept as a reference decode-in-GEMM implementation, not a recommended path.
  * The counter UPDATE: the one kernel that does pay off. memory_native.fused_update collapses
    the per-element RMS+stochastic-rounding update into one launch -- T4-VERIFIED against a CPU
    reference to within one SR quantum (chunked fp reduction, not bit-exact; the training dynamics
    match) and benchmarked at x45.9 on the update / x1.26 on the full step; it is
    wired into PackedRMSCounterLinear._fused_update. It still CONSUMES a materialized grad_w tile,
    though. The strict-memory analogue of the engine's OpenCL counter_*_fused -- an update kernel
    that takes (state, scale, v, x_or_Q(x), grad_out) and forms grad_w in registers so no dense
    gradient is ever materialized -- is the remaining open milestone.
  * When triton or CUDA is unavailable, TritonCounterLinear transparently falls back to the
    plain PackedRMSCounterLinear forward, so the class is safe to use anywhere.
"""
from __future__ import annotations

import torch

from .packed import PackedRMSCounterLinear

__all__ = ["HAS_TRITON", "triton_decode_matmul", "triton_grad_x", "TritonCounterLinear"]

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover - depends on env
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _decode_matmul_kernel(
        x_ptr, state_ptr, scale_ptr, y_ptr,
        M, N, K, C,
        stride_xm, stride_xk,
        stride_sn,  # state rows stride in bytes (= (K//4)*3)
        stride_ym, stride_yn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        # Computes Y[M,N] = X[M,K] @ W[N,K]^T where W[n,k] = scale[n] * t(state[n,k]).
        # state is packed 6-bit: 4 codes per 3 bytes along K. We require BLOCK_K % 4 == 0.
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        lv = 2 * C - 1
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0,
            )  # [BM, BK]

            # decode W tile [BN, BK]: for each (n, k), find packed byte group and 6-bit code.
            group = offs_k // 4                  # which 3-byte group along K
            lane = offs_k % 4                    # which code in the group
            base = offs_n[:, None] * stride_sn + group[None, :] * 3  # byte offset of group
            b0 = tl.load(state_ptr + base + 0,
                         mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0).to(tl.int32)
            b1 = tl.load(state_ptr + base + 1,
                         mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0).to(tl.int32)
            b2 = tl.load(state_ptr + base + 2,
                         mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0).to(tl.int32)
            code0 = b0 & 0x3F
            code1 = ((b0 >> 6) | (b1 << 2)) & 0x3F
            code2 = ((b1 >> 4) | (b2 << 4)) & 0x3F
            code3 = (b2 >> 2) & 0x3F
            ln = lane[None, :]
            code = tl.where(ln == 0, code0, tl.where(ln == 1, code1, tl.where(ln == 2, code2, code3)))
            t = code // lv - 1                   # ternary in {-1,0,1}
            scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=0.0)  # [BN]
            w = (t.to(tl.float32)) * scale[:, None]   # [BN, BK]

            acc += tl.dot(x, tl.trans(w))         # [BM,BK] x [BK,BN]

        y = acc.to(tl.float32)
        tl.store(
            y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
            y, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


if HAS_TRITON:

    @triton.jit
    def _decode_gradx_kernel(
        go_ptr, state_ptr, scale_ptr, gx_ptr,
        M, N, K, C,
        stride_gom, stride_gon,
        stride_sn,
        stride_gxm, stride_gxk,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        # grad_x[M,K] = grad_out[M,N] @ W[N,K], W[n,k] = scale[n]*t(state[n,k]); contract over N.
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        lv = 2 * C - 1
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

        for n0 in range(0, N, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            go = tl.load(
                go_ptr + offs_m[:, None] * stride_gom + offs_n[None, :] * stride_gon,
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0,
            )  # [BM, BN]
            # decode W tile [BN, BK]
            group = offs_k // 4
            lane = offs_k % 4
            base = offs_n[:, None] * stride_sn + group[None, :] * 3
            m_nk = (offs_n[:, None] < N) & (offs_k[None, :] < K)
            b0 = tl.load(state_ptr + base + 0, mask=m_nk, other=0).to(tl.int32)
            b1 = tl.load(state_ptr + base + 1, mask=m_nk, other=0).to(tl.int32)
            b2 = tl.load(state_ptr + base + 2, mask=m_nk, other=0).to(tl.int32)
            code0 = b0 & 0x3F
            code1 = ((b0 >> 6) | (b1 << 2)) & 0x3F
            code2 = ((b1 >> 4) | (b2 << 4)) & 0x3F
            code3 = (b2 >> 2) & 0x3F
            ln = lane[None, :]
            code = tl.where(ln == 0, code0, tl.where(ln == 1, code1, tl.where(ln == 2, code2, code3)))
            t = code // lv - 1
            scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=0.0)  # [BN]
            w = t.to(tl.float32) * scale[:, None]  # [BN, BK]
            acc += tl.dot(go, w)  # [BM,BN] x [BN,BK] -> [BM,BK]

        tl.store(
            gx_ptr + offs_m[:, None] * stride_gxm + offs_k[None, :] * stride_gxk,
            acc, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
        )


def triton_grad_x(grad_out: torch.Tensor, state: torch.Tensor, scale: torch.Tensor,
                  C: int, in_features: int, out_features: int) -> torch.Tensor:
    """grad_x = grad_out @ W with W decoded from packed 6-bit `state` in-kernel (no dense W)."""
    if not HAS_TRITON:
        raise RuntimeError("triton not available")
    assert grad_out.is_cuda and state.is_cuda, "triton_grad_x requires CUDA tensors"
    M, N = grad_out.shape
    K = in_features
    assert N == out_features and K % 4 == 0
    scale = scale.reshape(N).contiguous()
    grad_out = grad_out.contiguous()
    state = state.contiguous()
    gx = torch.empty((M, K), device=grad_out.device, dtype=torch.float32)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
    _decode_gradx_kernel[grid](
        grad_out, state, scale, gx,
        M, N, K, C,
        grad_out.stride(0), grad_out.stride(1),
        state.stride(0),
        gx.stride(0), gx.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return gx


def triton_decode_matmul(x: torch.Tensor, state: torch.Tensor, scale: torch.Tensor,
                         C: int, in_features: int, out_features: int) -> torch.Tensor:
    """y = x @ W^T with W decoded from packed 6-bit `state` inside the kernel (no dense W).

    x: [M, K] f32 (K = in_features), state: packed uint8 [N, (K//4)*3] (N = out_features),
    scale: [N] or [N,1] f32. Requires CUDA + triton; raises otherwise.
    """
    if not HAS_TRITON:
        raise RuntimeError("triton not available")
    assert x.is_cuda and state.is_cuda, "triton_decode_matmul requires CUDA tensors"
    M, K = x.shape
    N = out_features
    assert K == in_features and K % 4 == 0
    scale = scale.reshape(N).contiguous()
    x = x.contiguous()
    state = state.contiguous()
    y = torch.empty((M, N), device=x.device, dtype=torch.float32)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _decode_matmul_kernel[grid](
        x, state, scale, y,
        M, N, K, C,
        x.stride(0), x.stride(1),
        state.stride(0),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return y


class TritonCounterLinear(PackedRMSCounterLinear):
    """PackedRMSCounterLinear whose forward GEMM decodes the packed state in-kernel (Triton),
    so no dense weight is materialized on the forward. Falls back to the packed PyTorch forward
    when triton/CUDA are unavailable. T4-verified for correctness but net-negative vs torch-decode
    + cuBLAS (see module docstring); kept as a reference decode-in-GEMM, not a recommended path.
    The update kernel that actually pays off lives in memory_native.fused_update.
    """

    def _forward_matmul(self, x: torch.Tensor) -> torch.Tensor:
        if HAS_TRITON and x.is_cuda and x.dtype == torch.float32:
            # flatten any leading dims ([B,T,d] -> [B*T,d]); the kernel is 2D.
            x2 = x.reshape(-1, x.shape[-1])
            y2 = triton_decode_matmul(x2, self.state, self.scale, self.C,
                                      self.in_features, self.out_features)
            return y2.reshape(*x.shape[:-1], self.out_features)
        return super()._forward_matmul(x)

    def _has_fast_grad_x(self) -> bool:
        return HAS_TRITON and self.state.is_cuda

    def _backward_grad_x(self, grad_out2d: torch.Tensor) -> torch.Tensor:
        # grad_x straight from packed state -> no transient dense weight on the backward.
        # (The counter update still consumes per-tile grad_w; the dense weight is gone.)
        return triton_grad_x(grad_out2d, self.state, self.scale, self.C,
                             self.in_features, self.out_features)
