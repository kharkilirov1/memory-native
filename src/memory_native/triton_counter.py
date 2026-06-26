"""Triton packed kernel — the bandwidth/peak piece of "real sub-byte training".

PackedRMSCounterLinear already stores state at 0.75 byte/weight, but its forward still decodes
to a dense fp32 weight before the matmul, so a dense [out,in] weight is transiently resident.
This module fuses the decode INTO the GEMM: a Triton kernel reads the packed 6-bit state and
the per-row scale, reconstructs each weight tile in registers, and accumulates y = x @ W^T
without ever materializing the dense weight. That removes the transient dense weight from the
forward and reads 0.75 byte/weight from HBM instead of 4.

Scope / honesty:
  * This is the FORWARD kernel. The fused backward UPDATE kernel (forming grad_w in registers
    and applying the counter transition in-place, the analogue of the engine's OpenCL
    counter_*_fused) is the next milestone; here the backward still uses the PyTorch per-tile
    update inherited from PackedRMSCounterLinear (memory-native via packed storage, but it
    decodes tiles in torch).
  * !!! UNVERIFIED ON HARDWARE !!! It was written without a GPU to run it on. Before relying
    on it, run tests/test_triton.py on a CUDA machine with triton installed (it is skipped
    otherwise) -- it checks the Triton forward against the reference dense decode.
  * When triton or CUDA is unavailable, TritonCounterLinear transparently falls back to the
    plain PackedRMSCounterLinear forward, so the class is safe to use anywhere.
"""
from __future__ import annotations

import torch

from .packed import PackedRMSCounterLinear

__all__ = ["HAS_TRITON", "triton_decode_matmul", "TritonCounterLinear"]

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
    so no dense weight is materialized on the forward. Falls back to the packed PyTorch
    forward when triton/CUDA are unavailable. Backward update is inherited (PyTorch) — the
    fused backward kernel is the next milestone. UNVERIFIED ON HARDWARE (see module docstring).
    """

    def _forward_matmul(self, x: torch.Tensor) -> torch.Tensor:
        if HAS_TRITON and x.is_cuda and x.dtype == torch.float32:
            return triton_decode_matmul(x, self.state, self.scale, self.C,
                                        self.in_features, self.out_features)
        return super()._forward_matmul(x)
