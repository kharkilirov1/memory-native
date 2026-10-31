"""Triton fused decode+matmul kernel for PackedNA formats.

Eliminates the dense weight materialization by decoding codes on-the-fly
inside the GEMM.  This reduces peak VRAM to ~persistent bytes and removes
the decode kernel launch overhead.

Usage:
    from mx6.triton_kernel import packed_matmul
    out = packed_matmul(x, codes, scale_bytes, code_to_value_table, BLOCK=32)

The ``code_to_value_table`` is a precomputed float32 array of size n_codes
(25 for PackedNA5, 63 for PackedNA6) that maps each code directly to its
signed magnitude, avoiding integer division inside the kernel.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False
    triton = None
    tl = None

BLOCK = 32


def _jit(fn):
    """``triton.jit`` when available, else a passthrough.

    Without this the module could not even be imported on a CPU box: the
    decorator is evaluated at import time, so ``triton`` being ``None`` raised
    ``AttributeError`` before ``packed_matmul``'s own HAS_TRITON check ran.
    """
    return triton.jit(fn) if HAS_TRITON else fn


@_jit
def _packed_matmul_kernel(
    x_ptr,          # [M, K] float32
    codes_ptr,      # [N, K] uint8
    scale_ptr,      # [N, K // BLOCK] uint8 (E8M0)
    lut_ptr,        # [n_codes] float32 (code -> signed magnitude)
    out_ptr,        # [M, N] float32
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,   # must be multiple of BLOCK (32)
    BLOCK: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """Fused decode + matmul: out = x @ decode(codes, scale).T"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # bounds masks
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k = k_start + offs_k  # [BLOCK_K]
        mask_k = k < K

        # Load x tile [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + offs_m[:, None] * K + k[None, :]
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load codes tile [BLOCK_N, BLOCK_K]
        code_ptrs = codes_ptr + offs_n[:, None] * K + k[None, :]
        code_tile = tl.load(code_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0)

        # Decode: lookup magnitude from code_to_value table (gather)
        lut_ptrs = lut_ptr + code_tile.to(tl.int64)
        val_tile = tl.load(lut_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

        # Load scale for each 32-element block
        # k // BLOCK gives the block index within the row
        block_idx = k // BLOCK  # [BLOCK_K]
        scale_ptrs = scale_ptr + offs_n[:, None] * (K // BLOCK) + block_idx[None, :]
        scale_byte = tl.load(scale_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=127)
        scale_f = tl.exp2(scale_byte.to(tl.float32) - 127.0)

        # weight = magnitude * scale
        w_tile = val_tile * scale_f  # [BLOCK_N, BLOCK_K]

        # acc += x_tile @ w_tile.T  →  [BLOCK_M, BLOCK_N]
        # input_precision defaults to "ieee": Triton's own default is TF32 on
        # Ampere+, which would silently disagree with the fp32 torch reference.
        acc += tl.dot(x_tile, tl.trans(w_tile), input_precision=INPUT_PRECISION)

    # Store output
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@_jit
def _packed_matmul_dx_kernel(
    go_ptr,         # [M, N] float32 — upstream gradient
    codes_ptr,      # [N, K] uint8
    scale_ptr,      # [N, K // BLOCK] uint8
    lut_ptr,        # [n_codes] float32
    out_ptr,        # [M, K] float32
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """Fused decode + grad_x: out = go @ decode(codes, scale)."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_k = offs_k < K

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    block_idx = offs_k // BLOCK

    for n_start in range(0, N, BLOCK_N):
        n = n_start + tl.arange(0, BLOCK_N)
        mask_n = n < N

        go_tile = tl.load(go_ptr + offs_m[:, None] * N + n[None, :],
                          mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        code_tile = tl.load(codes_ptr + n[:, None] * K + offs_k[None, :],
                            mask=mask_n[:, None] & mask_k[None, :], other=0)
        val_tile = tl.load(lut_ptr + code_tile.to(tl.int64),
                           mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        scale_byte = tl.load(scale_ptr + n[:, None] * (K // BLOCK) + block_idx[None, :],
                             mask=mask_n[:, None] & mask_k[None, :], other=127)
        w_tile = val_tile * tl.exp2(scale_byte.to(tl.float32) - 127.0)

        acc += tl.dot(go_tile, w_tile, input_precision=INPUT_PRECISION)

    tl.store(out_ptr + offs_m[:, None] * K + offs_k[None, :], acc,
             mask=mask_m[:, None] & mask_k[None, :])


def packed_matmul(x: torch.Tensor, codes: torch.Tensor, scale: torch.Tensor,
                  code_to_value: torch.Tensor, block: int = BLOCK,
                  input_precision: str = "ieee") -> torch.Tensor:
    """Fused decode + matmul: out = x @ decode(codes, scale).T

    Args:
        x: [M, K] float32
        codes: [N, K] uint8 — packed (value, accumulator) codes
        scale: [N, K // block] uint8 — E8M0 block scales
        code_to_value: [n_codes] float32 — precomputed code → magnitude lookup
        block: block size (default 32)

    Returns:
        out: [M, N] float32
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton not available")
    M, K = x.shape
    N, K2 = codes.shape
    assert K == K2, f"K mismatch: x has {K}, codes has {K2}"
    assert K % block == 0
    assert scale.shape == (N, K // block)

    x = x.contiguous()
    codes = codes.contiguous()
    scale = scale.contiguous()
    code_to_value = code_to_value.contiguous()

    out = torch.empty(M, N, device=x.device, dtype=torch.float32)

    # Choose tile sizes
    BLOCK_M = 64 if M >= 64 else 16
    BLOCK_N = 64 if N >= 64 else 16
    BLOCK_K = 64  # 2 blocks of 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _packed_matmul_kernel[grid](
        x, codes, scale, code_to_value, out,
        M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK=block,
        INPUT_PRECISION=input_precision,
    )
    return out


def packed_matmul_dx(grad_out: torch.Tensor, codes: torch.Tensor, scale: torch.Tensor,
                     code_to_value: torch.Tensor, block: int = BLOCK,
                     input_precision: str = "ieee") -> torch.Tensor:
    """Fused decode + grad_x: out = grad_out @ decode(codes, scale)."""
    if not HAS_TRITON:
        raise RuntimeError("Triton not available")
    M, N = grad_out.shape
    N2, K = codes.shape
    assert N == N2, f"N mismatch: grad_out has {N}, codes has {N2}"
    assert K % block == 0
    assert scale.shape == (N, K // block)

    grad_out = grad_out.contiguous()
    codes = codes.contiguous()
    scale = scale.contiguous()
    code_to_value = code_to_value.contiguous()

    out = torch.empty(M, K, device=grad_out.device, dtype=torch.float32)

    BLOCK_M = 64 if M >= 64 else 16
    BLOCK_K = 64 if K >= 64 else 32
    BLOCK_N = 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
    _packed_matmul_dx_kernel[grid](
        grad_out, codes, scale, code_to_value, out,
        M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK=block,
        INPUT_PRECISION=input_precision,
    )
    return out
