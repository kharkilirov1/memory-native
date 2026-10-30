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


@triton.jit
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
        acc += tl.dot(x_tile, tl.trans(w_tile))

    # Store output
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def packed_matmul(x: torch.Tensor, codes: torch.Tensor, scale: torch.Tensor,
                  code_to_value: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
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
    )
    return out
