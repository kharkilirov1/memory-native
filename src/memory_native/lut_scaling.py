"""Multi-token LUT-GEMM (Method of Four Russians) for the packed group-ternary format.

EXPERIMENTAL research module (branch ``research/lut-multi-token``). Strict ternary readout
only: ``residual_alpha = 0``, no salient channel. The single-token kernel in
``lut_gemm.py`` is untouched; this module is the multi-token design plus its components,
so the cost of every stage can be measured and charged separately.

Pipeline (what a caller pays per training/inference step):

1. ``ternary_index_build``  state ``[N, 3P]`` (6-bit packed, act order) -> ``[N, P]`` int8
   of 4-ternary LUT indices. Depends on the WEIGHTS only, so it is built once per step
   and reused for every token.
2. ``lut_build``  activations ``[M, K]`` (act order) -> ``[M, P, 81]`` fp32 partial sums.
   Per token: 81 * P values.
3. ``lut_gather_multi``  out ``[M, N]`` = sum over groups of scale * sum over blocks of
   LUT[m, p, idx[n, p]] -- no multiplies per weight, no per-weight decode in the hot loop.

Why the multi-token form is the interesting question: the LUT build is O(M * P * 81) while
the gather is O(M * N * P) scalar table loads, i.e. both scale with M, but the gather uses
L1/L2 and ALU rather than tensor cores. The single-token win comes from streaming 0.75 B per
weight instead of 4 B (fp32) and from replacing decode ALU with one add per 4 weights; as M
grows the dense tensor-core path amortizes its weight stream and wins. This module exists to
locate that crossover empirically instead of asserting it.
"""
from __future__ import annotations

import torch

from .lut_gemm import LUT_ENTRIES, TERNARY_PER_BLOCK, build_lut_blocks, lut_group_matmul

__all__ = [
    "TERNARY_PER_BLOCK",
    "LUT_ENTRIES",
    "HAS_TRITON",
    "ternary_index_build",
    "lut_build",
    "lut_gather_multi",
    "lut_matmul_multi",
    "lut_group_matmul_reference",
]

try:  # pragma: no cover - platform probe
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False
    triton = None
    tl = None


def lut_group_matmul_reference(
    x: torch.Tensor,
    codes_perm: torch.Tensor,
    scale: torch.Tensor,
    perm: torch.Tensor,
    *,
    group: int = 128,
    C: int = 11,
) -> torch.Tensor:
    """Exact ``x @ W.T`` through the LUT identity, on any device (CPU reference).

    The CPU path in ``lut_gemm.lut_group_matmul`` loops over tokens in torch; it is the
    algebraic oracle the CUDA kernels are tested against, so it stays untouched here.
    """
    return lut_group_matmul(x, codes_perm, scale, perm, group=group, C=C)


def _ternary_index_cpu(codes_perm: torch.Tensor, k: int, C: int) -> torch.Tensor:
    """[N, K//4*3] packed state -> [N, K//4] int8 LUT indices (CPU reference)."""
    from .counter import decode_state
    from .packed import unpack_codes

    t, _ = decode_state(unpack_codes(codes_perm, k), C)
    w = torch.tensor([1, 3, 9, 27], device=codes_perm.device, dtype=torch.int64)
    idx = ((t.reshape(codes_perm.shape[0], k // TERNARY_PER_BLOCK, TERNARY_PER_BLOCK)
            .to(torch.int64) + 1) * w).sum(-1)
    return idx.to(torch.int8)


def _act_order(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return x[:, perm.to(torch.int64)] if perm is not None else x


def ternary_index_build(
    codes_perm: torch.Tensor, in_features: int, *, C: int = 11,
    block_n: int = 64, block_p: int = 32,
) -> torch.Tensor:
    """Weight-side preprocessing: packed state -> [N, P] int8 LUT index table.

    Cheap (N * K/4 bytes) and weight-only, so a training/inference loop builds it once per
    weight version and reuses it for all tokens. Returns an int8 tensor; indices are 0..80.
    """
    n = codes_perm.shape[0]
    p = in_features // TERNARY_PER_BLOCK
    if HAS_TRITON and codes_perm.is_cuda:
        out = torch.empty((n, p), device=codes_perm.device, dtype=torch.int8)
        grid = (triton.cdiv(n, block_n), triton.cdiv(p, block_p))
        with torch.cuda.device(codes_perm.device):
            _ternary_index_kernel[grid](
                codes_perm.contiguous(), out, n, p,
                C=C, TM=2 * C - 1, BLOCK_N=block_n, BLOCK_P=block_p,
            )
        return out
    return _ternary_index_cpu(codes_perm, in_features, C)


def lut_build(x_perm: torch.Tensor, *, block_p: int = 32) -> torch.Tensor:
    """[M, K] act-order activations -> [M, K//4, 81] fp32 LUT (CUDA) or torch reference."""
    if HAS_TRITON and x_perm.is_cuda:
        m, k = x_perm.shape
        p = k // TERNARY_PER_BLOCK
        lut = torch.empty((m, p, LUT_ENTRIES), device=x_perm.device, dtype=torch.float32)
        grid = (m, triton.cdiv(p, block_p))
        with torch.cuda.device(x_perm.device):
            _lut_build_kernel[grid](
                x_perm.contiguous(), lut, m, p, BLOCK_P=block_p, LUT=LUT_ENTRIES,
                LUT_PAD=128,
            )
        return lut
    return build_lut_blocks(x_perm)


def lut_gather_multi(
    lut: torch.Tensor, idx: torch.Tensor, scale: torch.Tensor, *,
    group: int, block_n: int = 128,
) -> torch.Tensor:
    """Gather-accumulate half of the algorithm: ``out[M, N]`` from LUT and index tables."""
    m, p, lut_size = lut.shape
    n, g = scale.shape
    if p * TERNARY_PER_BLOCK != idx.shape[1] * TERNARY_PER_BLOCK:
        raise ValueError("LUT / index tables disagree on the number of blocks")
    if not (HAS_TRITON and lut.is_cuda and idx.is_cuda):
        raise RuntimeError("lut_gather_multi needs CUDA + Triton")
    out = torch.empty((m, n), device=lut.device, dtype=torch.float32)
    grid = (m, triton.cdiv(n, block_n))
    bpg = group // TERNARY_PER_BLOCK
    with torch.cuda.device(lut.device):
        _lut_multi_kernel[grid](
            lut, idx, scale, out, m, n, p, g,
            GROUP=group, BLOCK_N=block_n, BP=bpg, LUT=lut_size,
        )
    return out


def lut_matmul_multi(
    x: torch.Tensor,
    codes_perm: torch.Tensor,
    scale: torch.Tensor,
    perm: torch.Tensor,
    *,
    group: int = 128,
    C: int = 11,
    block_n: int = 128,
    block_p: int = 32,
) -> torch.Tensor:
    """Full multi-token LUT matmul ``x @ W.T`` for a strict ternary packed layer.

    Components are timed separately by the benchmark; this convenience call runs the whole
    pipeline (index build + LUT build + gather) each time and is what the caller pays when
    it cannot cache the weight-side table.
    """
    m, k = x.shape
    x_perm = _act_order(x.to(torch.float32), perm).contiguous()
    idx = ternary_index_build(codes_perm, k, C=C)
    lut = lut_build(x_perm, block_p=block_p)
    return lut_gather_multi(lut, idx, scale, group=group, block_n=block_n)


if HAS_TRITON:  # pragma: no cover - compiled on CUDA only
    @triton.jit
    def _ternary_index_kernel(state_ptr, idx_ptr, N, P,
                              C: tl.constexpr, TM: tl.constexpr,
                              BLOCK_N: tl.constexpr, BLOCK_P: tl.constexpr):
        """Packed 6-bit state [N, 3P] -> LUT index [N, P] int8 (strict ternary)."""
        pid_n = tl.program_id(0)
        pid_p = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        mask = (offs_n[:, None] < N) & (offs_p[None, :] < P)
        base = state_ptr + offs_n[:, None] * (3 * P) + offs_p[None, :] * 3
        b0 = tl.load(base + 0, mask=mask, other=0).to(tl.int32)
        b1 = tl.load(base + 1, mask=mask, other=0).to(tl.int32)
        b2 = tl.load(base + 2, mask=mask, other=0).to(tl.int32)
        c0 = b0 & 63
        c1 = ((b0 >> 6) | (b1 << 2)) & 63
        c2 = ((b1 >> 4) | (b2 << 4)) & 63
        c3 = (b2 >> 2) & 63
        i0 = c0 // TM
        i1 = c1 // TM
        i2 = c2 // TM
        i3 = c3 // TM
        idx = i0 + 3 * i1 + 9 * i2 + 27 * i3
        tl.store(idx_ptr + offs_n[:, None] * P + offs_p[None, :], idx.to(tl.int8), mask=mask)

    @triton.jit
    def _lut_build_kernel(x_ptr, lut_ptr, M, P, BLOCK_P: tl.constexpr, LUT: tl.constexpr,
                          LUT_PAD: tl.constexpr):
        """One program per token: [4*P] activations -> [P, 81] partial sums.

        Entry ``i`` of the table is ``sum_k d_k(i) * x_k`` with ``d_k(i)`` the ternary
        digits of ``i`` in base 3 (digit k at 3**k), so the whole table is four broadcast
        FMAs over the 81 digits rather than 81 dot products. ``LUT_PAD`` is the next power
        of two above 81 (Triton ranges must be powers of two); the tail is masked out.
        """
        pid_m = tl.program_id(0)
        pid_p = tl.program_id(1)
        offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        mask_p = offs_p < P
        base = x_ptr + pid_m * (4 * P) + offs_p * 4
        x0 = tl.load(base + 0, mask=mask_p, other=0.0)
        x1 = tl.load(base + 1, mask=mask_p, other=0.0)
        x2 = tl.load(base + 2, mask=mask_p, other=0.0)
        x3 = tl.load(base + 3, mask=mask_p, other=0.0)
        d = tl.arange(0, LUT_PAD)
        q3 = d // 3
        q9 = d // 9
        q27 = d // 27
        d0 = d - q3 * 3
        d1 = q3 - q9 * 3
        d2 = q9 - q27 * 3
        d3 = q27
        acc = (x0[:, None] * (d0[None, :] - 1) + x1[:, None] * (d1[None, :] - 1)
               + x2[:, None] * (d2[None, :] - 1) + x3[:, None] * (d3[None, :] - 1))
        tl.store(lut_ptr + pid_m * (P * LUT) + offs_p[:, None] * LUT + d[None, :],
                 acc, mask=mask_p[:, None] & (d[None, :] < LUT))

    @triton.jit
    def _lut_multi_kernel(lut_ptr, idx_ptr, scale_ptr, out_ptr, M, N, P, G,
                          GROUP: tl.constexpr, BLOCK_N: tl.constexpr, BP: tl.constexpr,
                          LUT: tl.constexpr):
        """One program per (token, output-row block): no atomics, one store per element.

        Loops over all scale groups in-kernel so each output element is finalized once.
        The index table is read from global (int8, group-contiguous) and the LUT tile for
        this token is reused across the whole output-row block through L1/L2.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        lut_base = lut_ptr + pid_m * (P * LUT)
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for g in range(0, G):
            offs_p = g * BP + tl.arange(0, BP)
            idx = tl.load(idx_ptr + offs_n[:, None] * P + offs_p[None, :],
                          mask=mask_n[:, None], other=0).to(tl.int32)
            vals = tl.load(lut_base + offs_p[None, :] * LUT + idx,
                           mask=mask_n[:, None], other=0.0)
            acc += tl.sum(vals, axis=1) * tl.load(scale_ptr + offs_n * G + g,
                                                  mask=mask_n, other=0.0)
        tl.store(out_ptr + pid_m * N + offs_n, acc, mask=mask_n)
