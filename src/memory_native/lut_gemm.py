"""LUT-GEMM (Method of Four Russians) for the packed 6-bit counter format — EXPERIMENTAL.

Idea: a group-ternary weight has 3 levels, so 4 consecutive weights have 3^4 = 81 possible
contributions to a dot product (sign-symmetric: 41 distinct). One precomputes the 81 partial
sums ``sum_k w_k x_k`` per (token, 4-column block), after which the inner loop is a table
lookup plus an add: no multiplies, and no per-weight 6-bit decode in the accumulation loop.

Status: the torch reference is complete and CPU-tested for exact equivalence with
``x @ visible_weight().T``; the Triton kernel implements the decode case (M=1) and is
validated against that reference by ``tests/test_lut_gemm.py`` on CUDA. Throughput is
UNVERIFIED until ``scripts/bench_lut_gemm.py`` reports on a GPU — the roofline below says
where the win could come from, not that it exists.

Roofline, 4096x4096 layer, M=1 (RTX Pro 6000 Blackwell, ~1.8 TB/s):
  packed state 12.6 MiB -> ~7 us at peak;
  LUT for one token = 1024 blocks x 81 x 4 B = 324 KiB (fits in L2);
  gather traffic = 1024 x 4 B per output row (L2 hits).
So the weight stream is the floor and LUT can only remove decode ALU work. Whether that is
the current bottleneck is what the benchmark decides: the existing fused decode-GEMM is
plausibly ALU/occupancy-bound, which is the regime where LUT can win. If the kernel comes
within ~1.2x of the bandwidth floor and beats the existing descriptor, keep it; otherwise
this stays a documented negative.
"""
from __future__ import annotations

import torch

__all__ = [
    "TERNARY_PER_BLOCK",
    "LUT_ENTRIES",
    "HAS_TRITON",
    "build_lut_blocks",
    "lut_group_matmul",
    "triton_lut_group_matmul",
]

TERNARY_PER_BLOCK = 4
LUT_ENTRIES = 3 ** TERNARY_PER_BLOCK          # 81

try:                                          # pragma: no cover - import-time platform probe
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:                             # pragma: no cover
    HAS_TRITON = False
    triton = None
    tl = None


def _ternary_digits(x: torch.Tensor) -> torch.Tensor:
    """[81, 4] ternary digits for every LUT entry; entry ``i`` uses digit ``k`` at 3**k."""
    idx = torch.arange(LUT_ENTRIES, device=x.device, dtype=torch.int64)
    return torch.stack(
        [(torch.div(idx, 3 ** k, rounding_mode="floor") % 3 - 1).to(x.dtype)
         for k in range(TERNARY_PER_BLOCK)],
        dim=1,
    )


def build_lut_blocks(x: torch.Tensor) -> torch.Tensor:
    """[M, K] activations -> LUT [M, K//4, 81] of all 4-ternary partial sums.

    ``lut[m, p, i] = sum_k ((i // 3^k) % 3 - 1) * x[m, 4p + k]`` — the preprocess half of
    Method of Four Russians; costs ``M * K/4 * 81`` multiply-adds once per token.
    """
    if x.dim() != 2 or x.shape[1] % TERNARY_PER_BLOCK:
        raise ValueError("x must be [M, K] with K divisible by 4")
    m, k = x.shape
    blocks = x.reshape(m, k // TERNARY_PER_BLOCK, TERNARY_PER_BLOCK)
    return torch.einsum("mpk,ik->mpi", blocks, _ternary_digits(x))     # [M, K//4, 81]


def _ternary_block_indices(codes_perm: torch.Tensor, k: int, C: int) -> torch.Tensor:
    """Packed act-order state [N, K//4*3] -> LUT index [N, K//4] in [0, 81)."""
    from .counter import decode_state
    from .packed import unpack_codes

    t, _c = decode_state(unpack_codes(codes_perm, k), C)              # [N, K] int16
    digit_w = torch.tensor([1, 3, 9, 27], device=codes_perm.device, dtype=torch.int64)
    return ((t.reshape(codes_perm.shape[0], k // TERNARY_PER_BLOCK, TERNARY_PER_BLOCK)
             .to(torch.int64) + 1) * digit_w).sum(-1)                 # [N, K//4]


def lut_group_matmul(
    x: torch.Tensor,
    codes_perm: torch.Tensor,
    scale: torch.Tensor,
    perm: torch.Tensor,
    *,
    group: int = 128,
    C: int = 11,
) -> torch.Tensor:
    """Reference LUT matmul ``x @ W.T`` for a group-ternary weight, via lookup only.

    ``codes_perm`` is the packed act-order state [N, K//4*3], ``scale`` [N, K//group],
    ``perm`` the act-order permutation; returns [M, N] fp32. Algebraically identical to
    decoding the weight and multiplying — the tests pin that.
    """
    if codes_perm.dim() != 2:
        raise ValueError("codes_perm must be [N, K//4*3]")
    if x.dtype != torch.float32:
        x = x.float()
    m, k = x.shape
    n = codes_perm.shape[0]
    p = k // TERNARY_PER_BLOCK
    if k % group or group % TERNARY_PER_BLOCK:
        raise ValueError("K must be divisible by group and group by 4")
    # the state is stored act-ordered, so permute the activation columns once
    x_perm = x[:, perm.to(torch.int64)] if perm is not None else x
    lut = build_lut_blocks(x_perm)                                   # [M, P, 81]
    idx = _ternary_block_indices(codes_perm, k, C)                   # [N, P]
    p_range = torch.arange(p, device=x.device, dtype=torch.int64)
    flat = lut.reshape(m, p * LUT_ENTRIES)
    out = torch.empty(m, n, device=x.device, dtype=torch.float32)
    for mi in range(m):
        take = (p_range[None, :] * LUT_ENTRIES + idx).reshape(-1)
        picked = flat[mi].take(take).view(n, p)                      # [N, P]
        per_group = picked.view(n, k // group, group // TERNARY_PER_BLOCK).sum(-1)
        out[mi] = (per_group * scale).sum(-1)
    return out


if HAS_TRITON:  # pragma: no cover - needs CUDA to compile and run
    @triton.jit
    def _lut_matmul_kernel(
        lut_ptr, state_ptr, scale_ptr, out_ptr,
        N, P, G,
        C: tl.constexpr, TM: tl.constexpr,
        GROUP: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_P: tl.constexpr,
        LUT: tl.constexpr,
    ):
        """M=1: one group per program, BLOCK_N output rows, atomic-add into out.

        ``lut_ptr`` is the [P, 81] fp32 table for this token (built once by
        :func:`build_lut_blocks`); ``state_ptr`` is the packed act-order state.
        """
        pid_n = tl.program_id(0)
        g = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        bpg = GROUP // 4
        offs_p = g * bpg + tl.arange(0, BLOCK_P)
        mask_p = offs_p < P

        base = state_ptr + offs_n[:, None] * (3 * P) + offs_p[None, :] * 3
        b0 = tl.load(base, mask=mask_n[:, None] & mask_p[None, :], other=0).to(tl.int32)
        b1 = tl.load(base + 1, mask=mask_n[:, None] & mask_p[None, :], other=0).to(tl.int32)
        b2 = tl.load(base + 2, mask=mask_n[:, None] & mask_p[None, :], other=0).to(tl.int32)
        c0 = b0 & 0x3F
        c1 = ((b0 >> 6) | (b1 << 2)) & 0x3F
        c2 = ((b1 >> 4) | (b2 << 4)) & 0x3F
        c3 = (b2 >> 2) & 0x3F
        t0 = c0 // TM - 1
        t1 = c1 // TM - 1
        t2 = c2 // TM - 1
        t3 = c3 // TM - 1
        idx = (t0 + 1) + 3 * (t1 + 1) + 9 * (t2 + 1) + 27 * (t3 + 1)
        vals = tl.load(lut_ptr + offs_p[None, :] * LUT + idx,
                       mask=mask_n[:, None] & mask_p[None, :], other=0.0)
        acc = tl.sum(vals, axis=1)
        sc = tl.load(scale_ptr + offs_n * G + g, mask=mask_n, other=0.0)
        tl.atomic_add(out_ptr + offs_n, acc * sc, mask=mask_n)


def triton_lut_group_matmul(
    x: torch.Tensor,
    codes_perm: torch.Tensor,
    scale: torch.Tensor,
    perm: torch.Tensor,
    *,
    group: int = 128,
    C: int = 11,
    block_n: int = 64,
) -> torch.Tensor:
    """CUDA LUT matmul for one token (M=1). Falls back to the reference off-CUDA.

    The LUT build stays in torch (K/4*81 values is small next to the weight stream); the
    kernel does the decode + gather + group-scaled accumulation with one atomic add per
    (row, group). ``x`` is in ORIGINAL column order; the kernel wants act order, so the
    activations are permuted here (K elements, cheap next to an N x K weight stream).
    """
    k = x.shape[-1] if x.dim() == 2 else x.shape[0]
    cpu_or_unsupported = not (HAS_TRITON and x.is_cuda and codes_perm.is_cuda)
    if cpu_or_unsupported:
        return lut_group_matmul(x.reshape(1, -1), codes_perm, scale, perm, group=group, C=C)
    if x.dim() == 2:
        if x.shape[0] != 1:
            raise ValueError("triton_lut_group_matmul handles one token; use the reference")
        x = x[0]
    if k % group or group % TERNARY_PER_BLOCK:
        raise ValueError("K must be divisible by group and group by 4")
    p = k // TERNARY_PER_BLOCK
    n, g = scale.shape
    x_perm = x.to(torch.float32)[perm.to(torch.int64)]
    lut = build_lut_blocks(x_perm.reshape(1, k))[0].contiguous()      # [P, 81]
    out = torch.zeros(n, device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(n, block_n), g)
    _lut_matmul_kernel[grid](
        lut, codes_perm, scale, out,
        n, p, g,
        C=C, TM=2 * C - 1,
        GROUP=group, BLOCK_N=block_n, BLOCK_P=group // TERNARY_PER_BLOCK,
        LUT=LUT_ENTRIES,
    )
    return out
