"""Strict update-from-IO kernel — the counter update without a materialized weight-gradient.

`fused_update.triton_counter_update` consumes a dense grad_w (formed by cuBLAS). This module forms
grad_w[o,i] = sum_m grad_out[m,o] * x[m,i] INSIDE the kernel, per packed lane, so the dense [out,in]
gradient is never materialized -- the strict memory-native update (the analogue of the engine's
OpenCL apply_update_fused). One program per output row streams the M=batch*seq dimension,
accumulates the four packed-lane grad_w vectors, then does exact RMS + scale + deterministic
stochastic-rounding tick + re-pack in one pass.

Determinism (hash-SR) makes it reproducible, so counter_update_from_io_hashsr is the CPU reference
and the kernel is validated bit-quantified against it on a GPU (the FP reduction order differs from
cuBLAS, so a handful of weights round the other way -- the same SR-boundary noise as fused_update).

Honest scope: this is a MEMORY play, not a speed one -- forming grad_w in-kernel is a hand-written
GEMM and loses to cuBLAS. Use it when not materializing grad_w matters; otherwise the cuBLAS-grad_w
+ fused_update path is faster.
"""
from __future__ import annotations

import torch

from .fused_update import HAS_TRITON, counter_update_hashsr

__all__ = ["counter_update_from_io_hashsr", "HAS_TRITON", "triton_counter_update_from_io"]


@torch.no_grad()
def counter_update_from_io_hashsr(codes: torch.Tensor, scale: torch.Tensor, v: torch.Tensor,
                                  x: torch.Tensor, grad_out: torch.Tensor, **kw) -> torch.Tensor:
    """CPU reference: form grad_w = grad_out^T x, then the deterministic-SR update. Identical math
    to the kernel (which forms grad_w in registers instead of as a tensor)."""
    grad_w = grad_out.transpose(0, 1).to(torch.float32) @ x.to(torch.float32)
    return counter_update_hashsr(codes, scale, v, grad_w, **kw)


if HAS_TRITON:
    import triton
    import triton.language as tl

    from .fused_update import _tick

    @triton.jit
    def _counter_update_from_io_kernel(state_ptr, scale_ptr, v_ptr, x_ptr, go_ptr, seed_ptr,
                                       C, in_features, M, lr, lr_scale, rms_beta, rms_eps,
                                       BLOCK_G: tl.constexpr, BLOCK_M: tl.constexpr,
                                       LAGGED: tl.constexpr = False):
        # One program per output row. Forms grad_w[row, :] per packed lane (no dense grad_w), then
        # exact RMS + scale + SR tick + re-pack. state packed [out, (in/4)*3]; x [M,in]; go [M,out].
        row = tl.program_id(0)
        seed = tl.load(seed_ptr).to(tl.uint32)
        gpr = in_features // 4
        lv = 2 * C - 1
        Cf = C * 1.0
        gids = tl.arange(0, BLOCK_G)
        gmask = gids < gpr
        col0 = gids * 4
        # accumulate the four lane grad_w vectors over the M dimension
        gw0 = tl.zeros((BLOCK_G,), tl.float32); gw1 = tl.zeros((BLOCK_G,), tl.float32)
        gw2 = tl.zeros((BLOCK_G,), tl.float32); gw3 = tl.zeros((BLOCK_G,), tl.float32)
        for m0 in range(0, M, BLOCK_M):
            moff = m0 + tl.arange(0, BLOCK_M)
            mmask = moff < M
            gocol = tl.load(go_ptr + moff * tl.num_programs(0) + row, mask=mmask, other=0.0)  # [BLOCK_M]
            xb = x_ptr + moff[:, None] * in_features
            x0 = tl.load(xb + (col0 + 0)[None, :], mask=mmask[:, None] & gmask[None, :], other=0.0)
            x1 = tl.load(xb + (col0 + 1)[None, :], mask=mmask[:, None] & gmask[None, :], other=0.0)
            x2 = tl.load(xb + (col0 + 2)[None, :], mask=mmask[:, None] & gmask[None, :], other=0.0)
            x3 = tl.load(xb + (col0 + 3)[None, :], mask=mmask[:, None] & gmask[None, :], other=0.0)
            gw0 += tl.sum(gocol[:, None] * x0, axis=0)
            gw1 += tl.sum(gocol[:, None] * x1, axis=0)
            gw2 += tl.sum(gocol[:, None] * x2, axis=0)
            gw3 += tl.sum(gocol[:, None] * x3, axis=0)
        # decode the four lane ternary weights from the packed state
        base = row * gpr * 3 + gids * 3
        b0 = tl.load(state_ptr + base + 0, mask=gmask, other=0).to(tl.int32)
        b1 = tl.load(state_ptr + base + 1, mask=gmask, other=0).to(tl.int32)
        b2 = tl.load(state_ptr + base + 2, mask=gmask, other=0).to(tl.int32)
        c0 = b0 & 0x3F
        c1 = ((b0 >> 6) | (b1 << 2)) & 0x3F
        c2 = ((b1 >> 4) | (b2 << 4)) & 0x3F
        c3 = (b2 >> 2) & 0x3F
        t0 = (c0 // lv - 1).to(tl.float32); t1 = (c1 // lv - 1).to(tl.float32)
        t2 = (c2 // lv - 1).to(tl.float32); t3 = (c3 // lv - 1).to(tl.float32)
        # exact row stats over the whole row (masked lanes contributed 0)
        g_sq = (tl.sum(gw0 * gw0) + tl.sum(gw1 * gw1) + tl.sum(gw2 * gw2) + tl.sum(gw3 * gw3)) / in_features
        grad_s = (tl.sum(gw0 * t0) + tl.sum(gw1 * t1) + tl.sum(gw2 * t2) + tl.sum(gw3 * t3)) / tl.sqrt(in_features.to(tl.float32))
        s_old = tl.load(scale_ptr + row)
        v_old = tl.load(v_ptr + row)
        vv = rms_beta * v_old + (1.0 - rms_beta) * g_sq
        tl.store(v_ptr + row, vv)
        denom = tl.maximum(tl.sqrt(v_old if LAGGED else vv), rms_eps)
        s_new = tl.minimum(tl.maximum(s_old - lr_scale * grad_s, 1e-5), 10.0)
        # SR tick per lane, then re-pack 4 codes / 3 bytes
        nc0 = _tick(c0, gw0, row, col0 + 0, in_features, lv, C, Cf, lr, denom, s_old, s_new, seed)
        nc1 = _tick(c1, gw1, row, col0 + 1, in_features, lv, C, Cf, lr, denom, s_old, s_new, seed)
        nc2 = _tick(c2, gw2, row, col0 + 2, in_features, lv, C, Cf, lr, denom, s_old, s_new, seed)
        nc3 = _tick(c3, gw3, row, col0 + 3, in_features, lv, C, Cf, lr, denom, s_old, s_new, seed)
        p0 = (nc0 | (nc1 << 6)) & 0xFF
        p1 = ((nc1 >> 2) | (nc2 << 4)) & 0xFF
        p2 = ((nc2 >> 4) | (nc3 << 2)) & 0xFF
        tl.store(state_ptr + base + 0, p0.to(tl.uint8), mask=gmask)
        tl.store(state_ptr + base + 1, p1.to(tl.uint8), mask=gmask)
        tl.store(state_ptr + base + 2, p2.to(tl.uint8), mask=gmask)
        tl.store(scale_ptr + row, s_new)


def triton_counter_update_from_io(state_packed: torch.Tensor, scale: torch.Tensor, v: torch.Tensor,
                                  x: torch.Tensor, grad_out: torch.Tensor, *, C: int, lr: float,
                                  lr_scale: float, rms_beta: float, rms_eps: float, seed: int,
                                  lagged: bool = False) -> None:
    """One-launch strict update: forms grad_w in-kernel from (x, grad_out) -- no dense gradient --
    and applies the RMS + SR counter update. Mutates state_packed, scale, v in place.
    Bit-quantified-equal to counter_update_from_io_hashsr (verify on GPU). lagged=True uses the
    previous-step v for the denominator (per-element tick)."""
    if not HAS_TRITON:
        raise RuntimeError("triton not available")
    out = scale.numel()
    in_features = x.shape[1]
    M = x.shape[0]
    assert in_features % 4 == 0
    gpr = in_features // 4
    BLOCK_G = triton.next_power_of_2(gpr)
    seed_t = torch.tensor([int(seed) & 0xFFFFFFFF], dtype=torch.int64, device=x.device)
    _counter_update_from_io_kernel[(out,)](
        state_packed, scale.reshape(out), v.reshape(out), x.contiguous(), grad_out.contiguous(),
        seed_t, C, in_features, M, lr, lr_scale, rms_beta, rms_eps,
        BLOCK_G=BLOCK_G, BLOCK_M=128, LAGGED=lagged,
    )
