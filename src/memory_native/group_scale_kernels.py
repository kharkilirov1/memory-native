"""Packed group-scale counter kernels.

The packed state is stored in *act-order* (``W_perm = W[:, perm]``), so each scale group is
contiguous in the 6-bit stream. Forward/grad-x gather/scatter through ``perm`` without ever
materializing a dense weight. The strict update forms ``grad_w`` from ``(x, grad_out)`` inside
Triton and uses only O(out * n_groups) scratch -- never an [out, in] gradient.
"""
from __future__ import annotations

import torch

from .counter import decode_state, encode_state
from .fused_update import HAS_TRITON, hash_u32, uniform01

__all__ = [
    "HAS_TRITON",
    "group_counter_update_hashsr",
    "group_counter_update_from_io_hashsr",
    "group_update_scratch_bytes",
    "triton_group_decode_matmul",
    "triton_group_grad_x",
    "triton_group_counter_update_from_io",
    "triton_group_counter_update_dense",
]


def _validate_layout(codes: torch.Tensor, scale: torch.Tensor, perm: torch.Tensor, group: int) -> None:
    if codes.ndim != 2 or scale.ndim != 2 or perm.ndim != 1:
        raise ValueError("codes/scale/perm must be rank 2/2/1")
    out, in_features = codes.shape
    if scale.shape[0] != out or perm.numel() != in_features:
        raise ValueError("group counter layout mismatch")
    if in_features % 4 or group % 4:
        raise ValueError("in_features and group must be divisible by 4")
    expected_groups = (in_features + group - 1) // group
    if scale.shape[1] != expected_groups:
        raise ValueError(f"expected {expected_groups} groups, got {scale.shape[1]}")


def group_update_scratch_bytes(out_features: int, in_features: int, group: int = 128) -> int:
    """Peak auxiliary bytes of the strict GPU update (float32).

    The wrapper allocates old scales, group-scale gradients, group g² partials, and one effective
    RMS denominator per output row. This is O(out * ceil(in/group)), not O(out * in).
    """
    groups = (int(in_features) + int(group) - 1) // int(group)
    return 4 * (3 * int(out_features) * groups + int(out_features))


@torch.no_grad()
def group_counter_update_hashsr(
    codes_perm: torch.Tensor,
    scale: torch.Tensor,
    v: torch.Tensor,
    grad_w_perm: torch.Tensor,
    perm: torch.Tensor,
    *,
    group: int,
    C: int,
    lr: float,
    lr_scale: float,
    rms_beta: float,
    rms_eps: float,
    seed: int,
    residual_alpha: float = 0.0,
    lagged: bool = False,
    clip: float = 0.0,
) -> torch.Tensor:
    """Deterministic CPU/torch reference for the group-scale packed update.

    ``codes_perm`` and ``grad_w_perm`` are in act-order; ``perm[p]`` is the original input column.
    Scale/v are mutated in place and new unpacked codes (still act-ordered) are returned.
    """
    _validate_layout(codes_perm, scale, perm, group)
    out, in_features = codes_perm.shape
    if grad_w_perm.shape != codes_perm.shape:
        raise ValueError("grad_w_perm shape mismatch")
    t, c = decode_state(codes_perm, C)
    t = t.float()
    c = c.float()
    gw = grad_w_perm.float()

    g_sq = gw.square().mean(dim=1, keepdim=True)
    if lagged:
        denom = v.sqrt().clamp_min(rms_eps)
        v.mul_(rms_beta).add_(g_sq, alpha=1.0 - rms_beta)
    else:
        v.mul_(rms_beta).add_(g_sq, alpha=1.0 - rms_beta)
        denom = v.sqrt().clamp_min(rms_eps)
    if clip > 0:
        row_norm = (g_sq * in_features).sqrt() / denom
        denom = denom / (clip / row_norm.clamp_min(1e-30)).clamp_max(1.0)

    groups = scale.shape[1]
    group_idx = torch.div(
        torch.arange(in_features, device=codes_perm.device), group, rounding_mode="floor"
    )
    code_value = t + float(residual_alpha) * c / C
    grad_scale = torch.zeros_like(scale)
    grad_scale.scatter_add_(1, group_idx.unsqueeze(0).expand(out, -1), gw * code_value)
    counts = torch.bincount(group_idx, minlength=groups).to(scale.dtype).sqrt().clamp_min(1.0)
    grad_scale.div_(counts.unsqueeze(0))

    scale_old = scale.clone()
    scale_new = (scale_old - float(lr_scale) * grad_scale).clamp_(1e-5, 10.0)
    s_old_col = scale_old[:, group_idx]
    s_new_col = scale_new[:, group_idx]

    rows = torch.arange(out, device=codes_perm.device, dtype=torch.int64).unsqueeze(1)
    original_cols = perm.to(device=codes_perm.device, dtype=torch.int64).unsqueeze(0)
    elem = rows * in_features + original_cols
    rnd = uniform01((int(seed) ^ hash_u32(elem)) & 0xFFFFFFFF)
    tick = (-float(lr)) * (gw / denom) * (C / s_new_col)
    value = c * (s_old_col / s_new_col) + tick
    floor = torch.floor(value)
    cc = floor + (rnd < (value - floor)).to(value.dtype)
    carry = torch.trunc(cc / C)
    remainder = cc - carry * C
    proposed = t + carry
    new_t = proposed.clamp(-1, 1)
    remainder = torch.where(
        proposed != new_t, torch.sign(cc) * (C - 1), remainder
    ).clamp_(-(C - 1), C - 1)

    scale.copy_(scale_new)
    return encode_state(new_t.to(torch.int16), remainder.to(torch.int16), C)


@torch.no_grad()
def group_counter_update_from_io_hashsr(
    codes_perm: torch.Tensor,
    scale: torch.Tensor,
    v: torch.Tensor,
    x: torch.Tensor,
    grad_out: torch.Tensor,
    perm: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Reference strict update: forms the act-ordered correlation then calls the exact reference."""
    x2 = x.reshape(-1, x.shape[-1]).float()
    go2 = grad_out.reshape(-1, grad_out.shape[-1]).float()
    perm_long = perm.to(device=x2.device, dtype=torch.long)
    grad_w_perm = go2.transpose(0, 1) @ x2[:, perm_long]
    return group_counter_update_hashsr(
        codes_perm, scale, v, grad_w_perm, perm, **kwargs
    )


if HAS_TRITON:
    import triton
    import triton.language as tl

    from .fused_update import _tick

    @triton.jit
    def _decode6(b0, b1, b2, lane):
        c0 = b0 & 0x3F
        c1 = ((b0 >> 6) | (b1 << 2)) & 0x3F
        c2 = ((b1 >> 4) | (b2 << 4)) & 0x3F
        c3 = (b2 >> 2) & 0x3F
        return tl.where(lane == 0, c0, tl.where(lane == 1, c1, tl.where(lane == 2, c2, c3)))

    @triton.jit
    def _group_decode_matmul_kernel(
        x_ptr, state_ptr, scale_ptr, perm_ptr, y_ptr,
        M, N, K, G, C, group_size, residual_alpha,
        stride_xm, stride_xk, stride_sn, stride_scn, stride_scg,
        stride_ym, stride_yn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        X_BF16: tl.constexpr, X_FP16: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        lv = 2 * C - 1
        Cf = C * 1.0
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for p0 in range(0, K, BLOCK_K):
            offs_p = p0 + tl.arange(0, BLOCK_K)
            pmask = offs_p < K
            cols = tl.load(perm_ptr + offs_p, mask=pmask, other=0).to(tl.int32)
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + cols[None, :] * stride_xk,
                mask=(offs_m[:, None] < M) & pmask[None, :], other=0.0,
            )
            packed_group = offs_p // 4
            lane = offs_p % 4
            base = offs_n[:, None] * stride_sn + packed_group[None, :] * 3
            nk_mask = (offs_n[:, None] < N) & pmask[None, :]
            b0 = tl.load(state_ptr + base + 0, mask=nk_mask, other=0).to(tl.int32)
            b1 = tl.load(state_ptr + base + 1, mask=nk_mask, other=0).to(tl.int32)
            b2 = tl.load(state_ptr + base + 2, mask=nk_mask, other=0).to(tl.int32)
            code = _decode6(b0, b1, b2, lane[None, :])
            t = code // lv - 1
            c = code % lv - (C - 1)
            sg = offs_p // group_size
            s = tl.load(
                scale_ptr + offs_n[:, None] * stride_scn + sg[None, :] * stride_scg,
                mask=nk_mask, other=0.0,
            )
            w = s * (t.to(tl.float32) + residual_alpha * c.to(tl.float32) / Cf)
            if X_BF16:
                acc += tl.dot(x.to(tl.bfloat16), tl.trans(w.to(tl.bfloat16)))
            elif X_FP16:
                acc += tl.dot(x.to(tl.float16), tl.trans(w.to(tl.float16)))
            else:
                acc += tl.dot(x.to(tl.float32), tl.trans(w.to(tl.float32)))

        tl.store(
            y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
            acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )

    @triton.jit
    def _group_gradx_kernel(
        go_ptr, state_ptr, scale_ptr, perm_ptr, gx_ptr,
        M, N, K, G, C, group_size, residual_alpha,
        stride_gom, stride_gon, stride_sn, stride_scn, stride_scg,
        stride_gxm, stride_gxk,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GO_BF16: tl.constexpr, GO_FP16: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_p = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_p = pid_p * BLOCK_K + tl.arange(0, BLOCK_K)
        pmask = offs_p < K
        cols = tl.load(perm_ptr + offs_p, mask=pmask, other=0).to(tl.int32)
        lv = 2 * C - 1
        Cf = C * 1.0
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

        packed_group = offs_p // 4
        lane = offs_p % 4
        sg = offs_p // group_size
        for n0 in range(0, N, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            go = tl.load(
                go_ptr + offs_m[:, None] * stride_gom + offs_n[None, :] * stride_gon,
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0,
            )
            base = offs_n[:, None] * stride_sn + packed_group[None, :] * 3
            nk_mask = (offs_n[:, None] < N) & pmask[None, :]
            b0 = tl.load(state_ptr + base + 0, mask=nk_mask, other=0).to(tl.int32)
            b1 = tl.load(state_ptr + base + 1, mask=nk_mask, other=0).to(tl.int32)
            b2 = tl.load(state_ptr + base + 2, mask=nk_mask, other=0).to(tl.int32)
            code = _decode6(b0, b1, b2, lane[None, :])
            t = code // lv - 1
            c = code % lv - (C - 1)
            s = tl.load(
                scale_ptr + offs_n[:, None] * stride_scn + sg[None, :] * stride_scg,
                mask=nk_mask, other=0.0,
            )
            w = s * (t.to(tl.float32) + residual_alpha * c.to(tl.float32) / Cf)
            if GO_BF16:
                acc += tl.dot(go.to(tl.bfloat16), w.to(tl.bfloat16))
            elif GO_FP16:
                acc += tl.dot(go.to(tl.float16), w.to(tl.float16))
            else:
                acc += tl.dot(go.to(tl.float32), w.to(tl.float32))

        tl.store(
            gx_ptr + offs_m[:, None] * stride_gxm + cols[None, :] * stride_gxk,
            acc, mask=(offs_m[:, None] < M) & pmask[None, :],
        )

    @triton.jit
    def _group_stats_from_io_kernel(
        state_ptr, x_ptr, go_ptr, perm_ptr, grad_scale_ptr, gsq_ptr,
        M, N, K, G, C, group_size, residual_alpha,
        stride_sn, stride_xm, stride_xk, stride_gom, stride_gon,
        BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
    ):
        row = tl.program_id(0)
        grp = tl.program_id(1)
        offs = tl.arange(0, BLOCK_K)
        p = grp * group_size + offs
        pmask = p < K
        cols = tl.load(perm_ptr + p, mask=pmask, other=0).to(tl.int32)
        gw = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for m0 in range(0, M, BLOCK_M):
            moff = m0 + tl.arange(0, BLOCK_M)
            mmask = moff < M
            go = tl.load(
                go_ptr + moff * stride_gom + row * stride_gon,
                mask=mmask, other=0.0,
            ).to(tl.float32)
            xv = tl.load(
                x_ptr + moff[:, None] * stride_xm + cols[None, :] * stride_xk,
                mask=mmask[:, None] & pmask[None, :], other=0.0,
            ).to(tl.float32)
            gw += tl.sum(go[:, None] * xv, axis=0)

        packed_group = p // 4
        lane = p % 4
        base = row * stride_sn + packed_group * 3
        b0 = tl.load(state_ptr + base + 0, mask=pmask, other=0).to(tl.int32)
        b1 = tl.load(state_ptr + base + 1, mask=pmask, other=0).to(tl.int32)
        b2 = tl.load(state_ptr + base + 2, mask=pmask, other=0).to(tl.int32)
        code = _decode6(b0, b1, b2, lane)
        lv = 2 * C - 1
        Cf = C * 1.0
        t = code // lv - 1
        c = code % lv - (C - 1)
        visible = t.to(tl.float32) + residual_alpha * c.to(tl.float32) / Cf
        count = tl.sum(pmask.to(tl.float32), axis=0)
        gscale = tl.sum(gw * visible, axis=0) / tl.sqrt(tl.maximum(count, 1.0))
        g2 = tl.sum(gw * gw, axis=0)
        tl.store(grad_scale_ptr + row * G + grp, gscale)
        tl.store(gsq_ptr + row * G + grp, g2)

    @triton.jit
    def _group_finalize_stats_kernel(
        scale_ptr, v_ptr, grad_scale_ptr, gsq_ptr, denom_ptr,
        N, K, G, lr_scale, rms_beta, rms_eps, clip,
        BLOCK_G: tl.constexpr, LAGGED: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs_g = tl.arange(0, BLOCK_G)
        gmask = offs_g < G
        partial = tl.load(gsq_ptr + row * G + offs_g, mask=gmask, other=0.0)
        g_sq = tl.sum(partial, axis=0) / (K * 1.0)
        v_old = tl.load(v_ptr + row)
        v_new = rms_beta * v_old + (1.0 - rms_beta) * g_sq
        tl.store(v_ptr + row, v_new)
        denom = tl.maximum(tl.sqrt(v_old if LAGGED else v_new), rms_eps)
        row_norm = tl.sqrt(g_sq * (K * 1.0)) / denom
        cmult = tl.where(
            clip > 0.0,
            tl.minimum(clip / tl.maximum(row_norm, 1e-30), 1.0),
            1.0,
        )
        denom = denom / cmult
        tl.store(denom_ptr + row, denom)
        grad_s = tl.load(grad_scale_ptr + row * G + offs_g, mask=gmask, other=0.0)
        s_old = tl.load(scale_ptr + row * G + offs_g, mask=gmask, other=0.0)
        s_new = tl.minimum(tl.maximum(s_old - lr_scale * grad_s, 1e-5), 10.0)
        tl.store(scale_ptr + row * G + offs_g, s_new, mask=gmask)

    @triton.jit
    def _group_state_from_io_kernel(
        state_ptr, scale_old_ptr, scale_ptr, denom_ptr, x_ptr, go_ptr, perm_ptr, seed_ptr,
        M, N, K, G, C, group_size, lr,
        stride_sn, stride_xm, stride_xk, stride_gom, stride_gon,
        BLOCK_PG: tl.constexpr, BLOCK_M: tl.constexpr,
    ):
        row = tl.program_id(0)
        grp = tl.program_id(1)
        seed = tl.load(seed_ptr).to(tl.uint32)
        packed_per_group = group_size // 4
        local_pg = tl.arange(0, BLOCK_PG)
        packed_group = grp * packed_per_group + local_pg
        p0 = packed_group * 4
        gmask = (local_pg < packed_per_group) & (p0 < K)
        c0_orig = tl.load(perm_ptr + p0 + 0, mask=gmask, other=0).to(tl.int32)
        c1_orig = tl.load(perm_ptr + p0 + 1, mask=gmask, other=0).to(tl.int32)
        c2_orig = tl.load(perm_ptr + p0 + 2, mask=gmask, other=0).to(tl.int32)
        c3_orig = tl.load(perm_ptr + p0 + 3, mask=gmask, other=0).to(tl.int32)
        gw0 = tl.zeros((BLOCK_PG,), dtype=tl.float32)
        gw1 = tl.zeros((BLOCK_PG,), dtype=tl.float32)
        gw2 = tl.zeros((BLOCK_PG,), dtype=tl.float32)
        gw3 = tl.zeros((BLOCK_PG,), dtype=tl.float32)
        for m0 in range(0, M, BLOCK_M):
            moff = m0 + tl.arange(0, BLOCK_M)
            mmask = moff < M
            go = tl.load(
                go_ptr + moff * stride_gom + row * stride_gon,
                mask=mmask, other=0.0,
            ).to(tl.float32)
            base_x = x_ptr + moff[:, None] * stride_xm
            mask2 = mmask[:, None] & gmask[None, :]
            x0 = tl.load(base_x + c0_orig[None, :] * stride_xk, mask=mask2, other=0.0).to(tl.float32)
            x1 = tl.load(base_x + c1_orig[None, :] * stride_xk, mask=mask2, other=0.0).to(tl.float32)
            x2 = tl.load(base_x + c2_orig[None, :] * stride_xk, mask=mask2, other=0.0).to(tl.float32)
            x3 = tl.load(base_x + c3_orig[None, :] * stride_xk, mask=mask2, other=0.0).to(tl.float32)
            gw0 += tl.sum(go[:, None] * x0, axis=0)
            gw1 += tl.sum(go[:, None] * x1, axis=0)
            gw2 += tl.sum(go[:, None] * x2, axis=0)
            gw3 += tl.sum(go[:, None] * x3, axis=0)

        base = row * stride_sn + packed_group * 3
        b0 = tl.load(state_ptr + base + 0, mask=gmask, other=0).to(tl.int32)
        b1 = tl.load(state_ptr + base + 1, mask=gmask, other=0).to(tl.int32)
        b2 = tl.load(state_ptr + base + 2, mask=gmask, other=0).to(tl.int32)
        code0 = b0 & 0x3F
        code1 = ((b0 >> 6) | (b1 << 2)) & 0x3F
        code2 = ((b1 >> 4) | (b2 << 4)) & 0x3F
        code3 = (b2 >> 2) & 0x3F
        s_old = tl.load(scale_old_ptr + row * G + grp)
        s_new = tl.load(scale_ptr + row * G + grp)
        denom = tl.load(denom_ptr + row)
        lv = 2 * C - 1
        Cf = C * 1.0
        nc0 = _tick(code0, gw0, row, c0_orig, K, lv, C, Cf, lr, denom, s_old, s_new, seed)
        nc1 = _tick(code1, gw1, row, c1_orig, K, lv, C, Cf, lr, denom, s_old, s_new, seed)
        nc2 = _tick(code2, gw2, row, c2_orig, K, lv, C, Cf, lr, denom, s_old, s_new, seed)
        nc3 = _tick(code3, gw3, row, c3_orig, K, lv, C, Cf, lr, denom, s_old, s_new, seed)
        p0b = (nc0 | (nc1 << 6)) & 0xFF
        p1b = ((nc1 >> 2) | (nc2 << 4)) & 0xFF
        p2b = ((nc2 >> 4) | (nc3 << 2)) & 0xFF
        tl.store(state_ptr + base + 0, p0b.to(tl.uint8), mask=gmask)
        tl.store(state_ptr + base + 1, p1b.to(tl.uint8), mask=gmask)
        tl.store(state_ptr + base + 2, p2b.to(tl.uint8), mask=gmask)

    @triton.jit
    def _group_stats_dense_kernel(
        state_ptr, gw_ptr, grad_scale_ptr, gsq_ptr,
        K, G, C, group_size, residual_alpha,
        stride_sn, stride_gwn,
        BLOCK_K: tl.constexpr,
    ):
        # Same outputs as _group_stats_from_io_kernel, but gw is READ from a precomputed
        # act-ordered [N,K] fp32 correlation (cuBLAS) instead of an M-loop dot -- the L2
        # "semi-strict" path of the optimization plan. Group-boundary mask included.
        row = tl.program_id(0)
        grp = tl.program_id(1)
        offs = tl.arange(0, BLOCK_K)
        p = grp * group_size + offs
        pmask = (offs < group_size) & (p < K)
        gw = tl.load(gw_ptr + row * stride_gwn + p, mask=pmask, other=0.0)
        packed_group = p // 4
        lane = p % 4
        base = row * stride_sn + packed_group * 3
        b0 = tl.load(state_ptr + base + 0, mask=pmask, other=0).to(tl.int32)
        b1 = tl.load(state_ptr + base + 1, mask=pmask, other=0).to(tl.int32)
        b2 = tl.load(state_ptr + base + 2, mask=pmask, other=0).to(tl.int32)
        code = _decode6(b0, b1, b2, lane)
        lv = 2 * C - 1
        Cf = C * 1.0
        t = code // lv - 1
        c = code % lv - (C - 1)
        visible = t.to(tl.float32) + residual_alpha * c.to(tl.float32) / Cf
        count = tl.sum(pmask.to(tl.float32), axis=0)
        gscale = tl.sum(gw * visible, axis=0) / tl.sqrt(tl.maximum(count, 1.0))
        g2 = tl.sum(gw * gw, axis=0)
        tl.store(grad_scale_ptr + row * G + grp, gscale)
        tl.store(gsq_ptr + row * G + grp, g2)

    @triton.jit
    def _group_state_dense_kernel(
        state_ptr, scale_old_ptr, scale_ptr, denom_ptr, gw_ptr, perm_ptr, seed_ptr,
        K, G, C, group_size, lr,
        stride_sn, stride_gwn,
        BLOCK_PG: tl.constexpr,
    ):
        # Same transition as _group_state_from_io_kernel with the correlation read from gw_ptr.
        row = tl.program_id(0)
        grp = tl.program_id(1)
        seed = tl.load(seed_ptr).to(tl.uint32)
        packed_per_group = group_size // 4
        local_pg = tl.arange(0, BLOCK_PG)
        packed_group = grp * packed_per_group + local_pg
        p0 = packed_group * 4
        gmask = (local_pg < packed_per_group) & (p0 < K)
        c0_orig = tl.load(perm_ptr + p0 + 0, mask=gmask, other=0).to(tl.int32)
        c1_orig = tl.load(perm_ptr + p0 + 1, mask=gmask, other=0).to(tl.int32)
        c2_orig = tl.load(perm_ptr + p0 + 2, mask=gmask, other=0).to(tl.int32)
        c3_orig = tl.load(perm_ptr + p0 + 3, mask=gmask, other=0).to(tl.int32)
        gw_base = gw_ptr + row * stride_gwn
        gw0 = tl.load(gw_base + p0 + 0, mask=gmask, other=0.0)
        gw1 = tl.load(gw_base + p0 + 1, mask=gmask, other=0.0)
        gw2 = tl.load(gw_base + p0 + 2, mask=gmask, other=0.0)
        gw3 = tl.load(gw_base + p0 + 3, mask=gmask, other=0.0)
        base = row * stride_sn + packed_group * 3
        b0 = tl.load(state_ptr + base + 0, mask=gmask, other=0).to(tl.int32)
        b1 = tl.load(state_ptr + base + 1, mask=gmask, other=0).to(tl.int32)
        b2 = tl.load(state_ptr + base + 2, mask=gmask, other=0).to(tl.int32)
        code0 = b0 & 0x3F
        code1 = ((b0 >> 6) | (b1 << 2)) & 0x3F
        code2 = ((b1 >> 4) | (b2 << 4)) & 0x3F
        code3 = (b2 >> 2) & 0x3F
        s_old = tl.load(scale_old_ptr + row * G + grp)
        s_new = tl.load(scale_ptr + row * G + grp)
        denom = tl.load(denom_ptr + row)
        lv = 2 * C - 1
        Cf = C * 1.0
        nc0 = _tick(code0, gw0, row, c0_orig, K, lv, C, Cf, lr, denom, s_old, s_new, seed)
        nc1 = _tick(code1, gw1, row, c1_orig, K, lv, C, Cf, lr, denom, s_old, s_new, seed)
        nc2 = _tick(code2, gw2, row, c2_orig, K, lv, C, Cf, lr, denom, s_old, s_new, seed)
        nc3 = _tick(code3, gw3, row, c3_orig, K, lv, C, Cf, lr, denom, s_old, s_new, seed)
        p0b = (nc0 | (nc1 << 6)) & 0xFF
        p1b = ((nc1 >> 2) | (nc2 << 4)) & 0xFF
        p2b = ((nc2 >> 4) | (nc3 << 2)) & 0xFF
        tl.store(state_ptr + base + 0, p0b.to(tl.uint8), mask=gmask)
        tl.store(state_ptr + base + 1, p1b.to(tl.uint8), mask=gmask)
        tl.store(state_ptr + base + 2, p2b.to(tl.uint8), mask=gmask)


def _dtype_flags(tensor: torch.Tensor) -> tuple[bool, bool]:
    return tensor.dtype == torch.bfloat16, tensor.dtype == torch.float16


def triton_group_decode_matmul(
    x: torch.Tensor,
    state_packed_perm: torch.Tensor,
    scale: torch.Tensor,
    perm: torch.Tensor,
    *,
    C: int,
    group: int,
    residual_alpha: float = 0.0,
) -> torch.Tensor:
    if not HAS_TRITON:
        raise RuntimeError("triton not available")
    if not (x.is_cuda and state_packed_perm.is_cuda and scale.is_cuda and perm.is_cuda):
        raise ValueError("group Triton forward requires CUDA tensors")
    x = x.contiguous()
    state = state_packed_perm.contiguous()
    scale = scale.contiguous()
    perm = perm.contiguous()
    M, K = x.shape
    N, G = scale.shape
    if K % 4 or group % 4 or perm.numel() != K:
        raise ValueError("invalid packed group layout")
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    BM, BN, BK = 32, 32, 32
    bf16, fp16 = _dtype_flags(x)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _group_decode_matmul_kernel[grid](
        x, state, scale, perm, y,
        M, N, K, G, C, group, float(residual_alpha),
        x.stride(0), x.stride(1), state.stride(0), scale.stride(0), scale.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, X_BF16=bf16, X_FP16=fp16,
    )
    return y


def triton_group_grad_x(
    grad_out: torch.Tensor,
    state_packed_perm: torch.Tensor,
    scale: torch.Tensor,
    perm: torch.Tensor,
    *,
    in_features: int,
    C: int,
    group: int,
    residual_alpha: float = 0.0,
) -> torch.Tensor:
    if not HAS_TRITON:
        raise RuntimeError("triton not available")
    go = grad_out.contiguous()
    state = state_packed_perm.contiguous()
    scale = scale.contiguous()
    perm = perm.contiguous()
    M, N = go.shape
    K = int(in_features)
    if not (go.is_cuda and state.is_cuda and scale.is_cuda and perm.is_cuda):
        raise ValueError("group Triton grad_x requires CUDA tensors")
    gx = torch.empty((M, K), device=go.device, dtype=go.dtype)
    BM, BN, BK = 32, 32, 32
    bf16, fp16 = _dtype_flags(go)
    grid = (triton.cdiv(M, BM), triton.cdiv(K, BK))
    _group_gradx_kernel[grid](
        go, state, scale, perm, gx,
        M, N, K, scale.shape[1], C, group, float(residual_alpha),
        go.stride(0), go.stride(1), state.stride(0), scale.stride(0), scale.stride(1),
        gx.stride(0), gx.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GO_BF16=bf16, GO_FP16=fp16,
    )
    return gx


@torch.no_grad()
def triton_group_counter_update_from_io(
    state_packed_perm: torch.Tensor,
    scale: torch.Tensor,
    v: torch.Tensor,
    x: torch.Tensor,
    grad_out: torch.Tensor,
    perm: torch.Tensor,
    *,
    group: int,
    C: int,
    lr: float,
    lr_scale: float,
    rms_beta: float,
    rms_eps: float,
    seed: int,
    residual_alpha: float = 0.0,
    lagged: bool = False,
    clip: float = 0.0,
) -> None:
    """Strict group update with no dense grad_w.

    Three launches are used: group statistics, O(out*groups) scale/RMS finalization, then state
    update. The correlation is recomputed in the last launch; this trades FLOPs for bounded memory
    and avoids the giant one-row program used by the original strict kernel at Qwen FFN widths.
    """
    # Contract check first: the rejection must fire on any machine, with or without Triton.
    if group & (group - 1):
        raise ValueError(
            "strict Triton group update requires a power-of-two group size; "
            "use group=32/64/128/256 or the torch reference path"
        )
    if not HAS_TRITON:
        raise RuntimeError("triton not available")
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    go2 = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()
    state = state_packed_perm
    if not all(t.is_cuda for t in (state, scale, v, x2, go2, perm)):
        raise ValueError("strict group update requires CUDA tensors")
    N, G = scale.shape
    M, K = x2.shape
    if go2.shape != (M, N):
        raise ValueError("grad_out shape mismatch")
    if K % 4 or group % 4 or G != (K + group - 1) // group:
        raise ValueError("invalid group layout")
    scale_old = scale.clone()
    grad_scale = torch.empty_like(scale, dtype=torch.float32)
    gsq = torch.empty_like(scale, dtype=torch.float32)
    denom = torch.empty((N,), device=scale.device, dtype=torch.float32)
    seed_t = torch.tensor([int(seed) & 0xFFFFFFFF], dtype=torch.int64, device=x2.device)

    block_k = triton.next_power_of_2(group)
    _group_stats_from_io_kernel[(N, G)](
        state, x2, go2, perm, grad_scale, gsq,
        M, N, K, G, C, group, float(residual_alpha),
        state.stride(0), x2.stride(0), x2.stride(1), go2.stride(0), go2.stride(1),
        BLOCK_K=block_k, BLOCK_M=16,
    )
    block_g = triton.next_power_of_2(G)
    _group_finalize_stats_kernel[(N,)](
        scale, v.reshape(N), grad_scale, gsq, denom,
        N, K, G, float(lr_scale), float(rms_beta), float(rms_eps), float(clip),
        BLOCK_G=block_g, LAGGED=lagged,
    )
    block_pg = triton.next_power_of_2(group // 4)
    _group_state_from_io_kernel[(N, G)](
        state, scale_old, scale, denom, x2, go2, perm, seed_t,
        M, N, K, G, C, group, float(lr),
        state.stride(0), x2.stride(0), x2.stride(1), go2.stride(0), go2.stride(1),
        BLOCK_PG=block_pg, BLOCK_M=16,
    )


@torch.no_grad()
def triton_group_counter_update_dense(
    state_packed_perm: torch.Tensor,
    scale: torch.Tensor,
    v: torch.Tensor,
    grad_w_perm: torch.Tensor,
    perm: torch.Tensor,
    *,
    group: int,
    C: int,
    lr: float,
    lr_scale: float,
    rms_beta: float,
    rms_eps: float,
    seed: int,
    residual_alpha: float = 0.0,
    lagged: bool = False,
    clip: float = 0.0,
) -> None:
    """Semi-strict update (plan L2): the correlation arrives as a precomputed act-ordered
    [N,K] fp32 grad (one cuBLAS GEMM + gather at the caller), and three cheap launches do
    group stats, scale/RMS finalization and the SR state transition. Same math and SR keys
    as the from-IO kernels/reference; the O(M) in-kernel dot loops are gone."""
    if not HAS_TRITON:
        raise RuntimeError("triton not available")
    state = state_packed_perm
    N, G = scale.shape
    K = perm.numel()
    gw = grad_w_perm.contiguous()
    if not all(t.is_cuda for t in (state, scale, v, gw, perm)):
        raise ValueError("dense group update requires CUDA tensors")
    if gw.shape != (N, K) or gw.dtype != torch.float32:
        raise ValueError("grad_w_perm must be fp32 [out, in] in act-order")
    if K % 4 or group % 4 or G != (K + group - 1) // group:
        raise ValueError("invalid group layout")
    scale_old = scale.clone()
    grad_scale = torch.empty_like(scale, dtype=torch.float32)
    gsq = torch.empty_like(scale, dtype=torch.float32)
    denom = torch.empty((N,), device=scale.device, dtype=torch.float32)
    seed_t = torch.tensor([int(seed) & 0xFFFFFFFF], dtype=torch.int64, device=gw.device)

    block_k = triton.next_power_of_2(group)
    _group_stats_dense_kernel[(N, G)](
        state, gw, grad_scale, gsq,
        K, G, C, group, float(residual_alpha),
        state.stride(0), gw.stride(0),
        BLOCK_K=block_k,
    )
    block_g = triton.next_power_of_2(G)
    _group_finalize_stats_kernel[(N,)](
        scale, v.reshape(N), grad_scale, gsq, denom,
        N, K, G, float(lr_scale), float(rms_beta), float(rms_eps), float(clip),
        BLOCK_G=block_g, LAGGED=lagged,
    )
    block_pg = triton.next_power_of_2(group // 4)
    _group_state_dense_kernel[(N, G)](
        state, scale_old, scale, denom, gw, perm, seed_t,
        K, G, C, group, float(lr),
        state.stride(0), gw.stride(0),
        BLOCK_PG=block_pg,
    )
