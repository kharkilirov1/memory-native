"""Fused counter update — CPU reference + Triton kernel.

The hot part of the counter backward is the per-element update (RMS row-stats + stochastic-
rounding tick + carry/remainder + re-encode), today ~15 small torch ops. Untiling already cut
the launch overhead 2.9x; a single fused kernel collapses the rest into one launch.

The stochastic rounding here is DETERMINISTIC (a hash of seed^element index, like the engine's
OpenCL kernel), not torch.rand. That makes the update REPRODUCIBLE (same inputs+seed -> same
output every run, kernel-vs-kernel bit-for-bit), so:
  * counter_update_hashsr() is a pure-torch reference of exactly what the kernel computes;
  * triton_counter_update() (the kernel) matches the reference up to ONE SR quantum on an O(1)
    fraction of weights -- NOT bit-for-bit: the kernel reduces the per-row RMS stats (g_sq, grad_s)
    in BLOCK_I chunks, and fp addition is non-associative, so the denominator differs by ~1e-7 and
    can tip a stochastic-rounding boundary. The dynamics are identical; individual codes can differ
    by 1. (To make it truly bit-exact, reduce each row in a single pass.)
This module ships the verified reference now; the Triton kernel is validated against it on a GPU.
"""
from __future__ import annotations

import torch

from .counter import decode_state, encode_state

__all__ = ["hash_u32", "uniform01", "counter_update_hashsr", "HAS_TRITON", "triton_counter_update"]

_M1 = 0x7feb352d
_M2 = 0x846ca68b


def hash_u32(x: torch.Tensor) -> torch.Tensor:
    """MurmurHash-style uint32 hash (matches the OpenCL cc_hash_u32), computed in int64."""
    x = x.to(torch.int64) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * _M1) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * _M2) & 0xFFFFFFFF
    x ^= x >> 16
    return x & 0xFFFFFFFF


def uniform01(x: torch.Tensor) -> torch.Tensor:
    return (hash_u32(x) & 0x00FFFFFF).to(torch.float32) * (1.0 / 16777216.0)


@torch.no_grad()
def counter_update_hashsr(codes: torch.Tensor, scale: torch.Tensor, v: torch.Tensor,
                          grad_w: torch.Tensor, *, C: int, lr: float, lr_scale: float,
                          rms_beta: float, rms_eps: float, seed: int,
                          lagged: bool = False) -> torch.Tensor:
    """Deterministic-SR RMS counter update on UNPACKED codes [out,in]. Mutates scale and v in
    place, returns the new codes. Exactly the math the Triton kernel implements (per row).

    lagged=False (default, "exact"): the RMS denominator uses THIS step's freshly-updated v, so the
      per-element tick depends on a full-row reduction (g_sq) of the current grad -> two passes, and
      the state-write of weight [o,i] depends on the whole row's grad (NOT epilogue-fusable).
    lagged=True ("one-pass"): the denominator uses the PREVIOUS step's v (read before the EMA
      update), so given last step's v the tick of [o,i] depends ONLY on grad_w[o,i] -> the
      state-write is PER-ELEMENT and fuses into a tiled GEMM epilogue (fusion-plan lever #1). The
      v-EMA still needs the row's g_sq, but it emits only O(out) values (fold into split-K). This is
      the hash-SR analogue of RMSCounterLinear's rms_mode='lagged'."""
    out, in_ = codes.shape
    t, c = decode_state(codes, C)                       # int16 [out,in]
    t = t.float(); c = c.float()
    gw = grad_w.float()

    # --- row stats (per output row) ---
    g_sq = gw.pow(2).mean(dim=1, keepdim=True)          # [out,1]
    if lagged:
        denom = v.sqrt().clamp_min(rms_eps)             # previous-step v -> tick is per-element
        v.mul_(rms_beta).add_(g_sq, alpha=1.0 - rms_beta)
    else:
        v.mul_(rms_beta).add_(g_sq, alpha=1.0 - rms_beta)
        denom = v.sqrt().clamp_min(rms_eps)             # freshly-updated v -> full-row dependency
    grad_s = (gw * t).sum(dim=1, keepdim=True) / (in_ ** 0.5)
    s_old = scale.clone()
    s_new = (s_old - lr_scale * grad_s).clamp_(1e-5, 10.0)

    # --- per-element deterministic stochastic-rounding tick ---
    elem = torch.arange(out * in_, device=codes.device).reshape(out, in_)
    rnd = uniform01((seed ^ hash_u32(elem)) & 0xFFFFFFFF)
    tick = (-lr) * (gw / denom) * (C / s_new)
    c_reb = c * (s_old / s_new)
    val = c_reb + tick
    f = torch.floor(val)
    cc = f + (rnd < (val - f)).to(val.dtype)
    carry = torch.trunc(cc / C)
    rem = cc - carry * C
    nt = t + carry
    ct = nt.clamp(-1, 1)
    rem = torch.where(ct != nt, torch.sign(cc) * (C - 1), rem).clamp_(-(C - 1), C - 1)

    scale.copy_(s_new)
    return encode_state(ct, rem, C)


try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _hash_u32(x):
        # uint32 semantics: logical shifts + wrapping mul (0x846ca68b overflows int32).
        x = x.to(tl.uint32)
        x = x ^ (x >> 16)
        x = x * 0x7feb352d
        x = x ^ (x >> 15)
        x = x * 0x846ca68b
        x = x ^ (x >> 16)
        return x

    @triton.jit
    def _tick(code, gw, row, col, in_features, lv, C, Cf, lr, denom, s_old, s_new, seed):
        """SR counter tick for one weight; returns the new 6-bit code. Mirrors counter_update_hashsr."""
        t = code // lv - 1
        c = code % lv - (C - 1)
        tick = (-lr) * (gw / denom) * (Cf / s_new)
        val = c.to(tl.float32) * (s_old / s_new) + tick
        fl = tl.floor(val)
        elem = row * in_features + col
        rnd = (_hash_u32(seed ^ _hash_u32(elem)) & 0x00FFFFFF).to(tl.float32) * (1.0 / 16777216.0)
        cc = fl + tl.where(rnd < (val - fl), 1.0, 0.0)
        carry = tl.where(cc >= 0, tl.floor(cc / Cf), tl.ceil(cc / Cf))
        rem = cc - carry * Cf
        nt = t.to(tl.float32) + carry
        ct = tl.minimum(tl.maximum(nt, -1.0), 1.0)
        sgn = tl.where(cc > 0, 1.0, tl.where(cc < 0, -1.0, 0.0))
        rem = tl.where(ct != nt, sgn * (Cf - 1.0), rem)
        rem = tl.minimum(tl.maximum(rem, -(Cf - 1.0)), Cf - 1.0)
        return ((ct.to(tl.int32) + 1) * lv + (rem.to(tl.int32) + (C - 1))) & 0x3F

    @triton.jit
    def _counter_update_kernel(state_ptr, scale_ptr, v_ptr, grad_w_ptr, seed_ptr,
                               C, in_features, lr, lr_scale, rms_beta, rms_eps,
                               BLOCK_I: tl.constexpr, LAGGED: tl.constexpr = False):
        # One program per output row. state packed [out, (in/4)*3]; grad_w dense [out,in].
        # seed is loaded from a tensor (not a scalar arg) so Triton never specializes seed==0/1
        # to a Python int -- that would break the `seed ^ hash` uint32 arithmetic.
        row = tl.program_id(0)
        seed = tl.load(seed_ptr).to(tl.uint32)
        gpr = in_features // 4
        lv = 2 * C - 1
        # pass 1: row stats (needs t from the packed state)
        g_sq = tl.zeros((), dtype=tl.float32)
        grad_s = tl.zeros((), dtype=tl.float32)
        for i0 in range(0, in_features, BLOCK_I):
            offs = i0 + tl.arange(0, BLOCK_I)
            mask = offs < in_features
            gw = tl.load(grad_w_ptr + row * in_features + offs, mask=mask, other=0.0)
            group = offs // 4
            lane = offs % 4
            base = row * gpr * 3 + group * 3
            b0 = tl.load(state_ptr + base + 0, mask=mask, other=0).to(tl.int32)
            b1 = tl.load(state_ptr + base + 1, mask=mask, other=0).to(tl.int32)
            b2 = tl.load(state_ptr + base + 2, mask=mask, other=0).to(tl.int32)
            c0 = b0 & 0x3F
            c1 = ((b0 >> 6) | (b1 << 2)) & 0x3F
            c2 = ((b1 >> 4) | (b2 << 4)) & 0x3F
            c3 = (b2 >> 2) & 0x3F
            code = tl.where(lane == 0, c0, tl.where(lane == 1, c1, tl.where(lane == 2, c2, c3)))
            t = code // lv - 1
            g_sq += tl.sum(gw * gw, axis=0)
            grad_s += tl.sum(gw * t.to(tl.float32), axis=0)
        g_sq = g_sq / in_features
        grad_s = grad_s / tl.sqrt(in_features.to(tl.float32))
        s_old = tl.load(scale_ptr + row)
        v_old = tl.load(v_ptr + row)
        vv = rms_beta * v_old + (1.0 - rms_beta) * g_sq
        tl.store(v_ptr + row, vv)
        # lagged: denom from the PREVIOUS v (per-element tick); exact: from the freshly-updated v.
        denom = tl.maximum(tl.sqrt(v_old if LAGGED else vv), rms_eps)
        s_new = tl.minimum(tl.maximum(s_old - lr_scale * grad_s, 1e-5), 10.0)
        Cf = C * 1.0
        # pass 2: apply per packed group of 4 (in_features % 4 == 0 so col<in_features holds)
        for g0 in range(0, gpr, BLOCK_I):
            gids = g0 + tl.arange(0, BLOCK_I)
            gmask = gids < gpr
            base = row * gpr * 3 + gids * 3
            b0 = tl.load(state_ptr + base + 0, mask=gmask, other=0).to(tl.int32)
            b1 = tl.load(state_ptr + base + 1, mask=gmask, other=0).to(tl.int32)
            b2 = tl.load(state_ptr + base + 2, mask=gmask, other=0).to(tl.int32)
            col0 = gids * 4
            gw0 = tl.load(grad_w_ptr + row * in_features + col0 + 0, mask=gmask, other=0.0)
            gw1 = tl.load(grad_w_ptr + row * in_features + col0 + 1, mask=gmask, other=0.0)
            gw2 = tl.load(grad_w_ptr + row * in_features + col0 + 2, mask=gmask, other=0.0)
            gw3 = tl.load(grad_w_ptr + row * in_features + col0 + 3, mask=gmask, other=0.0)
            nc0 = _tick(b0 & 0x3F, gw0, row, col0 + 0, in_features, lv, C, Cf, lr, denom, s_old, s_new, seed)
            nc1 = _tick(((b0 >> 6) | (b1 << 2)) & 0x3F, gw1, row, col0 + 1, in_features, lv, C, Cf, lr, denom, s_old, s_new, seed)
            nc2 = _tick(((b1 >> 4) | (b2 << 4)) & 0x3F, gw2, row, col0 + 2, in_features, lv, C, Cf, lr, denom, s_old, s_new, seed)
            nc3 = _tick((b2 >> 2) & 0x3F, gw3, row, col0 + 3, in_features, lv, C, Cf, lr, denom, s_old, s_new, seed)
            p0 = (nc0 | (nc1 << 6)) & 0xFF
            p1 = ((nc1 >> 2) | (nc2 << 4)) & 0xFF
            p2 = ((nc2 >> 4) | (nc3 << 2)) & 0xFF
            tl.store(state_ptr + base + 0, p0.to(tl.uint8), mask=gmask)
            tl.store(state_ptr + base + 1, p1.to(tl.uint8), mask=gmask)
            tl.store(state_ptr + base + 2, p2.to(tl.uint8), mask=gmask)
        tl.store(scale_ptr + row, s_new)


def triton_counter_update(state_packed: torch.Tensor, scale: torch.Tensor, v: torch.Tensor,
                          grad_w: torch.Tensor, *, C: int, lr: float, lr_scale: float,
                          rms_beta: float, rms_eps: float, seed: int, lagged: bool = False) -> None:
    """One-launch fused RMS + stochastic-rounding counter update on packed state. Mutates
    state_packed, scale, v in place. Requires CUDA + triton; matches counter_update_hashsr (same
    `lagged`) up to one SR quantum on an O(1) fraction of weights (chunked fp reduction, not
    bit-exact). lagged=True uses the previous-step v for the denominator (per-element tick)."""
    if not HAS_TRITON:
        raise RuntimeError("triton not available")
    out = scale.numel()
    in_features = grad_w.shape[1]
    assert in_features % 4 == 0
    BLOCK_I = 256
    seed_t = torch.tensor([int(seed) & 0xFFFFFFFF], dtype=torch.int64, device=grad_w.device)
    _counter_update_kernel[(out,)](
        state_packed, scale.reshape(out), v.reshape(out), grad_w.contiguous(), seed_t,
        C, in_features, lr, lr_scale, rms_beta, rms_eps,
        BLOCK_I=BLOCK_I, LAGGED=lagged,
    )
