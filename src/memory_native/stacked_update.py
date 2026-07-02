"""Fused batched counter update for STACKED experts — CPU reference + Triton kernel.

The profile of the d=1536 MN-GLM step showed the batched torch update (_batched_rms_update:
decode/RMS/SR/encode as ~15 separate elementwise kernels over [E,out,in]) eating ~25-30% of the
step. This module collapses it into ONE launch: one program per (expert,row), codes stay uint8
in-register, RMS row-stats + deterministic hash-SR tick + re-encode in a single pass over HBM.

The stacked state is UNPACKED 6-bit codes in uint8 (1 B/weight; the expert stack is small), so no
bit-packing in the kernel — it reuses `_tick` from fused_update (same 6-bit code automaton, same
hash-SR as the packed kernel / the OpenCL engine).

SR family note: the torch fallback (_batched_rms_update) draws torch.rand; this kernel (and its CPU
reference here) uses DETERMINISTIC hash-SR — the same in-family switch PackedRMSCounterLinear makes
when its fused kernel fires. `stacked_update_hashsr` is the exact reference the kernel mirrors;
kernel-vs-reference matches up to one SR quantum on an O(1) fraction (chunked fp reduction), like
the packed kernel. grad_w may be fp32 OR bf16 — the kernel casts in-register (no fp32 copy), which
is what lets the bf16 GEMM path feed the update directly.
"""
from __future__ import annotations

import torch

from .fused_update import HAS_TRITON, counter_update_hashsr

__all__ = ["stacked_update_hashsr", "triton_stacked_update", "HAS_TRITON"]


@torch.no_grad()
def stacked_update_hashsr(state: torch.Tensor, scale: torch.Tensor, v: torch.Tensor,
                          grad_w: torch.Tensor, active: torch.Tensor, *, C: int, lr: float,
                          lr_scale: float, rms_beta: float, rms_eps: float, seed: int) -> None:
    """CPU/torch reference: the whole [E,out,in] stack updated as E*out independent rows with
    GLOBAL element indexing (elem = row*in + col), exactly what the kernel computes. Mutates
    state/scale/v in place; rows of inactive experts are left untouched."""
    E, out, in_ = state.shape
    snap = None
    if not bool(active.all()):
        idx = (~active).nonzero(as_tuple=True)[0]
        snap = (idx, state[idx].clone(), scale[idx].clone(), v[idx].clone())
    codes = state.reshape(E * out, in_)
    sc = scale.reshape(E * out, 1)
    vv = v.reshape(E * out, 1)
    new_codes = counter_update_hashsr(codes, sc, vv, grad_w.reshape(E * out, in_).float(),
                                      C=C, lr=lr, lr_scale=lr_scale, rms_beta=rms_beta,
                                      rms_eps=rms_eps, seed=seed)
    state.copy_(new_codes.reshape(E, out, in_))
    if snap is not None:
        idx, s0, sc0, v0 = snap
        state[idx] = s0
        scale[idx] = sc0
        v[idx] = v0


if HAS_TRITON:
    import triton
    import triton.language as tl

    from .fused_update import _tick

    @triton.jit
    def _stacked_update_kernel(state_ptr, scale_ptr, v_ptr, grad_ptr, active_ptr, seed_ptr,
                               C, IN, OUT, lr, lr_scale, rms_beta, rms_eps,
                               BLOCK_I: tl.constexpr):
        # One program per (expert,row) = one flat row of the [E*out, in] stack. Codes are unpacked
        # uint8, so decode is a load; the whole RMS+SR automaton runs in-register in one HBM pass.
        row = tl.program_id(0)
        e = row // OUT
        if tl.load(active_ptr + e) == 0:                       # expert got no tokens this step
            return
        seed = tl.load(seed_ptr).to(tl.uint32)
        lv = 2 * C - 1
        Cf = C * 1.0
        # pass 1: row stats (g_sq, grad_s) — needs t from the codes
        g_sq = tl.zeros((), dtype=tl.float32)
        grad_s = tl.zeros((), dtype=tl.float32)
        for i0 in range(0, IN, BLOCK_I):
            offs = i0 + tl.arange(0, BLOCK_I)
            mask = offs < IN
            gw = tl.load(grad_ptr + row * IN + offs, mask=mask, other=0.0).to(tl.float32)
            code = tl.load(state_ptr + row * IN + offs, mask=mask, other=0).to(tl.int32)
            t = code // lv - 1
            g_sq += tl.sum(gw * gw, axis=0)
            grad_s += tl.sum(gw * t.to(tl.float32), axis=0)
        g_sq = g_sq / IN
        grad_s = grad_s / tl.sqrt(IN.to(tl.float32))
        s_old = tl.load(scale_ptr + row)
        vv = rms_beta * tl.load(v_ptr + row) + (1.0 - rms_beta) * g_sq
        tl.store(v_ptr + row, vv)
        denom = tl.maximum(tl.sqrt(vv), rms_eps)               # exact mode: freshly-updated v
        s_new = tl.minimum(tl.maximum(s_old - lr_scale * grad_s, 1e-5), 10.0)
        # pass 2: per-element SR tick (global elem index row*IN+col matches the reference)
        for i0 in range(0, IN, BLOCK_I):
            offs = i0 + tl.arange(0, BLOCK_I)
            mask = offs < IN
            gw = tl.load(grad_ptr + row * IN + offs, mask=mask, other=0.0).to(tl.float32)
            code = tl.load(state_ptr + row * IN + offs, mask=mask, other=0).to(tl.int32)
            nc = _tick(code, gw, row, offs, IN, lv, C, Cf, lr, denom, s_old, s_new, seed)
            tl.store(state_ptr + row * IN + offs, nc.to(tl.uint8), mask=mask)
        tl.store(scale_ptr + row, s_new)


def triton_stacked_update(state: torch.Tensor, scale: torch.Tensor, v: torch.Tensor,
                          grad_w: torch.Tensor, active: torch.Tensor, *, C: int, lr: float,
                          lr_scale: float, rms_beta: float, rms_eps: float, seed: int) -> None:
    """One-launch fused RMS + hash-SR update over the whole [E,out,in] expert stack. Mutates
    state/scale/v in place; inactive experts untouched. grad_w may be fp32 or bf16 (cast
    in-register). Matches stacked_update_hashsr up to one SR quantum on an O(1) fraction."""
    if not HAS_TRITON:
        raise RuntimeError("triton not available")
    E, out, in_ = state.shape
    seed_t = torch.tensor([int(seed) & 0xFFFFFFFF], dtype=torch.int64, device=state.device)
    _stacked_update_kernel[(E * out,)](
        state.reshape(E * out, in_), scale.reshape(E * out), v.reshape(E * out),
        grad_w.reshape(E * out, in_).contiguous(), active.to(torch.uint8).contiguous(), seed_t,
        C, in_, out, lr, lr_scale, rms_beta, rms_eps, BLOCK_I=256,
    )
