"""Finite-state counter synapse — the memory lever for parameters+optimizer+gradients.

A weight is a per-synapse finite-state automaton: a ternary visible weight t in {-1,0,+1}
plus a residual counter c, packed into one small state. The optimizer state lives *inside*
that state (plus a per-row scale and a per-row RMS second moment), so there is no FP master
weight and no per-weight Adam moment. The update is fused into the layer's backward at row-
tile granularity, so a full-model gradient buffer is never retained.

This is pure PyTorch: it runs on CPU and CUDA with stock torch, no custom engine. It is a
*correctness / dynamics* implementation — it decodes states to ordinary tensors around the
GEMM, so it validates learning and persistent-state accounting but does not yet realize the
sub-byte bandwidth win (that needs a Triton/CUDA packed kernel; see README "Roadmap").
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "C_DEFAULT",
    "encode_state",
    "decode_state",
    "stochastic_round",
    "ternary_gradient_unbiased",
    "CompactCounterLinear",
    "RMSCounterLinear",
]

# C=8 -> counter c in {-7..+7} (15 levels), 3*15 = 45 reachable states (fits 6 bits / uint8).
# Larger C is allowed while 3*(2C-1) <= 256 (uint8); C=11 gives 63 states (best per ablation).
C_DEFAULT = 8


def stochastic_round(x: torch.Tensor) -> torch.Tensor:
    """Unbiased stochastic rounding to integers, valid for positive and negative x."""
    floor = torch.floor(x)
    return floor + (torch.rand_like(x) < (x - floor)).to(x.dtype)


def encode_state(t: torch.Tensor, c: torch.Tensor, C: int = C_DEFAULT) -> torch.Tensor:
    """Encode t in {-1,0,1}, c in {-(C-1),...,C-1} into one uint8 state."""
    levels = 2 * C - 1
    code = (t.to(torch.int16) + 1) * levels + (c.to(torch.int16) + (C - 1))
    return code.to(torch.uint8)


def decode_state(state: torch.Tensor, C: int = C_DEFAULT) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode a uint8 state into int16 ternary weight t and residual counter c."""
    levels = 2 * C - 1
    z = state.to(torch.int16)
    t = torch.div(z, levels, rounding_mode="floor") - 1
    c = torch.remainder(z, levels) - (C - 1)
    return t, c


def ternary_gradient_unbiased(g: torch.Tensor) -> torch.Tensor:
    """Row-wise unbiased ternary estimator: output in {-a,0,+a} with E[Q(g)|g]=g, a=max|g_j|."""
    amplitude = g.abs().amax(dim=1, keepdim=True)
    safe = amplitude.clamp_min(1e-30)
    p = (g.abs() / safe).clamp_(0.0, 1.0)
    event = (torch.rand_like(p) < p).to(g.dtype)
    return torch.sign(g) * event * amplitude


class _FusedCounterLinearFn(torch.autograd.Function):
    """Linear whose weight update is fused into backward at row-tile granularity.

    A scalar `tap` (requires_grad=True) is threaded through purely so the output requires
    grad and backward runs even when neither the input nor any Parameter needs a gradient --
    a counter layer fed raw input (e.g. the first layer) must still self-update. This mirrors
    the engine's "attach the update node regardless of x.requires_grad".
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, module: "CompactCounterLinear", tap: torch.Tensor) -> torch.Tensor:
        if module._outstanding_forward:
            raise RuntimeError(
                "CompactCounterLinear was reused before its previous backward. "
                "Weight sharing and gradient accumulation need an explicit scheduler."
            )
        module._outstanding_forward = True
        ctx.module = module
        ctx.save_for_backward(x)
        return module._forward_matmul(x)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x,) = ctx.saved_tensors
        module: CompactCounterLinear = ctx.module

        x2 = x.reshape(-1, x.shape[-1])
        go2 = grad_out.reshape(-1, grad_out.shape[-1])
        grad_x2 = torch.zeros((x2.shape[0], module.in_features), device=x.device, dtype=x.dtype)

        # The full weight-gradient never exists. Only [tile_rows, in_features] is live.
        for lo in range(0, module.out_features, module.tile_rows):
            hi = min(lo + module.tile_rows, module.out_features)
            t_i, c_i = module._decode_rows(lo, hi)
            s_i = module.scale[lo:hi]
            w_i = s_i.to(x.dtype) * t_i.to(x.dtype)

            go_i = go2[:, lo:hi]
            grad_x2.add_(go_i @ w_i)

            if module.training and module.update_enabled:
                grad_w_i = (go_i.transpose(0, 1) @ x2).float()
                module._update_tile(lo, hi, grad_w_i, t_i, c_i, s_i)

        module._outstanding_forward = False
        # grads for (x, module, tap); tap's grad is unused.
        return grad_x2.reshape_as(x), None, None


class CompactCounterLinear(nn.Module):
    """Ternary linear layer trained as a finite-state synaptic automaton.

    Persistent per-weight state in this implementation is 1 byte (uint8); the logical
    packed state is ceil(log2(states)) = 6 bits. Row scales stay FP32 (their O(out)
    cost is negligible). No FP master weight, no per-weight Adam moment.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        C: int = C_DEFAULT,
        lr: float = 0.04,
        lr_scale: float = 2e-4,
        init_gain: float = 1.0,
        tile_rows: int = 64,
        local_grad_clip: float = 0.0,
        pulse_mode: str = "direct",
    ) -> None:
        super().__init__()
        if 3 * (2 * C - 1) > 256:
            raise ValueError("C is too large for uint8 state encoding (need 3*(2C-1) <= 256)")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.C = int(C)
        self.lr = float(lr)
        self.lr_scale = float(lr_scale)
        self.tile_rows = int(tile_rows)
        self.local_grad_clip = float(local_grad_clip)
        if pulse_mode not in {"direct", "ternary"}:
            raise ValueError("pulse_mode must be 'direct' or 'ternary'")
        self.pulse_mode = pulse_mode
        self.update_enabled = True
        self._outstanding_forward = False

        t0 = torch.randint(-1, 2, (out_features, in_features), dtype=torch.int16)
        c0 = torch.zeros_like(t0)
        self.register_buffer("state", encode_state(t0, c0, C))

        # Var(t)=2/3 for uniform {-1,0,1}; choose Var(w)=gain^2/fan_in.
        s0 = init_gain * math.sqrt(3.0 / (2.0 * in_features))
        self.register_buffer("scale", torch.full((out_features, 1), s0, dtype=torch.float32))

        # Diagnostics only: O(1) scalars, not per-weight optimizer state.
        self.register_buffer("update_events", torch.zeros((), dtype=torch.int64), persistent=False)
        self.register_buffer("weight_flips", torch.zeros((), dtype=torch.int64), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            # tap forces the output to require grad so backward (and the self-update) runs
            # even when x and all Parameters need no gradient.
            tap = torch.zeros((), device=x.device, dtype=x.dtype, requires_grad=True)
            return _FusedCounterLinearFn.apply(x, self, tap)
        # inference: pure forward, no update, no graph.
        return self._forward_matmul(x)

    # --- storage abstraction (overridden by the packed-6bit subclass) -------------
    # Base storage is one uint8 code per weight in `state` [out, in]. A subclass may store
    # `state` packed (4 codes / 3 bytes) and override these three to pack/unpack at the
    # boundary; the autograd Function and _update_tile go through them, so the update math
    # is shared and the persistent footprint is whatever the storage chooses.
    def _forward_matmul(self, x: torch.Tensor) -> torch.Tensor:
        # y = x @ W^T. Base path materializes the dense weight; a kernel subclass (Triton)
        # may override this to decode the packed state inside the GEMM with no dense weight.
        return F.linear(x, self._dense_weight(x.dtype))

    def _dense_weight(self, dtype: torch.dtype) -> torch.Tensor:
        t, _ = decode_state(self.state, self.C)
        return self.scale.to(dtype) * t.to(dtype)

    def _decode_rows(self, lo: int, hi: int) -> tuple[torch.Tensor, torch.Tensor]:
        return decode_state(self.state[lo:hi], self.C)

    def _write_rows(self, lo: int, hi: int, t: torch.Tensor, c: torch.Tensor) -> None:
        self.state[lo:hi].copy_(encode_state(t, c, self.C))

    @torch.no_grad()
    def _update_tile(self, lo, hi, grad_w, t_i, c_i, s_i) -> None:
        if self.local_grad_clip > 0:
            row_norm = grad_w.norm(dim=1, keepdim=True).clamp_min(1e-30)
            grad_w = grad_w * (self.local_grad_clip / row_norm).clamp_max(1.0)

        # Learn one scale per output row; normalize by sqrt(fan_in) for width stability.
        grad_s = (grad_w * t_i.float()).sum(dim=1, keepdim=True) / math.sqrt(self.in_features)
        s_new = (s_i - self.lr_scale * grad_s).clamp_(1e-5, 10.0)

        update_signal = (
            grad_w if self.pulse_mode == "direct" else ternary_gradient_unbiased(grad_w)
        )
        # c stores a pending update in units of s/C; rebase it when the row scale changes.
        c_rebased = c_i.float() * (s_i / s_new)
        ticks = (-self.lr * update_signal) * (self.C / s_new)
        cc = stochastic_round(c_rebased + ticks)

        carry = torch.trunc(cc / self.C)
        remainder = cc - carry * self.C
        proposed_t = t_i.float() + carry
        new_t = proposed_t.clamp_(-1, 1)
        blocked = proposed_t != new_t
        remainder = torch.where(
            blocked, torch.sign(cc) * (self.C - 1), remainder
        ).clamp_(-(self.C - 1), self.C - 1)

        self._write_rows(lo, hi, new_t, remainder)
        self.scale[lo:hi].copy_(s_new)
        self.update_events.add_(int((cc != c_i).sum().item()))
        self.weight_flips.add_(int((new_t != t_i).sum().item()))

    @torch.no_grad()
    def state_statistics(self) -> dict[str, float]:
        t, c = decode_state(self.state, self.C)
        return {
            "minus": float((t == -1).float().mean()),
            "zero": float((t == 0).float().mean()),
            "plus": float((t == 1).float().mean()),
            "counter_abs_mean": float(c.float().abs().mean()),
            "counter_edge": float((c.abs() == self.C - 1).float().mean()),
            "scale_mean": float(self.scale.mean()),
        }


class RMSCounterLinear(CompactCounterLinear):
    """CompactCounterLinear + per-row RMS adaptive scaling (the cheap analogue of Adam's
    variance term). A per-output-row second moment v (O(out_features), negligible memory)
    normalizes the gradient before it drives the counter. This closes most of the gap to
    AdamW that vanilla counter-SGD leaves open."""

    def __init__(self, *args, rms_beta: float = 0.9, rms_eps: float = 1e-3,
                 use_rms: bool = True, **kw) -> None:
        super().__init__(*args, **kw)
        self.rms_beta = float(rms_beta)
        self.rms_eps = float(rms_eps)
        self.use_rms = bool(use_rms)
        self.register_buffer("v", torch.zeros((self.out_features, 1), dtype=torch.float32))

    @torch.no_grad()
    def _update_tile(self, lo, hi, grad_w, t_i, c_i, s_i) -> None:
        if self.use_rms:
            g_sq = grad_w.pow(2).mean(dim=1, keepdim=True)
            self.v[lo:hi].mul_(self.rms_beta).add_(g_sq, alpha=1.0 - self.rms_beta)
            denom = self.v[lo:hi].sqrt().clamp_min(self.rms_eps)
            grad_eff = grad_w / denom
        else:
            grad_eff = grad_w

        if self.local_grad_clip > 0:
            row_norm = grad_eff.norm(dim=1, keepdim=True).clamp_min(1e-30)
            grad_eff = grad_eff * (self.local_grad_clip / row_norm).clamp_max(1.0)

        # scale is still learned from the RAW gradient (its statistics are not normalised).
        grad_s = (grad_w * t_i.float()).sum(dim=1, keepdim=True) / math.sqrt(self.in_features)
        s_new = (s_i - self.lr_scale * grad_s).clamp_(1e-5, 10.0)

        update_signal = (
            grad_eff if self.pulse_mode == "direct" else ternary_gradient_unbiased(grad_eff)
        )
        c_rebased = c_i.float() * (s_i / s_new)
        ticks = (-self.lr * update_signal) * (self.C / s_new)
        cc = stochastic_round(c_rebased + ticks)

        carry = torch.trunc(cc / self.C)
        remainder = cc - carry * self.C
        proposed_t = t_i.float() + carry
        new_t = proposed_t.clamp_(-1, 1)
        blocked = proposed_t != new_t
        remainder = torch.where(
            blocked, torch.sign(cc) * (self.C - 1), remainder
        ).clamp_(-(self.C - 1), self.C - 1)

        self._write_rows(lo, hi, new_t, remainder)
        self.scale[lo:hi].copy_(s_new)
        self.update_events.add_(int((cc != c_i).sum().item()))
        self.weight_flips.add_(int((new_t != t_i).sum().item()))
