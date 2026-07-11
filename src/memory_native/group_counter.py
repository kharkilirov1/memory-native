"""GroupCounterLinear -- N:M (2:4) structured-sparse counter weight (method M2).

A counter group of 4 keeps hidden evidence for all 4 weights but makes only the top-2 VISIBLE:
within each contiguous group of 4 along the reduction (in_features) dim, the 2 weights with the
largest accumulated evidence |t*C + c| stay nonzero in the forward weight; the other 2 are forced
to 0. The forward and grad_x both use this 2:4-masked weight -> the visible weight is hardware-sparse
(2:4 maps to the Ampere/Hopper sparse Tensor Cores).

The crucial difference from pruning: the masked (invisible) weights are NOT dead. The update ticks
ALL four counters every step -- the weight-gradient evidence flows past the mask (straight-through),
so an invisible weight keeps accumulating and can FLIP back into the visible top-2 later. This is the
counter's own error-feedback applied to a structural mask: nothing is permanently zeroed.

    forward:  W_full = scale * t ;  mask = top-2 of |t*C+c| per group-of-4 ;  y = x @ (W_full*mask)^T
    grad_x:   grad_y @ (W_full*mask)            (the actually-used masked weight)
    update:   grad_w = grad_y^T x (DENSE, straight-through past the mask) -> RMS+SR tick ALL counters

CPU/CUDA pure PyTorch. The CPU gate (this module's witness) checks that 2:4 does not break teacher
recovery, the flip-rate is stable, and no dead-weight collapse occurs (the visible set rotates and
every weight keeps updating). The GPU sparse-Tensor-Core speedup (cuSPARSELt) is a later stage.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .counter import _carry_resolve, decode_state, encode_state, stochastic_round

__all__ = ["GroupCounterLinear", "two_four_mask"]


def two_four_mask(importance: torch.Tensor, group: int = 4, keep: int = 2) -> torch.Tensor:
    """[out, in] importance -> {0,1} mask keeping the top-`keep` of each `group` along in (2:4)."""
    out, in_ = importance.shape
    assert in_ % group == 0, "in_features must be divisible by the group size (4 for 2:4)"
    imp = importance.reshape(out, in_ // group, group)
    idx = imp.topk(keep, dim=-1).indices                       # [out, in//g, keep]
    mask = torch.zeros_like(imp)
    mask.scatter_(-1, idx, 1.0)
    return mask.reshape(out, in_)


class _GroupCounterFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, module, tap):
        t, c = module._decode()                                # [out,in] fp
        w_vis = (module.scale * t) * module.vis                # 2:4-masked by the COMMITTED mask
        flat = x.reshape(-1, x.shape[-1])
        y = flat @ w_vis.t()
        ctx.module = module
        ctx.save_for_backward(flat, w_vis, t, c)
        ctx.x_shape = x.shape
        return y.reshape(*x.shape[:-1], module.out_features)

    @staticmethod
    def backward(ctx, grad_y):
        flat, w_vis, t, c = ctx.saved_tensors
        m = ctx.module
        go2 = grad_y.reshape(-1, grad_y.shape[-1])
        grad_x = (go2 @ w_vis).reshape(ctx.x_shape)            # grad_x uses the MASKED weight
        if m.update_enabled and m.training:
            grad_w = go2.t() @ flat                            # DENSE evidence (straight-through)
            m._update(grad_w, t, c)
        return grad_x, None, None


class GroupCounterLinear(nn.Module):
    """Ternary counter linear whose VISIBLE weight is 2:4 structured-sparse (top-2 of every 4).

    keep:group = 2:4 by default. update_all=True (default) ticks every counter (error-feedback on
    the mask -> no dead weights); update_all=False ticks only visible weights (a pruning ablation,
    expected to collapse -- used by the witness to show error-feedback matters).
    """

    def __init__(self, in_features: int, out_features: int, *, C: int = 11, lr: float = 0.04,
                 lr_scale: float = 2e-4, init_gain: float = 1.0, rms_beta: float = 0.9,
                 rms_eps: float = 1e-3, group: int = 4, keep: int = 2,
                 update_all: bool = True, hysteresis: float = 1.0) -> None:
        super().__init__()
        if in_features % group != 0:
            raise ValueError(f"in_features {in_features} must be divisible by group {group}")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.C = int(C)
        self.lr = float(lr)
        self.lr_scale = float(lr_scale)
        self.rms_beta = float(rms_beta)
        self.rms_eps = float(rms_eps)
        self.group = int(group)
        self.keep = int(keep)
        self.update_all = bool(update_all)
        # Mask stability: a currently-visible weight gets a +hysteresis*C evidence bonus, so a masked
        # weight must EXCEED it by that margin to take its slot. Without it, full error-feedback ticks
        # every counter and masked weights constantly displace visible ones -> the 2:4 set thrashes
        # and learning is worse than pruning. hysteresis=0 reproduces the naive (thrashing) behaviour.
        self.hysteresis = float(hysteresis)
        self.update_enabled = True

        t0 = torch.randint(-1, 2, (out_features, in_features), dtype=torch.int16)
        c0 = torch.zeros_like(t0)
        self.register_buffer("state", encode_state(t0, c0, C))
        s0 = init_gain * math.sqrt(3.0 / (2.0 * in_features))
        self.register_buffer("scale", torch.full((out_features, 1), s0, dtype=torch.float32))
        self.register_buffer("v", torch.zeros((out_features, 1), dtype=torch.float32))
        # the COMMITTED 2:4 visible mask (state, so hysteresis has memory). Init from |t*C+c|.
        init_mask = two_four_mask((t0.float() * C + c0.float()).abs(), group, keep)
        self.register_buffer("vis", init_mask)
        # diagnostics (O(1) / O(out)); not optimizer state
        self.register_buffer("weight_flips", torch.zeros((), dtype=torch.int64), persistent=False)
        self.register_buffer("ever_visible", torch.zeros((out_features, in_features),
                                                         dtype=torch.bool), persistent=False)

    def _decode(self):
        t, c = decode_state(self.state, self.C)
        return t.to(torch.float32), c.to(torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (self.training and torch.is_grad_enabled()):
            t, _ = self._decode()
            return x @ ((self.scale * t) * self.vis).t()       # committed mask
        tap = torch.zeros((), device=x.device, dtype=x.dtype, requires_grad=True)
        return _GroupCounterFn.apply(x, self, tap)

    @torch.no_grad()
    def _update(self, grad_w, t, c) -> None:
        """RMS+SR counter tick, then re-select the 2:4 visible set with hysteresis.

        update_all -> tick every weight (error-feedback past the mask); else only the currently
        -visible weights tick (pruning ablation). After ticking, the new visible set is the top-2
        of (|t*C+c| + hysteresis*C*vis_prev) per group -- the bonus makes a visible weight sticky,
        so a masked weight must beat it by the margin to swap in (prevents the 2:4 set thrashing)."""
        gate = torch.ones_like(self.vis) if self.update_all else self.vis
        g = grad_w * gate
        s_i = self.scale
        g_sq = g.pow(2).mean(dim=1, keepdim=True)
        self.v.mul_(self.rms_beta).add_(g_sq, alpha=1.0 - self.rms_beta)
        denom = self.v.sqrt().clamp_min(self.rms_eps)
        grad_eff = g / denom

        grad_s = (g * t).sum(dim=1, keepdim=True) / math.sqrt(self.in_features)
        s_new = (s_i - self.lr_scale * grad_s).clamp_(1e-5, 10.0)

        c_rebased = c * (s_i / s_new)
        ticks = (-self.lr * grad_eff) * (self.C / s_new)
        cc = stochastic_round(c_rebased + ticks)
        new_t, remainder = _carry_resolve(cc, t, self.C)   # blocked flips pin to the edge
        self.state.copy_(encode_state(new_t.to(torch.int16), remainder.to(torch.int16), self.C))
        self.scale.copy_(s_new)
        self.weight_flips += int((new_t != t).sum().item())
        # re-select the visible 2:4 set with a stickiness bonus for the currently-visible weights.
        importance = (new_t * self.C + remainder).abs() + self.hysteresis * self.C * self.vis
        self.vis.copy_(two_four_mask(importance, self.group, self.keep))
        self.ever_visible |= self.vis.bool()

    @torch.no_grad()
    def visible_mask(self) -> torch.Tensor:
        return self.vis.clone()

    def persistent_bytes(self) -> int:
        return self.state.numel() + self.scale.numel() * 4 + self.v.numel() * 4

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, {self.keep}:{self.group} sparse, "
                f"update_all={self.update_all}")
