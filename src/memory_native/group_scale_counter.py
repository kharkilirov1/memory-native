"""Trainable group-scale finite-state counter linear.

This is the missing bridge between group-128 PTQ and counter recovery. Each output row owns
one scale per act-ordered input group while every weight keeps the same finite-state (t, c)
automaton. ``perm`` is the GPTQ act-order permutation; ``group_index`` maps original input
columns back to their permuted group without changing the runtime input layout.

The optional residual homotopy exposes ``t + alpha*c/C`` during early recovery and anneals
``alpha -> 0``. No FP master weight is introduced: c is the counter's existing finite state.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .counter import decode_state, encode_state, stochastic_round

__all__ = ["GroupScaleCounterLinear"]


def _carry_resolve(cc: torch.Tensor, t: torch.Tensor, C: int):
    carry = torch.trunc(cc / C)
    remainder = cc - carry * C
    proposed = t + carry
    new_t = proposed.clamp(-1, 1)
    blocked = proposed != new_t
    remainder = torch.where(blocked, torch.sign(cc) * (C - 1), remainder)
    return new_t, remainder.clamp(-(C - 1), C - 1)


class _GroupScaleCounterFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, module: "GroupScaleCounterLinear", tap: torch.Tensor):
        if module._outstanding_forward:
            raise RuntimeError("GroupScaleCounterLinear was reused before its previous backward")
        module._outstanding_forward = True
        w = module.visible_weight(dtype=x.dtype)
        y = x.reshape(-1, x.shape[-1]) @ w.t()
        ctx.module = module
        ctx.x_shape = x.shape
        ctx.save_for_backward(x.reshape(-1, x.shape[-1]), w)
        return y.reshape(*x.shape[:-1], module.out_features)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x2, w = ctx.saved_tensors
        module: GroupScaleCounterLinear = ctx.module
        go2 = grad_out.reshape(-1, grad_out.shape[-1])
        grad_x = go2 @ w
        if module.training and module.update_enabled:
            module._update(go2.t().float() @ x2.float())
        module._outstanding_forward = False
        return grad_x.reshape(ctx.x_shape), None, None


class GroupScaleCounterLinear(nn.Module):
    """Ternary counter linear with FP group scales and act-order metadata.

    Persistent weight state is one uint8 code per coefficient plus one FP32 scale per
    ``(output row, group)``. ``perm`` follows ``W_perm = W[:, perm]``. State itself remains
    in original column order, avoiding an input permutation in forward.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        group: int = 128,
        C: int = 11,
        lr: float = 2e-3,
        lr_scale: float = 2e-4,
        rms_beta: float = 0.9,
        rms_eps: float = 1e-3,
        local_grad_clip: float = 0.0,
        residual_alpha: float = 0.0,
        perm: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if 3 * (2 * C - 1) > 256:
            raise ValueError("C is too large for uint8 state encoding")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group = int(group)
        self.C = int(C)
        self.lr = float(lr)
        self.lr_scale = float(lr_scale)
        self.rms_beta = float(rms_beta)
        self.rms_eps = float(rms_eps)
        self.local_grad_clip = float(local_grad_clip)
        self.residual_alpha = float(residual_alpha)
        self.update_enabled = True
        self._outstanding_forward = False

        self.n_groups = (self.in_features + self.group - 1) // self.group
        t0 = torch.zeros((out_features, in_features), dtype=torch.int16)
        c0 = torch.zeros_like(t0)
        self.register_buffer("state", encode_state(t0, c0, self.C))
        self.register_buffer(
            "scale", torch.full((out_features, self.n_groups), 1e-2, dtype=torch.float32)
        )
        self.register_buffer("v", torch.zeros((out_features, 1), dtype=torch.float32))
        self.register_buffer("weight_flips", torch.zeros((), dtype=torch.int64), persistent=False)
        self.register_buffer("update_events", torch.zeros((), dtype=torch.int64), persistent=False)
        self.register_buffer("perm", torch.empty(in_features, dtype=torch.long))
        self.register_buffer("group_index", torch.empty(in_features, dtype=torch.long))
        self.set_permutation(torch.arange(in_features) if perm is None else perm)

    @torch.no_grad()
    def set_permutation(self, perm: torch.Tensor) -> None:
        perm = perm.detach().to(device=self.state.device, dtype=torch.long)
        if perm.numel() != self.in_features or not torch.equal(
            torch.sort(perm).values, torch.arange(self.in_features, device=perm.device)
        ):
            raise ValueError("perm must be a permutation of range(in_features)")
        group_perm = torch.div(
            torch.arange(self.in_features, device=perm.device), self.group, rounding_mode="floor"
        )
        gidx = torch.empty_like(group_perm)
        gidx[perm] = group_perm
        self.perm.copy_(perm)
        self.group_index.copy_(gidx)

    def _decode(self):
        t, c = decode_state(self.state, self.C)
        return t.float(), c.float()

    def column_scales(self) -> torch.Tensor:
        return self.scale[:, self.group_index]

    def visible_weight(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        t, c = self._decode()
        code = t + self.residual_alpha * c / self.C
        w = self.column_scales() * code
        return w if dtype is None else w.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (self.training and torch.is_grad_enabled()):
            return x @ self.visible_weight(dtype=x.dtype).t()
        tap = torch.zeros((), device=x.device, dtype=x.dtype, requires_grad=True)
        return _GroupScaleCounterFn.apply(x, self, tap)

    @torch.no_grad()
    def load_group_state(
        self,
        scales: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
        perm: torch.Tensor | None = None,
    ) -> None:
        scales = scales.to(device=self.state.device, dtype=torch.float32)
        t = t.to(device=self.state.device, dtype=torch.int16)
        c = torch.zeros_like(t) if c is None else c.to(device=self.state.device, dtype=torch.int16)
        if scales.shape != self.scale.shape:
            raise ValueError(f"scale shape {tuple(scales.shape)} != {tuple(self.scale.shape)}")
        if t.shape != self.state.shape or c.shape != self.state.shape:
            raise ValueError("t/c shape mismatch")
        if not set(t.unique().tolist()) <= {-1, 0, 1}:
            raise ValueError("t must be ternary")
        if c.abs().max().item() > self.C - 1:
            raise ValueError("counter residual exceeds representable range")
        if perm is not None:
            self.set_permutation(perm)
        self.scale.copy_(scales.clamp_min(1e-8))
        self.state.copy_(encode_state(t, c, self.C))
        self.v.zero_()
        self.weight_flips.zero_()
        self.update_events.zero_()

    @torch.no_grad()
    def set_lr(self, lr: float) -> None:
        self.lr = float(lr)

    @torch.no_grad()
    def set_residual_alpha(self, alpha: float) -> None:
        self.residual_alpha = float(min(1.0, max(0.0, alpha)))

    @torch.no_grad()
    def _update(self, grad_w: torch.Tensor) -> None:
        t, c = self._decode()
        g_sq = grad_w.pow(2).mean(dim=1, keepdim=True)
        self.v.mul_(self.rms_beta).add_(g_sq, alpha=1.0 - self.rms_beta)
        grad_eff = grad_w / self.v.sqrt().clamp_min(self.rms_eps)
        if self.local_grad_clip > 0:
            norm = grad_eff.norm(dim=1, keepdim=True).clamp_min(1e-30)
            grad_eff.mul_((self.local_grad_clip / norm).clamp_max(1.0))

        code = t + self.residual_alpha * c / self.C
        contrib = grad_w * code
        gidx = self.group_index.unsqueeze(0).expand(self.out_features, -1)
        grad_scale = torch.zeros_like(self.scale).scatter_add_(1, gidx, contrib)
        counts = torch.bincount(self.group_index, minlength=self.n_groups).to(self.scale.dtype)
        grad_scale = grad_scale / counts.sqrt().clamp_min(1.0).unsqueeze(0)
        old_scale = self.scale.clone()
        new_scale = (old_scale - self.lr_scale * grad_scale).clamp_(1e-5, 10.0)

        old_col = old_scale[:, self.group_index]
        new_col = new_scale[:, self.group_index]
        c_rebased = c * (old_col / new_col)
        ticks = (-self.lr * grad_eff) * (self.C / new_col)
        cc = stochastic_round(c_rebased + ticks)
        new_t, new_c = _carry_resolve(cc, t, self.C)
        self.weight_flips.add_((new_t != t).sum().to(self.weight_flips.dtype))
        self.update_events.add_(grad_w.numel())
        self.scale.copy_(new_scale)
        self.state.copy_(encode_state(new_t.to(torch.int16), new_c.to(torch.int16), self.C))

    @torch.no_grad()
    def state_statistics(self) -> dict[str, float]:
        t, c = self._decode()
        return {
            "counter_edge": float((c.abs() >= self.C - 1).float().mean()),
            "ternary_zero": float((t == 0).float().mean()),
            "scale_mean": float(self.scale.mean()),
            "residual_alpha": float(self.residual_alpha),
        }

    def persistent_bytes(self) -> int:
        return (
            self.state.numel() * self.state.element_size()
            + self.scale.numel() * self.scale.element_size()
            + self.v.numel() * self.v.element_size()
            + self.group_index.numel() * self.group_index.element_size()
            + self.perm.numel() * self.perm.element_size()
        )

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, group={self.group}, "
            f"C={self.C}, alpha={self.residual_alpha:.3f}"
        )
