"""StackCounterLinear -- the M-STACK integration (verification-plan M-STACK): do the per-step levers
COMPOSE without conflicting?

Combines two independently-witnessed levers in one weight:
  * M2 base: a 2:4 structured-sparse group-counter (the visible weight is top-2 of every 4) -- the
    forward/grad_x lever (maps to sparse Tensor Cores);
  * M3 fast: a low-rank fp residual A@B^T trained every step, folded into the 2:4 base every K steps
    (CounterProject merge) -- the wgrad-frequency lever.

    forward:  Y = X · (2:4-masked s·T)^T  +  (X B) A^T
    base is FROZEN between merges (update_enabled=False); only A,B learn each step.
    merge (every K): re-encode  s·T_full + A@B^T  into the 2:4 group-counter (ternary + counter +
                     per-row scale), re-select the 2:4 visible set, reset A,B.

HONEST SCOPE: this witnesses COMPOSABILITY (do 2:4 + slow-fast train together, what is the val-gap
vs dense) -- NOT raw per-step speed. The per-step win needs the sparse-Tensor-Core (cuSPARSELt) 2:4
kernel and is gated on hardware/kernels not built here; measuring tok/s on the dense PyTorch fallback
would be meaningless. int8 forward (M5) is a separate, deterministic, already-validated lever that
composes trivially, so it is left out of this module to keep the novel 2:4 + slow-fast question clean.
Pure PyTorch, CPU/CUDA.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .counter import stochastic_round
from .group_counter import GroupCounterLinear, two_four_mask

__all__ = ["StackCounterLinear"]


class StackCounterLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, rank: int = 16,
                 merge_every: int = 16, hysteresis: float = 2.0, group: int = 4, keep: int = 2,
                 C: int = 11, lr: float = 0.04, lr_scale: float = 2e-4, init_gain: float = 1.0,
                 fast_init: float = 1e-3) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.merge_every = int(merge_every)
        self.fast_init = float(fast_init)
        self.base = GroupCounterLinear(in_features, out_features, C=C, lr=lr, lr_scale=lr_scale,
                                       init_gain=init_gain, hysteresis=hysteresis,
                                       group=group, keep=keep, update_all=True)
        self._has_fast = self.rank > 0
        # base frozen between merges when a fast path carries the learning (M3); else it self-updates.
        self.base.update_enabled = not self._has_fast
        if self._has_fast:
            self.A = nn.Parameter(torch.randn(out_features, self.rank) * self.fast_init)
            self.B = nn.Parameter(torch.zeros(in_features, self.rank))
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)
        self.register_buffer("step_count", torch.zeros((), dtype=torch.int64), persistent=False)
        self.register_buffer("merge_count", torch.zeros((), dtype=torch.int64), persistent=False)
        self._merge_pending = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._has_fast and self.training and torch.is_grad_enabled() and self._merge_pending:
            self.merge()
            self._merge_pending = False
        y = self.base(x)                                  # 2:4-masked forward (base frozen if fast)
        if self._has_fast:
            y = y + (x @ self.B) @ self.A.t()             # low-rank fp residual
            if self.training and torch.is_grad_enabled():
                self.step_count += 1
                if int(self.step_count) % self.merge_every == 0:
                    self._merge_pending = True
        return y

    @torch.no_grad()
    def merge(self) -> None:
        """Fold s·T_full + A@B^T into the 2:4 group-counter base, re-select the visible set, reset A,B."""
        if not self._has_fast:
            return
        b = self.base
        C = b.C
        t, c = b._decode()
        w = b.scale * t + (b.scale * c / C) + self.A @ self.B.t()      # effective weight (+ residual)
        scale_new = w.abs().mean(dim=1, keepdim=True).clamp_(1e-5, 10.0)
        q = w / scale_new
        t_new = q.round().clamp_(-1, 1)
        c_new = stochastic_round((q - t_new) * C).clamp_(-(C - 1), C - 1)
        from .counter import encode_state
        b.state.copy_(encode_state(t_new.to(torch.int16), c_new.to(torch.int16), C))
        b.scale.copy_(scale_new)
        b.v.zero_()
        importance = (t_new * C + c_new).abs() + b.hysteresis * C * b.vis
        b.vis.copy_(two_four_mask(importance, b.group, b.keep))
        self.A.normal_(0.0, self.fast_init)
        self.B.zero_()
        self.merge_count += 1

    def fast_parameters(self):
        return [p for p in (self.A, self.B) if p is not None]

    def persistent_bytes(self) -> int:
        return self.base.persistent_bytes()

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, 2:4 base + rank={self.rank} "
                f"fast, merge_every={self.merge_every}")
