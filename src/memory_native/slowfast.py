"""Slow-fast low-rank residual counter linear (method M3).

A counter weight is expensive to UPDATE: the base update needs the full base correlation
G = Delta^T X every step (an [out, in] GEMM). Slow-fast decomposes the effective weight as

    W_eff = s*T  +  A @ B^T          (base counter weight s*T  +  low-rank fp residual, rank r << d)

and trains only the small fp residual (A:[out, r], B:[in, r]) with a normal optimizer each step
(cheap, O(M d r)). The base counter state s*T is *frozen* between merges -- its full Delta^T X is
NOT recomputed every step. Every K steps the fast residual is folded into the base ("merge"):

    s*T  <-  reencode( s*T + A @ B^T )         then  A, B  <-  0

so the base absorbs the accumulated residual and the fast path restarts from zero. This cuts the
full-base-correlation frequency by ~K x (it runs only on merge steps).

Forward keeps the base s*T DENSE (Y = X (sT)^T + (X B) A^T), exactly like RMSCounterLinear -- the
low-rank term is an *additive* correction, never folded into the GEMM weight outside a merge.

HONEST BOUNDARY (the standalone speed ceiling is small): slow-fast cheapens the UPDATE of the
base, not its forward / grad_x USE. The forward still materializes the dense s*T and runs the full
X (sT)^T GEMM every step, and grad_x still uses the dense weight. With the base update being ~1 of
the 3 per-layer GEMMs, removing it on (K-1)/K of steps gives only a ~1.43x standalone ceiling. The
real win is in COMBINATION with structured sparsity (a later method) that also cheapens the USE.
The point of THIS witness is correctness + the update-frequency reduction, not raw speed.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .counter import RMSCounterLinear, decode_state, encode_state, stochastic_round

__all__ = ["SlowFastCounterLinear"]


class SlowFastCounterLinear(nn.Module):
    """RMSCounterLinear base + low-rank fp residual A@B^T, merged into the base every K steps.

    Args:
        in_features, out_features: linear dims.
        rank: r of the fast residual. r=0 disables the fast path entirely -> this layer is
            then exactly an RMSCounterLinear (the base self-updates every step, parity arm).
        merge_every: K. Fold A@B^T into the base counter state every K forward(train) calls and
            reset A,B<-0. The base full correlation Delta^T X then runs only on merge steps.
        fast_init: std of B init (A is init 0 so A@B^T == 0 at construction -> exact parity start).
        counter_kw: forwarded to the RMSCounterLinear base (C, lr, lr_scale, init_gain, ...).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int = 16,
        merge_every: int = 16,
        fast_init: float = 1e-3,
        init_gain: float = 1.0,
        **counter_kw,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.merge_every = int(merge_every)
        if self.merge_every < 1:
            raise ValueError("merge_every must be >= 1")

        self.base = RMSCounterLinear(in_features, out_features, init_gain=init_gain, **counter_kw)
        # With a fast path, the base does NOT self-update every step; it only changes on merges.
        # r=0 -> no fast path -> let the base behave as a plain RMSCounterLinear (updates each step).
        self._has_fast = self.rank > 0
        self.base.update_enabled = not self._has_fast
        self._merge_pending = False

        if self._has_fast:
            # A init random, B init ZERO -> A@B^T == 0 at start (exact parity with the base) while
            # B's gradient (X^T grad_y) A is nonzero from step 1 (A != 0), so the residual bootstraps
            # immediately. (Init A=0 would zero BOTH gradients and the residual could never grow --
            # and the every-K reset would keep re-killing it. LoRA-style: one side zero, one random.)
            self.fast_init = float(fast_init)
            self.A = nn.Parameter(torch.randn(out_features, self.rank) * self.fast_init)
            self.B = nn.Parameter(torch.zeros(in_features, self.rank))
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

        # Diagnostics (O(1), not optimizer state).
        self.register_buffer("step_count", torch.zeros((), dtype=torch.int64), persistent=False)
        self.register_buffer("merge_count", torch.zeros((), dtype=torch.int64), persistent=False)
        self.register_buffer("base_corr_steps", torch.zeros((), dtype=torch.int64), persistent=False)

    # -- forward ---------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # A merge mutates A,B in place, so it must NOT run inside a live autograd graph (the
        # previous step's backward may not have run yet). We mark the step as "merge pending" and
        # actually fold it at the START of the next training forward, before A,B enter the graph.
        if self._has_fast and self.training and torch.is_grad_enabled() and self._merge_pending:
            self.merge()
            self._merge_pending = False
        y = self.base(x)                       # X (sT)^T, dense base; base self-updates only if r==0
        if self._has_fast:
            # additive low-rank correction (X B) A^T -- O(M d r), never folded into the GEMM weight.
            y = y + (x @ self.B) @ self.A.t()
            if self.training and torch.is_grad_enabled():
                self.step_count += 1
                if int(self.step_count) % self.merge_every == 0:
                    self._merge_pending = True
        return y

    @torch.no_grad()
    def flush_merge(self) -> bool:
        """Apply a pending merge now (safe to call AFTER backward()/opt.step()). Returns whether
        a merge fired. Lets a caller fold the residual deterministically between steps instead of
        waiting for the lazy fold at the next forward -- used by the witness to measure the exact
        post-merge loss against the pre-merge loss (stability)."""
        if self._has_fast and self._merge_pending:
            self.merge()
            self._merge_pending = False
            return True
        return False

    # -- merge -----------------------------------------------------------------
    @torch.no_grad()
    def merge(self) -> None:
        """Fold the fast residual A@B^T into the base counter state, then restart A,B at parity.

        Re-encodes W_merged = s*T + A@B^T directly into (ternary T, residual counter c, per-row
        scale s): per-row absmean scale (BitNet-style, the best ternary scale), ternary sign by
        rounding, and the leftover fraction stored in the counter c (units of s/C). This is the
        full base "update" and is the only place the base state changes -- it runs once per K steps,
        so the full base correlation frequency is K x lower than an every-step counter."""
        if not self._has_fast:
            return
        base = self.base
        C = base.C
        # decode the current base visible weight  W = s*T  (+ hidden residual c, folded back too).
        t, c = decode_state(base.state, C)
        s = base.scale                                   # [out, 1]
        # include the hidden counter residual so the merge conserves the layer's effective weight:
        # decoded effective weight ~ s * (t + c/C).
        w_base = s * (t.float() + c.float() / C)
        w_merged = w_base + self.A @ self.B.t()          # [out, in] fp

        # per-row absmean scale (clamped like the base): best ternary scale for this row.
        scale_new = w_merged.abs().mean(dim=1, keepdim=True).clamp_(1e-5, 10.0)
        q = w_merged / scale_new                         # ~ t + residual, in scale units
        t_new = q.round().clamp_(-1, 1)
        # leftover fraction -> counter (units of s/C), stochastic-rounded and clamped like an update.
        c_new = stochastic_round((q - t_new) * C).clamp_(-(C - 1), C - 1)

        base.state.copy_(encode_state(t_new.to(torch.int16), c_new.to(torch.int16), C))
        base.scale.copy_(scale_new)
        if hasattr(base, "s_base"):
            base.s_base.copy_(scale_new)                 # counter is now calibrated to the new scale
        if hasattr(base, "v"):
            base.v.zero_()                               # reset RMS second moment after a re-encode
        if base.cache_mode != "none" and hasattr(base, "_t_cache"):
            base._t_cache.copy_(t_new.to(base._t_cache.dtype))

        # restart the fast path at the parity state (A@B^T == 0) that still bootstraps: A random,
        # B zero -> B's gradient is nonzero on the very next step so the residual re-grows.
        self.A.normal_(0.0, self.fast_init)
        self.B.zero_()
        self.merge_count += 1
        self.base_corr_steps += 1                        # a merge IS one full base-correlation step

    # -- convenience -----------------------------------------------------------
    @torch.no_grad()
    def effective_weight(self) -> torch.Tensor:
        """Dense W_eff = s*T + A@B^T (for testing / comparison)."""
        w = self.base._dense_weight(torch.float32)
        if self._has_fast:
            w = w + self.A @ self.B.t()
        return w

    def fast_parameters(self):
        """The low-rank residual Parameters (A, B) for a normal optimizer. Empty if rank==0."""
        return [p for p in (self.A, self.B) if p is not None]

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, rank={self.rank}, "
                f"merge_every={self.merge_every}, has_fast={self._has_fast}")
