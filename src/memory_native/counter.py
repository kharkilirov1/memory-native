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
import torch.distributed as dist
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


def _allreduce_grad_w_(grad_w: torch.Tensor) -> None:
    """Average a counter weight-gradient across data-parallel ranks, in place. No-op unless a
    process group is initialized. Called inside backward so all replicas apply an identical
    counter update and their packed states stay synchronized (the counter has no Parameter
    gradient for DDP to handle -- the optimizer is the in-place state update itself)."""
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(grad_w, op=dist.ReduceOp.AVG)

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
        y = module._forward_matmul(x)
        # Activation-memory lever: if act_save_bits is set, store an UNBIASED low-bit
        # quantization of x (codes + per-row scale) instead of fp x. The update only needs
        # E[Q(x)|x] = x, so it stays unbiased; the saved activation shrinks from fp to b bits.
        if module.act_save_bits:
            from .actquant import pack_int4, quantize_codes
            x2 = x.reshape(-1, x.shape[-1])
            codes, scale = quantize_codes(x2, module.act_save_bits, dim=-1)
            if module.act_save_bits == 4:
                # true 4-bit packing: 2 codes per byte -> 0.5 byte/elem (codes in [-7,7]).
                store = pack_int4(codes)
                ctx.packed4 = True
                ctx.n_codes = x2.numel()
            else:
                # int8 (1 byte) for bits<=8, int16 for 9..15.
                store = codes.to(torch.int8) if module.act_save_bits <= 8 else codes.to(torch.int16)
                ctx.packed4 = False
            ctx.save_for_backward(store, scale)
            ctx.x_shape = x.shape
            ctx.quantized = True
        else:
            ctx.save_for_backward(x)
            ctx.quantized = False
        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        module: CompactCounterLinear = ctx.module
        if ctx.quantized:
            store, scale = ctx.saved_tensors
            if ctx.packed4:
                from .actquant import unpack_int4
                codes = unpack_int4(store, ctx.n_codes).reshape(-1, module.in_features)
            else:
                codes = store
            x2 = (codes.to(scale.dtype) * scale)            # dequant Q(x): [-1, in]
            x = x2.reshape(ctx.x_shape)
        else:
            (x,) = ctx.saved_tensors
            x2 = x.reshape(-1, x.shape[-1])
        go2 = grad_out.reshape(-1, grad_out.shape[-1])

        # grad_x: a kernel subclass (Triton) computes it straight from packed state with no
        # dense weight at all; the base path accumulates it tile-by-tile in the loop below.
        use_kernel = module._has_fast_grad_x()
        if use_kernel:
            grad_x2 = module._backward_grad_x(go2)
        else:
            grad_x2 = torch.zeros((x2.shape[0], module.in_features), device=x.device, dtype=x.dtype)

        # The full weight-gradient never exists. Only [tile_rows, in_features] is live.
        # Adaptive decimation (memo M8): once a layer's flip-rate is tiny it is near-stable, so
        # apply the update only every _dec_period steps with lr scaled to compensate. Decided once
        # per backward; grad_x is always computed, only the update is skipped.
        do_update = module.training and module.update_enabled
        fire = module._decimation_apply() if do_update else False
        if do_update or not use_kernel:
            for lo in range(0, module.out_features, module.tile_rows):
                hi = min(lo + module.tile_rows, module.out_features)
                t_i, c_i = module._decode_rows(lo, hi)
                s_i = module.scale[lo:hi]
                if not use_kernel:
                    w_i = s_i.to(x.dtype) * t_i.to(x.dtype)
                    grad_x2.add_(go2[:, lo:hi] @ w_i)
                if fire:
                    if module.update_compute == "int8":
                        from .int8_compute import int8_correlation
                        grad_w_i = int8_correlation(go2[:, lo:hi], x2)
                    else:
                        grad_w_i = (go2[:, lo:hi].transpose(0, 1) @ x2).float()
                    # proxy RMS: take the row second-moment from grad_out norms (* activation
                    # energy) instead of the full ||G_o||^2 reduction (post-LN E[r_o^2]~||D_o||^2).
                    proxy_gsq = None
                    if getattr(module, "rms_mode", "exact") == "proxy":
                        # E[r_o^2] ~ (sum_m D_mo^2) * E[X^2]: SUM over the M tokens, not mean --
                        # mean is off by 1/M, so the denominator scales ~1/sqrt(M) (batch/seq).
                        proxy_gsq = go2[:, lo:hi].pow(2).sum(dim=0, keepdim=True).t() * x2.pow(2).mean()
                    # Data-parallel: the counter optimizer lives in the state and is applied
                    # in-place here, so there is no Parameter .grad for DDP to all-reduce. Sync
                    # the counter gradient itself across ranks -> every replica applies the same
                    # update (same SR seed) and the packed state stays bit-identical everywhere.
                    _allreduce_grad_w_(grad_w_i)
                    if not module._fused_update(lo, hi, grad_w_i):
                        module._update_tile(lo, hi, grad_w_i, t_i, c_i, s_i, proxy_gsq=proxy_gsq)
            if fire:
                module._decimation_observe()

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
        tile_rows: int = 256,
        local_grad_clip: float = 0.0,
        pulse_mode: str = "direct",
        act_save_bits: int = 0,
        decimate_updates: bool = False,
        cache_mode: str = "none",
        update_compute: str = "fp",
    ) -> None:
        super().__init__()
        if 3 * (2 * C - 1) > 256:
            raise ValueError("C is too large for uint8 state encoding (need 3*(2C-1) <= 256)")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.C = int(C)
        self.lr = float(lr)
        self.lr_scale = float(lr_scale)
        # tile_rows=R: the update runs over R-row tiles, materializing only an [R,in] grad_w tile.
        # With the row-slice fused Triton kernel this is the fast-low-peak mode -- T4 frontier
        # (results/ACCELERATION.md) shows R~256 at ~0.83x the full-matrix step time at half the
        # transient gradient. Default 256. tile_rows=0 means untiled (one shot, full [out,in] grad
        # tile); set it for the no-row-slicing path. (On the torch fallback the per-tile loop adds
        # a little launch overhead but the result is identical -- the update is per-row independent.)
        self.tile_rows = int(tile_rows) if int(tile_rows) > 0 else int(out_features)
        self.local_grad_clip = float(local_grad_clip)
        # 0 = store fp activation; >0 = store unbiased act_save_bits-bit Q(x) for the update.
        self.act_save_bits = int(act_save_bits)
        if pulse_mode not in {"direct", "ternary"}:
            raise ValueError("pulse_mode must be 'direct' or 'ternary'")
        self.pulse_mode = pulse_mode
        self.update_enabled = True
        self._outstanding_forward = False
        # Adaptive update decimation (memo M8): when on, a near-stable layer (tiny flip-rate)
        # updates only every _dec_period steps with lr scaled by the period to compensate.
        self.decimate_updates = bool(decimate_updates)
        self._dec_period = 1
        self._dec_since = 0
        self._dec_flip_rate = 1.0
        self._lr_mult = 1.0

        t0 = torch.randint(-1, 2, (out_features, in_features), dtype=torch.int16)
        c0 = torch.zeros_like(t0)
        self.register_buffer("state", encode_state(t0, c0, C))

        # Var(t)=2/3 for uniform {-1,0,1}; choose Var(w)=gain^2/fan_in.
        s0 = init_gain * math.sqrt(3.0 / (2.0 * in_features))
        self.register_buffer("scale", torch.full((out_features, 1), s0, dtype=torch.float32))

        # Diagnostics only: O(1) scalars, not per-weight optimizer state.
        self.register_buffer("update_events", torch.zeros((), dtype=torch.int64), persistent=False)
        self.register_buffer("weight_flips", torch.zeros((), dtype=torch.int64), persistent=False)

        # Derived visible-weight compute cache (memo M5): the visible ternary T is the only thing
        # forward/grad_x need; the hidden counter c is only the update's. cache_mode keeps T in a
        # GEMM-friendly dtype so the matmul never pays the 6-bit decode tax. It is a *derived view*
        # (not truth, not optimizer state): rebuilt from the state, refreshed on visible flips.
        if cache_mode not in {"none", "fp16", "int8"}:
            raise ValueError("cache_mode must be 'none', 'fp16' or 'int8'")
        if update_compute not in {"fp", "int8"}:
            raise ValueError("update_compute must be 'fp' or 'int8'")
        # "int8" forms grad_w with an unbiased int8 GEMM estimator (Tensor Cores on CUDA) instead
        # of the fp32 correlation. Unbiased -> training-neutral in expectation; opt-in, fp default.
        self.update_compute = update_compute
        self.cache_mode = cache_mode
        if cache_mode != "none":
            # Built lazily on first use -- subclasses (e.g. the packed layout) finish constructing
            # their state after this base __init__. Rebuilt from the truth state after any load
            # (the cache is persistent=False, so it is never the source of truth).
            self.register_load_state_dict_post_hook(
                lambda m, _ic: m._build_t_cache() if hasattr(m, "_t_cache") else None)

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

    def _has_fast_grad_x(self) -> bool:
        # Base path computes grad_x tile-by-tile in the backward loop. A kernel subclass
        # returns True and provides _backward_grad_x to form grad_x straight from state.
        return False

    def _backward_grad_x(self, grad_out2d: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def _fused_update(self, lo: int, hi: int, grad_w: torch.Tensor) -> bool:
        # A subclass with a one-launch fused update (Triton) returns True after applying it;
        # the base path always returns False so the caller runs the torch tile update.
        return False

    def _eff_lr(self) -> float:
        return self.lr * self._lr_mult

    def _decimation_apply(self) -> bool:
        """Advance the per-layer decimation clock; return whether to apply the update this step.
        Off (default) -> always fire at lr*1. On -> fire every _dec_period steps with lr scaled by
        the period (sum of r similar grads ~ r*grad)."""
        if not self.decimate_updates:
            self._lr_mult = 1.0
            return True
        self._dec_since += 1
        if self._dec_since >= self._dec_period:
            self._lr_mult = float(self._dec_period)
            self._dec_since = 0
            return True
        return False

    def _decimation_observe(self) -> None:
        """Set the next update period from the flip-rate just observed (smaller flips -> rarer)."""
        if not self.decimate_updates:
            return
        r = self._dec_flip_rate
        self._dec_period = 1 if r > 1e-3 else 2 if r > 1e-4 else 4 if r > 1e-5 else 8

    def _build_t_cache(self) -> None:
        dt = torch.float16 if self.cache_mode == "fp16" else torch.int8
        t, _ = self._decode_rows(0, self.out_features)
        cache = t.to(dt)
        if hasattr(self, "_t_cache"):
            self._t_cache.copy_(cache)
        else:
            self.register_buffer("_t_cache", cache, persistent=False)

    def _visible_t(self, dtype: torch.dtype) -> torch.Tensor:
        # the visible ternary weight T, from the derived cache when enabled (no decode), else decoded.
        if self.cache_mode != "none":
            if not hasattr(self, "_t_cache"):
                self._build_t_cache()
            return self._t_cache.to(dtype)
        t, _ = self._decode_rows(0, self.out_features)
        return t.to(dtype)

    def _dense_weight(self, dtype: torch.dtype) -> torch.Tensor:
        return self.scale.to(dtype) * self._visible_t(dtype)

    def _decode_rows(self, lo: int, hi: int) -> tuple[torch.Tensor, torch.Tensor]:
        return decode_state(self.state[lo:hi], self.C)

    def _refresh_t_cache(self, lo: int, hi: int, t: torch.Tensor) -> None:
        if self.cache_mode != "none":
            if not hasattr(self, "_t_cache"):   # state for these rows is already written -> snapshot
                self._build_t_cache()
            else:
                self._t_cache[lo:hi] = t.to(self._t_cache.dtype)

    def _write_rows(self, lo: int, hi: int, t: torch.Tensor, c: torch.Tensor) -> None:
        self.state[lo:hi].copy_(encode_state(t, c, self.C))
        self._refresh_t_cache(lo, hi, t)

    @torch.no_grad()
    def _update_tile(self, lo, hi, grad_w, t_i, c_i, s_i, proxy_gsq=None) -> None:
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
        ticks = (-self._eff_lr() * update_signal) * (self.C / s_new)
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
        flips = int((new_t != t_i).sum().item())
        self._dec_flip_rate = flips / max(new_t.numel(), 1)
        self.update_events.add_(int((cc != c_i).sum().item()))
        self.weight_flips.add_(flips)

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
                 use_rms: bool = True, rms_mode: str = "exact",
                 scale_rebase: str = "eager", **kw) -> None:
        super().__init__(*args, **kw)
        self.rms_beta = float(rms_beta)
        self.rms_eps = float(rms_eps)
        self.use_rms = bool(use_rms)
        # rms_mode: "exact" uses the freshly-updated v for the denominator (the tick depends on a
        #   row-stat of THIS step's grad -> two passes); "lagged" uses last step's v so the tick
        #   needs no row-stat of the current grad -> one pass (the strict update-from-IO kernel).
        # scale_rebase: "eager" rebases the counter to s_new before ticking (needs s_new first);
        #   "lazy" keeps a per-row calibration scale s_base and rebases at the next step's read,
        #   so the tick uses the current scale only. lagged+lazy together are the one-pass update.
        # "proxy": post-LayerNorm E[r_o^2|D] ~ ||D_o||^2, so the RMS denominator comes from cheap
        #   grad_out row-norms (x a scalar activation-energy) instead of the full ||G_o||^2 stat.
        assert rms_mode in ("exact", "lagged", "proxy")
        assert scale_rebase in ("eager", "lazy")
        self.rms_mode = rms_mode
        self.scale_rebase = scale_rebase
        self.register_buffer("v", torch.zeros((self.out_features, 1), dtype=torch.float32))
        self.register_buffer("s_base", self.scale.clone())  # scale the counter is calibrated to

    @torch.no_grad()
    def _update_tile(self, lo, hi, grad_w, t_i, c_i, s_i, proxy_gsq=None) -> None:
        if self.use_rms:
            # proxy mode takes the row second-moment from grad_out norms (passed in) instead of
            # the full ||G_o||^2 reduction; lagged uses the previous v, else the freshly-updated v.
            g_sq = proxy_gsq if (self.rms_mode == "proxy" and proxy_gsq is not None) \
                else grad_w.pow(2).mean(dim=1, keepdim=True)
            if self.rms_mode == "lagged":
                denom = self.v[lo:hi].sqrt().clamp_min(self.rms_eps)          # previous v
                self.v[lo:hi].mul_(self.rms_beta).add_(g_sq, alpha=1.0 - self.rms_beta)
            else:
                self.v[lo:hi].mul_(self.rms_beta).add_(g_sq, alpha=1.0 - self.rms_beta)
                denom = self.v[lo:hi].sqrt().clamp_min(self.rms_eps)          # freshly-updated v
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
        if self.scale_rebase == "lazy":
            # counter is calibrated to s_base; bring it to the current scale s_i, tick in s_i units
            # (no dependence on s_new), and record that the stored counter is now calibrated to s_i.
            c_cur = stochastic_round(c_i.float() * (self.s_base[lo:hi] / s_i))
            ticks = (-self._eff_lr() * update_signal) * (self.C / s_i)
            cc = stochastic_round(c_cur + ticks)
            self.s_base[lo:hi].copy_(s_i)
            return self._finish_update(lo, hi, cc, t_i, c_i, s_new)
        c_rebased = c_i.float() * (s_i / s_new)
        ticks = (-self._eff_lr() * update_signal) * (self.C / s_new)
        cc = stochastic_round(c_rebased + ticks)
        self._finish_update(lo, hi, cc, t_i, c_i, s_new)

    @torch.no_grad()
    def _finish_update(self, lo, hi, cc, t_i, c_i, s_new) -> None:
        # carry/remainder -> ternary flip + residual, then write state + scale (shared by the
        # eager and the lazy-rebase paths).
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
        flips = int((new_t != t_i).sum().item())
        self._dec_flip_rate = flips / max(new_t.numel(), 1)
        self.update_events.add_(int((cc != c_i).sum().item()))
        self.weight_flips.add_(flips)
