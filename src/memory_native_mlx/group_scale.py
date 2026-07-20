"""Trainable group-scale finite-state counter linear on MLX — the Bonsai-format layer.

MLX port of the solver-v3 branch's `GroupScaleCounterLinear` (agent/solver-v3-group-recovery):
each output row owns one FP scale per input group (group=128 default — the Bonsai/PTQ
granularity) while every weight keeps the same finite-state (t, c) automaton. `perm` carries
the GPTQ act-order permutation; state stays in original column order so forward needs no
input permutation. The optional residual homotopy exposes `t + alpha*c/C` during early
recovery and anneals alpha -> 0; no FP master weight is introduced — c is the counter's
existing finite state.

Differences from the torch original, consistent with the rest of the MLX port: stochastic
rounding is the deterministic hash-SR stream (not torch.rand), the update is functional,
and the self-update runs in the layer's custom VJP. With one group spanning the whole row
(group=in_features, alpha=0) this layer's update degenerates to exactly RMSCounterLinear's
(tested bit-for-bit) — the group layer is a strict generalization.
"""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .counter import decode_state, encode_state, hash_u32, uniform01

__all__ = ["GroupScaleCounterLinear"]


class GroupScaleCounterLinear(nn.Module):
    """Ternary counter linear with FP group scales and act-order metadata.

    Persistent state: one uint8 code per weight (`codes`), one FP32 scale per (row, group)
    (`scale` [out, n_groups]), one RMS second moment per row (`v` [out,1]), plus the
    act-order permutation. All frozen; the layer self-updates in its VJP. `tap` is the
    trainable 0-scalar that threads the layer into every value_and_grad diff set."""

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
        residual_alpha: float = 0.0,
        perm: mx.array | None = None,
        sr_seed: int = 0,
        key: mx.array | None = None,
    ) -> None:
        super().__init__()
        if 3 * (2 * C - 1) > 256:
            raise ValueError("C is too large for uint8 state encoding")
        if in_features % group != 0:
            raise ValueError(f"in_features {in_features} must be divisible by group {group}")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group = int(group)
        self.n_groups = in_features // group
        self.C = int(C)
        self.lr = float(lr)
        self.lr_scale = float(lr_scale)
        self.rms_beta = float(rms_beta)
        self.rms_eps = float(rms_eps)
        self.residual_alpha = float(residual_alpha)
        self.update_enabled = True
        self._sr_step = int(sr_seed)

        k = key if key is not None else mx.random.key(0)
        t0 = mx.random.randint(-1, 2, (out_features, in_features), key=k)
        self.codes = encode_state(t0, mx.zeros_like(t0), self.C)
        s0 = math.sqrt(3.0 / (2.0 * in_features))
        self.scale = mx.full((out_features, self.n_groups), s0, dtype=mx.float32)
        self.v = mx.zeros((out_features, 1), dtype=mx.float32)
        self.tap = mx.zeros(())
        self.perm = mx.arange(in_features, dtype=mx.int32)
        self.group_index = mx.arange(in_features, dtype=mx.int32) // self.group
        self.freeze(keys=["codes", "scale", "v", "perm", "group_index"], recurse=False)
        self._identity_perm = True
        if perm is not None:
            self.set_permutation(perm)
        self._fn = self._make_fn()

    def set_permutation(self, perm: mx.array) -> None:
        perm = perm.astype(mx.int32)
        if perm.size != self.in_features:
            raise ValueError("perm must be a permutation of range(in_features)")
        sorted_ok = mx.all(mx.sort(perm) == mx.arange(self.in_features, dtype=mx.int32))
        if not sorted_ok.item():
            raise ValueError("perm must be a permutation of range(in_features)")
        # group_index[j] = (position of column j in the act-ordered layout) // group
        inv = mx.argsort(perm).astype(mx.int32)
        self.perm = perm
        self.group_index = inv // self.group
        self._identity_perm = bool(mx.all(perm == mx.arange(self.in_features, dtype=mx.int32)).item())

    # --- decoded views ---------------------------------------------------------------
    def _decode(self) -> tuple[mx.array, mx.array]:
        t, c = decode_state(self.codes, self.C)
        return t.astype(mx.float32), c.astype(mx.float32)

    def column_scales(self, scale: mx.array | None = None) -> mx.array:
        s = self.scale if scale is None else scale
        if self._identity_perm and self.group == self.in_features:
            return s  # [out,1] broadcasts
        return mx.take(s, self.group_index, axis=1)

    def visible_weight(self) -> mx.array:
        t, c = self._decode()
        code = t + self.residual_alpha * c / self.C if self.residual_alpha > 0 else t
        return self.column_scales() * code

    # --- state I/O (the PTQ / Bonsai entry point) --------------------------------------
    def load_group_state(
        self,
        scales: mx.array,
        t: mx.array,
        c: mx.array | None = None,
        perm: mx.array | None = None,
    ) -> None:
        if tuple(scales.shape) != (self.out_features, self.n_groups):
            raise ValueError(f"scale shape {tuple(scales.shape)} != {(self.out_features, self.n_groups)}")
        if tuple(t.shape) != (self.out_features, self.in_features):
            raise ValueError("t shape mismatch")
        t = t.astype(mx.int32)
        if not mx.all(mx.abs(t) <= 1).item():
            raise ValueError("t must be ternary")
        c = mx.zeros_like(t) if c is None else c.astype(mx.int32)
        if mx.max(mx.abs(c)).item() > self.C - 1:
            raise ValueError("counter residual exceeds representable range")
        if perm is not None:
            self.set_permutation(perm)
        self.scale = mx.maximum(scales.astype(mx.float32), 1e-8)
        self.codes = encode_state(t, c, self.C)
        self.v = mx.zeros_like(self.v)
        mx.eval(self.parameters())

    def set_residual_alpha(self, alpha: float) -> None:
        self.residual_alpha = float(min(1.0, max(0.0, alpha)))

    # --- the fused-backward linear -----------------------------------------------------
    def _make_fn(self):
        @mx.custom_function
        def gs_linear(x2: mx.array, w: mx.array, tap: mx.array) -> mx.array:
            return x2 @ w.T

        @gs_linear.vjp
        def gs_linear_vjp(primals, cotangents, outputs):
            x2, w, tap = primals
            go = cotangents[0] if isinstance(cotangents, (list, tuple)) else cotangents
            grad_x = go @ w
            if self.training and self.update_enabled:
                grad_w = mx.matmul(go.T.astype(mx.float32), x2.astype(mx.float32))
                seed = self._sr_step & 0xFFFFFFFF
                self._sr_step += 1
                self._apply_update(grad_w, seed)
            return grad_x, mx.zeros_like(w), mx.zeros_like(tap)

        return gs_linear

    def __call__(self, x: mx.array) -> mx.array:
        w = self.visible_weight()
        shape = x.shape
        y2 = self._fn(x.reshape(-1, shape[-1]), w, self.tap)
        return y2.reshape(*shape[:-1], self.out_features)

    def _apply_update(self, grad_w: mx.array, seed: int) -> None:
        """Group-scale RMS + hash-SR counter update (functional; mirrors the solver-v3 torch
        `_update` with the deterministic SR stream of the rest of the MLX port)."""
        C = self.C
        t, c = self._decode()
        gw = grad_w.astype(mx.float32)

        g_sq = mx.mean(gw * gw, axis=1, keepdims=True)
        v_new = self.rms_beta * self.v + (1.0 - self.rms_beta) * g_sq
        denom = mx.maximum(mx.sqrt(v_new), self.rms_eps)
        grad_eff = gw / denom

        code = t + self.residual_alpha * c / C if self.residual_alpha > 0 else t
        contrib = gw * code
        # per-(row, group) scale gradient: sum the act-ordered groups, normalize by sqrt(group)
        if not self._identity_perm:
            contrib = mx.take(contrib, self.perm, axis=1)
        grad_scale = contrib.reshape(self.out_features, self.n_groups, self.group).sum(axis=-1)
        grad_scale = grad_scale / math.sqrt(self.group)
        new_scale = mx.clip(self.scale - self.lr_scale * grad_scale, 1e-5, 10.0)

        old_col = self.column_scales(self.scale)
        new_col = self.column_scales(new_scale)
        c_reb = c * (old_col / new_col)
        ticks = (-self.lr) * grad_eff * (C / new_col)
        val = c_reb + ticks
        elem = mx.arange(self.out_features * self.in_features, dtype=mx.uint32
                         ).reshape(self.out_features, self.in_features)
        rnd = uniform01(mx.array(int(seed) & 0xFFFFFFFF, dtype=mx.uint32) ^ hash_u32(elem))
        f = mx.floor(val)
        cc = f + (rnd < (val - f)).astype(mx.float32)
        carry = mx.trunc(cc / C)
        rem = cc - carry * C
        nt = t + carry
        ct = mx.clip(nt, -1, 1)
        rem = mx.where(ct != nt, mx.sign(cc) * (C - 1), rem)
        rem = mx.clip(rem, -(C - 1), C - 1)

        self.codes = encode_state(ct.astype(mx.int32), rem.astype(mx.int32), C)
        self.scale = new_scale
        self.v = v_new

    # --- diagnostics -------------------------------------------------------------------
    def state_statistics(self) -> dict[str, float]:
        t, c = self._decode()
        return {
            "counter_edge": mx.mean((mx.abs(c) >= self.C - 1).astype(mx.float32)).item(),
            "ternary_zero": mx.mean((t == 0).astype(mx.float32)).item(),
            "scale_mean": mx.mean(self.scale).item(),
            "residual_alpha": float(self.residual_alpha),
        }

    def persistent_bytes(self) -> int:
        return (self.codes.size + self.scale.size * 4 + self.v.size * 4
                + self.perm.size * 4 + self.group_index.size * 4)
