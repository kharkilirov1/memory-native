"""Finite-state counter synapse on MLX — the parameters+optimizer+gradients lever, on Metal.

Port notes (vs src/memory_native/counter.py, the PyTorch reference):

* The state layout, encode/decode and the RMS+stochastic-rounding update are the SAME MATH.
  Stochastic rounding here is ALWAYS the deterministic hash-SR of `fused_update.py` (the
  Triton/OpenCL kernels' scheme), not a global RNG: MLX has no global RNG stream to rely on,
  and hash-SR makes the update reproducible across backends (macOS/Metal == Linux/CPU ==
  the torch reference, which is how this port is validated without a Mac in the loop).
* The in-backward fused update maps to `mx.custom_function`: the layer's VJP computes
  grad_x, forms grad_w transiently, applies the counter transition to the state buffers and
  returns. No nn.Parameter-like weight exists; there is nothing for an optimizer to hold.
* The torch `tap` trick carries over as a real trainable scalar `tap` (always zero, zero
  gradient): `nn.value_and_grad` only differentiates paths that reach a trainable parameter,
  so tap guarantees the VJP — and therefore the self-update — fires even for a first layer
  fed raw input. AdamW over the rest of the model sees tap's zero grad and leaves it alone.
* MLX modules are pytrees: `codes`/`scale`/`v` are public frozen arrays (saved by
  `save_weights`, evaluated by `mx.eval(model.parameters())`), excluded from
  `trainable_parameters()` so optimizers never touch them.

Contract (mirrors the torch eager-only contract): one forward -> one VJP per step; train
through `nn.value_and_grad` (a plain call never runs the VJP, so it never updates — that is
the inference path); no `mx.compile` over the update path (the SR seed advances in Python).
"""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

__all__ = [
    "C_DEFAULT",
    "encode_state",
    "decode_state",
    "hash_u32",
    "uniform01",
    "counter_update_hashsr",
    "RMSCounterLinear",
]

# C=8 -> counter c in {-7..+7} (15 levels), 3*15 = 45 reachable states (fits 6 bits / uint8).
# Larger C is allowed while 3*(2C-1) <= 256 (uint8); C=11 gives 63 states (best per ablation).
C_DEFAULT = 8

_M1 = 0x7FEB352D
_M2 = 0x846CA68B


def encode_state(t: mx.array, c: mx.array, C: int = C_DEFAULT) -> mx.array:
    """Encode t in {-1,0,1}, c in {-(C-1),...,C-1} into one uint8 state code."""
    levels = 2 * C - 1
    code = (t.astype(mx.int32) + 1) * levels + (c.astype(mx.int32) + (C - 1))
    return code.astype(mx.uint8)


def decode_state(state: mx.array, C: int = C_DEFAULT) -> tuple[mx.array, mx.array]:
    """Decode a uint8 state code into int32 ternary weight t and residual counter c."""
    levels = 2 * C - 1
    z = state.astype(mx.int32)
    t = z // levels - 1
    c = z % levels - (C - 1)
    return t, c


def hash_u32(x: mx.array) -> mx.array:
    """MurmurHash-style uint32 hash; bit-identical to fused_update.hash_u32 (torch) and the
    OpenCL/Triton cc_hash_u32 (uint32 wrapping arithmetic is native here)."""
    x = x.astype(mx.uint32)
    x = x ^ (x >> 16)
    x = x * mx.array(_M1, dtype=mx.uint32)
    x = x ^ (x >> 15)
    x = x * mx.array(_M2, dtype=mx.uint32)
    x = x ^ (x >> 16)
    return x


def uniform01(x: mx.array) -> mx.array:
    return (hash_u32(x) & 0x00FFFFFF).astype(mx.float32) * (1.0 / 16777216.0)


def counter_update_hashsr(
    codes: mx.array,
    scale: mx.array,
    v: mx.array,
    grad_w: mx.array,
    *,
    C: int,
    lr: float,
    lr_scale: float,
    rms_beta: float,
    rms_eps: float,
    seed: int,
    lagged: bool = False,
    use_rms: bool = True,
) -> tuple[mx.array, mx.array, mx.array]:
    """Deterministic-SR RMS counter update on unpacked codes [out, in]. FUNCTIONAL (MLX
    style): returns (new_codes, new_scale, new_v) instead of mutating. Same math as the
    torch reference `memory_native.fused_update.counter_update_hashsr`; with use_rms=True
    it matches that reference bit-for-bit up to fp reduction order (~one SR quantum on a
    vanishing fraction of weights — the same caveat the Triton kernel carries)."""
    out, in_ = codes.shape
    t_i, c_i = decode_state(codes, C)
    t = t_i.astype(mx.float32)
    c = c_i.astype(mx.float32)
    gw = grad_w.astype(mx.float32)

    # --- row stats (per output row) ---
    g_sq = mx.mean(gw * gw, axis=1, keepdims=True)  # [out,1]
    if use_rms:
        if lagged:
            denom = mx.maximum(mx.sqrt(v), rms_eps)  # previous-step v -> per-element tick
            v_new = rms_beta * v + (1.0 - rms_beta) * g_sq
        else:
            v_new = rms_beta * v + (1.0 - rms_beta) * g_sq
            denom = mx.maximum(mx.sqrt(v_new), rms_eps)  # freshly-updated v
        grad_eff = gw / denom
    else:
        v_new = v
        grad_eff = gw
    # scale is learned from the RAW gradient (its statistics are not normalised).
    grad_s = mx.sum(gw * t, axis=1, keepdims=True) / math.sqrt(in_)
    s_new = mx.clip(scale - lr_scale * grad_s, 1e-5, 10.0)

    # --- per-element deterministic stochastic-rounding tick ---
    elem = mx.arange(out * in_, dtype=mx.uint32).reshape(out, in_)
    seed_u = mx.array(int(seed) & 0xFFFFFFFF, dtype=mx.uint32)
    rnd = uniform01(seed_u ^ hash_u32(elem))
    tick = (-lr) * grad_eff * (C / s_new)
    val = c * (scale / s_new) + tick
    f = mx.floor(val)
    cc = f + (rnd < (val - f)).astype(mx.float32)
    carry = mx.trunc(cc / C)
    rem = cc - carry * C
    nt = t + carry
    ct = mx.clip(nt, -1, 1)
    rem = mx.where(ct != nt, mx.sign(cc) * (C - 1), rem)
    rem = mx.clip(rem, -(C - 1), C - 1)
    new_codes = encode_state(ct.astype(mx.int32), rem.astype(mx.int32), C)
    return new_codes, s_new, v_new


class RMSCounterLinear(nn.Module):
    """Ternary linear layer trained as a finite-state synaptic automaton, with per-row RMS
    adaptive scaling — the MLX analogue of `memory_native.RMSCounterLinear`.

    Exposes NO trainable weight: `codes` is a frozen uint8 state buffer (1 byte/weight here;
    the packed subclass stores 0.75), `scale`/`v` are frozen O(out) row vectors, and the layer
    updates itself inside its custom VJP. Mix it with normal MLX modules; an mlx AdamW over
    the rest of the model (embeddings, norms, head) trains those as usual.
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
        rms_beta: float = 0.9,
        rms_eps: float = 1e-3,
        use_rms: bool = True,
        rms_mode: str = "exact",
        sr_seed: int = 0,
        key: mx.array | None = None,
    ) -> None:
        super().__init__()
        if 3 * (2 * C - 1) > 256:
            raise ValueError("C is too large for uint8 state encoding (need 3*(2C-1) <= 256)")
        if rms_mode not in ("exact", "lagged"):
            raise ValueError("rms_mode must be 'exact' or 'lagged' (MLX port)")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.C = int(C)
        self.lr = float(lr)
        self.lr_scale = float(lr_scale)
        self.rms_beta = float(rms_beta)
        self.rms_eps = float(rms_eps)
        self.use_rms = bool(use_rms)
        self.rms_mode = rms_mode
        self.update_enabled = True
        # hash-SR stream position; advances one per applied update (mirrors _sr_step in the
        # torch packed layer). Python-side state => keep the update path out of mx.compile.
        self._sr_step = int(sr_seed)

        k = key if key is not None else mx.random.key(0)
        t0 = mx.random.randint(-1, 2, (self.out_features, self.in_features), key=k)
        c0 = mx.zeros_like(t0)
        # NOTE: named `codes`, not `state` — mlx.nn.Module already owns a `.state` property.
        self.codes = encode_state(t0, c0, self.C)

        # Var(t)=2/3 for uniform {-1,0,1}; choose Var(w)=gain^2/fan_in.
        s0 = init_gain * math.sqrt(3.0 / (2.0 * self.in_features))
        self.scale = mx.full((self.out_features, 1), s0, dtype=mx.float32)
        self.v = mx.zeros((self.out_features, 1), dtype=mx.float32)
        # tap: genuine trainable scalar (stays 0, gets 0 grad) that threads the layer into
        # every value_and_grad diff set so the VJP-side self-update always fires.
        self.tap = mx.zeros(())
        self.freeze(keys=["codes", "scale", "v"], recurse=False)

        # diagnostics: lazily-evaluated scalars for the LAST applied update (replaced, not
        # accumulated, so no graph growth when never evaluated).
        self._last_update_flips: mx.array | None = None

        self._fn = self._make_fn()

    # --- storage hooks (overridden by the packed subclass) -------------------------
    def _codes(self) -> mx.array:
        return self.codes

    def _store_codes(self, codes: mx.array) -> None:
        self.codes = codes

    def _dense_weight(self) -> mx.array:
        t, _ = decode_state(self._codes(), self.C)
        return self.scale * t.astype(mx.float32)

    def _apply_update(self, grad_w: mx.array, seed: int) -> None:
        old_t, _ = decode_state(self._codes(), self.C)
        new_codes, s_new, v_new = counter_update_hashsr(
            self._codes(), self.scale, self.v, grad_w,
            C=self.C, lr=self.lr, lr_scale=self.lr_scale,
            rms_beta=self.rms_beta, rms_eps=self.rms_eps,
            seed=seed, lagged=(self.rms_mode == "lagged"), use_rms=self.use_rms,
        )
        new_t, _ = decode_state(new_codes, self.C)
        self._last_update_flips = mx.sum((new_t != old_t).astype(mx.int32))
        self._store_codes(new_codes)
        self.scale = s_new
        self.v = v_new

    # --- the fused-backward linear ---------------------------------------------------
    def _make_fn(self):
        @mx.custom_function
        def counter_linear(x2: mx.array, w: mx.array, tap: mx.array) -> mx.array:
            return x2 @ w.T

        @counter_linear.vjp
        def counter_linear_vjp(primals, cotangents, outputs):
            x2, w, tap = primals
            go = cotangents[0] if isinstance(cotangents, (list, tuple)) else cotangents
            # grad_x from the pre-update weight (same order as the torch backward).
            grad_x = go @ w
            if self.training and self.update_enabled:
                # the [out,in] grad_w tile exists only inside this VJP — transient, freed
                # with the step graph; no full-model gradient buffer is ever retained.
                grad_w = mx.matmul(go.T.astype(mx.float32), x2.astype(mx.float32))
                seed = self._sr_step & 0xFFFFFFFF
                self._sr_step += 1
                self._apply_update(grad_w, seed)
            return grad_x, mx.zeros_like(w), mx.zeros_like(tap)

        return counter_linear

    def __call__(self, x: mx.array) -> mx.array:
        w = self._dense_weight()
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        y2 = self._fn(x2, w, self.tap)
        return y2.reshape(*shape[:-1], self.out_features)

    # --- diagnostics ------------------------------------------------------------------
    def state_statistics(self) -> dict[str, float]:
        t, c = decode_state(self._codes(), self.C)
        return {
            "minus": mx.mean((t == -1).astype(mx.float32)).item(),
            "zero": mx.mean((t == 0).astype(mx.float32)).item(),
            "plus": mx.mean((t == 1).astype(mx.float32)).item(),
            "counter_abs_mean": mx.mean(mx.abs(c).astype(mx.float32)).item(),
            "counter_edge": mx.mean((mx.abs(c) == self.C - 1).astype(mx.float32)).item(),
            "scale_mean": mx.mean(self.scale).item(),
        }
