"""Fused counter update as a custom Metal kernel (mx.fast.metal_kernel).

This is the Metal analogue of the Triton `_counter_update_kernel` in
memory_native/fused_update.py: one thread per output row, two passes over the packed row —
(1) decode + row stats (g_sq, grad_s), (2) per-element hash-SR tick, carry/clamp, repack.
The hash-SR stream is the SAME deterministic MurmurHash scheme, so the kernel agrees with
the pure-MLX / pure-torch references up to fp reduction order (~one SR quantum on an O(1)
fraction of weights — the same caveat the Triton kernel documents), and the dynamics are
identical.

MLX metal kernels are functional: the kernel writes fresh (state, scale, v) outputs and the
caller swaps them in. That keeps the packed state packed end to end — no unpacked uint8
tensor, no dense fp weight, ever, on the update path.

STATUS: written to mirror the T4-verified Triton kernel line for line, and gated behind
`metal_available()`; the pure-MLX fallback (identical math) is what CI on Linux exercises.
Run tests/test_mlx_port.py on an Apple-silicon Mac to gate kernel-vs-reference parity there
(`test_metal_fused_update_matches_reference` engages automatically on Metal).
"""
from __future__ import annotations

import mlx.core as mx

__all__ = ["metal_available", "fused_counter_update_metal"]

_HEADER = """
static inline uint cc_hash_u32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}
"""

# Kernel body. Inputs: state (packed uint8), scale [out,1], v [out,1], grad_w [out,in],
# params_i = [out, in_features, C, lagged], params_f = [lr, lr_scale, rms_beta, rms_eps],
# params_u = [seed]. Outputs: state_out, scale_out, v_out (same shapes as the inputs).
_SOURCE = """
    uint row = thread_position_in_grid.x;
    uint n_out = params_i[0];
    uint in_features = params_i[1];
    int  C = int(params_i[2]);
    bool lagged = params_i[3] != 0;
    if (row >= n_out) return;

    float lr = params_f[0], lr_scale = params_f[1];
    float rms_beta = params_f[2], rms_eps = params_f[3];
    uint seed = params_u[0];

    uint gpr = in_features / 4;
    int lv = 2 * C - 1;
    float Cf = float(C);

    // pass 1: row stats (needs t from the packed state)
    float g_sq = 0.0f, grad_s = 0.0f;
    for (uint g = 0; g < gpr; ++g) {
        uint base = row * gpr * 3 + g * 3;
        uint b0 = uint(state[base + 0]);
        uint b1 = uint(state[base + 1]);
        uint b2 = uint(state[base + 2]);
        int code[4];
        code[0] = int(b0 & 0x3F);
        code[1] = int(((b0 >> 6) | (b1 << 2)) & 0x3F);
        code[2] = int(((b1 >> 4) | (b2 << 4)) & 0x3F);
        code[3] = int((b2 >> 2) & 0x3F);
        for (int l = 0; l < 4; ++l) {
            float gw = grad_w[row * in_features + g * 4 + uint(l)];
            int t = code[l] / lv - 1;
            g_sq += gw * gw;
            grad_s += gw * float(t);
        }
    }
    g_sq /= float(in_features);
    grad_s /= metal::sqrt(float(in_features));

    float s_old = scale[row];
    float v_old = v[row];
    float vv = rms_beta * v_old + (1.0f - rms_beta) * g_sq;
    // lagged: denom from the PREVIOUS v (per-element tick); exact: from the fresh v.
    float denom = metal::max(metal::sqrt(lagged ? v_old : vv), rms_eps);
    float s_new = metal::clamp(s_old - lr_scale * grad_s, 1e-5f, 10.0f);
    v_out[row] = vv;
    scale_out[row] = s_new;

    // pass 2: per-element SR tick, applied per packed group of 4
    for (uint g = 0; g < gpr; ++g) {
        uint base = row * gpr * 3 + g * 3;
        uint b0 = uint(state[base + 0]);
        uint b1 = uint(state[base + 1]);
        uint b2 = uint(state[base + 2]);
        int code[4];
        code[0] = int(b0 & 0x3F);
        code[1] = int(((b0 >> 6) | (b1 << 2)) & 0x3F);
        code[2] = int(((b1 >> 4) | (b2 << 4)) & 0x3F);
        code[3] = int((b2 >> 2) & 0x3F);
        uint ncode[4];
        for (int l = 0; l < 4; ++l) {
            uint col = g * 4 + uint(l);
            float gw = grad_w[row * in_features + col];
            int t = code[l] / lv - 1;
            int c = code[l] % lv - (C - 1);
            float tick = (-lr) * (gw / denom) * (Cf / s_new);
            float val = float(c) * (s_old / s_new) + tick;
            float fl = metal::floor(val);
            uint elem = row * in_features + col;
            uint r = cc_hash_u32(seed ^ cc_hash_u32(elem)) & 0x00FFFFFF;
            float rnd = float(r) * (1.0f / 16777216.0f);
            float cc = fl + ((rnd < (val - fl)) ? 1.0f : 0.0f);
            float carry = (cc >= 0.0f) ? metal::floor(cc / Cf) : metal::ceil(cc / Cf);
            float rem = cc - carry * Cf;
            float nt = float(t) + carry;
            float ct = metal::clamp(nt, -1.0f, 1.0f);
            float sgn = (cc > 0.0f) ? 1.0f : ((cc < 0.0f) ? -1.0f : 0.0f);
            if (ct != nt) rem = sgn * (Cf - 1.0f);
            rem = metal::clamp(rem, -(Cf - 1.0f), Cf - 1.0f);
            ncode[l] = uint((int(ct) + 1) * lv + (int(rem) + (C - 1))) & 0x3F;
        }
        state_out[base + 0] = uchar((ncode[0] | (ncode[1] << 6)) & 0xFF);
        state_out[base + 1] = uchar(((ncode[1] >> 2) | (ncode[2] << 4)) & 0xFF);
        state_out[base + 2] = uchar(((ncode[2] >> 4) | (ncode[3] << 2)) & 0xFF);
    }
"""

_kernel = None


def metal_available() -> bool:
    try:
        return bool(mx.metal.is_available())
    except Exception:  # pragma: no cover - very old mlx
        return False


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="cc_counter_update",
            input_names=["state", "scale", "v", "grad_w", "params_i", "params_f", "params_u"],
            output_names=["state_out", "scale_out", "v_out"],
            header=_HEADER,
            source=_SOURCE,
        )
    return _kernel


def fused_counter_update_metal(
    state_packed: mx.array,
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
) -> tuple[mx.array, mx.array, mx.array]:
    """One-launch fused RMS + hash-SR counter update on PACKED state. Functional: returns
    (new_state_packed, new_scale, new_v). Requires an Apple GPU (`metal_available()`)."""
    out, in_features = grad_w.shape
    assert in_features % 4 == 0
    assert state_packed.shape == (out, (in_features // 4) * 3)
    params_i = mx.array([out, in_features, C, 1 if lagged else 0], dtype=mx.uint32)
    params_f = mx.array([lr, lr_scale, rms_beta, rms_eps], dtype=mx.float32)
    params_u = mx.array([int(seed) & 0xFFFFFFFF], dtype=mx.uint32)
    kernel = _get_kernel()
    tg = 64 if out >= 64 else out
    new_state, new_scale, new_v = kernel(
        inputs=[state_packed, scale, v, grad_w.astype(mx.float32), params_i, params_f, params_u],
        output_shapes=[state_packed.shape, scale.shape, v.shape],
        output_dtypes=[mx.uint8, mx.float32, mx.float32],
        grid=(out, 1, 1),
        threadgroup=(tg, 1, 1),
    )
    return new_state, new_scale, new_v
