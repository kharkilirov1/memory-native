"""Bonsai-format import/export: ternary group-scale checkpoints <-> trainable counter layers.

PrismML's Bonsai releases (e.g. Ternary-Bonsai-27B, Apache 2.0) store ternary {-1,0,+1}
weights with one FP16 scale per group of 128 — exactly the visible part of a
`GroupScaleCounterLinear` state (t = their ternary, scale = their group scales, counter
c = 0). These helpers turn such checkpoints into layers that FINE-TUNE as ternary (the part
no inference stack provides), and export the visible weight back into MLX's native grouped
quantization for `mx.quantized_matmul` inference.

Entry points:
  * `group_counter_from_dense(w)` — from a dequantized/"unpacked" fp weight matrix that is
    exactly group-wise ternary (each group's values in {-s, 0, +s}). Recovers (t, s) and
    verifies ternarity; raises on a non-ternary matrix.
  * `group_counter_from_quantized(q, scales, biases)` — from MLX affine-quant tensors (the
    `-mlx-2bit` builds): dequantize with `mx.dequantize`, then the dense path.
  * `to_mlx_quantized(layer)` — visible weight -> (q, scales, biases) via `mx.quantize`,
    verified lossless by round-trip before returning.

No network access here: these operate on tensors you loaded (e.g. via mlx.core.load from a
downloaded safetensors shard). Model-level orchestration (walking a full checkpoint's
layers) is deliberately left to scripts — layer naming differs per release.
"""
from __future__ import annotations

import mlx.core as mx

from .group_scale import GroupScaleCounterLinear

__all__ = ["group_counter_from_dense", "group_counter_from_quantized", "to_mlx_quantized",
           "ternary_to_mlx_quant"]


def ternary_to_mlx_quant(t: mx.array, scales: mx.array, *, group: int = 128
                         ) -> tuple[mx.array, mx.array, mx.array]:
    """Construct MLX affine 2-bit tensors (q, scales, biases) EXACTLY from ternary t
    [out, in] and per-group scales [out, in/group]: q = t+1 in {0,1,2}, scale = s,
    bias = -s, so dequantize(w) == s*t bit-for-bit on MLX's grid (0 included).

    NOTE: `mx.quantize` itself must NOT be used for ternary weights — it fits affine
    params to the group min/max, whose 2-bit grid {-s, -s/3, +s/3, +s} cannot represent
    0; manual construction is exact where mx.quantize loses every zero weight."""
    out, in_ = t.shape
    assert in_ % 32 == 0, "in_features must be divisible by 32 for 2-bit uint32 packing"
    codes = (t.astype(mx.int32) + 1).astype(mx.uint32).reshape(out, in_ // 16, 16)
    packed = mx.zeros((out, in_ // 16), dtype=mx.uint32)
    for k in range(16):  # little-endian: element k occupies bits [2k, 2k+2)
        packed = packed | (codes[..., k] << (2 * k))
    s = scales.astype(mx.float32)
    return packed, s, -s


def group_counter_from_dense(
    w: mx.array,
    *,
    group: int = 128,
    C: int = 11,
    tol: float = 1e-2,
    **layer_kw,
) -> tuple[GroupScaleCounterLinear, float]:
    """Dense fp weight [out, in], exactly group-wise ternary -> trainable counter layer.

    Recovers per-group scales as max|w| over the group and t = round(w/s), then verifies the
    reconstruction: returns (layer, max_rel_err). Raises ValueError if the matrix is not
    group-ternary within `tol` (relative to each group's scale) — that means the source is
    not a Bonsai-style ternary checkpoint and importing it would silently change the model."""
    out, in_ = w.shape
    if in_ % group != 0:
        raise ValueError(f"in_features {in_} not divisible by group {group}")
    wf = w.astype(mx.float32)
    wg = wf.reshape(out, in_ // group, group)
    s = mx.max(mx.abs(wg), axis=-1)                       # [out, n_groups]
    s_safe = mx.maximum(s, 1e-8)
    t = mx.round(wg / s_safe[..., None])
    if not mx.all(mx.abs(t) <= 1).item():
        raise ValueError("weight is not group-wise ternary: |round(w/s)| > 1")
    err = mx.max(mx.abs(wg - t * s_safe[..., None]) / s_safe[..., None]).item()
    if err > tol:
        raise ValueError(
            f"weight is not group-wise ternary within tol: max_rel_err={err:.3g} > {tol}")
    layer = GroupScaleCounterLinear(in_, out, group=group, C=C, **layer_kw)
    layer.load_group_state(s_safe, t.reshape(out, in_).astype(mx.int32))
    return layer, err


def group_counter_from_quantized(
    q: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    group: int = 128,
    bits: int = 2,
    C: int = 11,
    tol: float = 1e-2,
    **layer_kw,
) -> tuple[GroupScaleCounterLinear, float]:
    """MLX affine-quant tensors (a `-mlx-2bit` Bonsai build) -> trainable counter layer."""
    w = mx.dequantize(q, scales, biases, group_size=group, bits=bits)
    return group_counter_from_dense(w, group=group, C=C, tol=tol, **layer_kw)


def to_mlx_quantized(
    layer: GroupScaleCounterLinear,
    *,
    bits: int = 2,
    atol: float = 1e-3,
) -> tuple[mx.array, mx.array, mx.array]:
    """Visible ternary weight -> MLX native grouped quantization (q, scales, biases), for
    `mx.quantized_matmul` / mlx-lm inference on the optimized kernels.

    Requires residual_alpha == 0 (the deployed weight must be the pure ternary one) and
    verifies the round-trip dequantization matches the visible weight before returning."""
    if layer.residual_alpha != 0.0:
        raise ValueError("set residual_alpha=0 before exporting for inference")
    if bits != 2:
        raise ValueError("ternary export targets the 2-bit MLX format")
    t, _ = layer._decode()
    q, s, b = ternary_to_mlx_quant(t.astype(mx.int32), layer.scale, group=layer.group)
    back = mx.dequantize(q, s, b, group_size=layer.group, bits=2)
    w = layer.visible_weight()
    err = mx.max(mx.abs(back - w)).item()
    scale_floor = mx.max(mx.abs(w)).item()
    if err > atol * max(scale_floor, 1e-8):
        raise ValueError(f"ternary quant round-trip failed: err={err:.3g}")
    return q, s, b
