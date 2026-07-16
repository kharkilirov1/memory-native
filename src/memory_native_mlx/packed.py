"""Packed 6-bit storage on MLX: the real sub-byte persistent footprint (0.75 byte/weight).

Bit-identical packing to the torch/OpenCL layout (packed.py / kernels cc_pack4): 4 codes per
3 bytes. A counter model trained on CUDA exports straight into this layer (see interop.py)
and vice versa. On an Apple GPU the packed layer routes its update through the fused Metal
kernel (metal_update.py) — one launch per step, state stays packed end to end; elsewhere it
falls back to the pure-MLX unpack -> update -> repack path with identical hash-SR math.
"""
from __future__ import annotations

import mlx.core as mx

from .counter import RMSCounterLinear
from .metal_update import fused_counter_update_metal, metal_available

__all__ = ["pack_codes", "unpack_codes", "PackedRMSCounterLinear"]


def pack_codes(codes: mx.array) -> mx.array:
    """codes uint8 [out, in] in [0,63] -> packed uint8 [out, (in//4)*3] (4 codes / 3 bytes)."""
    out, in_ = codes.shape
    assert in_ % 4 == 0, "in_features must be divisible by 4 for 6-bit packing"
    c = codes.astype(mx.uint32).reshape(out, in_ // 4, 4)
    c0, c1, c2, c3 = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    p0 = (c0 & 0x3F) | ((c1 & 0x03) << 6)
    p1 = ((c1 >> 2) & 0x0F) | ((c2 & 0x0F) << 4)
    p2 = ((c2 >> 4) & 0x03) | ((c3 & 0x3F) << 2)
    packed = mx.stack([p0, p1, p2], axis=-1)  # [out, in//4, 3]
    return (packed & 0xFF).astype(mx.uint8).reshape(out, (in_ // 4) * 3)


def unpack_codes(packed: mx.array, in_features: int) -> mx.array:
    """packed uint8 [out, (in//4)*3] -> codes uint8 [out, in] in [0,63]."""
    out = packed.shape[0]
    gpr = in_features // 4
    p = packed.astype(mx.uint32).reshape(out, gpr, 3)
    p0, p1, p2 = p[..., 0], p[..., 1], p[..., 2]
    c0 = p0 & 0x3F
    c1 = ((p0 >> 6) | (p1 << 2)) & 0x3F
    c2 = ((p1 >> 4) | (p2 << 4)) & 0x3F
    c3 = (p2 >> 2) & 0x3F
    codes = mx.stack([c0, c1, c2, c3], axis=-1)  # [out, gpr, 4]
    return codes.astype(mx.uint8).reshape(out, in_features)


class PackedRMSCounterLinear(RMSCounterLinear):
    """RMSCounterLinear whose persistent `codes` buffer is packed to 6 bits (0.75 B/weight).

    Same learning dynamics as RMSCounterLinear — with hash-SR both layers are bit-identical
    step for step (tested). Only the storage layout and the update launch differ: on Metal
    the update reads and writes the packed state directly (no unpacked tensor at all)."""

    def __init__(self, *args, **kw) -> None:
        super().__init__(*args, **kw)  # builds unpacked codes [out, in], scale, v
        assert self.in_features % 4 == 0, "in_features must be divisible by 4 for 6-bit packing"
        self.codes = pack_codes(self.codes)  # reassignment keeps the frozen flag

    # storage hooks ------------------------------------------------------------
    def _codes(self) -> mx.array:
        return unpack_codes(self.codes, self.in_features)

    def _store_codes(self, codes: mx.array) -> None:
        self.codes = pack_codes(codes)

    def _apply_update(self, grad_w: mx.array, seed: int) -> None:
        # Fused Metal path: the kernel is the packed-state analogue of the Triton
        # `_counter_update_kernel` — decode, row stats, SR tick and repack in one launch.
        # Same eligibility gates as the torch packed layer's _fused_update.
        if metal_available() and self.use_rms:
            new_packed, s_new, v_new = fused_counter_update_metal(
                self.codes, self.scale, self.v, grad_w,
                C=self.C, lr=self.lr, lr_scale=self.lr_scale,
                rms_beta=self.rms_beta, rms_eps=self.rms_eps,
                seed=seed, lagged=(self.rms_mode == "lagged"),
            )
            self.codes = new_packed
            self.scale = s_new
            self.v = v_new
            # no flip diagnostic here: counting flips would unpack the state, which is
            # exactly what the fused path exists to avoid.
            self._last_update_flips = None
            return
        super()._apply_update(grad_w, seed)
