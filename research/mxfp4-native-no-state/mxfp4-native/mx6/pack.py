"""Bit layout for `mx6`: E2M1 values + 2-bit residual + one E8M0 scale per 32.

Normative description lives in docs/FORMAT.md. One block of 32 parameters is 25 bytes:

    bytes  0..15  32 value nibbles   (element 2i low nibble of byte i, element 2i+1 high)
    bytes 16..23  32 residual codes  (element i in bits 2*(i%4) of byte 16 + i//4)
    byte  24      E8M0 block scale   (value = 2 ** (byte - 127))

A block never straddles a row: `in_features` must be a multiple of 32.
"""
from __future__ import annotations

import torch

BLOCK = 32
BYTES_PER_BLOCK = 25
VALUE_BYTES = 16
RESIDUAL_BYTES = 8
MAX_LEVEL = 6.0

#: E2M1 magnitudes, indexed by the 3-bit magnitude field of the code.
LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
#: Midpoints between consecutive magnitudes; the rounding boundaries.
MIDPOINTS = (LEVELS[1:] + LEVELS[:-1]) / 2  # 0.25 .75 1.25 1.75 2.5 3.5 5.0


def check_shape(in_features: int, out_features: int) -> int:
    if in_features % BLOCK:
        raise ValueError(f"in_features must be a multiple of {BLOCK}, got {in_features}")
    return in_features // BLOCK


def value_codes(w: torch.Tensor) -> torch.Tensor:
    """Round a float tensor to E2M1 codes (uint8 0..15, sign in bit 3)."""
    a = w.detach().abs().clamp(max=MAX_LEVEL)
    mag = torch.bucketize(a, MIDPOINTS.to(a.device))  # 0..7
    neg = (w.detach() < 0).to(torch.int16)
    codes = (mag.to(torch.int16) | (neg << 3)).to(torch.uint8)
    return torch.where(mag == 0, torch.zeros_like(codes), codes)  # never encode -0


def value_floats(codes: torch.Tensor) -> torch.Tensor:
    """Decode E2M1 codes to their magnitudes with sign (in units of the block scale)."""
    mag = LEVELS.to(codes.device)[(codes & 0x07).to(torch.long)]
    return torch.where(codes >= 8, -mag, mag)


def ceil_log2(x: torch.Tensor) -> torch.Tensor:
    """Smallest integer e with 2**e >= x, computed exactly (no libm `log2` rounding).

    `torch.log2` is not guaranteed to be exact on powers of two, and an off-by-one
    there silently doubles a block scale. `frexp` splits x = m * 2**e with m in
    [0.5, 1), so x is a power of two exactly when m == 0.5.
    """
    mantissa, exponent = torch.frexp(x)
    return exponent - (mantissa == 0.5).to(exponent.dtype)


def scale_bytes(w: torch.Tensor, reference: torch.Tensor | None = None,
                headroom: float = 6.0) -> torch.Tensor:
    """E8M0 scale per 32-element block: largest power of two keeping amax/s <= 6.

    With `reference` (the layer's current scale bytes) the scale is sticky: it is kept
    whenever the block already fits, i.e. `3*s <= amax <= 6*s`. Without this, quantising
    an already-quantised block whose maximum sits on level 3 re-derives `s/2` and rewrites
    every code in the block on a step with a zero gradient.
    """
    out_features, in_features = w.shape
    blocks = w.reshape(out_features, in_features // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=-1).clamp_min(1e-30)
    exponent = ceil_log2(amax / headroom)
    candidate = (exponent + 127).clamp(0, 254).to(torch.uint8)  # 255 is NaN in E8M0
    if reference is None:
        return candidate
    s_ref = scale_floats(reference)
    # move at most one octave per step, so a moving distribution cannot ratchet the scale
    ref16 = reference.to(torch.int16)
    candidate = candidate.to(torch.int16).clamp(ref16 - 1, ref16 + 1).to(torch.uint8)
    keep = (amax <= headroom * s_ref) & (amax >= 0.5 * headroom * s_ref)
    return torch.where(keep, reference, candidate)


def scale_floats(scale: torch.Tensor) -> torch.Tensor:
    """E8M0 bytes -> float scales, broadcastable over the 32 elements of a block."""
    return torch.pow(2.0, scale.to(torch.float32) - 127.0)


def quantize(w: torch.Tensor, headroom: float = 6.0,
             mode: str = "headroom") -> tuple[torch.Tensor, torch.Tensor]:
    """Full MXFP4 quantisation: (codes, scale) for the visible weight."""
    if mode == "mse":
        s = scale_bytes_mse(w, headroom=headroom)
    elif mode == "auto":
        s = scale_bytes_auto(w)
    else:
        s = scale_bytes(w, headroom=headroom)
    s_expanded = scale_floats(s).repeat_interleave(BLOCK, dim=1)
    codes = value_codes(w / s_expanded)
    return codes, s


def dequantize(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """(codes, scale bytes) -> float weight."""
    return value_floats(codes) * scale_floats(scale).repeat_interleave(BLOCK, dim=1)


def pack(codes: torch.Tensor, residual: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Pack the three planes into the flat 25-bytes-per-block blob (uint8, 1-D)."""
    out_features, in_features = codes.shape
    blocks = in_features // BLOCK
    c = codes.reshape(out_features, blocks, BLOCK).to(torch.int16)
    r = residual.reshape(out_features, blocks, BLOCK).to(torch.int16)

    lo, hi = c[..., 0::2], c[..., 1::2]
    values = (lo | (hi << 4)).to(torch.uint8)

    r0, r1, r2, r3 = r[..., 0::4], r[..., 1::4], r[..., 2::4], r[..., 3::4]
    residuals = (r0 | (r1 << 2) | (r2 << 4) | (r3 << 6)).to(torch.uint8)

    blob = torch.cat([values, residuals, scale.reshape(out_features, blocks, 1)], dim=-1)
    return blob.reshape(-1)


def unpack(blob: torch.Tensor, out_features: int, in_features: int
           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse of `pack`."""
    blocks = in_features // BLOCK
    b = blob.reshape(out_features, blocks, BYTES_PER_BLOCK).to(torch.int16)

    values = b[..., :VALUE_BYTES]
    lo = values & 0x0F
    hi = (values >> 4) & 0x0F
    codes = torch.stack([lo, hi], dim=-1).reshape(out_features, blocks, BLOCK).to(torch.uint8)

    residuals = b[..., VALUE_BYTES:VALUE_BYTES + RESIDUAL_BYTES]
    parts = [(residuals >> shift) & 0x03 for shift in (0, 2, 4, 6)]
    residual = torch.stack(parts, dim=-1).reshape(out_features, blocks, BLOCK).to(torch.uint8)

    scale = b[..., VALUE_BYTES + RESIDUAL_BYTES].to(torch.uint8)
    return (codes.reshape(out_features, in_features),
            residual.reshape(out_features, in_features),
            scale.reshape(out_features, blocks))


def persistent_bytes(out_features: int, in_features: int) -> int:
    """Bytes of the training state for a bias-free layer, excluding the per-row RMS."""
    blocks = check_shape(in_features, out_features)
    return out_features * blocks * BYTES_PER_BLOCK


def deployed_bytes(out_features: int, in_features: int) -> int:
    """Bytes of the shipped artifact: the residual plane is dropped."""
    blocks = check_shape(in_features, out_features)
    return out_features * blocks * (VALUE_BYTES + 1)

def _reconstruct_levels(x: torch.Tensor) -> torch.Tensor:
    """Round values (in block-scale units) onto the E2M1 ladder and back."""
    a = x.abs().clamp(max=MAX_LEVEL)
    mag = LEVELS.to(x.device)[torch.bucketize(a, MIDPOINTS.to(x.device))]
    return torch.where(x < 0, -mag, mag)


def scale_bytes_mse(w: torch.Tensor, headroom: float = 2.0, reference: torch.Tensor | None = None
                    ) -> torch.Tensor:
    """Per-block scale that minimises reconstruction MSE over three adjacent octaves.

    The headroom rule only centres the search; the winning exponent is chosen by
    sum((w - Q_s(w))**2) per block. This removes the width-dependent headroom
    hyperparameter (h=1.5 at d=128, h=4 at d=256) in favour of the classical
    MSE-optimal clipping criterion.
    """
    out_features, in_features = w.shape
    blocks = w.reshape(out_features, in_features // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    base = ceil_log2(amax / headroom)
    best_err = torch.full(base.shape, float("inf"), device=base.device)
    best_exp = base.clone()
    for delta in (-1, 0, 1):
        exponent = base + delta
        scale = torch.pow(2.0, exponent.to(torch.float32))
        approx = _reconstruct_levels(blocks / scale) * scale
        err = (blocks - approx).square().sum(dim=-1, keepdim=True)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_exp = torch.where(better, exponent, best_exp)
    candidate = (best_exp.squeeze(-1) + 127).clamp(0, 254).to(torch.uint8)
    if reference is None:
        return candidate
    s_ref = scale_floats(reference)
    # move at most one octave per step, so a moving distribution cannot ratchet the scale
    ref16 = reference.to(torch.int16)
    candidate = candidate.to(torch.int16).clamp(ref16 - 1, ref16 + 1).to(torch.uint8)
    keep = (amax <= headroom * s_ref) & (amax >= 0.5 * headroom * s_ref)
    return torch.where(keep, reference, candidate)

# ── block-accumulator: split scale byte (5-bit exp + 3-bit acc-high) ──────────
# See docs/BLOCK_ACCUMULATOR.md for the full design.

SCALE_BITS = 5
ACC_HIGH_BITS = 3
SCALE_BIAS = 15
ACC_HIGH_MASK = (1 << ACC_HIGH_BITS) - 1   # 0x07
SCALE_SHIFT = ACC_HIGH_BITS                 # 3

BLOCK_ACC_BYTES = 2  # scale+acc-high byte + acc-low byte per 32-param block


def scale_bytes_split(w: torch.Tensor, headroom: float = 6.0,
                      reference: torch.Tensor | None = None) -> torch.Tensor:
    """Per-block scale packed into 5 bits with 3-bit acc-high zeroed.

    Returns a uint8 tensor [out, blocks] where bits [7:3] are the 5-bit exponent
    (bias ``SCALE_BIAS``) and bits [2:0] are zero (reserved for acc-high).
    The scale is sticky like ``scale_bytes``: kept while the block fits.
    """
    out_features, in_features = w.shape
    blocks = w.reshape(out_features, in_features // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=-1).clamp_min(1e-30)
    exponent = ceil_log2(amax / headroom)
    exp5 = (exponent + SCALE_BIAS).clamp(0, (1 << SCALE_BITS) - 1)
    candidate = (exp5 << SCALE_SHIFT).to(torch.uint8)
    if reference is None:
        return candidate
    ref_exp = reference >> SCALE_SHIFT
    ref_exp16 = ref_exp.to(torch.int16)
    cand_exp = exp5.to(torch.int16).clamp(ref_exp16 - 1, ref_exp16 + 1)
    candidate = (cand_exp << SCALE_SHIFT).to(torch.uint8)
    s_ref = scale_floats_split(reference)
    keep = (amax <= headroom * s_ref) & (amax >= 0.5 * headroom * s_ref)
    # preserve acc-high bits from reference when keeping
    return torch.where(keep, reference, candidate)


def scale_floats_split(scale_byte: torch.Tensor) -> torch.Tensor:
    """Extract the 5-bit exponent from a split scale byte and return float scales."""
    exp = (scale_byte >> SCALE_SHIFT).to(torch.float32)
    return torch.pow(2.0, exp - SCALE_BIAS)


def acc_high_get(scale_byte: torch.Tensor) -> torch.Tensor:
    """Extract the 3-bit acc-high field from a split scale byte."""
    return (scale_byte & ACC_HIGH_MASK).to(torch.int32)


def acc_high_set(scale_byte: torch.Tensor, acc_high: torch.Tensor) -> torch.Tensor:
    """Write the 3-bit acc-high field into a split scale byte (preserves exponent)."""
    exp = scale_byte & ~ACC_HIGH_MASK
    return (exp | (acc_high.to(torch.int32) & ACC_HIGH_MASK)).to(torch.uint8)


def block_acc_combine(acc_high: torch.Tensor, acc_low: torch.Tensor) -> torch.Tensor:
    """Combine 3-bit acc-high and 8-bit acc-low into an 11-bit unsigned accumulator."""
    return (acc_high.to(torch.int32) << 8) | acc_low.to(torch.int32)


def block_acc_split(acc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split an 11-bit unsigned accumulator into 3-bit acc-high and 8-bit acc-low."""
    acc = acc.to(torch.int32)
    return ((acc >> 8) & ACC_HIGH_MASK).to(torch.uint8), (acc & 0xFF).to(torch.uint8)


def quantize_split(w: torch.Tensor, headroom: float = 6.0
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """Full quantisation with split scale: (codes, scale_byte with acc-high=0)."""
    s = scale_bytes_split(w, headroom=headroom)
    s_expanded = scale_floats_split(s).repeat_interleave(BLOCK, dim=1)
    codes = value_codes(w / s_expanded)
    return codes, s


def dequantize_split(codes: torch.Tensor, scale_byte: torch.Tensor) -> torch.Tensor:
    """(codes, split scale bytes) -> float weight (acc-high bits ignored)."""
    return value_floats(codes) * scale_floats_split(scale_byte).repeat_interleave(BLOCK, dim=1)


def persistent_bytes_block(out_features: int, in_features: int) -> int:
    """Bytes of training state for a block-acc layer (codes + 2 bytes/block)."""
    blocks = check_shape(in_features, out_features)
    return out_features * in_features // 2 + out_features * blocks * BLOCK_ACC_BYTES


def scale_bytes_auto(w: torch.Tensor, reference: torch.Tensor | None = None,
                     h_min: float = 1.5, h_max: float = 6.0, flush: float = 4.0
                     ) -> torch.Tensor:
    """Per-block headroom derived from the block's own dynamic range.

    A block whose smallest meaningful weight (10th percentile of |w|) sits far below its
    amax cannot afford aggressive headroom: shrinking s would flush those weights to zero.
    So h grows with the block spread max / (flush * q10), clamped to [h_min, h_max].
    Tight blocks get the full headroom; wide blocks get little. This is scale-free by
    construction, unlike a per-layer constant whose knee moved from 1.5 to 4 with width.
    """
    out_features, in_features = w.shape
    blocks = w.reshape(out_features, in_features // BLOCK, BLOCK)
    abs_blocks = blocks.abs()
    amax = abs_blocks.amax(dim=-1).clamp_min(1e-30)
    q10 = abs_blocks.sort(dim=-1).values[..., max(0, BLOCK // 10)].clamp_min(1e-12)
    spread = amax / (flush * q10)
    headroom = spread.clamp(h_min, h_max)
    exponent = ceil_log2(amax / headroom)
    return (exponent + 127).clamp(0, 254).to(torch.uint8)


# ── nibble / 2-bit storage (compact_state) ────────────────────────────────────
# The compact_state path referenced these helpers before they existed, so it
# raised AttributeError on construction and was never exercised.

def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """[out, in] uint8 codes in [0, 15] -> [out, in // 2] uint8 (low nibble first)."""
    out_f, in_f = codes.shape
    if in_f % 2:
        raise ValueError("nibble packing needs an even in_features")
    c = codes.reshape(out_f, in_f // 2, 2).to(torch.int32)
    return ((c[..., 0] & 0x0F) | ((c[..., 1] & 0x0F) << 4)).to(torch.uint8)


def unpack_nibbles(packed: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    out_f, in_f = shape
    p = packed.to(torch.int32)
    lo = p & 0x0F
    hi = (p >> 4) & 0x0F
    return torch.stack((lo, hi), dim=-1).reshape(out_f, in_f).to(torch.uint8)


def pack_2bit_signed(acc: torch.Tensor) -> torch.Tensor:
    """[out, in] int8 in [-2, 1] -> [out, in // 4] uint8, biased by +2."""
    out_f, in_f = acc.shape
    if in_f % 4:
        raise ValueError("2-bit packing needs in_features divisible by 4")
    a = (acc.to(torch.int32) + 2).clamp(0, 3).reshape(out_f, in_f // 4, 4)
    return (a[..., 0] | (a[..., 1] << 2) | (a[..., 2] << 4) | (a[..., 3] << 6)).to(torch.uint8)


def unpack_2bit_signed(packed: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    out_f, in_f = shape
    p = packed.to(torch.int32)
    parts = [((p >> (2 * i)) & 0x03) - 2 for i in range(4)]
    return torch.stack(parts, dim=-1).reshape(out_f, in_f).to(torch.int8)
