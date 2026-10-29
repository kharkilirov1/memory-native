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
