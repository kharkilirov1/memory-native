"""Bit layout for mx6: E2M1 values + 2-bit residual + one E8M0 scale per 32."""
from __future__ import annotations
import torch

BLOCK = 32
BYTES_PER_BLOCK = 25
VALUE_BYTES = 16
RESIDUAL_BYTES = 8
MAX_LEVEL = 6.0
LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
MIDPOINTS = (LEVELS[1:] + LEVELS[:-1]) / 2


def check_shape(in_features: int, out_features: int) -> int:
    if in_features % BLOCK:
        raise ValueError(f"in_features must be a multiple of {BLOCK}, got {in_features}")
    return in_features // BLOCK


def value_codes(w: torch.Tensor) -> torch.Tensor:
    a = w.detach().abs().clamp(max=MAX_LEVEL)
    mag = torch.bucketize(a, MIDPOINTS.to(a.device))
    neg = (w.detach() < 0).to(torch.int16)
    codes = (mag.to(torch.int16) | (neg << 3)).to(torch.uint8)
    return torch.where(mag == 0, torch.zeros_like(codes), codes)


def value_floats(codes: torch.Tensor) -> torch.Tensor:
    mag = LEVELS.to(codes.device)[(codes & 0x07).to(torch.long)]
    return torch.where(codes >= 8, -mag, mag)


def ceil_log2(x: torch.Tensor) -> torch.Tensor:
    mantissa, exponent = torch.frexp(x)
    return exponent - (mantissa == 0.5).to(exponent.dtype)


def scale_bytes(w: torch.Tensor, reference: torch.Tensor | None = None,
                headroom: float = 6.0) -> torch.Tensor:
    out_features, in_features = w.shape
    blocks = w.reshape(out_features, in_features // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=-1).clamp_min(1e-30)
    exponent = ceil_log2(amax / headroom)
    candidate = (exponent + 127).clamp(0, 254).to(torch.uint8)
    if reference is None:
        return candidate
    s_ref = scale_floats(reference)
    ref16 = reference.to(torch.int16)
    candidate = candidate.to(torch.int16).clamp(ref16 - 1, ref16 + 1).to(torch.uint8)
    keep = (amax <= headroom * s_ref) & (amax >= 0.5 * headroom * s_ref)
    return torch.where(keep, reference, candidate)


def scale_floats(scale: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, scale.to(torch.float32) - 127.0)


def quantize(w: torch.Tensor, headroom: float = 6.0, mode: str = "headroom"):
    if mode == "mse":
        s = scale_bytes_mse(w, headroom=headroom)
    elif mode == "auto":
        s = scale_bytes_auto(w)
    else:
        s = scale_bytes(w, headroom=headroom)
    s_expanded = scale_floats(s).repeat_interleave(BLOCK, dim=1)
    return value_codes(w / s_expanded), s


def dequantize(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return value_floats(codes) * scale_floats(scale).repeat_interleave(BLOCK, dim=1)


def pack(codes: torch.Tensor, residual: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out_features, in_features = codes.shape
    blocks = in_features // BLOCK
    c = codes.reshape(out_features, blocks, BLOCK).to(torch.int16)
    r = residual.reshape(out_features, blocks, BLOCK).to(torch.int16)
    values = (c[..., 0::2] | (c[..., 1::2] << 4)).to(torch.uint8)
    residuals = (r[..., 0::4] | (r[..., 1::4] << 2) |
                 (r[..., 2::4] << 4) | (r[..., 3::4] << 6)).to(torch.uint8)
    blob = torch.cat([values, residuals, scale.reshape(out_features, blocks, 1)], dim=-1)
    return blob.reshape(-1)


def unpack(blob: torch.Tensor, out_features: int, in_features: int):
    blocks = in_features // BLOCK
    b = blob.reshape(out_features, blocks, BYTES_PER_BLOCK).to(torch.int16)
    values = b[..., :VALUE_BYTES]
    lo, hi = values & 0x0F, (values >> 4) & 0x0F
    codes = torch.stack([lo, hi], dim=-1).reshape(out_features, blocks, BLOCK).to(torch.uint8)
    residuals = b[..., VALUE_BYTES:VALUE_BYTES + RESIDUAL_BYTES]
    parts = [(residuals >> shift) & 0x03 for shift in (0, 2, 4, 6)]
    residual = torch.stack(parts, dim=-1).reshape(out_features, blocks, BLOCK).to(torch.uint8)
    scale = b[..., VALUE_BYTES + RESIDUAL_BYTES].to(torch.uint8)
    return (codes.reshape(out_features, in_features),
            residual.reshape(out_features, in_features), scale.reshape(out_features, blocks))


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    flat = codes.reshape(-1).to(torch.uint8)
    if flat.numel() % 2:
        raise ValueError("4-bit code packing requires an even element count")
    pair = flat.reshape(-1, 2).to(torch.int16)
    return (pair[:, 0] | (pair[:, 1] << 4)).to(torch.uint8)


def unpack_nibbles(blob: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    b = blob.reshape(-1).to(torch.int16)
    out = torch.stack((b & 0x0F, (b >> 4) & 0x0F), dim=-1).reshape(-1).to(torch.uint8)
    expected = 1
    for dim in shape: expected *= dim
    if out.numel() != expected:
        raise ValueError(f"packed code count {out.numel()} != requested shape size {expected}")
    return out.reshape(shape)


def pack_2bit_signed(values: torch.Tensor, min_value: int = -1, max_value: int = 2) -> torch.Tensor:
    if min_value != -1 or max_value != 2:
        raise ValueError("only the canonical signed 2-bit state mapping is supported")
    flat = values.reshape(-1).to(torch.int16)
    if flat.numel() % 4:
        raise ValueError("2-bit state packing requires an element count divisible by 4")
    if bool(((flat < -1) | (flat > 1)).any()):
        raise ValueError("2-bit integrator state must be in {-1, 0, +1}")
    enc = (flat + 1).reshape(-1, 4)
    return (enc[:, 0] | (enc[:, 1] << 2) | (enc[:, 2] << 4) | (enc[:, 3] << 6)).to(torch.uint8)


def unpack_2bit_signed(blob: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    b = blob.reshape(-1).to(torch.int16)
    parts = torch.stack(tuple((b >> shift) & 0x03 for shift in (0, 2, 4, 6)), dim=-1)
    enc = parts.reshape(-1)
    expected = 1
    for dim in shape: expected *= dim
    if enc.numel() != expected:
        raise ValueError(f"packed state count {enc.numel()} != requested shape size {expected}")
    if bool((enc == 3).any()):
        raise ValueError("reserved 2-bit state code 3 encountered")
    return (enc - 1).to(torch.int8).reshape(shape)


def persistent_bytes(out_features: int, in_features: int) -> int:
    blocks = check_shape(in_features, out_features)
    return out_features * blocks * BYTES_PER_BLOCK


def deployed_bytes(out_features: int, in_features: int) -> int:
    blocks = check_shape(in_features, out_features)
    return out_features * blocks * (VALUE_BYTES + 1)


def _reconstruct_levels(x: torch.Tensor) -> torch.Tensor:
    a = x.abs().clamp(max=MAX_LEVEL)
    mag = LEVELS.to(x.device)[torch.bucketize(a, MIDPOINTS.to(x.device))]
    return torch.where(x < 0, -mag, mag)


def scale_bytes_mse(w: torch.Tensor, headroom: float = 2.0,
                    reference: torch.Tensor | None = None) -> torch.Tensor:
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
    ref16 = reference.to(torch.int16)
    candidate = candidate.to(torch.int16).clamp(ref16 - 1, ref16 + 1).to(torch.uint8)
    keep = (amax <= headroom * s_ref) & (amax >= 0.5 * headroom * s_ref)
    return torch.where(keep, reference, candidate)


def scale_bytes_auto(w: torch.Tensor, reference: torch.Tensor | None = None,
                     h_min: float = 1.5, h_max: float = 6.0, flush: float = 4.0) -> torch.Tensor:
    out_features, in_features = w.shape
    blocks = w.reshape(out_features, in_features // BLOCK, BLOCK)
    abs_blocks = blocks.abs()
    amax = abs_blocks.amax(dim=-1).clamp_min(1e-30)
    q10 = abs_blocks.sort(dim=-1).values[..., max(0, BLOCK // 10)].clamp_min(1e-12)
    spread = amax / (flush * q10)
    headroom = spread.clamp(h_min, h_max)
    exponent = ceil_log2(amax / headroom)
    return (exponent + 127).clamp(0, 254).to(torch.uint8)
