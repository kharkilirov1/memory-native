"""The update: stochastic rounding on the E2M1 ladder + 2-bit error feedback."""
from __future__ import annotations
import torch
from . import pack

KAPPA = torch.tensor([0.0, 0.0, -0.5, 0.5])
_GAPS = pack.LEVELS[1:] - pack.LEVELS[:-1]


def _min_gap_per_magnitude() -> torch.Tensor:
    out = []
    for i in range(8):
        adjacent = []
        if i > 0: adjacent.append(float(_GAPS[i - 1]))
        if i < 7: adjacent.append(float(_GAPS[i]))
        out.append(min(adjacent))
    return torch.tensor(out)


MIN_GAP = _min_gap_per_magnitude()


def delta(codes: torch.Tensor, mode: str = "rel") -> torch.Tensor:
    if mode == "abs":
        return torch.ones(codes.shape, dtype=torch.float32, device=codes.device)
    if mode != "rel":
        raise ValueError(f"unknown residual mode {mode!r}")
    return MIN_GAP.to(codes.device)[(codes & 0x07).to(torch.long)]


def stochastic_round_ladder(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    sign = torch.sign(x)
    a = x.abs().clamp(max=pack.MAX_LEVEL)
    upper = torch.bucketize(a, pack.LEVELS.to(x.device)).clamp(1, 7)
    low = pack.LEVELS.to(x.device)[upper - 1]
    high = pack.LEVELS.to(x.device)[upper]
    p = ((a - low) / (high - low)).clamp(0.0, 1.0)
    uniform = torch.rand(a.shape, generator=generator, device=a.device)
    magnitude = torch.where(uniform < p, high, low)
    return torch.where(a == 0, torch.zeros_like(magnitude), sign * magnitude)


def balanced_stochastic_round_ladder(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    if x.shape[-1] % pack.BLOCK:
        raise ValueError(f"last dimension must be divisible by {pack.BLOCK}")
    original_shape = x.shape
    blocks = x.reshape(-1, pack.BLOCK).float()
    sign = torch.sign(blocks)
    a = blocks.abs().clamp(max=pack.MAX_LEVEL)
    levels = pack.LEVELS.to(x.device)
    upper = torch.bucketize(a, levels).clamp(1, 7)
    low, high = levels[upper - 1], levels[upper]
    gap = high - low
    p = ((a - low) / gap).clamp(0.0, 1.0)
    gap_id = torch.where(gap < 0.75, 0, torch.where(gap < 1.5, 1, 2))
    category = (sign < 0).to(torch.long) * 3 + gap_id
    choose_high = torch.zeros_like(p, dtype=torch.bool)
    for group in range(6):
        pg = torch.where(category == group, p, torch.zeros_like(p))
        cumulative = pg.cumsum(dim=-1)
        previous = cumulative - pg
        offset = torch.rand((blocks.shape[0], 1), generator=generator, device=x.device)
        selected = torch.floor(cumulative + offset) - torch.floor(previous + offset)
        choose_high |= (category == group) & (selected > 0)
    magnitude = torch.where(choose_high, high, low)
    result = torch.where(a == 0, torch.zeros_like(magnitude), sign * magnitude)
    return result.reshape(original_shape).to(x.dtype)


def deterministic_round_ladder(x: torch.Tensor) -> torch.Tensor:
    sign = torch.sign(x)
    a = x.abs().clamp(max=pack.MAX_LEVEL)
    return sign * pack.LEVELS.to(x.device)[torch.bucketize(a, pack.MIDPOINTS.to(x.device))]


def quantize_residual(error: torch.Tensor, delta_units: torch.Tensor) -> torch.Tensor:
    normalised = error / delta_units
    distance = (normalised.unsqueeze(-1) - KAPPA.to(error.device)).abs()
    return distance.argmin(dim=-1).to(torch.uint8)


def step(values: torch.Tensor, kappa: torch.Tensor, scale: torch.Tensor, grad_hat: torch.Tensor,
         lr: float, mode: str = "rel", generator: torch.Generator | None = None,
         rounding: str = "sr", feedback: bool = True, headroom: float = 6.0,
         scale_mode: str = "headroom"):
    s = pack.scale_floats(scale).repeat_interleave(pack.BLOCK, dim=1)
    d = delta(values, mode) * s
    visible = pack.value_floats(values) * s
    residual = KAPPA.to(visible.device)[kappa.to(torch.long)] * d
    proposed = visible + residual - lr * grad_hat
    new_scale = (pack.scale_bytes_mse(proposed, headroom=headroom, reference=scale)
                 if scale_mode == "mse"
                 else pack.scale_bytes(proposed, reference=scale, headroom=headroom))
    s_new = pack.scale_floats(new_scale).repeat_interleave(pack.BLOCK, dim=1)
    if rounding == "sr":
        rounded = stochastic_round_ladder(proposed / s_new, generator)
    elif rounding == "bsr":
        rounded = balanced_stochastic_round_ladder(proposed / s_new, generator)
    elif rounding == "rne":
        rounded = deterministic_round_ladder(proposed / s_new)
    else:
        raise ValueError(f"unknown rounding mode {rounding!r}")
    rounded = rounded * s_new
    new_values = pack.value_codes(rounded / s_new)
    error = proposed - rounded
    new_kappa = (quantize_residual(error / s_new, delta(new_values, mode))
                 if feedback else torch.zeros_like(values))
    return new_values, new_kappa, new_scale


def rms_update(grad: torch.Tensor, v_row: torch.Tensor, beta: float, eps: float):
    g_sq = grad.square().mean(dim=1, keepdim=True)
    v_new = v_row.mul(beta).add(g_sq, alpha=1.0 - beta)
    return v_new, grad / v_new.sqrt().clamp_min(eps)


def signed_index(codes: torch.Tensor) -> torch.Tensor:
    magnitude = (codes & 0x07).to(torch.int16)
    return torch.where(codes >= 8, -magnitude, magnitude)


def code_from_index(index: torch.Tensor) -> torch.Tensor:
    index = index.to(torch.int16).clamp(-7, 7)
    codes = index.abs().to(torch.uint8) | torch.where(index < 0, 8, 0).to(torch.uint8)
    return torch.where(index == 0, torch.zeros_like(codes), codes)


def stochastic_round_int(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    floor = torch.floor(x)
    return floor + (torch.rand(x.shape, generator=generator, device=x.device) < (x - floor)).to(x.dtype)


def step_with_integrator(values: torch.Tensor, acc: torch.Tensor, scale: torch.Tensor,
                         grad_hat: torch.Tensor, lr: float, states: int,
                         generator: torch.Generator | None = None, cautious: bool = False,
                         decay: float = 0.0, headroom: float = 6.0,
                         scale_mode: str = "headroom", lr_mode: str = "abs"):
    C = (states + 1) // 2
    s = pack.scale_floats(scale).repeat_interleave(pack.BLOCK, dim=1)
    d = delta(values, "rel") * s
    step_size = -lr * grad_hat
    if lr_mode == "scale":
        step_size = step_size * s
    ticks = step_size / d * C
    if decay > 0.0:
        acc = torch.trunc(acc.to(torch.float32) * decay)
    if cautious:
        opposed = (acc != 0) & (torch.sign(ticks) != torch.sign(acc.to(torch.float32)))
        ticks = torch.where(opposed, torch.zeros_like(ticks), ticks)
    cc = stochastic_round_int(acc.to(torch.float32) + ticks, generator)
    carry = torch.trunc(cc / C)
    index = signed_index(values)
    moved = index + carry.to(torch.int16)
    new_index = moved.clamp(-7, 7)
    blocked = moved != new_index
    new_acc = cc - carry * C
    new_values = code_from_index(new_index)
    new_scale = scale
    new_acc = torch.where(blocked, torch.sign(cc) * (C - 1), new_acc).clamp(-(C - 1), C - 1)
    return new_values, new_acc.to(torch.int8), new_scale


def encode_pow2(x: torch.Tensor) -> torch.Tensor:
    mantissa, exponent = torch.frexp(x.clamp_min(1e-30))
    exponent = exponent - (mantissa < 0.7071067811865476).to(exponent.dtype)
    return (exponent + 127).clamp(0, 254).to(torch.uint8)


def decode_pow2(byte: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, byte.to(torch.float32) - 127.0)


def rms_update_block(grad: torch.Tensor, v_block: torch.Tensor, beta: float, eps: float,
                     codec=None, decoder=None):
    out_features, in_features = grad.shape
    blocks = grad.reshape(out_features, in_features // pack.BLOCK, pack.BLOCK)
    g_sq = blocks.square().mean(dim=-1)
    codec = codec or encode_pow2
    decoder = decoder or decode_pow2
    v_new = beta * decoder(v_block) + (1.0 - beta) * g_sq
    new_bytes = codec(v_new)
    denom = decoder(new_bytes).sqrt().repeat_interleave(pack.BLOCK, dim=1).clamp_min(eps)
    return new_bytes, grad / denom


def encode_fp8(x: torch.Tensor) -> torch.Tensor:
    mantissa, exponent = torch.frexp(x.clamp_min(1e-30))
    mantissa = mantissa * 2.0
    exp = (exponent - 1 + 8).clamp(0, 15)
    mant = ((mantissa - 1.0) * 16.0).round().clamp(0, 15)
    return ((exp << 4) | mant.to(torch.int64)).to(torch.uint8)


def decode_fp8(byte: torch.Tensor) -> torch.Tensor:
    exp = (byte.to(torch.int64) >> 4) - 8
    mant = (byte.to(torch.int64) & 0x0F).to(torch.float32) / 16.0 + 1.0
    return mant * torch.pow(2.0, exp.to(torch.float32))


def rms_update_col(grad: torch.Tensor, r_bytes: torch.Tensor, c_bytes: torch.Tensor,
                   beta: float, eps: float, codec=None, decoder=None):
    codec = codec or encode_fp8
    decoder = decoder or decode_fp8
    g_sq = grad.square()
    r_old, c_old = decoder(r_bytes), decoder(c_bytes)
    r_new = beta * r_old + (1.0 - beta) * g_sq.mean(dim=1, keepdim=True)
    c_new = beta * c_old + (1.0 - beta) * g_sq.mean(dim=0, keepdim=True)
    v = r_new * c_new / r_new.mean().clamp_min(1e-30)
    return codec(r_new), codec(c_new), grad / v.sqrt().clamp_min(eps)


def rms_update_layer(grad: torch.Tensor, v_byte: torch.Tensor, beta: float, eps: float):
    g_sq = grad.square().mean()
    v_old = decode_fp8(v_byte).reshape(())
    v_new = beta * v_old + (1.0 - beta) * g_sq
    new_byte = encode_fp8(v_new.reshape(1))
    denom = decode_fp8(new_byte).sqrt().clamp_min(eps)
    return new_byte, grad / denom
