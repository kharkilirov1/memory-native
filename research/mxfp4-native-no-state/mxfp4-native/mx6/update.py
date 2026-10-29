"""The update: stochastic rounding on the E2M1 ladder + 2-bit error feedback.

Framework-free maths (docs/UPDATE.md). No autograd, no modules РІР‚вЂќ just tensors in,
tensors out, so the same functions can be tested on CPU and called from a kernel
reference later.

    u        = s * value(v) + kappa * delta(v) - lr * grad_hat
    s'       = E8M0 block scale of u
    v'       = SR_e2m1(u / s')
    kappa'   = Q2((u - s' * value(v')) / (delta(v') * s'))

`kappa` is the only thing standing in for a momentum buffer.
"""
from __future__ import annotations

import torch

from . import pack

#: The four residual coefficients, normalised to the local gap. Two codes are the same
#: zero on purpose: an even number of levels cannot hold zero symmetrically, and the
#: duplicated zero gives a dead zone (+-0.25 gap) that stops the residual dithering the
#: weight at convergence. The mid-rise alternative (+-3/8, +-1/8) is unbiased but never
#: settles to a zero residual; it is a Phase-2 ablation, not the default.
KAPPA = torch.tensor([0.0, 0.0, -0.5, 0.5])

#: Gap between consecutive E2M1 magnitudes, in units of the block scale.
_GAPS = (pack.LEVELS[1:] - pack.LEVELS[:-1])  # 0.5 0.5 0.5 0.5 1 1 2


def _min_gap_per_magnitude() -> torch.Tensor:
    """Smallest adjacent gap for each of the 8 magnitudes (the conservative step)."""
    out = []
    for i in range(8):
        adjacent = []
        if i > 0:
            adjacent.append(float(_GAPS[i - 1]))
        if i < 7:
            adjacent.append(float(_GAPS[i]))
        out.append(min(adjacent))
    return torch.tensor(out)


MIN_GAP = _min_gap_per_magnitude()  # [0.5, 0.5, 0.5, 0.5, 1, 1, 1, 2]


def delta(codes: torch.Tensor, mode: str = "rel") -> torch.Tensor:
    """The step that a residual coefficient multiplies, in units of the block scale.

    ``rel`` (default) uses the local gap, so the residual keeps the same *relative*
    precision everywhere. ``abs`` uses one scale step everywhere РІР‚вЂќ simpler, but the
    feedback becomes useless at the top of the ladder (gap 2.0 vs 0.5 at the bottom).
    """
    if mode == "abs":
        return torch.ones(codes.shape, dtype=torch.float32, device=codes.device)
    if mode != "rel":
        raise ValueError(f"unknown residual mode {mode!r}")
    return MIN_GAP.to(codes.device)[(codes & 0x07).to(torch.long)]


def stochastic_round_ladder(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Unbiased rounding of real values (in level units) onto the E2M1 magnitudes.

    The probability of picking the upper neighbour is the relative distance on the
    *non-uniform* ladder, so the result is unbiased in value, not in index.
    """
    sign = torch.sign(x)
    a = x.abs().clamp(max=pack.MAX_LEVEL)
    upper = torch.bucketize(a, pack.LEVELS.to(x.device)).clamp(1, 7)
    low = pack.LEVELS.to(x.device)[upper - 1]
    high = pack.LEVELS.to(x.device)[upper]
    p = ((a - low) / (high - low)).clamp(0.0, 1.0)
    uniform = torch.rand(a.shape, generator=generator, device=a.device)
    magnitude = torch.where(uniform < p, high, low)
    return torch.where(a == 0, torch.zeros_like(magnitude), sign * magnitude)


def deterministic_round_ladder(x: torch.Tensor) -> torch.Tensor:
    """Nearest E2M1 magnitude (used for the RNE + feedback ablation)."""
    sign = torch.sign(x)
    a = x.abs().clamp(max=pack.MAX_LEVEL)
    levels = pack.LEVELS.to(x.device)
    return sign * levels[torch.bucketize(a, pack.MIDPOINTS.to(x.device))]


def quantize_residual(error: torch.Tensor, delta_units: torch.Tensor) -> torch.Tensor:
    """Nearest of the four kappa coefficients for a normalised error; returns 0..3."""
    normalised = error / (delta_units * 1.0)
    distance = (normalised.unsqueeze(-1) - KAPPA.to(error.device)).abs()
    return distance.argmin(dim=-1).to(torch.uint8)


def step(values: torch.Tensor, kappa: torch.Tensor, scale: torch.Tensor,
         grad_hat: torch.Tensor, lr: float, mode: str = "rel",
         generator: torch.Generator | None = None, rounding: str = "sr",
         feedback: bool = True, headroom: float = 6.0,
         scale_mode: str = "headroom") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One counter update. Returns (codes, kappa, scale); inputs are not modified."""
    s = pack.scale_floats(scale).repeat_interleave(pack.BLOCK, dim=1)
    d = delta(values, mode) * s

    visible = pack.value_floats(values) * s
    residual = KAPPA.to(visible.device)[kappa.to(torch.long)] * d
    proposed = visible + residual - lr * grad_hat

    new_scale = (pack.scale_bytes_mse(proposed, headroom=headroom, reference=scale)
                 if scale_mode == "mse"
                 else pack.scale_bytes(proposed, reference=scale, headroom=headroom))
    s_new = pack.scale_floats(new_scale).repeat_interleave(pack.BLOCK, dim=1)

    ladder = stochastic_round_ladder if rounding == "sr" else deterministic_round_ladder
    rounded = ladder(proposed / s_new, generator) if rounding == "sr" else ladder(proposed / s_new)
    rounded = rounded * s_new
    new_values = pack.value_codes(rounded / s_new)

    error = proposed - rounded
    new_kappa = (quantize_residual(error / s_new, delta(new_values, mode))
                 if feedback else torch.zeros_like(values))
    return new_values, new_kappa, new_scale


def rms_update(grad: torch.Tensor, v_row: torch.Tensor, beta: float, eps: float
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row second moment and the normalised gradient (the only extra state)."""
    g_sq = grad.square().mean(dim=1, keepdim=True)
    v_new = v_row.mul(beta).add(g_sq, alpha=1.0 - beta)
    return v_new, grad / v_new.sqrt().clamp_min(eps)

def signed_index(codes: torch.Tensor) -> torch.Tensor:
    """E2M1 code <-> signed ladder index in [-7, 7] (index +-1 is one ladder step)."""
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
                         scale_mode: str = "headroom", lr_mode: str = "abs"
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ternary-style accumulator: integrate gradient ticks, carry into the value at +-C.

    cc holds integers in [-(C-1), C-1] with C = (states + 1) // 2, i.e. the visible
    weight only moves when the accumulated pressure crosses a full ladder step. This is
    momentum stored in the weight word, and it is what the ternary design had with C=11.
    """
    C = (states + 1) // 2
    s = pack.scale_floats(scale).repeat_interleave(pack.BLOCK, dim=1)
    d = delta(values, "rel") * s
    step = -lr * grad_hat
    if lr_mode == "scale":
        step = step * s          # step measured in fractions of the block scale
    ticks = step / d * C
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
    visible = pack.value_floats(new_values) * s
    # isible is already on the ladder, so its amax/s sits in [3, 6]; the maintenance
    # band must be that, not the init headroom, or the scale multiplies every step.
    new_scale = scale  # frozen: headroom is an initialisation choice, not an invariant
    new_acc = torch.where(blocked, torch.sign(cc) * (C - 1), new_acc).clamp(-(C - 1), C - 1)
    return new_values, new_acc.to(torch.int8), new_scale


def encode_pow2(x: torch.Tensor) -> torch.Tensor:
    """Nearest power-of-two exponent as an E8M0-style byte (1 byte per value)."""
    mantissa, exponent = torch.frexp(x.clamp_min(1e-30))
    exponent = exponent - (mantissa < 0.7071067811865476).to(exponent.dtype)
    return (exponent + 127).clamp(0, 254).to(torch.uint8)


def decode_pow2(byte: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, byte.to(torch.float32) - 127.0)


def rms_update_block(grad: torch.Tensor, v_block: torch.Tensor, beta: float, eps: float,
                     codec=None, decoder=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Second moment per 32-element block, stored as one E8M0 byte (0.25 bit/param)."""
    out_features, in_features = grad.shape
    blocks = grad.reshape(out_features, in_features // pack.BLOCK, pack.BLOCK)
    g_sq = blocks.square().mean(dim=-1)
    codec = codec or encode_pow2
    decoder = decoder or decode_pow2
    v = decoder(v_block)
    v_new = beta * v + (1.0 - beta) * g_sq
    new_bytes = codec(v_new)
    denom = decoder(new_bytes).sqrt().repeat_interleave(pack.BLOCK, dim=1).clamp_min(eps)
    return new_bytes, grad / denom


def encode_fp8(x: torch.Tensor) -> torch.Tensor:
    """1 byte per value: 4-bit exponent (bias 8) + 4-bit mantissa, ~3% relative error."""
    mantissa, exponent = torch.frexp(x.clamp_min(1e-30))
    mantissa = mantissa * 2.0            # [1, 2)
    exp = (exponent - 1 + 8).clamp(0, 15)
    mant = ((mantissa - 1.0) * 16.0).round().clamp(0, 15)
    return ((exp << 4) | mant.to(torch.int64)).to(torch.uint8)


def decode_fp8(byte: torch.Tensor) -> torch.Tensor:
    exp = (byte.to(torch.int64) >> 4) - 8
    mant = (byte.to(torch.int64) & 0x0F).to(torch.float32) / 16.0 + 1.0
    return mant * torch.pow(2.0, exp.to(torch.float32))


def rms_update_col(grad: torch.Tensor, r_bytes: torch.Tensor, c_bytes: torch.Tensor,
                   beta: float, eps: float, codec=None, decoder=None
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adafactor-style rank-1 second moment: v_ij ~ R_i * C_j / mean(R), fp8 each.

    One byte per output row plus one byte per input column (~0.08 bit/param for a
    128x512 layer), i.e. per-element-ish scaling for the price of two vectors.
    """
    codec = codec or encode_fp8
    decoder = decoder or decode_fp8
    g_sq = grad.square()
    r_old, c_old = decoder(r_bytes), decoder(c_bytes)
    r_new = beta * r_old + (1.0 - beta) * g_sq.mean(dim=1, keepdim=True)
    c_new = beta * c_old + (1.0 - beta) * g_sq.mean(dim=0, keepdim=True)
    v = r_new * c_new / r_new.mean().clamp_min(1e-30)
    denom = v.sqrt().clamp_min(eps)
    return codec(r_new), codec(c_new), grad / denom


def rms_update_layer(grad: torch.Tensor, v_byte: torch.Tensor, beta: float, eps: float
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """One second moment for the whole layer, stored as a single fp8 byte.

    Removes every per-row/per-block amplification: all weights share one step scale,
    so the only difference from Adam is that our direction is a normalised-gradient
    momentum. This isolates 'normalisation granularity' from 'adaptivity'.
    """
    g_sq = grad.square().mean()
    v_old = decode_fp8(v_byte).reshape(())
    v_new = beta * v_old + (1.0 - beta) * g_sq
    new_byte = encode_fp8(v_new.reshape(1))
    denom = decode_fp8(new_byte).sqrt().clamp_min(eps)
    return new_byte, grad / denom
