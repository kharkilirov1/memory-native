"""Balanced dependent stochastic rounding for 32-value MXFP4 blocks.

This is a standalone research implementation extracted from the uploaded experiment.
It adds temporary work only; it does not maintain optimizer state.
"""
from __future__ import annotations

import torch

BLOCK = 32
LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
MAX_LEVEL = 6.0


def balanced_stochastic_round_ladder(
    x: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Unbiased dependent rounding with bounded aggregate error per MX block.

    Values are grouped by sign and adjacent E2M1 gap. A single uniform offset is applied
    to cumulative probabilities in each group. Each coordinate keeps its SR marginal;
    groupwise rounding-error sum is bounded by one local ladder gap.

    The final dimension must be divisible by 32.
    """
    if x.shape[-1] % BLOCK:
        raise ValueError(f"last dimension must be divisible by {BLOCK}")
    original_shape = x.shape
    blocks = x.reshape(-1, BLOCK).float()
    sign = torch.sign(blocks)
    a = blocks.abs().clamp(max=MAX_LEVEL)
    levels = LEVELS.to(x.device)
    upper = torch.bucketize(a, levels).clamp(1, 7)
    low = levels[upper - 1]
    high = levels[upper]
    gap = high - low
    p = ((a - low) / gap).clamp(0.0, 1.0)
    gap_id = torch.where(gap < 0.75, 0, torch.where(gap < 1.5, 1, 2))
    sign_id = (sign < 0).to(torch.long)
    category = sign_id * 3 + gap_id
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
