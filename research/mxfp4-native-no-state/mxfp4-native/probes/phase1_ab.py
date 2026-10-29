#!/usr/bin/env python
"""Phase-2 preview: does a 2-bit residual buy anything once rounding is unbiased?

Four cells, same teacher, same protocol, medians over 5 seeds:

    rounding x error-feedback

Deterministic rounding (RNE) without feedback cannot learn at all — every update is
smaller than half a grid step and vanishes. Stochastic rounding (SR) fixes that for
free, which is why the error buffer has nothing left to compensate.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mx6 import pack, update  # noqa: E402


def rne(x: torch.Tensor) -> torch.Tensor:
    sign = torch.sign(x)
    a = x.abs().clamp(max=pack.MAX_LEVEL)
    return sign * pack.LEVELS.to(x.device)[torch.bucketize(a, pack.MIDPOINTS.to(x.device))]


def run(seed: int, lr: float, steps: int, rounding: str, feedback: bool, size: int = 32) -> float:
    torch.manual_seed(seed)
    x = torch.randn(128, size)
    teacher = torch.randn(size, size) * 0.14
    target = x @ teacher.t()
    variance = target.var().item()

    codes, scale = pack.quantize(0.25 * torch.randint(-1, 2, (size, size)).float())
    weight = pack.dequantize(codes, scale)
    error = torch.zeros_like(weight)
    v_rms = torch.full((size, 1), 1e-3)

    for _ in range(steps):
        grad = (2.0 / (x.shape[0] * size)) * (x @ weight.t() - target).t() @ x
        v_rms, grad_hat = update.rms_update(grad, v_rms, 0.9, 1e-3)
        proposed = weight + (error if feedback else 0.0) - lr * grad_hat
        scale = pack.scale_bytes(proposed)
        s = pack.scale_floats(scale).repeat_interleave(pack.BLOCK, dim=1)
        quant = update.stochastic_round_ladder if rounding == "sr" else rne
        rounded = quant(proposed / s).to(torch.float32) * s
        if feedback:
            error = proposed - rounded
        weight = rounded

    with torch.no_grad():
        return (x @ weight.t() - target).square().mean().item() / variance


if __name__ == "__main__":
    print(f"{'rounding':>8s} {'feedback':>9s} {'lr=0.003':>9s} {'lr=0.01':>9s}")
    for rounding in ("sr", "rne"):
        for feedback in (False, True):
            row = [statistics.median(run(s, lr, 600, rounding, feedback) for s in range(5))
                   for lr in (0.003, 0.01)]
            print(f"{rounding:>8s} {str(feedback):>9s} " + " ".join(f"{v:9.4f}" for v in row))
