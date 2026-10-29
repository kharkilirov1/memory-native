"""Stateless Muon-style polar-direction primitive for experiments.

This is intentionally not standard Muon: no momentum matrix is retained. It only
transforms the current gradient into an approximate polar direction using temporary
FP32 matrices. Early char-LM results were worse than the SR baseline; do not enable
this by default without a controlled learning-rate/shape sweep.
"""
from __future__ import annotations

import torch


def zeropower_ns(grad: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate a matrix polar factor with Newton–Schulz iterations."""
    if grad.ndim != 2:
        raise ValueError("stateless Muon-style direction expects a 2-D gradient matrix")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    x = grad.detach().float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        a = x @ x.T
        b = -4.7750 * a + 2.0315 * (a @ a)
        x = 3.4445 * x + b @ x
    if transposed:
        x = x.T
    return x
