"""Unbiased low-bit activation quantization — the activation-memory lever for the counter update.

The counter update needs grad_w = Delta^T X. It does NOT need X exactly: if X is replaced by an
*unbiased* low-bit quantization Q(X) with E[Q(X) | X] = X, then

    E[ Delta^T Q(X) | X, Delta ] = Delta^T X,

so the update stays unbiased (it only gains variance). Storing Q(X) at b bits instead of fp X
shrinks the saved-activation memory the backward needs. This module provides per-row symmetric
stochastic quantization (the unbiasedness comes from stochastic rounding) and an effective-bits
accounting that includes the per-row scale overhead.
"""
from __future__ import annotations

import torch

__all__ = ["stochastic_quantize", "quantize_codes", "dequantize_codes", "effective_bits"]


def _levels(bits: int) -> int:
    # signed symmetric: codes in [-levels, levels], levels = 2^(bits-1) - 1
    return (1 << (bits - 1)) - 1


def quantize_codes(x: torch.Tensor, bits: int, dim: int = -1):
    """Per-row symmetric stochastic quantization. Returns (codes int, scale fp) with
    codes in [-levels, levels] and E[codes * scale | x] = x. Storing codes (b bits) + one
    scale per row is the low-bit saved activation."""
    levels = _levels(bits)
    amax = x.abs().amax(dim=dim, keepdim=True).clamp_min(1e-12)
    scale = amax / levels
    y = x / scale
    floor = torch.floor(y)
    codes = floor + (torch.rand_like(y) < (y - floor)).to(y.dtype)  # stochastic round (unbiased)
    codes = codes.clamp_(-levels, levels)
    return codes, scale


def dequantize_codes(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return codes * scale


def stochastic_quantize(x: torch.Tensor, bits: int, dim: int = -1) -> torch.Tensor:
    """Convenience: round-trip x through unbiased b-bit quantization, returning the dequantized
    tensor (same shape/dtype as x). E[stochastic_quantize(x) | x] = x."""
    codes, scale = quantize_codes(x, bits, dim)
    return dequantize_codes(codes, scale).to(x.dtype)


def effective_bits(bits: int, row_len: int, scale_bits: int = 16) -> float:
    """Bits per element including the amortized per-row scale (one scale_bits value per row)."""
    return bits + scale_bits / float(row_len)
