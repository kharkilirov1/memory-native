"""Fused QKV counter layer — the no-method-change speed layout (acceleration memo M2/M3).

Attention's q/k/v counter linears all read the *same* input h = LN(x). Three separate counter
layers therefore save/quantize h three times, run three small GEMMs, and fire three update
kernels for one activation. A single counter linear of shape [3d, d] is mathematically identical
(each output row owns its scale + RMS second moment, independent across rows -> stacking q,k,v
rows changes nothing), but pays the activation save / quantize / update / launch ONCE and runs one
larger, better-occupied GEMM. This is the "share the activation" win for free.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .baselines import make_linear

__all__ = ["CounterQKVLinear"]


class CounterQKVLinear(nn.Module):
    """One counter linear d -> 3*d whose output is split into (q, k, v).

    Drop-in for three separate make_linear(kind, d, d) layers in an attention block: identical
    math (verified bit-for-bit against the split layers), one saved activation, one update.
    `kind` / `counter_kw` are forwarded to make_linear, so it inherits counter_packed + the fused
    update kernel + act_save_bits exactly like a normal counter linear.
    """

    def __init__(self, d: int, kind: str = "counter_packed", **counter_kw) -> None:
        super().__init__()
        self.d = int(d)
        self.proj = make_linear(kind, d, 3 * d, 1.0, **counter_kw)

    def forward(self, x: torch.Tensor):
        q, k, v = self.proj(x).split(self.d, dim=-1)
        return q, k, v
