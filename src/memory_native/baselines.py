"""Baselines to compare the counter synapse against — in plain PyTorch.

Right now: a BitNet-b1.58-style ternary QAT linear (full FP32 master + Adam, ternary in the
forward) which isolates "cost of the 6-bit counter optimizer" vs "cost of ternarization".
The README roadmap lists the memory-efficient-training baselines worth adding next (8-bit
Adam / GaLore / LoMo), so the memory claim is measured against real competitors, not only
FP32+Adam.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TernaryQATLinear", "make_linear"]


class TernaryQATLinear(nn.Module):
    """FP32 master weight, per-row absmean ternary in forward, straight-through gradient to
    the master; Adam updates the master. Same ternary inference as the counter layer, but a
    full FP32 optimizer -- so (counter vs this) isolates the 6-bit optimizer cost."""

    def __init__(self, fin: int, fout: int, init_gain: float = 1.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(fout, fin))
        nn.init.normal_(self.weight, mean=0.0, std=init_gain * math.sqrt(1.0 / fin))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        scale = w.abs().mean(dim=1, keepdim=True).clamp_min(1e-5)
        wq = (w / scale).round().clamp_(-1, 1) * scale
        wq = w + (wq - w).detach()  # straight-through estimator
        return F.linear(x, wq)


def make_linear(kind: str, fin: int, fout: int, init_gain: float = 1.0, **counter_kw):
    """Factory used by the GPT harness.

    Counter kinds include row-scale packed/triton layers, 2:4 ``group``, and solver-v3's
    ``group_scale`` / ``group_scale_packed`` act-ordered group-scale formats.
    """
    from .counter import CompactCounterLinear, RMSCounterLinear
    from .packed import PackedRMSCounterLinear

    if kind == "counter":
        return CompactCounterLinear(fin, fout, init_gain=init_gain, **counter_kw)
    if kind == "counter_rms":
        return RMSCounterLinear(fin, fout, init_gain=init_gain, **counter_kw)
    if kind == "counter_packed":
        return PackedRMSCounterLinear(fin, fout, init_gain=init_gain, **counter_kw)
    if kind == "counter_triton":
        from .triton_counter import TritonCounterLinear
        return TritonCounterLinear(fin, fout, init_gain=init_gain, **counter_kw)
    if kind == "slowfast":                                  # M3
        from .slowfast import SlowFastCounterLinear
        return SlowFastCounterLinear(fin, fout, init_gain=init_gain, **counter_kw)
    if kind == "group":                                     # M2
        from .group_counter import GroupCounterLinear
        return GroupCounterLinear(fin, fout, init_gain=init_gain, **counter_kw)
    if kind == "group_scale":
        from .group_scale_counter import GroupScaleCounterLinear
        return GroupScaleCounterLinear(fin, fout, **counter_kw)
    if kind in {"group_scale_packed", "group_packed"}:
        from .group_scale_packed import PackedGroupScaleCounterLinear
        return PackedGroupScaleCounterLinear(fin, fout, init_gain=init_gain, **counter_kw)
    if kind == "qat":
        return TernaryQATLinear(fin, fout, init_gain)
    if kind == "dense":
        lin = nn.Linear(fin, fout, bias=False)
        nn.init.normal_(lin.weight, mean=0.0, std=init_gain * math.sqrt(1.0 / fin))
        return lin
    raise ValueError(f"unknown linear kind: {kind!r}")
