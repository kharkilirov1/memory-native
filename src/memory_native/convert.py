"""Model-level warm-start: swap the nn.Linear layers of a pretrained module for counter layers.

Phase 2 core (donor-independent). `swap_linears_to_counter` walks an arbitrary nn.Module and
replaces every nn.Linear with a counter layer initialized from that linear's pretrained weight
(via RMSCounterLinear.from_linear). Embeddings, norms and the LM head are left untouched -- they
stay fp and are trained by AdamW, exactly as the GPT/GLM harness expects (the counter method
targets the transformer-body linears, which are the optimizer/gradient/activation pools it zeros).

Bias handling: counter layers are bias-free. A donor linear that carries a bias is wrapped so the
pretrained bias is preserved as a small fp parameter (kept unless keep_bias=False). Modern donors
(Llama/Qwen/Gemma) are mostly bias-free, so this is usually a no-op.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .baselines import make_linear
from .counter import C_DEFAULT, CompactCounterLinear

__all__ = ["CounterLinearWithBias", "SwapReport", "swap_linears_to_counter"]


class CounterLinearWithBias(nn.Module):
    """A counter linear plus a preserved fp bias (only used when a donor linear had a bias)."""

    def __init__(self, counter: CompactCounterLinear, bias: torch.Tensor) -> None:
        super().__init__()
        self.counter = counter
        self.bias = nn.Parameter(bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.counter(x) + self.bias


@dataclass
class SwapReport:
    swapped: list[str] = field(default_factory=list)      # module paths that became counter layers
    skipped: list[str] = field(default_factory=list)      # nn.Linear paths left as-is (by predicate)
    coeffs: int = 0                                        # total counter weights created

    def __str__(self) -> str:
        return (f"SwapReport(swapped={len(self.swapped)} linears, "
                f"{self.coeffs:,} counter coeffs; skipped={len(self.skipped)})")


def _should_skip(path: str, skip) -> bool:
    if skip is None:
        return False
    if callable(skip):
        return bool(skip(path))
    return any(s in path for s in skip)


def swap_linears_to_counter(
    model: nn.Module,
    *,
    kind: str = "counter_rms",
    skip=None,
    C: int = C_DEFAULT,
    threshold_ratio: float = 0.7,
    keep_bias: bool = True,
    **counter_kw,
) -> SwapReport:
    """In place, replace every nn.Linear in `model` with a warm-started counter layer.

    kind: any counter kind from make_linear ("counter_rms" default, "counter_packed" for 6-bit
      storage, ...). skip: None, an iterable of substrings, or a predicate path->bool to leave
      selected linears untouched (e.g. an untied lm_head, or router projections). Returns a
      SwapReport. Packed kinds require each swapped linear's in_features % 4 == 0.
    """
    report = SwapReport()
    # snapshot first: we mutate the tree while iterating, so collect targets up front.
    targets: list[tuple[nn.Module, str, nn.Linear, str]] = []
    for parent_path, parent in model.named_modules():
        for child_name, child in parent.named_children():
            if isinstance(child, nn.Linear):
                path = f"{parent_path}.{child_name}" if parent_path else child_name
                targets.append((parent, child_name, child, path))

    for parent, child_name, lin, path in targets:
        if _should_skip(path, skip):
            report.skipped.append(path)
            continue
        counter = make_linear(
            kind, lin.in_features, lin.out_features, 1.0, C=C, **counter_kw
        )
        counter.load_dense_weight(lin.weight, threshold_ratio=threshold_ratio)
        if lin.bias is not None and keep_bias:
            replacement: nn.Module = CounterLinearWithBias(counter, lin.bias)
        else:
            replacement = counter
        setattr(parent, child_name, replacement)
        report.swapped.append(path)
        report.coeffs += lin.in_features * lin.out_features

    return report
