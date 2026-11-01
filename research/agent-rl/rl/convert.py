"""Replace nn.Linear layers inside a HF decoder with LatentGridLinear (mx6)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

# mx6 search path
_MX6 = Path(__file__).resolve().parents[3] / "research" / "mxfp4-native-no-state" / "mxfp4-native"
if str(_MX6) not in sys.path:
    sys.path.insert(0, str(_MX6))

from mx6.latent_grid import LatentGridLinear, lg_layers  # noqa: E402
from mx6 import lg_accum  # noqa: E402


class LGLinearBias(nn.Module):
    """Wraps a bias-free LatentGridLinear + a separate bias parameter trained by AdamW."""

    def __init__(self, lg: LatentGridLinear, bias: nn.Parameter):
        super().__init__()
        self.lg = lg
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lg(x) + self.bias


def _is_decoder_linear(name: str, mod: nn.Module, skip: tuple[str, ...]) -> bool:
    """True if *mod* is an nn.Linear inside a decoder layer and not in skip."""
    if not isinstance(mod, nn.Linear):
        return False
    if any(s in name for s in skip):
        return False
    parts = name.split(".")
    for i, p in enumerate(parts[:-1]):
        if p == "layers" and parts[i + 1].isdigit():
            return True
    return False


def convert_to_lg(
    model: nn.Module,
    k: int = 6,
    headroom: float = 2.0,
    lr: float = 0.001,
    momentum: float = 0.9,
    mom_dtype: str = "fp8",
    n_micro: int = 4,
    skip: tuple[str, ...] = ("lm_head", "embed"),
    backend: str = "torch",
) -> dict[str, Any]:
    """Replace decoder nn.Linear with LatentGridLinear, enable momentum.

    Returns a report dict with keys: n_layers, params_in_lg, resident_bytes, report_lines.
    """
    report: list[str] = []
    n_layers = 0
    params_in_lg = 0

    replacements: list[tuple[str, nn.Module, str, nn.Module]] = []
    for name, mod in model.named_modules():
        if not _is_decoder_linear(name, mod, skip):
            continue
        parts = name.split(".")
        attr = parts[-1]
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        replacements.append((name, parent, attr, mod))

    for name, parent, attr, mod in replacements:
        w = mod.weight.detach()
        lg = LatentGridLinear.from_dense(
            w, k=k, headroom=headroom, lr=lr, backend=backend,
            update_mode="exact", rms_beta=0.9, rms_eps=1e-3,
        )
        if mod.bias is not None:
            wrapper = LGLinearBias(lg, nn.Parameter(mod.bias.detach()))
            setattr(parent, attr, wrapper)
        else:
            setattr(parent, attr, lg)

        n_layers += 1
        params_in_lg += w.numel()
        report.append(f"  {name}: Linear({mod.in_features},{mod.out_features}) -> LG k={k} (bias={mod.bias is not None})")

    if momentum > 0.0 and n_layers > 0:
        st = lg_accum.set_momentum(model, mu=momentum, dtype=mom_dtype)
        if st is not None:
            report.append(f"  momentum: mu={momentum}, dtype={mom_dtype}, buffer ~{lg_accum.accum_resident_bytes(model)} B")

    res = 0
    for m in lg_layers(model):
        res += m.resident_bytes() + m.persistent_bytes()

    return {
        "n_layers": n_layers,
        "params_in_lg": params_in_lg,
        "resident_bytes": res,
        "report_lines": report,
    }
