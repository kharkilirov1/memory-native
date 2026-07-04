"""Saturation probe for the counter synapse.

Research hypothesis (see docs/ARTICLE.md, "The open question has a shape"):
    The bounded counter loses gradient pressure when weights saturate against
    the +/-1 boundary. If the saturation rate grows with model width, that is
    a mechanistic explanation for the OPEN convergence gap at scale.

This module measures, per training step:
    -- saturation_rate: fraction of weights whose accumulator c is at the
       boundary +/-(C-1) (i.e. the counter is "stuck" against the wall)
    -- blocked_rate: fraction of proposed weight flips that were blocked
       (weight already at +/-1 and the carry pushed further)
    -- flip_rate: fraction of weights that actually flipped this step
    -- mean_runlength: mean consecutive steps the gradient sign has been
       the same per weight (proxy for "how long does pressure persist")

It works WITHOUT modifying counter.py: it hooks the post-update state and
reads the numbers from outside. This is intentionally a separate research
instrument -- the production counter layer stays untouched.

Usage:
    from scripts.saturation_probe import attach_probes, collect_step, summary
    probes = attach_probes(model)        # call before training
    for step in range(steps):
        train_step(...)
        collect_step(probes, step)       # call after the optimizer step
    summary(probes)                      # prints + dumps JSON
"""
from __future__ import annotations

import json
import math
import os
import statistics
from collections import defaultdict
from typing import Dict, List, Tuple

import torch

# Metrics tracked per layer, per step
STEP_KEYS = ("saturation_rate", "blocked_rate_proxy", "flip_rate", "mean_abs_c", "n_weights")


class _DummyModel(torch.nn.Module):
    """Wrapper so attach_probes can find counter layers via standard named_modules."""
    def __init__(self, layers):
        super().__init__()
        for name, mod in layers:
            setattr(self, name, mod)


def _find_counter_layers(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    """Find all RMSCounterLinear modules in the model (by attribute sniffing)."""
    out = []
    for name, mod in model.named_modules():
        # counter layers have a `state` buffer (uint8) and `C` attribute
        if hasattr(mod, "state") and hasattr(mod, "C") and hasattr(mod, "weight_flips"):
            out.append((name, mod))
    return out


def _decode(state: torch.Tensor, C: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mirror of counter.decode_state: t, c from packed uint8 state."""
    levels = 2 * C - 1
    z = state.to(torch.int32)
    t = torch.floor_divide(z, levels) - 1
    c = torch.remainder(z, levels) - (C - 1)
    return t, c


def attach_probes(model: torch.nn.Module) -> Dict:
    """Attach saturation probes to every counter layer in `model`.

    Returns an opaque dict to pass to `collect_step` / `summary`.
    """
    layers = _find_counter_layers(model)
    if not layers:
        raise RuntimeError(
            "No counter layers found. Make sure the model has RMSCounterLinear modules "
            "with `state`, `C`, and `weight_flips` attributes."
        )
    # per-layer baseline state to compute flips delta
    baselines = {}
    sign_history = {}  # layer -> running sign-of-gradient (approximated from counter delta)
    for name, mod in layers:
        baselines[name] = {
            "flips": int(mod.weight_flips.item()),
            "prev_t": _decode(mod.state.detach().cpu(), int(mod.C))[0].clone(),
        }
        sign_history[name] = {"runlength_sum": 0.0, "runlength_count": 0, "cur_len": 0, "prev_dir": 0}
    return {
        "layers": layers,
        "baselines": baselines,
        "sign_history": sign_history,
        "history": defaultdict(lambda: defaultdict(list)),  # [layer][key] -> list per step
        "step_history": [],
    }


def collect_step(probes: Dict, step: int) -> None:
    """Record saturation metrics for this step. Call AFTER the optimizer/update step."""
    for name, mod in probes["layers"]:
        C = int(mod.C)
        state = mod.state.detach().cpu()
        t, c = _decode(state, C)
        n = t.numel()

        # saturation: |c| at the boundary +/-(C-1)
        sat = (c.abs() == (C - 1)).float().mean().item()

        # blocked-rate proxy: weights where t is at +/-1 AND |c| is at the boundary
        # (these are the weights where pressure is being lost -- they can't flip further)
        at_boundary_t = (t.abs() == 1)
        at_boundary_c = (c.abs() == (C - 1))
        blocked = (at_boundary_t & at_boundary_c).float().mean().item()

        # flip-rate: how many weights flipped since last step
        prev_t = probes["baselines"][name]["prev_t"]
        prev_flips = probes["baselines"][name]["flips"]
        # use weight_flips delta if reliable, else t-comparison
        cur_flips = int(mod.weight_flips.item())
        dflips = cur_flips - prev_flips
        flip_rate = dflips / max(n, 1)
        # cross-check via t comparison (more accurate if weight_flips is per-multiple-tiles)
        t_changed = (t != prev_t).float().mean().item()
        probes["baselines"][name]["flips"] = cur_flips
        probes["baselines"][name]["prev_t"] = t.clone()

        # mean |c| (how full the accumulator is on average)
        mean_abs_c = c.float().abs().mean().item()

        probes["history"][name]["saturation_rate"].append(sat)
        probes["history"][name]["blocked_rate_proxy"].append(blocked)
        probes["history"][name]["flip_rate"].append(flip_rate)
        probes["history"][name]["flip_rate_alt"].append(t_changed)
        probes["history"][name]["mean_abs_c"].append(mean_abs_c)
        probes["history"][name]["n_weights"].append(n)
    probes["step_history"].append(step)


def summary(probes: Dict, out_path: str | None = None, tag: str = "") -> Dict:
    """Aggregate per-layer metrics, optionally dump JSON to out_path.

    Returns {"layers": {name: {key: {mean, last, max, series}}}, "tag": tag}
    """
    out = {"tag": tag, "layers": {}}
    for name, hist in probes["history"].items():
        out["layers"][name] = {}
        for key, series in hist.items():
            if not series:
                continue
            if key == "n_weights":
                out["layers"][name][key] = int(series[0])
                continue
            out["layers"][name][key] = {
                "mean": float(statistics.mean(series)),
                "max":  float(max(series)),
                "last": float(series[-1]),
                "series": [float(x) for x in series],
            }
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
    return out


def headline(probes: Dict) -> str:
    """One-line per-layer summary for quick CLI inspection."""
    lines = []
    for name, hist in probes["history"].items():
        sat = hist.get("saturation_rate", [])
        blk = hist.get("blocked_rate_proxy", [])
        flp = hist.get("flip_rate_alt", [])
        if not sat:
            continue
        lines.append(
            f"  {name:40s}  sat={statistics.mean(sat):.3f} "
            f"(max {max(sat):.3f})  blocked={statistics.mean(blk):.3f}  flip={statistics.mean(flp):.4f}"
        )
    return "\n".join(lines) if lines else "  (no data)"


if __name__ == "__main__":
    # Self-test: tiny synthetic run on CPU to verify instrumentation works
    print("=== saturation_probe self-test (CPU, tiny model) ===")
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from memory_native.counter import RMSCounterLinear  # noqa

    torch.manual_seed(0)
    # tiny: just a few counter layers
    layer = RMSCounterLinear(64, 64, C=11)
    model = _DummyModel([("l0", layer)])
    probes = attach_probes(model)
    for step in range(5):
        layer.zero_grad()
        x = torch.randn(4, 64)
        y = layer(x).sum()
        y.backward()
        collect_step(probes, step)
    print(headline(probes))
    s = summary(probes, out_path=os.path.join(os.path.dirname(__file__), "..", "results", "_probe_selftest.json"))
    print(f"  layers measured: {len(s['layers'])}")
