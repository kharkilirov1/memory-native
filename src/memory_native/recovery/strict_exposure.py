"""Opt-in strict-readout training exposure; zero extra backward calls.

This is an ablation switch, NOT a proven remedy for model-scale quality loss.
Counter layers update state inside backward; running two independent backward
passes before scheduling a single counter update is not currently safe.
"""
from __future__ import annotations

import math


def strict_exposure_alpha(base_alpha: float, step: int, every: int = 0) -> float:
    """Use alpha=0 on each every-th 1-indexed step; zero disables the ablation.

    Must be invoked once BEFORE the only forward/backward of the training step.
    The current autograd path must update counters exactly once per step.
    """
    if not isinstance(every, int) or isinstance(every, bool) or every < 0:
        raise ValueError("strict exposure period must be a nonnegative integer")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("step must be a nonnegative integer")
    if not math.isfinite(base_alpha) or not 0 <= base_alpha <= 1:
        raise ValueError("base_alpha must be finite and in [0, 1]")
    return 0.0 if every and (step + 1) % every == 0 else float(base_alpha)
