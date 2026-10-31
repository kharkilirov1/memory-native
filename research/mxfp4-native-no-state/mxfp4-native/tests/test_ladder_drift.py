"""Ladder-drift regression: PackedNA.step must be unbiased in VALUE space.

Under zero-mean gradient noise with a frozen scale the expected value change per
step is 0, so (a) the mean value must not drift from any start rung and (b) mass
must not pile onto the outermost rungs. The legacy ``gap_mode="min"`` normalises
the tick by the smallest neighbour gap but carries one index, so from the second
rung outward a carry jumps +5 (e2m1) / +2.5 (wide) while the tick is calibrated
to 0.5 -> biased outward drift. (c) is the null twin: legacy must still show it.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mx6.packed_na import PackedNA  # noqa: E402

LADDERS = {"e2m1": ([0.0, 0.5, 1.0, 6.0], 5), "wide": ([0.0, 0.5, 3.0], 3)}
N = 20000          # weights per run (multiple of 32)
STEPS = 100
LR = 0.05          # per-step rms displacement 0.05 -> ~0.5 after 100 steps (scale=1)


def make(name, mode):
    mags, C = LADDERS[name]
    if mode == "min":
        try:
            return PackedNA(mags, C, gap_mode="min")
        except TypeError:        # pre-fix code: legacy was the only behaviour
            return PackedNA(mags, C)
    return PackedNA(mags, C)     # default (directional after the fix)


def run(fmt, start_val, seed=0):
    """Returns (value, value+pending) per weight; scale is frozen at 1."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1)
    sb = torch.full((1, N // 32), 127, dtype=torch.uint8)       # scale == 1, frozen
    i = int((fmt.signed - start_val).abs().argmin())
    c = torch.full((1, N), i * fmt.acc_levels + fmt.C - 1, dtype=torch.uint8)
    for _ in range(STEPS):
        c = fmt.step(c, sb, torch.randn(1, N, generator=gen), LR, gen=gen)
    v = fmt.value_float(c).flatten()
    mag = (c.long() // fmt.acc_levels).flatten()
    acc = ((c.long() % fmt.acc_levels) - (fmt.C - 1)).flatten().float()
    gap = torch.where(acc >= 0, fmt._gaps_up[mag] if hasattr(fmt, "_gaps_up") else fmt._gaps[mag],
                      fmt._gaps_down[mag] if hasattr(fmt, "_gaps_down") else fmt._gaps[mag])
    return v, v + acc / fmt.C * gap


def _starts(fmt):
    return [float(v) for v in fmt.signed.tolist() if 0.0 <= v < fmt.signed.max() - 1e-6]


def _outer(fmt, v):
    return float((v.abs() == float(fmt.signed.abs().max())).float().mean())


@pytest.mark.parametrize("name", ["e2m1", "wide"])
def test_mean_value_drift_zero(name):
    """value+pending-accumulator is the exact martingale of the integrator.

    (The bare rung value alone legitimately lags it by the pending progress, which
    is stationary and not a drift.)  sd of one weight is <= ~1, so the 5-sigma
    band of the mean at N=20000 is ~0.035; 0.05 absorbs wall clipping (<1 %).
    """
    fmt = make(name, "directional")
    for v0 in _starts(fmt):
        _, true = run(fmt, v0)
        tol = 5 * float(true.std()) / N ** 0.5 + 0.02
        assert abs(float(true.mean()) - v0) < tol, (name, v0, float(true.mean()), tol)


@pytest.mark.parametrize("name,cap", [("e2m1", 0.02), ("wide", 0.15)])
def test_no_outer_rung_pileup(name, cap):
    """Outer-rung mass stays at its martingale level, far below the legacy pile-up.

    An unbiased walk still puts ~Var/outer^2 on the outer rung (hence the wide cap);
    the biased legacy walk puts 4-57 %. Criterion: absolute cap and < 0.35x legacy.
    """
    fmt, old = make(name, "directional"), make(name, "min")
    for v0 in _starts(fmt):
        f_new = _outer(fmt, run(fmt, v0)[0])
        f_old = _outer(old, run(old, v0)[0])
        assert f_new < cap, (name, v0, f_new)
        if f_old > 0.03:
            assert f_new < 0.35 * f_old, (name, v0, f_new, f_old)


@pytest.mark.parametrize("name", ["e2m1", "wide"])
def test_legacy_min_still_drifts(name):
    """Null twin: the legacy mode must keep reproducing the old biased drift."""
    fmt = make(name, "min")
    v0 = 1.0 if name == "e2m1" else 0.5
    v, _ = run(fmt, v0)
    assert float(v.mean()) - v0 > 0.3
    assert _outer(fmt, v) > 0.05
