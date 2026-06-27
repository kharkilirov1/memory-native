"""Adaptive update decimation (acceleration memo M8).

Once a counter layer's flip-rate is tiny it is near-stable, so the update fires only every
_dec_period steps (lr scaled by the period to compensate). This must (a) leave the default path
untouched and (b) still recover a teacher while actually engaging decimation once stable.
"""
import math

import torch

from memory_native import RMSCounterLinear


def _teacher(decimate, steps=700):
    torch.manual_seed(0)
    n, N, C = 24, 256, 11
    ts = 0.25
    base = math.sqrt(3.0 / (2.0 * n))
    tw = torch.randint(-1, 2, (n, n)).float()
    x = torch.randn(N, n)
    y = x @ (ts * tw).t()
    lay = RMSCounterLinear(n, n, C=C, lr=0.02, lr_scale=2e-4, init_gain=ts / base,
                           decimate_updates=decimate).train()
    for _ in range(steps):
        (lay(x) - y).pow(2).mean().backward()
    with torch.no_grad():
        mse = (lay(x) - y).pow(2).mean().item()
    return lay, mse


def test_decimation_off_keeps_period_one():
    lay, mse = _teacher(decimate=False)
    assert lay._dec_period == 1 and lay._lr_mult == 1.0   # never decimates
    assert mse < 0.02


def test_decimation_recovers_and_engages():
    lay, mse = _teacher(decimate=True)
    assert mse < 0.02, f"decimated training failed to recover teacher: {mse:.4f}"
    assert lay._dec_period > 1, "decimation never engaged though the layer stabilized"
