"""Parity gate for the one-pass update modes (acceleration memo M4).

`rms_mode="lagged"` (denominator from the previous step's v) and `scale_rebase="lazy"` (counter
rebased at the next read via a per-row s_base) remove the two-pass dependency in the counter
update. They change the dynamics slightly, so this gate requires every combination to still
recover a ternary teacher about as well as the exact/eager default.
"""
import math

import pytest
import torch

from memory_native import RMSCounterLinear


def _recover(rms_mode, scale_rebase, steps=500):
    torch.manual_seed(0)
    n, N, C = 24, 256, 11
    ts = 0.25
    base = math.sqrt(3.0 / (2.0 * n))
    tw = torch.randint(-1, 2, (n, n)).float()
    x = torch.randn(N, n)
    y = x @ (ts * tw).t()
    lay = RMSCounterLinear(n, n, C=C, lr=0.02, lr_scale=2e-4, init_gain=ts / base,
                           rms_mode=rms_mode, scale_rebase=scale_rebase).train()
    for _ in range(steps):
        loss = (lay(x) - y).pow(2).mean()
        loss.backward()                         # layer self-updates here
    with torch.no_grad():
        mse = (lay(x) - y).pow(2).mean().item()
    return mse


@pytest.mark.parametrize("rms_mode,scale_rebase", [
    ("exact", "eager"),     # the default / fused-kernel math
    ("lagged", "eager"),    # one-pass RMS
    ("exact", "lazy"),      # lazy scale rebase
    ("lagged", "lazy"),     # full one-pass update
])
def test_one_pass_modes_recover_teacher(rms_mode, scale_rebase):
    mse = _recover(rms_mode, scale_rebase)
    assert mse < 0.02, f"{rms_mode}/{scale_rebase} failed to recover teacher: mse={mse:.4f}"


def test_lagged_lazy_close_to_exact():
    """The full one-pass update must land in the same ballpark as the exact/eager baseline."""
    base = _recover("exact", "eager")
    one_pass = _recover("lagged", "lazy")
    assert one_pass < max(3 * base, 0.01), f"one-pass {one_pass:.5f} vs exact {base:.5f}"
