"""refresh_scale_ must be sign-symmetric: refresh(-w) == -refresh(w).

Regression: ``torch.bucketize`` (right=False) sends an exact midpoint tie to the
LOWER rung, i.e. towards -inf. When the new scale is one octave larger than the old
one, every inner-rung weight (+-0.5 s) lands exactly on the 0/0.5 midpoint (+-0.25),
so +0.5 collapsed to 0 while -0.5 stayed at -0.5: a one-sided drift that destroyed the
symmetry of a symmetric task.
"""
from __future__ import annotations

import pytest
import torch

from mx6.layer import MXNLinear
from mx6.packed_na import PackedNA

LADDERS = {
    "wide_c3": ([0.0, 0.5, 3.0], 3, 1.5),
    "e2m1_c5": ([0.0, 0.5, 1.0, 6.0], 5, 3.0),
}


def _layer_with_signed_rungs(fmt: PackedNA, sign: float, headroom: float) -> MXNLinear:
    """One 32-block per row: a single outer-rung weight (sets amax) and the rest on
    the smallest non-zero rung, so refresh picks a scale one octave above the old."""
    layer = MXNLinear(32, 4, fmt, scale_headroom=headroom)
    n = fmt.n_vals
    mid = n // 2
    outer = (n - 1) if sign > 0 else 0
    inner = mid + 1 if sign > 0 else mid - 1
    mag = torch.full((4, 32), inner, dtype=torch.long)
    mag[:, 0] = outer
    acc = torch.zeros(4, 32, dtype=torch.long)
    with torch.no_grad():
        layer._state.copy_(fmt.pack_state(fmt.encode(mag, acc).to(torch.uint8)))
        layer._scale.fill_(127)           # s = 1.0
    layer._codes_cache = None
    layer._weight_cache = None
    return layer


@pytest.mark.parametrize("name", list(LADDERS))
def test_refresh_is_sign_symmetric(name):
    mags, C, hr = LADDERS[name]
    fmt = PackedNA(mags_unsigned=mags, C=C)
    pos = _layer_with_signed_rungs(fmt, +1.0, headroom=hr)
    neg = _layer_with_signed_rungs(fmt, -1.0, headroom=hr)
    assert torch.equal(pos.weight(), -neg.weight())      # sanity: mirrored inputs
    pos.refresh_scale_()
    neg.refresh_scale_()
    assert torch.equal(pos.weight(), -neg.weight()), (
        "refresh_scale_ is not sign-symmetric: ties round toward -inf\n"
        f"w(+)={pos.weight()[0, :4].tolist()}  w(-)={neg.weight()[0, :4].tolist()}")
