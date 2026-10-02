"""Analytic witnesses for the eager-rebase latent dynamics in the preprint.

Enumerating each scalar's two stochastic-rounding outcomes computes its exact
conditional mean; it avoids a Monte Carlo tolerance that could hide scale drift.
"""
import math

import pytest
import torch

from memory_native.counter import _carry_resolve, _rms_eager_pre_sr


def _enumerate_sr(pre, t, scale, C):
    lower = pre.floor()
    fraction = pre - lower
    outcomes = []
    mean = torch.zeros_like(pre)
    for rounded, probability in ((lower, 1 - fraction), (lower + 1, fraction)):
        new_t, new_c = _carry_resolve(rounded, t, C)
        latent = (new_t + new_c / C) * scale
        mean += probability * latent
        outcomes.append((rounded, probability, latent))
    return mean, outcomes


@pytest.mark.parametrize(
    "ternary,counter,gradient,lr,lr_scale,scale,clip,use_rms",
    [
        ([1], [0], [0.1], 0.1, 0.2, 1.0, 0.0, True),
        ([-1], [3], [-0.2], 0.1, 0.2, 1.0, 0.0, True),
        ([1, -1, 0], [0, 3, 10], [0.1, -0.2, -0.3],
         0.05, 0.2, 1.0, 0.0, True),
        ([0], [10], [-0.1], 0.4, 0.0, 1.0, 0.0, True),
        ([1], [-10], [0.1], 0.1, 0.0, 1.0, 0.0, True),
        ([1], [0], [0.1], 0.1, 0.2, 1.0, 0.04, True),
        ([1], [0], [0.1], 1e-8, 0.2, 1e-5, 0.0, True),
        ([-1], [2], [-0.2], 0.1, 0.2, 1.0, 0.0, False),
    ],
    ids=[
        "learned-scale-positive-visible", "learned-scale-negative-visible",
        "shared-row-scale-and-carry", "fixed-scale-positive-carry",
        "fixed-scale-negative-carry", "clipped-normalized-gradient",
        "clipped-scale", "without-rms",
    ],
)
def test_eager_rebase_exact_expected_latent_update(
    ternary, counter, gradient, lr, lr_scale, scale, clip, use_rms,
):
    C, beta, eps = 11, 0.9, 1e-3
    t = torch.tensor([ternary], dtype=torch.float64)
    c = torch.tensor([counter], dtype=torch.float64)
    g = torch.tensor([gradient], dtype=torch.float64)
    old_scale = torch.full((1, 1), scale, dtype=torch.float64)
    old_v = torch.ones_like(old_scale)
    v = old_v.clone()
    pre, new_scale = _rms_eager_pre_sr(
        g, t, c, old_scale, v, g.square().mean(dim=1, keepdim=True),
        use_rms, eps, beta, clip, lr, lr_scale, C, math.sqrt(g.shape[1]),
    )

    # Derive the target from the stated optimizer, independently of its ticks.
    expected_v = beta * old_v + (1 - beta) * g.square().mean(dim=1, keepdim=True)
    signal = g / expected_v.sqrt().clamp_min(eps) if use_rms else g
    if clip:
        signal = signal * (clip / signal.norm(dim=1, keepdim=True)).clamp_max(1)
    latent_before = (t + c / C) * old_scale
    scale_displacement = t * (new_scale - old_scale)
    expected_latent = latent_before - lr * signal + scale_displacement

    exact_mean, outcomes = _enumerate_sr(pre, t, new_scale, C)
    for rounded, probability, latent in outcomes:
        proposed_t = t + torch.trunc(rounded / C)
        assert torch.all((probability == 0) | (proposed_t.abs() <= 1))
        assert torch.all((latent - expected_latent).abs() <= new_scale / C + 1e-12)
    torch.testing.assert_close(exact_mean, expected_latent, rtol=0, atol=1e-12)

    if lr_scale == 0:
        torch.testing.assert_close(new_scale, old_scale, rtol=0, atol=0)
        torch.testing.assert_close(exact_mean, latent_before - lr * signal, rtol=0, atol=1e-12)
    elif scale == 1e-5:
        # Clipping the scale does not invalidate the identity using its actual value.
        torch.testing.assert_close(new_scale, old_scale, rtol=0, atol=0)
    else:
        # The omitted visible-scale term is nonzero, with either sign of t.
        assert scale_displacement.abs().max() > 0
        assert not torch.allclose(exact_mean, latent_before - lr * signal, rtol=0, atol=1e-6)

    if use_rms:
        torch.testing.assert_close(v, expected_v, rtol=0, atol=1e-12)
    else:
        torch.testing.assert_close(v, old_v, rtol=0, atol=0)


def test_preprint_changing_scale_counterexample_without_saturation():
    """The old formula misses exactly -0.02 even when neither SR outcome carries."""
    C = 11
    t, s, v = (torch.ones((1, 1), dtype=torch.float64) for _ in range(3))
    c = torch.zeros_like(t)
    g = torch.full_like(t, 0.1)
    pre, new_scale = _rms_eager_pre_sr(
        g, t, c, s, v, g.square(), True, 1e-3, 0.9, 0.0, 0.1, 0.2, C, 1.0,
    )
    exact_mean, outcomes = _enumerate_sr(pre, t, new_scale, C)
    for rounded, _, _ in outcomes:
        assert torch.equal(torch.trunc(rounded / C), torch.zeros_like(t))
    old_claim = (t + c / C) * s - 0.1 * g / math.sqrt(0.901)
    torch.testing.assert_close(exact_mean - old_claim, torch.full_like(t, -0.02),
                               rtol=0, atol=1e-12)
    assert exact_mean.item() == pytest.approx(0.9694649253, abs=1e-9)


def test_saturation_is_biased_even_with_unbiased_rounding_and_fixed_scale():
    """A blocked carry pins both outcomes to the boundary, below the raw mean."""
    C = 11
    t, scale, v = (torch.ones((1, 1), dtype=torch.float64) for _ in range(3))
    c, g = torch.full_like(t, 10), torch.full_like(t, -0.1)
    pre, new_scale = _rms_eager_pre_sr(
        g, t, c, scale, v, g.square(), True, 1e-3, 0.9, 0.0, 0.5, 0.0, C, 1.0,
    )
    exact_mean, outcomes = _enumerate_sr(pre, t, new_scale, C)
    assert all(0 < probability.item() < 1 for _, probability, _ in outcomes)
    assert outcomes[1][0].item() == C  # positive-probability blocked carry
    assert exact_mean.item() == pytest.approx((2 * C - 1) / C, abs=1e-12)
    unsaturated_mean = (t + c / C) * scale - 0.5 * g / v.sqrt()
    assert exact_mean.item() < unsaturated_mean.item()
