import torch
import pytest

from probes.bsr import balanced_stochastic_round_ladder


def test_requires_multiple_of_block_width():
    with pytest.raises(ValueError):
        balanced_stochastic_round_ladder(torch.zeros(31))


def test_output_is_on_e2m1_ladder():
    x = torch.linspace(-6, 6, 256).reshape(8, 32)
    y = balanced_stochastic_round_ladder(x, torch.Generator().manual_seed(1))
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    assert torch.isfinite(y).all()
    assert torch.all(torch.isin(y.abs(), levels))


def test_marginals_are_unbiased_in_monte_carlo():
    # A value halfway between adjacent levels should round up in about half of trials.
    x = torch.full((1, 32), 1.25)
    trials = []
    for seed in range(200):
        y = balanced_stochastic_round_ladder(x, torch.Generator().manual_seed(seed))
        trials.append(y[0, 0].item())
    mean = sum(trials) / len(trials)
    assert abs(mean - 1.25) < 0.08


def test_zero_stays_zero():
    x = torch.zeros(4, 32)
    y = balanced_stochastic_round_ladder(x, torch.Generator().manual_seed(3))
    assert torch.equal(x, y)
