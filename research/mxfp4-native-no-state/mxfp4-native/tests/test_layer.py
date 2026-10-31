"""Layer-level invariants: strict readout, the training guard, and a smoke train."""
from __future__ import annotations

import pytest
import torch

from mx6 import MX6Linear, pack, update


def test_forward_ignores_the_residual_plane():
    torch.manual_seed(0)
    dense = torch.randn(16, 64) * 0.05
    layer = MX6Linear.from_dense(dense)
    x = torch.randn(4, 64)

    with torch.no_grad():
        before = layer(x).clone()
        layer.kappa = torch.randint(0, 4, layer.kappa.shape, dtype=torch.uint8)
        after = layer(x)
    assert torch.equal(before, after), "the residual must never reach the forward pass"


def test_one_forward_per_backward():
    layer = MX6Linear(32, 8)
    x = torch.randn(4, 32)
    layer(x)
    with pytest.raises(RuntimeError, match="one forward per backward"):
        layer(x)


def test_eval_mode_allows_repeated_forwards():
    layer = MX6Linear(32, 8).eval()
    x = torch.randn(4, 32)
    with torch.no_grad():
        layer(x)
        layer(x)  # no guard: evaluation is not a training step


def test_update_is_in_place_and_state_stays_packed():
    torch.manual_seed(3)
    layer = MX6Linear.from_dense(torch.randn(16, 64) * 0.05)
    before = layer.pack_state().clone()
    out = layer(torch.randn(4, 64))
    out.square().mean().backward()

    assert layer.codes.dtype == torch.uint8 and layer.kappa.dtype == torch.uint8
    assert layer.scale.dtype == torch.uint8
    assert layer.pack_state().numel() == before.numel()
    assert not torch.equal(before, layer.pack_state()), "the counter must have moved"
    # persistent_bytes() reports the tensors actually held (see layer.py:151), not an
    # idealized code size: deployed_bytes() is the shipped artifact and drops the
    # accumulator plane, so the two must differ on purpose.
    real = sum(t.numel() * t.element_size() for t in layer.buffers() if t.numel())
    assert layer.persistent_bytes() == real
    assert layer.persistent_bytes() > pack.deployed_bytes(16, 64)
    feedback = MX6Linear.from_dense(torch.zeros(16, 64), feedback=True)
    assert feedback.persistent_bytes() > layer.persistent_bytes()


def test_zero_gradient_does_not_move_the_codes():
    torch.manual_seed(4)
    layer = MX6Linear.from_dense(torch.randn(8, 32) * 0.05)
    codes = layer.codes.clone()
    layer.apply_update_(torch.zeros(8, 32))
    assert torch.equal(codes, layer.codes), "a zero gradient must not flip a weight"


def test_stochastic_rounding_is_unbiased_on_the_ladder():
    torch.manual_seed(5)
    gen = torch.Generator().manual_seed(0)
    x = torch.full((20000,), 0.75)  # midway between 0.5 and 1.0
    rounded = update.stochastic_round_ladder(x, gen)
    assert abs(float(rounded.mean()) - 0.75) < 0.02


def test_residual_quantizer_has_a_dead_zone():
    zeros = torch.zeros(4, 32)
    deltas = torch.full((4, 32), 0.5)
    assert torch.all(update.quantize_residual(zeros, deltas) == 0), "zero error must map to zero kappa"
    assert torch.all(update.quantize_residual(0.4 * deltas, deltas) == 3)
    assert torch.all(update.quantize_residual(-0.4 * deltas, deltas) == 2)


def test_a_regression_actually_learns():
    torch.manual_seed(6)
    x = torch.randn(128, 32)
    teacher = torch.randn(32, 32) * 0.1
    target = x @ teacher.t()

    layer = MX6Linear.from_dense(torch.zeros(32, 32))
    layer.lr = 0.02
    losses = []
    for _ in range(150):
        loss = (layer(x) - target).square().mean()
        loss.backward()
        losses.append(float(loss))

    assert losses[-1] < 0.5 * losses[0], f"no learning: {losses[0]:.4f} -> {losses[-1]:.4f}"
