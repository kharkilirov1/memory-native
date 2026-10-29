"""Layer invariants: strict readout, training guard, and compact-state smoke train."""
from __future__ import annotations
import pytest
import torch
from mx6 import MX6Linear
from mx6 import pack, update


def test_forward_ignores_residual_plane():
    torch.manual_seed(0)
    layer = MX6Linear.from_dense(torch.randn(16, 64) * 0.05)
    x = torch.randn(4, 64)
    with torch.no_grad():
        before = layer(x).clone()
        layer.kappa = torch.randint(0, 4, layer.kappa.shape, dtype=torch.uint8)
        after = layer(x)
    assert torch.equal(before, after)


def test_one_forward_per_backward():
    layer = MX6Linear(32, 8)
    x = torch.randn(4, 32)
    layer(x)
    with pytest.raises(RuntimeError, match="one forward per backward"):
        layer(x)


def test_eval_mode_allows_repeated_forwards():
    layer = MX6Linear(32, 8).eval()
    with torch.no_grad():
        layer(torch.randn(4, 32))
        layer(torch.randn(4, 32))


def test_zero_gradient_does_not_move_codes():
    layer = MX6Linear.from_dense(torch.randn(8, 32) * 0.05)
    codes = layer.codes.clone()
    layer.apply_update_(torch.zeros(8, 32))
    assert torch.equal(codes, layer.codes)


def test_stochastic_rounding_is_unbiased():
    gen = torch.Generator().manual_seed(0)
    x = torch.full((20000,), 0.75)
    rounded = update.stochastic_round_ladder(x, gen)
    assert abs(float(rounded.mean()) - 0.75) < 0.02


def test_balanced_rounding_is_unbiased():
    x = torch.tensor([[0.2, 0.3, 0.7, 0.8] * 8], dtype=torch.float32)
    samples = [update.balanced_stochastic_round_ladder(
        x, torch.Generator().manual_seed(seed)) for seed in range(128)]
    assert torch.max(torch.abs(torch.stack(samples).mean(0) - x)) < 0.06


def test_compact_integrator_state_is_packed_and_trains():
    torch.manual_seed(31)
    dense = torch.randn(8, 64) * 0.05
    layer = MX6Linear.from_dense(dense, integrator=3, compact_state=True, lr=0.02)
    n = dense.numel()
    assert layer._codes_packed.numel() == n // 2
    assert layer._acc_packed.numel() == n // 4
    before = layer.pack_state().clone()
    (layer(torch.randn(4, 64)).square().mean()).backward()
    assert not torch.equal(before, layer.pack_state())


def test_compact_state_checkpoint_roundtrip():
    source = MX6Linear(32, 8, integrator=3, compact_state=True)
    source._store_acc(torch.randint(-1, 2, source.acc.shape, dtype=torch.int8))
    blob = source.pack_state()
    restored = MX6Linear(32, 8, integrator=3, compact_state=True)
    restored.load_state(blob)
    assert torch.equal(source.codes, restored.codes)
    assert torch.equal(source.acc, restored.acc)
    assert torch.equal(source.scale, restored.scale)


def test_compact_state_requires_integrator_three():
    with pytest.raises(ValueError, match="integrator=3"):
        MX6Linear(32, 8, integrator=33, compact_state=True)


def test_compact_state_rejects_feedback_plane():
    with pytest.raises(ValueError, match="does not support"):
        MX6Linear(32, 8, integrator=3, compact_state=True, feedback=True)
