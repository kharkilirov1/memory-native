"""Equation-level checks for opt-in RMS correction and cumulative scale rebasing."""
import copy

import pytest
import torch

from memory_native import PackedRMSCounterLinear, RMSCounterLinear, configure_rounding_streams


def zero_layer(cls=RMSCounterLinear, **kw):
    layer = cls(8, 3, C=11, lr=0.01, lr_scale=0, **kw)
    z = torch.zeros(3, 8, dtype=torch.int16)
    layer.load_counter_state(torch.ones(3, 1), z, z)
    return layer


@pytest.mark.parametrize("beta", [0.0, 0.9, 0.99, 0.999999])
@pytest.mark.parametrize("mode", ["exact", "proxy"])
def test_bias_correction_matches_explicit_history(beta, mode):
    layer = zero_layer(rms_beta=beta, rms_mode=mode, rms_bias_correction=True)
    history = torch.zeros_like(layer.v, dtype=torch.float64)
    captured = []
    layer._round_counter = lambda value, lo: captured.append(value.clone()) or torch.zeros_like(value)
    for step in range(1, 5):
        grad = torch.arange(1, 25).reshape(3, 8).float() * step / 25
        sq = grad.square().mean(1, keepdim=True)
        if mode == "proxy":
            sq = sq * 2
        history = beta * history + (1 - beta) * sq.double()
        t, c = layer._decode_rows(0, 3)
        layer._update_tile(0, 3, grad, t, c, layer.scale, proxy_gsq=sq)
        v_hat = history / (1 - beta ** step)
        expected = -layer.lr * layer.C * grad / v_hat.sqrt().float().clamp_min(layer.rms_eps)
        torch.testing.assert_close(captured[-1], expected, atol=2e-7, rtol=2e-6)
        assert torch.equal(layer.rms_updates, torch.full((3, 1), step))


def test_correction_keeps_scale_learning_on_raw_gradient():
    layer = zero_layer(rms_bias_correction=True)
    t = torch.ones(3, 8, dtype=torch.int16)
    layer.load_counter_state(torch.ones(3, 1), t, torch.zeros_like(t))
    layer.lr_scale = 0.002
    grad = torch.arange(24).reshape(3, 8).float() / 12
    expected = layer.scale - layer.lr_scale * grad.sum(1, keepdim=True) / 8 ** 0.5
    layer.apply_update_from_grad_w(grad)
    torch.testing.assert_close(layer.scale, expected)


def test_counts_follow_actual_updates_and_reset_on_import():
    layer = zero_layer(PackedRMSCounterLinear, rms_bias_correction=True, tile_rows=2,
                       decimate_updates=True)
    layer._dec_period = 2
    layer._decimation_observe = lambda: None
    x = torch.ones(4, 8)
    layer(x).sum().backward()
    assert not layer.rms_updates.any()
    layer(x).sum().backward()
    assert torch.equal(layer.rms_updates, torch.ones(3, 1, dtype=torch.int64))
    layer.update_enabled = False
    layer(x).sum().backward()
    assert layer.rms_updates.max() == 1
    layer.load_dense_weight(torch.ones(3, 8))
    assert not layer.rms_updates.any()
    assert not layer.v.any()
    torch.testing.assert_close(layer.s_base, layer.scale)


def test_corrected_checkpoint_needs_update_counts():
    # We cannot infer the age of nonzero historical moments. Refuse silent reset.
    old = zero_layer().state_dict()
    with pytest.raises(RuntimeError, match="rms_updates"):
        zero_layer(rms_bias_correction=True).load_state_dict(old)


@pytest.mark.parametrize("cls", [RMSCounterLinear, PackedRMSCounterLinear])
def test_checkpoint_resumes_corrected_threshold_updates(cls):
    layer = zero_layer(cls, rms_bias_correction=True, scale_rebase="threshold")
    configure_rounding_streams(layer, 23)
    layer.lr_scale = 0.001
    gen = torch.Generator().manual_seed(74)
    for _ in range(4):
        layer.apply_update_from_grad_w(torch.randn(3, 8, generator=gen))
    resumed = zero_layer(cls, rms_bias_correction=True, scale_rebase="threshold")
    resumed.lr_scale = layer.lr_scale
    resumed.load_state_dict(copy.deepcopy(layer.state_dict()))
    for _ in range(7):
        grad = torch.randn(3, 8, generator=gen)
        layer.apply_update_from_grad_w(grad)
        torch.rand(37)  # replica-local RNG activity must not affect counter SR
        resumed.apply_update_from_grad_w(grad)
        for key, value in layer.state_dict().items():
            assert torch.equal(value, resumed.state_dict()[key]), key


def test_threshold_ignores_small_oscillation_but_catches_cumulative_drift():
    layer = zero_layer(use_rms=False, scale_rebase="threshold")
    t = torch.zeros(3, 8, dtype=torch.int16)
    c = torch.full_like(t, 5)
    layer.load_counter_state(torch.ones(3, 1), t, c)
    for step in range(100):
        layer.scale.fill_(1 + (-1) ** step * 0.002)
        layer.apply_update_from_grad_w(torch.zeros(3, 8))
        assert torch.equal(layer._decode_rows(0, 3)[1], c)
        assert torch.equal(layer.s_base, torch.ones(3, 1))
    rebases = 0
    previous = layer.s_base.clone()
    for step in range(1, 1001):
        layer.scale.fill_(1.002 ** step)
        layer.apply_update_from_grad_w(torch.zeros(3, 8))
        assert ((layer.s_base / layer.scale - 1).abs() <= layer.rebase_threshold + 1e-6).all()
        rebases += int(not torch.equal(previous, layer.s_base))
        previous = layer.s_base.clone()
    assert rebases > 20


def test_threshold_ticks_use_retained_units_and_rebase_before_carry():
    layer = zero_layer(use_rms=False, scale_rebase="threshold")
    layer.scale.fill_(1.02)
    captured = []
    layer._round_counter = lambda value, lo: captured.append(value.clone()) or value.round()
    grad = torch.full((3, 8), -0.1)
    layer.apply_update_from_grad_w(grad)
    torch.testing.assert_close(captured[-1], -layer.lr * grad * layer.C)  # s_base=1
    assert torch.equal(layer.s_base, torch.ones(3, 1))
    # A carry at a stale basis would jump by the wrong physical weight. Rebase first.
    t = torch.zeros(3, 8, dtype=torch.int16)
    c = torch.full_like(t, 10)
    layer._write_rows(0, 3, t, c)
    grad.fill_(-10)
    expected = (c.float() - layer.lr * grad * layer.C) / 1.02
    layer.apply_update_from_grad_w(grad)
    torch.testing.assert_close(captured[-1], expected)
    torch.testing.assert_close(layer.s_base, layer.scale)
    assert (layer._decode_rows(0, 3)[0] == 1).all()


@pytest.mark.parametrize("kw", [
    {"rms_beta": 1}, {"rms_beta": -0.1}, {"rms_eps": 0},
    {"rebase_threshold": float("nan")}, {"rebase_threshold": -0.1},
    {"rms_bias_correction": True, "rms_mode": "lagged"},
    {"rms_bias_correction": True, "use_rms": False},
])
def test_invalid_stability_configuration_is_rejected(kw):
    with pytest.raises(ValueError):
        zero_layer(**kw)
