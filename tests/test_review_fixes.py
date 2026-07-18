import copy

import torch
import torch.nn as nn

from memory_native.convert import CounterLinearWithBias, SwapReport
from memory_native.counter import decode_state, encode_state
from memory_native.donor.ptq import ptq_warm_start
from memory_native.group_scale_packed import PackedGroupScaleCounterLinear
from memory_native.packed import pack_codes, unpack_codes
from memory_native.recovery.runtime import (
    build_ptq_counter_kwargs,
    evaluate_at_alpha,
    observe_counter_telemetry,
    restore_counter_structure,
)


def make_layer(*, group=8, k=24, out=5, seed=0, flip_sample_size=4096):
    torch.manual_seed(seed)
    perm = torch.randperm(k)
    layer = PackedGroupScaleCounterLinear(
        k, out, group=group, C=11, lr=2e-3, lr_scale=2e-4,
        perm=perm, kernel_mode="torch", flip_sample_size=flip_sample_size,
    )
    t = torch.randint(-1, 2, (out, k), dtype=torch.int16)
    c = torch.randint(-10, 11, (out, k), dtype=torch.int16)
    scales = torch.rand(out, (k + group - 1) // group) * 0.1 + 0.05
    layer.load_group_state(scales, t, c, perm)
    return layer


def test_sr_step_persists_and_resume_continues_bit_exactly():
    layer = make_layer(seed=1)
    x1, go1 = torch.randn(13, 24), torch.randn(13, 5)
    layer._update_from_io(x1, go1)
    assert int(layer.sr_step) == 1

    resumed = make_layer(seed=2)
    resumed.load_state_dict(copy.deepcopy(layer.state_dict()))
    assert int(resumed.sr_step) == 1

    x2, go2 = torch.randn(11, 24), torch.randn(11, 5)
    layer._update_from_io(x2, go2)
    resumed._update_from_io(x2, go2)
    assert torch.equal(layer.state, resumed.state)
    assert torch.equal(layer.scale, resumed.scale)
    assert torch.equal(layer.v, resumed.v)
    assert int(layer.sr_step) == int(resumed.sr_step) == 2


def test_decoded_diff_sampler_catches_forced_visible_change_and_no_false_positive():
    layer = make_layer(k=16, out=3, group=8, seed=3, flip_sample_size=10_000)
    assert layer.observe_flip_sample(reset=True)["flip_rate_alt"] == 0.0
    assert layer.observe_flip_sample()["flip_rate_alt"] == 0.0

    codes = unpack_codes(layer.state, layer.in_features)
    t, c = decode_state(codes, layer.C)
    t[0, 0] = 0 if int(t[0, 0]) != 0 else 1
    layer.state.copy_(pack_codes(encode_state(t, c, layer.C)))
    telemetry = layer.observe_flip_sample()
    assert telemetry["flip_rate_alt"] > 0.0
    assert telemetry["sample_size"] == layer.out_features * layer.in_features
    assert layer.observe_flip_sample()["flip_rate_alt"] == 0.0


def test_telemetry_aggregates_fused_safe_channel():
    a = make_layer(k=16, out=2, group=8, seed=4, flip_sample_size=10_000)
    b = make_layer(k=16, out=2, group=8, seed=5, flip_sample_size=10_000)
    a.observe_flip_sample(reset=True)
    b.observe_flip_sample(reset=True)
    codes = unpack_codes(a.state, a.in_features)
    t, c = decode_state(codes, a.C)
    t[0, 0] = -1 if int(t[0, 0]) != -1 else 1
    a.state.copy_(pack_codes(encode_state(t, c, a.C)))
    telemetry = observe_counter_telemetry([a, b])
    assert telemetry["flip_rate_alt"] > 0
    assert telemetry["flip_sample_size"] == 64


def test_strict_alpha_eval_is_honest_and_restores_training_alpha():
    layer = make_layer(k=16, out=3, group=8, seed=6)
    layer.set_residual_alpha(0.8)
    x = torch.randn(7, 16)
    train_output = layer(x).detach()
    strict_output = evaluate_at_alpha(layer, 0.0, lambda: {"y": layer(x).detach()})["y"]
    assert not torch.allclose(train_output, strict_output)
    assert layer.residual_alpha == 0.8
    assert layer.training


def test_per_row_ablation_kwargs_do_not_receive_group_only_controls():
    kwargs = build_ptq_counter_kwargs(
        "gptq", lr=1e-3, lr_scale=2e-4, local_grad_clip=1.0,
        residual_alpha=1.0, cache_mode="int8", kernel_mode="auto",
        strict_update=True, flip_sample_size=4096,
    )
    assert kwargs == {
        "lr": 1e-3, "lr_scale": 2e-4,
        "local_grad_clip": 1.0, "cache_mode": "int8",
    }


def test_ptq_api_filters_group_only_kwargs_for_legacy_path(monkeypatch):
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 4, bias=False)

        def forward(self, x):
            return self.proj(x)

    class DummyCounter(nn.Module):
        def __init__(self):
            super().__init__()
            self.loaded = False

        def load_counter_state(self, s, t, c):
            self.loaded = True

        def forward(self, x):
            return x[..., :4]

    captured = {}

    def fake_swap(model, **kwargs):
        captured.update(kwargs)
        model.proj = DummyCounter()
        return SwapReport(swapped=["proj"], coeffs=32)

    monkeypatch.setattr("memory_native.convert.swap_linears_to_counter", fake_swap)
    model = Tiny()
    ptq_warm_start(
        model, [], mode="optimal", kind="counter_packed", C=11, progress=False,
        residual_alpha=1.0, kernel_mode="auto", strict_update=True, flip_sample_size=1024,
    )
    assert model.proj.loaded
    for key in ("residual_alpha", "kernel_mode", "strict_update", "flip_sample_size"):
        assert key not in captured


def test_restore_structure_skips_solver_and_reproduces_checkpoint_output():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(16, 6, bias=True)

        def forward(self, x):
            return self.proj(x)

    torch.manual_seed(8)
    source = Tiny()
    original_bias = source.proj.bias.detach().clone()
    counter = make_layer(k=16, out=6, group=8, seed=9)
    source.proj = CounterLinearWithBias(counter, original_bias)
    source.proj.bias.data.add_(0.03)
    source.proj.counter.sr_step.fill_(37)
    source.proj.counter._sr_step = 37
    checkpoint = copy.deepcopy(source.state_dict())
    x = torch.randn(4, 16)
    expected = source(x).detach()

    restored = Tiny()
    report = restore_counter_structure(
        restored, checkpoint, kind="counter_packed", group=8, C=11,
        kernel_mode="torch", strict_update=True, flip_sample_size=256,
    )
    assert report.coeffs == 16 * 6
    restored.load_state_dict(checkpoint)
    assert isinstance(restored.proj, CounterLinearWithBias)
    assert isinstance(restored.proj.counter, PackedGroupScaleCounterLinear)
    assert int(restored.proj.counter.sr_step) == 37
    assert restored.proj.counter._sr_step == 37
    assert torch.allclose(restored(x), expected)


def test_non_power_of_two_group_is_valid_in_reference_path():
    layer = make_layer(group=12, k=36, out=4, seed=10)
    x = torch.randn(9, 36)
    y = layer(x)
    assert y.shape == (9, 4)
    y.square().mean().backward()
    assert torch.isfinite(layer.scale).all()


def test_non_power_of_two_strict_triton_is_rejected_before_launch():
    from memory_native.group_scale_kernels import triton_group_counter_update_from_io
    import pytest

    with pytest.raises(ValueError, match="power-of-two"):
        triton_group_counter_update_from_io(
            None, None, None, None, None, None,
            group=12, C=11, lr=2e-3, lr_scale=2e-4,
            rms_beta=0.9, rms_eps=1e-3, seed=0,
        )
