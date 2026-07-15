"""CPU/reference gates for packed group-scale solver-v3 recovery."""
import math

import torch

from memory_native.counter import decode_state, encode_state
from memory_native.group_scale_kernels import (
    group_counter_update_from_io_hashsr,
    group_counter_update_hashsr,
    group_update_scratch_bytes,
)
from memory_native.group_scale_packed import PackedGroupScaleCounterLinear
from memory_native.packed import unpack_codes


def _state(out=3, in_features=16, C=11):
    torch.manual_seed(0)
    t = torch.randint(-1, 2, (out, in_features), dtype=torch.int16)
    c = torch.randint(-(C - 1), C, (out, in_features), dtype=torch.int16)
    return t, c


def test_act_order_packed_roundtrip_and_visible_weight():
    out, k, group, C = 2, 16, 8, 11
    perm = torch.tensor([8, 3, 14, 1, 6, 10, 0, 12, 5, 15, 2, 9, 7, 4, 13, 11])
    layer = PackedGroupScaleCounterLinear(k, out, group=group, C=C, perm=perm)
    t, c = _state(out, k, C)
    scale = torch.tensor([[0.25, 0.5], [0.75, 1.0]])
    layer.load_group_state(scale, t, c, perm)

    codes_perm = unpack_codes(layer.state, k)
    td, cd = decode_state(codes_perm, C)
    assert torch.equal(td, t[:, perm])
    assert torch.equal(cd, c[:, perm])

    layer.set_residual_alpha(0.4)
    expected_perm = scale[:, torch.arange(k) // group] * (
        t[:, perm].float() + 0.4 * c[:, perm].float() / C
    )
    expected = torch.empty_like(expected_perm)
    expected[:, perm] = expected_perm
    assert torch.allclose(layer.visible_weight(), expected)
    x = torch.randn(5, k)
    assert torch.allclose(layer(x), x @ expected.t(), atol=1e-6)


def test_from_io_reference_matches_explicit_group_grad_path():
    torch.manual_seed(2)
    out, k, group, C = 5, 24, 8, 11
    perm = torch.randperm(k)
    t, c = _state(out, k, C)
    codes_perm = encode_state(t[:, perm], c[:, perm], C)
    scale_a = torch.rand(out, math.ceil(k / group)) * 0.2 + 0.1
    scale_b = scale_a.clone()
    v_a = torch.rand(out, 1) * 1e-2
    v_b = v_a.clone()
    x = torch.randn(17, k)
    go = torch.randn(17, out)
    kw = dict(
        group=group, C=C, lr=1e-3, lr_scale=2e-4, rms_beta=0.9,
        rms_eps=1e-3, seed=7, residual_alpha=0.35, clip=1.0,
    )
    got = group_counter_update_from_io_hashsr(
        codes_perm.clone(), scale_a, v_a, x, go, perm, **kw
    )
    grad_perm = go.t() @ x[:, perm]
    ref = group_counter_update_hashsr(
        codes_perm.clone(), scale_b, v_b, grad_perm, perm, **kw
    )
    assert torch.equal(got, ref)
    assert torch.equal(scale_a, scale_b)
    assert torch.equal(v_a, v_b)


def test_packed_layer_backward_updates_without_parameters():
    torch.manual_seed(3)
    out, k, group, C = 7, 32, 8, 11
    perm = torch.randperm(k)
    layer = PackedGroupScaleCounterLinear(
        k, out, group=group, C=C, lr=2e-3, lr_scale=2e-4,
        local_grad_clip=1.0, perm=perm, kernel_mode="torch",
    )
    t, c = _state(out, k, C)
    scale = torch.rand(out, k // group) * 0.1 + 0.05
    layer.load_group_state(scale, t, c, perm)
    assert sum(p.numel() for p in layer.parameters()) == 0
    before_state = layer.state.clone()
    before_scale = layer.scale.clone()
    x = torch.randn(13, k)
    target = torch.randn(13, out)
    loss = (layer(x) - target).square().mean()
    loss.backward()
    assert not torch.equal(layer.state, before_state) or not torch.equal(layer.scale, before_scale)
    assert int(layer.update_events) == out * k


def test_grad_x_uses_preupdate_weight():
    torch.manual_seed(4)
    out, k, group, C = 6, 16, 8, 11
    perm = torch.randperm(k)
    layer = PackedGroupScaleCounterLinear(
        k, out, group=group, C=C, lr=0.2, lr_scale=0.01,
        perm=perm, kernel_mode="torch",
    )
    t, c = _state(out, k, C)
    scale = torch.rand(out, k // group) * 0.2 + 0.1
    layer.load_group_state(scale, t, c, perm)
    x = torch.randn(9, k, requires_grad=True)
    old_w = layer.visible_weight().clone()
    go = torch.randn(9, out)
    y = layer(x)
    y.backward(go)
    assert torch.allclose(x.grad, go @ old_w, atol=1e-6)


def test_strict_scratch_is_group_scaled_not_dense():
    # Both dominant Qwen2.5-1.5B FFN orientations have ~52.5 MiB dense fp32 grad_w.
    for out, in_features in [(8960, 1536), (1536, 8960)]:
        scratch = group_update_scratch_bytes(out, in_features, 128)
        dense = out * in_features * 4
        assert scratch < dense / 30, (scratch, dense)


def test_homotopy_endpoints_do_not_change_packed_truth():
    torch.manual_seed(5)
    out, k, group, C = 3, 16, 8, 11
    perm = torch.randperm(k)
    layer = PackedGroupScaleCounterLinear(k, out, group=group, C=C, perm=perm)
    t, c = _state(out, k, C)
    scale = torch.rand(out, k // group) + 0.1
    layer.load_group_state(scale, t, c, perm)
    truth = layer.state.clone()
    layer.set_residual_alpha(1.0)
    w1 = layer.visible_weight()
    layer.set_residual_alpha(0.0)
    w0 = layer.visible_weight()
    assert not torch.allclose(w1, w0)
    assert torch.equal(layer.state, truth)


def test_persistent_state_is_subbyte_plus_group_metadata():
    out, k, group = 64, 256, 128
    layer = PackedGroupScaleCounterLinear(k, out, group=group)
    packed_state = out * k * 3 // 4
    assert layer.state.numel() == packed_state
    # Scale/v/perm are O(out*k/group + out + k), not per-weight optimizer state.
    overhead = layer.persistent_bytes() - packed_state
    assert overhead < packed_state // 4


def test_packed_group_counter_recovers_synthetic_teacher():
    torch.manual_seed(9)
    k, out, group, C, n = 24, 8, 8, 11, 96
    perm = torch.randperm(k)
    teacher_w = torch.randn(out, k) * 0.12
    layer = PackedGroupScaleCounterLinear(
        k, out, group=group, C=C, lr=5e-4, lr_scale=2e-5,
        perm=perm, kernel_mode="torch", residual_alpha=0.0,
    )
    t = torch.randint(-1, 2, (out, k), dtype=torch.int16)
    c = torch.zeros_like(t)
    scale = torch.full((out, k // group), 0.12)
    layer.load_group_state(scale, t, c, perm)
    x = torch.randn(n, k)
    y = x @ teacher_w.t()
    layer.eval()
    with torch.no_grad():
        start = float((layer(x) - y).square().mean())
    layer.train()
    for _ in range(150):
        (layer(x) - y).square().mean().backward()
    layer.eval()
    with torch.no_grad():
        end = float((layer(x) - y).square().mean())
    assert end < start * 0.9, (start, end)


def test_partial_last_group_supported():
    torch.manual_seed(11)
    out, k, group, C = 4, 20, 8, 11
    perm = torch.randperm(k)
    layer = PackedGroupScaleCounterLinear(k, out, group=group, C=C, perm=perm, kernel_mode="torch")
    t, c = _state(out, k, C)
    scale = torch.rand(out, 3) + 0.05
    layer.load_group_state(scale, t, c, perm)
    x = torch.randn(7, k)
    y = layer(x)
    assert y.shape == (7, out)
    y.square().mean().backward()
    assert torch.isfinite(layer.scale).all()
