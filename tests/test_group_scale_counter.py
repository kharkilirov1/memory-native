import math

import torch

from memory_native.donor.ptq import gptq_group_ternary, group_residual_counter
from memory_native.group_scale_counter import GroupScaleCounterLinear


def test_group_scale_mapping_and_homotopy():
    layer = GroupScaleCounterLinear(
        8, 2, group=4, C=11, residual_alpha=0.0,
        perm=torch.tensor([4, 5, 6, 7, 0, 1, 2, 3]),
    )
    scales = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
    ternary = torch.tensor(
        [[1, 0, -1, 1, 1, 1, 0, -1], [0, 1, 1, -1, -1, 0, 1, 1]],
        dtype=torch.int16,
    )
    residual = torch.ones_like(ternary) * 5
    layer.load_group_state(scales, ternary, residual, layer.perm)

    expected_scales = torch.tensor(
        [[3, 3, 3, 3, 2, 2, 2, 2], [7, 7, 7, 7, 5, 5, 5, 5]],
        dtype=torch.float32,
    )
    assert torch.equal(layer.column_scales(), expected_scales)
    assert torch.allclose(layer.visible_weight(), expected_scales * ternary.float())

    layer.set_residual_alpha(1.0)
    assert torch.allclose(
        layer.visible_weight(), expected_scales * (ternary.float() + residual.float() / 11)
    )


def test_backward_updates_without_master_parameter():
    torch.manual_seed(0)
    layer = GroupScaleCounterLinear(16, 5, group=4, C=11, lr=0.01, lr_scale=1e-3)
    ternary = torch.randint(-1, 2, (5, 16), dtype=torch.int16)
    residual = torch.zeros_like(ternary)
    scales = torch.full((5, 4), 0.1)
    layer.load_group_state(scales, ternary, residual)
    assert sum(p.numel() for p in layer.parameters()) == 0
    before_state = layer.state.clone()
    before_scale = layer.scale.clone()
    x = torch.randn(32, 16)
    target = torch.randn(32, 5)
    loss = (layer(x) - target).square().mean()
    loss.backward()
    assert not torch.equal(layer.state, before_state) or not torch.equal(layer.scale, before_scale)
    assert int(layer.update_events) == layer.state.numel()


def test_solver_refinement_monotone_and_reconstructable():
    torch.manual_seed(4)
    n, in_f, out_f = 384, 32, 12
    base = torch.randn(n, in_f)
    x = base @ (torch.eye(in_f) + 0.15 * torch.randn(in_f, in_f))
    hessian = x.t() @ x
    weight = torch.randn(out_f, in_f) * 0.08
    weight[:, :2] *= 6

    q0, _, _ = gptq_group_ternary(weight, hessian, group=8, refine_scale=False)
    q2, s2, t2, perm, wadj = gptq_group_ternary(
        weight,
        hessian,
        group=8,
        refine_scale=True,
        refine_iters=2,
        scale_refit="hdiag",
        return_perm=True,
    )

    def error(q):
        e = weight - q
        return float(((e @ hessian) * e).sum())

    assert error(q2) <= error(q0) * (1 + 1e-6)
    group_perm = torch.arange(in_f) // 8
    group_index = torch.empty_like(group_perm)
    group_index[perm] = group_perm
    reconstructed = s2[:, group_index] * t2.float()
    assert torch.allclose(reconstructed, q2, atol=1e-5)
    counter = group_residual_counter(wadj, s2, t2, perm, 8, C=11)
    assert counter.abs().max() <= 10


def test_synthetic_teacher_low_lr_recovery_decreases_pure_ternary_loss():
    torch.manual_seed(7)
    in_f, out_f = 24, 10
    teacher_weight = torch.randn(out_f, in_f) * 0.12
    xcal = torch.randn(512, in_f)
    hessian = xcal.t() @ xcal
    _, scales, ternary, perm, wadj = gptq_group_ternary(
        teacher_weight, hessian, group=6, refine_iters=2, return_perm=True
    )
    residual = group_residual_counter(wadj, scales, ternary, perm, 6, C=11)
    layer = GroupScaleCounterLinear(
        in_f, out_f, group=6, C=11, lr=1e-4, lr_scale=1e-5,
        residual_alpha=0.0, perm=perm,
    )
    layer.load_group_state(scales, ternary, residual, perm)
    x = torch.randn(128, in_f)
    y = x @ teacher_weight.t()
    layer.eval()
    with torch.no_grad():
        start = float((layer(x) - y).square().mean())
    layer.train()
    for _ in range(500):
        (layer(x) - y).square().mean().backward()
    layer.eval()
    with torch.no_grad():
        end = float((layer(x) - y).square().mean())
    assert math.isfinite(end)
    assert end < start, (start, end)
