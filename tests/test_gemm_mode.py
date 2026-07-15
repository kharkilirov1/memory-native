"""Gates for kernel_mode="gemm" (optimization plan L1/L2): cuBLAS matmuls + semi-strict update."""
import torch

from memory_native.group_scale_packed import PackedGroupScaleCounterLinear


def _loaded(kernel_mode, *, k=32, out=6, group=8, seed=11):
    torch.manual_seed(seed)
    perm = torch.randperm(k)
    layer = PackedGroupScaleCounterLinear(
        k, out, group=group, C=11, lr=2e-3, lr_scale=2e-4,
        local_grad_clip=1.0, perm=perm, kernel_mode=kernel_mode,
    )
    t = torch.randint(-1, 2, (out, k), dtype=torch.int16)
    c = torch.randint(-10, 11, (out, k), dtype=torch.int16)
    scales = torch.rand(out, k // group) * 0.1 + 0.05
    layer.load_group_state(scales, t, c, perm)
    return layer


def test_gemm_mode_matches_dense_reference():
    layer = _loaded("gemm")
    layer.set_residual_alpha(0.4)
    x = torch.randn(9, 32)
    go = torch.randn(9, 6)
    w = layer.visible_weight()
    assert torch.allclose(layer(x), x @ w.t(), atol=1e-6)
    assert torch.allclose(layer._grad_x_2d(go), go @ w, atol=1e-6)


def test_gemm_update_bit_identical_to_torch_mode():
    a = _loaded("gemm", seed=3)
    b = _loaded("torch", seed=3)
    x, go = torch.randn(13, 32), torch.randn(13, 6)
    a._update_from_io(x, go)
    b._update_from_io(x, go)
    assert torch.equal(a.state, b.state)
    assert torch.equal(a.scale, b.scale)
    assert torch.equal(a.v, b.v)


def test_auto_mode_trains_off_cuda():
    layer = _loaded("auto", seed=5)
    before = layer.state.clone()
    x = torch.randn(17, 32)
    (layer(x) - torch.randn(17, 6)).square().mean().backward()
    assert not torch.equal(layer.state, before) or int(layer.update_events) > 0
