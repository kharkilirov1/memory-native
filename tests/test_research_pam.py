"""Numerical and learning witnesses for the PAM linear research reference."""
import math

import pytest
import torch

from memory_native.research.pam import PAMLinear, pam


def test_known_signed_values_and_mantissa_carry():
    a = torch.tensor([1.5, 1.25, 1.75, -1.5, -1.5, 0.375, 0.0])
    b = torch.tensor([1.5, 1.5, 1.5, 1.5, -1.5, 12.0, -3.5])
    expected = torch.tensor([2.0, 1.75, 2.5, -2.0, 2.0, 4.0, 0.0])
    torch.testing.assert_close(pam(a, b), expected, rtol=0, atol=0)
    assert pam(1.5, 1.5).item() == 2.0  # ordinary multiplication gives 2.25


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_powers_of_two_are_exact_for_both_signs(dtype):
    values = torch.tensor([-15.75, -1.3, -0.0625, 0.0, 0.03, 1.5, 11.25], dtype=dtype)
    powers = torch.ldexp(torch.ones(9, dtype=dtype), torch.arange(-4, 5))
    torch.testing.assert_close(pam(values[:, None], powers[None, :]),
                               values[:, None] * powers[None, :], rtol=0, atol=0)
    torch.testing.assert_close(pam(-powers[None, :], values[:, None]),
                               -powers[None, :] * values[:, None], rtol=0, atol=0)


def test_exact_signed_derivatives_match_finite_differences_away_from_boundaries():
    a = torch.tensor([1.1, 1.4, -2.6, -0.8, 0.0375], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([1.2, -1.8, -0.7, 3.1, -12.2], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(pam, (a, b), eps=1e-7, atol=1e-7, rtol=1e-6)
    delta = torch.tensor([0.3, -0.7, 1.2, -0.5, 0.9], dtype=torch.float64)
    grad_a, grad_b = torch.autograd.grad(pam(a, b), (a, b), delta)
    # First pair is in the no-carry segment: d_a=2**0, d_b=2**0.
    assert grad_a[0].item() == pytest.approx(0.3)
    assert grad_b[0].item() == pytest.approx(0.3)
    # Second pair carries; sign(b) enters d_a and sign(a) enters d_b.
    assert grad_a[1].item() == pytest.approx(1.4)
    assert grad_b[1].item() == pytest.approx(-1.4)


def test_exact_carry_boundary_uses_the_upper_segment():
    a = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.grad(pam(a, b), (a, b)) == (torch.tensor(2.0), torch.tensor(2.0))


def test_primitive_broadcast_gradients_reduce_to_original_shapes():
    a = torch.tensor([[1.1], [-2.6]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([1.2, -1.8, -0.7], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(pam, (a, b), eps=1e-7, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_subnormals_and_underflow_are_explicitly_supported(dtype):
    zero, one = torch.tensor(0.0, dtype=dtype), torch.tensor(1.0, dtype=dtype)
    smallest = torch.nextafter(zero, one)
    small = torch.tensor(torch.finfo(dtype).tiny / 4, dtype=dtype)
    torch.testing.assert_close(pam(small, 1.5), small * 1.5, rtol=0, atol=0)
    torch.testing.assert_close(pam(smallest, 2.0), smallest * 2, rtol=0, atol=0)
    assert pam(smallest, 0.25).item() == 0.0
    # Normalization works for an extremely small A and extremely large B too.
    exponent = 127 if dtype == torch.float32 else 1023
    huge = torch.ldexp(one, torch.tensor(exponent))
    torch.testing.assert_close(pam(smallest, huge), smallest * huge, rtol=0, atol=0)


def test_exact_zero_convention_is_zero_for_both_operands():
    a = torch.tensor([0.0, -0.0, 1.5, 0.0], requires_grad=True)
    b = torch.tensor([1.5, -3.0, 0.0, 0.0], requires_grad=True)
    result = pam(a, b)
    grad_a, grad_b = torch.autograd.grad(result.sum(), (a, b))
    assert torch.equal(result, torch.zeros_like(result))
    assert torch.equal(grad_a, torch.zeros_like(a))
    assert torch.equal(grad_b, torch.zeros_like(b))


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_inputs_are_rejected_even_when_other_operand_is_zero(bad):
    with pytest.raises(ValueError, match="finite"):
        pam(torch.tensor(bad), torch.tensor(0.0))
    layer = PAMLinear(2, 1)
    with pytest.raises(ValueError, match="finite"):
        layer(torch.tensor([[bad, 1.0]]))


def test_overflow_and_nonfinite_upstream_gradient_fail_explicitly():
    with pytest.raises(ValueError, match="overflow"):
        pam(torch.tensor(torch.finfo(torch.float32).max), torch.tensor(2.0))
    a = torch.tensor(1.1, requires_grad=True)
    with pytest.raises(ValueError, match="upstream"):
        pam(a, 1.2).backward(torch.tensor(math.inf))


def test_dtype_promotion_and_unsupported_formats():
    result = pam(torch.tensor(1.5, dtype=torch.float32), torch.tensor(1.5, dtype=torch.float64))
    assert result.dtype == torch.float64
    with pytest.raises(TypeError, match="float32 and float64"):
        pam(torch.ones(2, dtype=torch.float16), 1.5)
    with pytest.raises(TypeError, match="float32 and float64"):
        pam(torch.ones(2, dtype=torch.int64), 2)


@pytest.mark.parametrize("mode", ["exact", "surrogate"])
@pytest.mark.parametrize("chunk_size", [1, 8, 16, 32])
def test_linear_chunked_forward_and_all_gradients_match_scalar_oracles(mode, chunk_size):
    torch.manual_seed(11)
    layer = PAMLinear(19, 7, backward=mode, bias=True, chunk_size=chunk_size).double()
    x = torch.randn(2, 3, 19, dtype=torch.float64, requires_grad=True)
    delta = torch.randn(2, 3, 7, dtype=torch.float64)
    result = layer(x)
    grad_x, grad_w, grad_b = torch.autograd.grad(result, (x, layer.weight, layer.bias), delta)

    # Small untiled oracle: independent autograd reduction for the exact mode.
    xo = x.detach().clone().requires_grad_(True)
    wo = layer.weight.detach().clone().requires_grad_(True)
    bo = layer.bias.detach().clone().requires_grad_(True)
    flat, go = xo.reshape(-1, 19), delta.reshape(-1, 7)
    oracle = pam(flat[:, None, :], wo[None, :, :]).sum(dim=-1) + bo
    torch.testing.assert_close(result.reshape_as(oracle), oracle, rtol=1e-12, atol=1e-12)
    if mode == "exact":
        dx, dw, db = torch.autograd.grad(oracle, (xo, wo, bo), go)
    else:
        with torch.no_grad():
            dx = pam(wo[None, :, :], go[:, :, None]).sum(dim=1).reshape_as(x)
            dw = pam(flat[:, None, :], go[:, :, None]).sum(dim=0)
            db = go.sum(dim=0)
    torch.testing.assert_close(grad_x, dx, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(grad_w, dw, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(grad_b, db, rtol=1e-12, atol=1e-12)


def test_surrogate_zero_operand_follows_paper_instead_of_exact_zero_convention():
    layer = PAMLinear(1, 1, backward="surrogate").double()
    with torch.no_grad():
        layer.weight.fill_(1.5)
    x = torch.zeros((1, 1), dtype=torch.float64, requires_grad=True)
    dx, dw = torch.autograd.grad(layer(x), (x, layer.weight), torch.ones((1, 1), dtype=torch.float64))
    assert dx.item() == 1.5
    assert dw.item() == 0.0
    with torch.no_grad():
        layer.weight.zero_()
    x = torch.full((1, 1), 1.5, dtype=torch.float64, requires_grad=True)
    dx, dw = torch.autograd.grad(layer(x), (x, layer.weight), torch.ones((1, 1), dtype=torch.float64))
    assert dx.item() == 0.0
    assert dw.item() == 1.5


def test_float32_chunk_parity_and_noncontiguous_input():
    torch.manual_seed(12)
    layers = [PAMLinear(23, 9, bias=True, chunk_size=chunk) for chunk in (1, 16, 32)]
    for layer in layers[1:]:
        layer.load_state_dict(layers[0].state_dict())
    source = torch.randn(3, 23, 2).transpose(1, 2)
    assert not source.is_contiguous()
    delta = torch.randn(3, 2, 9)
    witnesses = []
    for layer in layers:
        x = source.detach().requires_grad_(True)
        output = layer(x)
        witnesses.append((output, *torch.autograd.grad(output, (x, layer.weight, layer.bias), delta)))
    for witness in witnesses[1:]:
        for actual, expected in zip(witness, witnesses[0]):
            torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("mode", ["exact", "surrogate"])
def test_checkpoint_reload_preserves_forward_and_training_rule(mode):
    torch.manual_seed(14)
    source = PAMLinear(19, 7, backward=mode, bias=True, chunk_size=16).double()
    restored = PAMLinear(19, 7, backward=mode, bias=True, chunk_size=32).double()
    restored.load_state_dict(source.state_dict())
    assert restored._pam_config.tolist() == [19, 7, int(mode == "surrogate")]
    x = torch.randn(6, 19, dtype=torch.float64, requires_grad=True)
    delta = torch.randn(6, 7, dtype=torch.float64)
    witnesses = []
    for layer in (source, restored):
        y = layer(x)
        witnesses.append((y, *torch.autograd.grad(y, (x, layer.weight, layer.bias), delta)))
    for actual, expected in zip(witnesses[1], witnesses[0]):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("mismatch", ["backward", "dimensions", "missing"])
def test_checkpoint_mismatch_is_rejected_before_parameter_mutation(mismatch):
    source = PAMLinear(4, 2, backward="exact", bias=True)
    target = PAMLinear(4, 2, backward="surrogate" if mismatch == "backward" else "exact", bias=True)
    state = source.state_dict()
    if mismatch == "dimensions":
        state["_pam_config"] = torch.tensor([5, 2, 0], dtype=torch.int64)
    elif mismatch == "missing":
        del state["_pam_config"]
    before_weight, before_bias = target.weight.detach().clone(), target.bias.detach().clone()
    with pytest.raises(RuntimeError, match="configuration"):
        target.load_state_dict(state)
    torch.testing.assert_close(target.weight, before_weight, rtol=0, atol=0)
    torch.testing.assert_close(target.bias, before_bias, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["exact", "surrogate"])
def test_regression_training_improves_held_out_error(mode):
    """Actual optimizer/backward iterations generalize beyond the training batch."""
    torch.manual_seed(13)
    teacher = PAMLinear(4, 2, bias=True)
    with torch.no_grad():
        teacher.weight.copy_(torch.tensor([[0.75, -0.55, 0.33, 1.0], [-0.2, 0.5, 0.9, -0.4]]))
        teacher.bias.copy_(torch.tensor([0.15, -0.25]))
    train_x, val_x = torch.randn(96, 4), torch.randn(64, 4)
    with torch.no_grad():
        train_y, val_y = teacher(train_x), teacher(val_x)
    student = PAMLinear(4, 2, backward=mode, init_gain=0.2, bias=True, chunk_size=3)
    optimizer = torch.optim.Adam(student.parameters(), lr=0.03)
    with torch.no_grad():
        initial = (student(val_x) - val_y).square().mean().item()
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        loss = (student(train_x) - train_y).square().mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = (student(val_x) - val_y).square().mean().item()
    assert final < initial * 0.1, (mode, initial, final)


@pytest.mark.parametrize("kwargs", [dict(chunk_size=0), dict(chunk_size=True), dict(backward="unknown"),
                                     dict(init_gain=math.inf), dict(in_features=0), dict(out_features=-1)])
def test_invalid_layer_configuration_is_rejected(kwargs):
    defaults = dict(in_features=4, out_features=2)
    defaults.update(kwargs)
    with pytest.raises(ValueError):
        PAMLinear(**defaults)
