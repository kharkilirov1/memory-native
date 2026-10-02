"""Independent dense witnesses for the experimental factored BLAST passes."""
import copy

import pytest
import torch
from torch.nn import functional as F

from memory_native.counter import RMSCounterLinear, decode_state
from memory_native.research.structured import BLASTLinear, CounterBLASTLinear, LowRankLinear


def _dense_weight(model):
    """Diagnostic reconstruction outside the operator under test."""
    def weight(factor):
        if isinstance(factor, RMSCounterLinear):
            return factor._dense_weight(torch.float32)
        return factor.weight

    s = torch.stack([weight(factor) for factor in model.S], dim=-1)
    return torch.cat([
        torch.cat([weight(u) @ torch.diag(s[i, j]) @ weight(v)
                   for j, v in enumerate(model.V)], dim=1)
        for i, u in enumerate(model.U)
    ], dim=0)


@pytest.mark.parametrize("fin,fout", [(32, 32), (32, 48), (35, 21), (7, 11)])
def test_fp_forward_and_all_gradients_match_dense_reconstruction(fin, fout):
    torch.manual_seed(37)
    model = BLASTLinear(fin, fout, block_size=16, rank=8, bias=True).double()
    reference = copy.deepcopy(model)
    x = torch.randn(2, 3, fin, dtype=torch.float64, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    delta = torch.randn(2, 3, fout, dtype=torch.float64)
    actual = model(x)
    expected = F.linear(xr, _dense_weight(reference), reference.bias)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    (actual * delta).sum().backward()
    (expected * delta).sum().backward()
    torch.testing.assert_close(x.grad, xr.grad, rtol=1e-11, atol=1e-11)
    ref_parameters = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, ref_parameters[name].grad,
                                   rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("fin,fout", [(32, 32), (35, 21)])
def test_counter_chain_uses_preupdate_weights_and_correct_factor_gradients(fin, fout, monkeypatch):
    torch.manual_seed(41)
    model = CounterBLASTLinear(fin, fout, block_size=16, rank=8, bias=True,
                               counter_lr=.2, lr_scale=.002)
    floating = BLASTLinear(fin, fout, block_size=16, rank=8, bias=True)
    before = {}
    recorded = {}
    with torch.no_grad():
        floating.bias.copy_(model.bias)
        for group in ("U", "V", "S"):
            for index, (counter, fp) in enumerate(zip(getattr(model, group), getattr(floating, group))):
                # A sign-only state avoids accidental all-zero tiny factors in
                # this witness. Both nonzero and zero states are exercised by
                # the default-init reload and shape witnesses below.
                t, c = decode_state(counter.state, counter.C)
                t = torch.where(t == 0, torch.ones_like(t), t)
                counter.load_counter_state(counter.scale, t, c)
                fp.weight.copy_(counter._dense_weight(torch.float32))
                key = f"{group}.{index}"
                before[key] = counter.state.clone()
                original_update = counter._update_tile

                def record_update(lo, hi, grad_w, *args, _key=key,
                                  _original=original_update, **kwargs):
                    assert _key not in recorded, "factor updated more than once"
                    recorded[_key] = grad_w.detach().clone()
                    return _original(lo, hi, grad_w, *args, **kwargs)

                monkeypatch.setattr(counter, "_update_tile", record_update)
    x = torch.randn(2, 3, fin, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    delta = torch.randn(2, 3, fout)
    actual = model(x)
    expected = F.linear(xr, _dense_weight(floating), floating.bias)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    (actual * delta).sum().backward()
    (expected * delta).sum().backward()
    torch.testing.assert_close(x.grad, xr.grad, rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(model.bias.grad, floating.bias.grad)
    assert len(recorded) == len(before)
    for group in ("U", "V", "S"):
        for index, (counter, fp) in enumerate(zip(getattr(model, group), getattr(floating, group))):
            key = f"{group}.{index}"
            torch.testing.assert_close(recorded[key], fp.weight.grad, rtol=3e-5, atol=3e-5)
            assert not torch.equal(before[key], counter.state), f"no state update at {key}"
            assert not counter._outstanding_forward
            assert torch.isfinite(counter.scale).all()


@pytest.mark.parametrize("cls", [BLASTLinear, CounterBLASTLinear])
def test_rectangular_edge_blocks_and_checkpoint_roundtrip(cls):
    torch.manual_seed(19)
    model = cls(35, 21, block_size=16, rank=8, bias=True)
    x = torch.randn(3, 35)
    model(x).square().mean().backward()
    reloaded = cls(35, 21, block_size=16, rank=8, bias=True)
    reloaded.load_state_dict(copy.deepcopy(model.state_dict()))
    with torch.no_grad():
        torch.testing.assert_close(reloaded(x), model(x), rtol=0, atol=0)
    assert model.input_block_sizes == (16, 16, 3)
    assert model.output_block_sizes == (16, 5)
    assert model.rank_bound == 13


def test_counter_has_no_fp_master_factors_or_registered_weight_gradients():
    model = CounterBLASTLinear(64, 48, block_size=16, rank=8)
    assert list(model.parameters()) == []
    states = [layer.state for layer in model.modules() if isinstance(layer, RMSCounterLinear)]
    assert sum(state.numel() for state in states) == model.coefficient_count
    assert all(state.dtype == torch.uint8 for state in states)
    assert not any(tuple(buffer.shape) == (48, 64) for buffer in model.buffers())
    model(torch.randn(2, 64)).square().mean().backward()
    assert all(state.grad is None for state in states)


def test_structural_arithmetic_reduction_and_rank_limit_are_explicit():
    model = CounterBLASTLinear(64, 256, block_size=16, rank=8)
    expected_coefficients = 8 * (64 + 256 + 4 * 16)
    assert model.coefficient_count == expected_coefficients
    assert model.operation_counts(12) == {
        "forward_macs": 12 * expected_coefficients,
        "backward_macs": 24 * expected_coefficients,
    }
    assert expected_coefficients < 64 * 256
    assert model.rank_bound == 32


def test_lowrank_replacement_matches_dense_forward_and_backward():
    torch.manual_seed(43)
    model = LowRankLinear(35, 21, rank=9, bias=True).double()
    reference = copy.deepcopy(model)
    x = torch.randn(2, 3, 35, dtype=torch.float64, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    delta = torch.randn(2, 3, 21, dtype=torch.float64)
    actual = model(x)
    expected = F.linear(xr, reference.U.weight @ reference.V.weight, reference.U.bias)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    (actual * delta).sum().backward()
    (expected * delta).sum().backward()
    torch.testing.assert_close(x.grad, xr.grad, rtol=1e-11, atol=1e-11)
    for (name, parameter), (_, ref_parameter) in zip(model.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(parameter.grad, ref_parameter.grad, rtol=1e-11, atol=1e-11)
    assert model.coefficient_count == 9 * (35 + 21)
    assert model.rank_bound == 9
    assert model.operation_counts(6)["backward_macs"] == 12 * model.coefficient_count


def test_counter_reuse_guard_is_preserved_and_clears_after_backward():
    model = CounterBLASTLinear(32, 48, block_size=16, rank=8)
    x = torch.randn(2, 32)
    y = model(x)
    with pytest.raises(RuntimeError, match="reused before its previous backward"):
        model(x)
    y.sum().backward()
    model(x).sum().backward()
    assert not any(layer._outstanding_forward for layer in model.modules()
                   if isinstance(layer, RMSCounterLinear))


def test_eval_backward_preserves_counter_state():
    model = CounterBLASTLinear(32, 48, block_size=16, rank=8).eval()
    before = copy.deepcopy(model.state_dict())
    x = torch.randn(2, 32, requires_grad=True)
    model(x).sum().backward()
    assert torch.isfinite(x.grad).all()
    for key, value in model.state_dict().items():
        assert torch.equal(before[key], value), key


def test_counter_checkpoint_refuses_different_C_before_mutating_buffers():
    saved = CounterBLASTLinear(32, 48, C=8).state_dict()
    target = CounterBLASTLinear(32, 48, C=9)
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(RuntimeError, match="topology or counter C mismatch"):
        target.load_state_dict(saved)
    assert all(torch.equal(value, before[key]) for key, value in target.state_dict().items())


@pytest.mark.parametrize("kwargs", [{"rank": 0}, {"block_size": 0}, {"rank": 2.5},
                                    {"init_gain": float("nan")}, {"init_gain": 0},
                                    {"C": 0}, {"C": 44}, {"counter_lr": -1},
                                    {"lr_scale": float("inf")}])
def test_invalid_counter_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        CounterBLASTLinear(32, 48, **kwargs)
