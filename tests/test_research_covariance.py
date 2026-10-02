import copy

import pytest
import torch

from memory_native.counter import RMSCounterLinear
from memory_native.research.covariance import CovarianceRMSCounterLinear, precondition_counter_blast
from memory_native.research.structured import CounterBLASTLinear


def test_precondition_matches_closed_form():
    layer = CovarianceRMSCounterLinear(3, 2, cov_ridge=.1, cov_eps=1e-6)
    layer.input_gram.copy_(torch.tensor([[2., .3, 0.], [.3, 1., .2], [0., .2, .5]]))
    g = torch.tensor([[1., -2., 3.], [-.5, .2, 1.]])
    metric = layer.input_gram + (.1 * layer.input_gram.trace() / 3 + 1e-6) * torch.eye(3)
    torch.testing.assert_close(layer.precondition(g), g @ torch.linalg.inv(metric))


def test_current_batch_uncentered_ema_and_reuse_guard():
    layer = CovarianceRMSCounterLinear(2, 1, cov_beta=.75, lr=0., lr_scale=0.)
    x = torch.tensor([[2., 1.], [0., 3.]])
    y = layer(x)
    expected = .75 * torch.eye(2) + .25 * (x.T @ x / 2)
    torch.testing.assert_close(layer.input_gram, expected)
    assert int(layer.gram_updates) == 1
    with pytest.raises(RuntimeError, match="reused"):
        layer(10*x)
    torch.testing.assert_close(layer.input_gram, expected)
    y.sum().backward()
    with torch.no_grad():
        layer(10*x)
    layer.eval()
    layer(10*x).sum().backward()
    assert int(layer.gram_updates) == 1


def test_old_input_gradient_and_preconditioned_update(monkeypatch):
    layer = CovarianceRMSCounterLinear(3, 2, lr=.2, lr_scale=.01, tile_rows=1)
    x = torch.tensor([[1., -.5, 2.], [.5, 1., -1.]], requires_grad=True)
    upstream = torch.tensor([[1., -.5], [.2, 1.5]])
    old_weight = layer._forward_matmul(torch.eye(3)).T.detach().clone()
    seen = []
    original = RMSCounterLinear._update_tile

    def capture(self, lo, hi, grad_w, *args, **kwargs):
        seen.append((lo, hi, grad_w.clone()))
        return original(self, lo, hi, grad_w, *args, **kwargs)

    monkeypatch.setattr(RMSCounterLinear, "_update_tile", capture)
    layer(x).backward(upstream)
    torch.testing.assert_close(x.grad, upstream @ old_weight)
    raw = upstream.T @ x.detach()
    for lo, hi, signal in seen:
        torch.testing.assert_close(signal, layer.precondition(raw[lo:hi]))
    assert len(seen) == 2


def test_factory_preserves_state_output_rng_and_local_sizes():
    torch.manual_seed(4)
    model = CounterBLASTLinear(32, 32, block_size=8, rank=4, C=2)
    baseline = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        before = model(torch.eye(32))
    rng = torch.get_rng_state().clone()
    precondition_counter_blast(model)
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in baseline.items():
        assert torch.equal(value, model.state_dict()[name])
    with torch.no_grad():
        torch.testing.assert_close(model(torch.eye(32)), before, rtol=0, atol=0)
    assert all(f.input_gram.shape == (8, 8) for f in model.V)
    assert all(f.input_gram.shape == (4, 4) for f in [*model.U, *model.S])
    assert sum(f.input_gram.numel()*4 for f in [*model.V, *model.S, *model.U]) == 1536


def test_roundtrip_preserves_metric_and_next_update():
    a = CovarianceRMSCounterLinear(3, 2, cov_ridge=.01)
    a(torch.randn(4, 3)).square().sum().backward()
    b = CovarianceRMSCounterLinear(3, 2, cov_ridge=.01)
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    x = torch.randn(4, 3)
    torch.manual_seed(23)
    a(x).square().sum().backward()
    torch.manual_seed(23)
    b(x).square().sum().backward()
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key])


def test_config_mismatch_refuses_before_state_mutation():
    a = CovarianceRMSCounterLinear(3, 2, cov_ridge=.01)
    b = CovarianceRMSCounterLinear(3, 2, cov_ridge=.1)
    before = copy.deepcopy(b.state_dict())
    with pytest.raises(RuntimeError, match="configuration mismatch"):
        b.load_state_dict(a.state_dict())
    for key, value in before.items():
        assert torch.equal(value, b.state_dict()[key])


@pytest.mark.parametrize("config", [dict(cov_beta=1), dict(cov_beta=-.1),
    dict(cov_ridge=-1), dict(cov_eps=0), dict(rms_mode="proxy"), dict(update_compute="int8")])
def test_invalid_config(config):
    with pytest.raises(ValueError):
        CovarianceRMSCounterLinear(3, 2, **config)


def test_zero_inputs_remain_spd_and_disabled_updates_do_not_observe():
    layer = CovarianceRMSCounterLinear(3, 2, cov_beta=0., cov_ridge=0.)
    layer(torch.zeros(2, 3)).sum().backward()
    assert bool(torch.isfinite(layer.precondition(torch.ones(2, 3))).all())
    layer.update_enabled = False
    layer(torch.ones(2, 3)).sum().backward()
    assert int(layer.gram_updates) == 1


def test_factory_refuses_pending_forward_without_replacing_factors():
    model = CounterBLASTLinear(8, 8, block_size=4, rank=2)
    factors = list(model.V)
    output = model(torch.randn(2, 8))
    with pytest.raises(RuntimeError, match="outstanding forward"):
        precondition_counter_blast(model)
    assert list(model.V) == factors
    output.sum().backward()


def test_factory_refuses_non_float32_without_lossy_conversion():
    model = CounterBLASTLinear(8, 8, block_size=4, rank=2).double()
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(TypeError, match="float32"):
        precondition_counter_blast(model)
    for key, value in before.items():
        assert torch.equal(value, model.state_dict()[key])


@pytest.mark.parametrize("x", [torch.empty(0, 3), torch.full((2, 3), float("nan")),
                              torch.full((2, 3), float("inf"))])
def test_invalid_observation_refuses_before_mutation(x):
    layer = CovarianceRMSCounterLinear(3, 2)
    before = copy.deepcopy(layer.state_dict())
    with pytest.raises(ValueError, match="nonempty finite"):
        layer(x)
    assert not layer._outstanding_forward
    for key, value in before.items():
        assert torch.equal(value, layer.state_dict()[key])
