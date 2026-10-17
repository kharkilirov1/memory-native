import copy
import io
from types import SimpleNamespace

import pytest
import torch

from memory_native import (
    PackedGroupScaleCounterLinear, PackedRMSCounterLinear, RMSCounterLinear,
    configure_rounding_streams, pack_codes, unpack_codes,
)


def test_layer_paths_separate_rounding_and_replicas_reproduce_it():
    model = torch.nn.Sequential(RMSCounterLinear(256, 256), RMSCounterLinear(256, 256))
    replica = copy.deepcopy(model)
    streams = configure_rounding_streams(model, 42)
    assert streams == configure_rounding_streams(replica, 42)
    assert streams["0"] != streams["1"]
    value = torch.full((256, 256), 0.5)
    draws = [layer._round_counter(value) for layer in model]
    assert 0.47 < (draws[0] != draws[1]).float().mean() < 0.53
    for i in range(2):
        torch.rand(1000)
        assert torch.equal(draws[i], replica[i]._round_counter(value))
        assert 0.48 < draws[i].mean() < 0.52


@pytest.mark.parametrize("scope,decimation", [("row", 1), ("group", 2)])
def test_group_streams_resume_and_keep_decimation_schedule(scope, decimation):
    def make():
        return PackedGroupScaleCounterLinear(32, 4, group=16, kernel_mode="torch",
                                            stats_scope=scope, decimation=decimation)
    layer = make()
    configure_rounding_streams(layer, 17)
    gen = torch.Generator().manual_seed(41)
    for _ in range(3):
        layer._update_from_io(torch.randn(7, 32, generator=gen), torch.randn(7, 4, generator=gen))
    buf = io.BytesIO()
    torch.save(layer.state_dict(), buf)
    buf.seek(0)
    replica = make()
    replica.load_state_dict(torch.load(buf, weights_only=True))
    for _ in range(4):
        x, go = torch.randn(7, 32, generator=gen), torch.randn(7, 4, generator=gen)
        layer._update_from_io(x, go)
        torch.rand(51)
        replica._update_from_io(x, go)
        for key, value in layer.state_dict().items():
            assert torch.equal(value, replica.state_dict()[key]), key
    assert layer._sr_step == replica._sr_step == 7
    assert layer._rounding_step == replica._rounding_step == 7


def test_old_checkpoint_restores_legacy_rng_strictly():
    model = torch.nn.Sequential(PackedRMSCounterLinear(8, 3),
                               PackedGroupScaleCounterLinear(8, 3, group=4))
    old = copy.deepcopy(model.state_dict())
    configure_rounding_streams(model, 15)
    model.load_state_dict(old, strict=True)
    for layer in model:
        assert layer._rounding_salt is None
        assert layer._rounding_step == 0
    assert set(model.state_dict()) == set(old)


def test_packed_fused_scheduler_passes_stream_seed_to_kernel_and_resumes(monkeypatch):
    import memory_native.fused_update as fused

    seeds = []
    def cpu_kernel(state, scale, v, grad_w, **kw):
        seeds.append(kw["seed"])
        codes = fused.counter_update_hashsr(unpack_codes(state, grad_w.shape[1]), scale, v, grad_w, **kw)
        state.copy_(pack_codes(codes))
    monkeypatch.setattr(fused, "HAS_TRITON", True)
    monkeypatch.setattr(fused, "triton_counter_update", cpu_kernel)
    model = torch.nn.Sequential(PackedRMSCounterLinear(8, 3), PackedRMSCounterLinear(8, 3))
    configure_rounding_streams(model, 19)
    grad = torch.full((3, 8), 0.01)
    cuda_grad = SimpleNamespace(is_cuda=True, shape=grad.shape, contiguous=lambda: grad)
    for layer in model:
        assert layer._fused_update(0, 3, cuda_grad)
    assert seeds[0] != seeds[1]
    replica = copy.deepcopy(model)
    replica.load_state_dict(copy.deepcopy(model.state_dict()))
    for a, b in zip(model, replica):
        a._fused_update(0, 3, cuda_grad)
        b._fused_update(0, 3, cuda_grad)
        assert seeds[-2] == seeds[-1]
        assert torch.equal(a.state, b.state)
    corrected = PackedRMSCounterLinear(8, 3, rms_bias_correction=True)
    assert not corrected._fused_update(0, 3, cuda_grad)
    thresholded = PackedRMSCounterLinear(8, 3, scale_rebase="threshold")
    assert not thresholded._fused_update(0, 3, cuda_grad)
