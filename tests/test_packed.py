import io
import math
from types import SimpleNamespace

import pytest
import torch

from memory_native import (
    PackedRMSCounterLinear,
    RMSCounterLinear,
    decode_state,
    encode_state,
    pack_codes,
    unpack_codes,
)


@pytest.mark.parametrize("C", [0, -1, 12, 43, 1.5, 11.5])
def test_packed_rejects_unrepresentable_counter_configuration(C):
    with pytest.raises(ValueError, match="6-bit packing"):
        PackedRMSCounterLinear(4, 1, C=C)


@pytest.mark.parametrize("C", [1, 11])
def test_packed_preserves_every_reachable_state_at_supported_boundaries(C):
    levels = 2 * C - 1
    # Four lanes for every reachable state include t=+1, c=C-1, the old overflow case.
    t = torch.arange(-1, 2, dtype=torch.int16).repeat_interleave(levels).reshape(-1, 1).repeat(1, 4)
    c = torch.arange(-(C - 1), C, dtype=torch.int16).repeat(3).reshape(-1, 1).repeat(1, 4)
    layer = PackedRMSCounterLinear(4, t.shape[0], C=C)
    layer.load_counter_state(torch.ones(t.shape[0], 1), t, c)
    got_t, got_c = decode_state(layer._all_codes(), C)
    assert torch.equal(got_t, t)
    assert torch.equal(got_c, c)
    assert torch.equal(layer._dense_weight(torch.float32), t.float())


def test_hash_sr_checkpoint_resume_preserves_tile_stream(monkeypatch):
    """Exercise the row layer's seed scheduler with the CPU hash-SR kernel reference."""
    import memory_native.fused_update as fused

    seeds = []

    def cpu_kernel(state, scale, v, grad_w, **kw):
        seeds.append(kw["seed"])
        codes = fused.counter_update_hashsr(unpack_codes(state, grad_w.shape[1]), scale, v, grad_w, **kw)
        state.copy_(pack_codes(codes))

    monkeypatch.setattr(fused, "HAS_TRITON", True)
    monkeypatch.setattr(fused, "triton_counter_update", cpu_kernel)
    layer = PackedRMSCounterLinear(8, 5, C=11, lr=0.004, lr_scale=0.001, tile_rows=2)
    model = torch.nn.Sequential(layer)
    generator = torch.Generator().manual_seed(27)
    gradients = [torch.randn(5, 8, generator=generator) * 0.1 for _ in range(6)]

    def update(module, grad_w):
        for lo in range(0, module.out_features, module.tile_rows):
            hi = min(lo + module.tile_rows, module.out_features)
            tile = grad_w[lo:hi]
            # The production CUDA entrypoint is replaced by its exact CPU math; only
            # the gradient's CUDA eligibility flag is adapted so the real scheduler runs.
            cuda_tile = SimpleNamespace(is_cuda=True, shape=tile.shape, contiguous=lambda: tile)
            assert module._fused_update(lo, hi, cuda_tile)

    for grad in gradients[:3]:
        update(layer, grad)
    assert seeds == list(range(9))
    saved = io.BytesIO()
    torch.save(model.state_dict(), saved)
    saved.seek(0)
    checkpoint = torch.load(saved, weights_only=True)
    assert int(checkpoint["0.sr_step"]) == 9
    resumed = PackedRMSCounterLinear(8, 5, C=11, lr=0.004, lr_scale=0.001, tile_rows=2)
    torch.nn.Sequential(resumed).load_state_dict(checkpoint, strict=True)
    assert resumed._sr_step == layer._sr_step == 9
    for grad in gradients[3:]:
        update(layer, grad)
        update(resumed, grad)
        assert resumed._sr_step == layer._sr_step
        for name in ("state", "scale", "v"):
            assert torch.equal(getattr(resumed, name), getattr(layer, name))
    assert layer._sr_step == 18


def test_packed_old_checkpoint_without_sr_step_loads_strictly():
    source = torch.nn.Sequential(PackedRMSCounterLinear(8, 3, C=11))
    checkpoint = source.state_dict()
    del checkpoint["0.sr_step"]
    restored = torch.nn.Sequential(PackedRMSCounterLinear(8, 3, C=11))
    restored[0]._sr_step = 23
    incompatible = restored.load_state_dict(checkpoint, strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    assert restored[0]._sr_step == 0
    assert torch.equal(restored[0].state, source[0].state)


def test_pack_unpack_roundtrip():
    torch.manual_seed(0)
    for C in (8, 11):
        out, in_ = 7, 64
        t = torch.randint(-1, 2, (out, in_), dtype=torch.int16)
        c = torch.randint(-(C - 1), C, (out, in_), dtype=torch.int16)
        codes = encode_state(t, c, C)            # uint8 [out,in] in [0,63]
        packed = pack_codes(codes)
        assert packed.shape == (out, (in_ // 4) * 3)
        back = unpack_codes(packed, in_)
        assert torch.equal(back, codes)


def test_packed_state_is_three_quarter_byte():
    layer = PackedRMSCounterLinear(64, 64, C=11)
    logical = layer.in_features * layer.out_features
    assert layer.state.dtype == torch.uint8
    assert layer.state.numel() == logical * 3 // 4   # 0.75 byte/weight, not 1.0


def test_packed_matches_unpacked_dynamics():
    """Same seed -> packed and unpacked layers must train identically (storage-only change)."""
    n, N, C = 16, 256, 11
    teacher_scale = 0.25
    base = math.sqrt(3.0 / (2.0 * n))

    def make(cls):
        torch.manual_seed(0)
        m = cls(n, n, C=C, lr=0.005, lr_scale=0.0, init_gain=teacher_scale / base)
        m.train()
        return m

    torch.manual_seed(123)
    tw = torch.randint(-1, 2, (n, n)).float()
    x = torch.randn(N, n)
    y = x @ (teacher_scale * tw).t()

    plain, packed = make(RMSCounterLinear), make(PackedRMSCounterLinear)
    for _ in range(200):
        torch.manual_seed(1)  # same SR randomness for both
        ((plain(x) - y) ** 2).mean().backward()
        torch.manual_seed(1)
        ((packed(x) - y) ** 2).mean().backward()

    with torch.no_grad():
        wp = plain._dense_weight(torch.float32)
        wq = packed._dense_weight(torch.float32)
    assert torch.allclose(wp, wq), (wp - wq).abs().max()
