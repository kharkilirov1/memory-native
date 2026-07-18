"""Row clip inside the fused update (hashsr reference + Triton kernel).

The clip folds into the RMS denominator with zero extra passes:
tick = -lr*(gw/denom)*mult = -lr*gw/(denom/mult), and mult needs only row stats the kernel
already computes: ||gw/denom||_row = sqrt(mean(gw^2)*in)/denom. GPU parity of the Triton
kernel vs this reference at clip>0 runs on the T4 witness kernel (same tolerance contract
as test_triton_update_matches_reference)."""
import pytest

torch = pytest.importorskip("torch")

from memory_native.counter import encode_state
from memory_native.fused_update import counter_update_hashsr

OUT, IN = 32, 64
KW = dict(C=8, lr=0.05, lr_scale=2e-4, rms_beta=0.9, rms_eps=1e-3, seed=3)


def _fresh(seed=0, grad_scale=1.0):
    torch.manual_seed(seed)
    codes = encode_state(torch.randint(-1, 2, (OUT, IN), dtype=torch.int16),
                         torch.randint(-7, 8, (OUT, IN), dtype=torch.int16), 8)
    gw = torch.randn(OUT, IN) * grad_scale
    scale = torch.full((OUT, 1), 0.05)
    v = torch.rand(OUT, 1) * 1e-2
    return codes, gw, scale, v


def test_folded_row_norm_equals_layer_clip_norm():
    # the identity the fold relies on: sqrt(mean(gw^2)*in)/denom == ||gw/denom||_row
    _, gw, _, v = _fresh()
    denom = (v * 0.9 + gw.pow(2).mean(dim=1, keepdim=True) * 0.1).sqrt().clamp_min(1e-3)
    lhs = (gw.pow(2).mean(dim=1, keepdim=True) * IN).sqrt() / denom
    rhs = (gw / denom).norm(dim=1, keepdim=True)
    assert torch.allclose(lhs, rhs, rtol=1e-6)


def test_clip_zero_is_bit_identical_to_unclipped():
    codes, gw, scale, v = _fresh()
    a = counter_update_hashsr(codes.clone(), scale.clone(), v.clone(), gw, **KW)
    b = counter_update_hashsr(codes.clone(), scale.clone(), v.clone(), gw, clip=0.0, **KW)
    assert torch.equal(a, b)


def test_huge_clip_is_bit_identical_to_unclipped():
    # every row_norm < clip -> mult clamps to exactly 1.0 -> denom/1.0 is bit-exact
    codes, gw, scale, v = _fresh()
    a = counter_update_hashsr(codes.clone(), scale.clone(), v.clone(), gw, **KW)
    b = counter_update_hashsr(codes.clone(), scale.clone(), v.clone(), gw, clip=1e9, **KW)
    assert torch.equal(a, b)


def test_clip_limits_update_magnitude_on_huge_grads():
    codes, gw, scale, v = _fresh(grad_scale=100.0)   # RMS-normalized rows far above clip=1
    from memory_native.counter import decode_state
    t0, c0 = decode_state(codes, 8)
    pos0 = t0.float() + c0.float() / 8

    def total_move(clip):
        out = counter_update_hashsr(codes.clone(), scale.clone(), v.clone(), gw,
                                    clip=clip, **KW)
        t1, c1 = decode_state(out, 8)
        return (t1.float() + c1.float() / 8 - pos0).abs().sum()

    unclipped = total_move(0.0)
    clipped = total_move(1.0)
    assert clipped < unclipped * 0.7, (clipped, unclipped)
