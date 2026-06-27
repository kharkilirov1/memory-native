import math

import pytest
import torch

from memory_native.counter import decode_state, encode_state
from memory_native.fused_update import (
    HAS_TRITON,
    counter_update_hashsr,
    hash_u32,
    triton_counter_update,
    uniform01,
)

CUDA = torch.cuda.is_available()


def test_hash_rng_unbiased():
    e = torch.arange(200_000)
    u = uniform01((12345 ^ hash_u32(e)) & 0xFFFFFFFF)
    assert abs(u.mean().item() - 0.5) < 0.01
    assert 0.0 <= u.min().item() and u.max().item() <= 1.0


def test_reference_recovers_teacher():
    """The deterministic-SR reference update must recover a ternary teacher (algorithm correct)."""
    torch.manual_seed(0)
    n, N, C = 16, 256, 11
    ts = 0.25
    tw = torch.randint(-1, 2, (n, n)).to(torch.int16)
    x = torch.randn(N, n)
    y = x @ (ts * tw.float()).t()
    codes = encode_state(torch.randint(-1, 2, (n, n), dtype=torch.int16),
                         torch.zeros((n, n), dtype=torch.int16), C)
    scale = torch.full((n, 1), 0.25)
    v = torch.zeros((n, 1))
    for step in range(800):
        t, _ = decode_state(codes, C)
        w = scale * t.float()
        grad_out = (2.0 / (N * n)) * (x @ w.t() - y)
        codes = counter_update_hashsr(codes, scale, v, grad_out.t() @ x,
                                      C=C, lr=0.005, lr_scale=0.0, rms_beta=0.9, rms_eps=1e-3, seed=step)
    t, _ = decode_state(codes, C)
    mse = ((x @ (scale * t.float()).t() - y) ** 2).mean().item()
    assert mse < 5e-3 and (t == tw).float().mean().item() > 0.9


@pytest.mark.skipif(not (CUDA and HAS_TRITON), reason="needs CUDA + triton")
def test_triton_update_matches_reference_bitexact():
    """The fused Triton update must equal the deterministic-SR reference bit-for-bit."""
    from memory_native.packed import pack_codes, unpack_codes
    torch.manual_seed(0)
    out, in_, C = 24, 64, 11
    codes = encode_state(torch.randint(-1, 2, (out, in_), dtype=torch.int16),
                         torch.randint(-(C - 1), C, (out, in_), dtype=torch.int16), C).cuda()
    grad_w = torch.randn(out, in_, device="cuda")
    scale = torch.full((out, 1), 0.3, device="cuda")
    v = torch.rand(out, 1, device="cuda") * 1e-2
    kw = dict(C=C, lr=5e-3, lr_scale=2e-4, rms_beta=0.9, rms_eps=1e-3, seed=7)

    # reference on unpacked codes (mutates ref_scale/ref_v in place)
    ref_scale, ref_v = scale.clone(), v.clone()
    ref_codes = counter_update_hashsr(codes.clone(), ref_scale, ref_v, grad_w, **kw)
    # kernel on packed state (mutates sc2/v2 in place)
    packed = pack_codes(codes).contiguous()
    sc2, v2 = scale.clone(), v.clone()
    triton_counter_update(packed, sc2, v2, grad_w, **kw)
    got_codes = unpack_codes(packed, in_)
    assert torch.equal(got_codes, ref_codes), (got_codes.int() - ref_codes.int()).abs().max()
    assert torch.allclose(sc2, ref_scale, atol=1e-6)
    assert torch.allclose(v2, ref_v, atol=1e-6)
