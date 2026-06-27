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
@pytest.mark.parametrize("out,in_,C", [(24, 64, 11), (512, 2048, 11), (256, 4096, 8)])
def test_triton_update_matches_reference(out, in_, C):
    """The fused Triton update must match the deterministic-SR reference.

    Bit-identity is *not* required: the kernel reduces row-stats (g_sq, grad_s) in BLOCK_I
    chunks while the torch reference reduces in one pass, so s_new/denom differ by ~1e-7. That
    feeds every element's `val` and can flip a stochastic-rounding boundary on a tiny fraction of
    weights (rnd granularity 1/2**24 over ~out*in elements -> O(1) flips). A flip moves one weight
    by exactly one counter quantum -- the same unbiased noise SR already injects -- so we assert
    the divergence is a negligible fraction of weights, each off by at most one quantum.
    """
    from memory_native.packed import pack_codes, unpack_codes
    torch.manual_seed(0)
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

    # row-stat outputs match closely (different reduction order -> ~1e-7)
    assert torch.allclose(sc2, ref_scale, atol=1e-5)
    assert torch.allclose(v2, ref_v, atol=1e-6)
    # where codes differ, the weight moved by at most one counter quantum (t + c/C)
    gt, gc = decode_state(got_codes, C)
    rt, rc = decode_state(ref_codes, C)
    pos_got = gt.float() + gc.float() / C
    pos_ref = rt.float() + rc.float() / C
    mism = (got_codes != ref_codes)
    frac = mism.float().mean().item()
    max_quanta = (pos_got - pos_ref).abs().max().item()
    assert frac < 1e-3, f"too many SR-boundary flips: {frac:.2e}"
    assert max_quanta < 1.0 / C + 1e-6, f"a flip moved a weight by >1 quantum: {max_quanta}"
