"""Strict update-from-IO (forms grad_w in-kernel, no dense gradient).

The CPU reference equals the grad_w path fed grad_out^T x (by construction) and trains a teacher;
the Triton kernel forms grad_w in registers and must match the reference bit-quantified on GPU
(FP reduction order differs -> a few SR-boundary flips, like fused_update).
"""
import pytest
import torch

from memory_native.counter import decode_state, encode_state
from memory_native.fused_update import counter_update_hashsr
from memory_native.update_from_io import (
    HAS_TRITON,
    counter_update_from_io_hashsr,
    triton_counter_update_from_io,
)

CUDA = torch.cuda.is_available()


def test_from_io_reference_equals_grad_w_path():
    torch.manual_seed(0)
    out, in_, C = 24, 64, 11
    codes = encode_state(torch.randint(-1, 2, (out, in_), dtype=torch.int16),
                         torch.randint(-(C - 1), C, (out, in_), dtype=torch.int16), C)
    x = torch.randn(40, in_); go = torch.randn(40, out)
    kw = dict(C=C, lr=5e-3, lr_scale=2e-4, rms_beta=0.9, rms_eps=1e-3, seed=7)
    a_s, a_v = torch.full((out, 1), 0.3), torch.zeros(out, 1)
    b_s, b_v = a_s.clone(), a_v.clone()
    ref_io = counter_update_from_io_hashsr(codes.clone(), a_s, a_v, x, go, **kw)
    ref_gw = counter_update_hashsr(codes.clone(), b_s, b_v, go.t() @ x, **kw)
    assert torch.equal(ref_io, ref_gw)
    assert torch.equal(a_s, b_s) and torch.equal(a_v, b_v)


def test_from_io_reference_recovers_teacher():
    torch.manual_seed(0)
    n, N, C = 16, 64, 11
    ts = 0.25
    tw = torch.randint(-1, 2, (n, n)).to(torch.int16)
    x = torch.randn(N, n); y = x @ (ts * tw.float()).t()
    codes = encode_state(torch.randint(-1, 2, (n, n), dtype=torch.int16),
                         torch.zeros((n, n), dtype=torch.int16), C)
    scale = torch.full((n, 1), 0.25); v = torch.zeros((n, 1))
    for step in range(800):
        t, _ = decode_state(codes, C)
        grad_out = (2.0 / (N * n)) * (x @ (scale * t.float()).t() - y)
        codes = counter_update_from_io_hashsr(codes, scale, v, x, grad_out,
                                              C=C, lr=0.005, lr_scale=0.0, rms_beta=0.9, rms_eps=1e-3, seed=step)
    t, _ = decode_state(codes, C)
    assert ((x @ (scale * t.float()).t() - y) ** 2).mean().item() < 5e-3


@pytest.mark.skipif(not (CUDA and HAS_TRITON), reason="needs CUDA + triton")
def test_from_io_kernel_bitquantified():
    from memory_native.packed import pack_codes, unpack_codes
    torch.manual_seed(0)
    out, in_, C = 64, 256, 11
    codes = encode_state(torch.randint(-1, 2, (out, in_), dtype=torch.int16),
                         torch.randint(-(C - 1), C, (out, in_), dtype=torch.int16), C).cuda()
    x = torch.randn(128, in_, device="cuda"); go = torch.randn(128, out, device="cuda")
    scale = torch.full((out, 1), 0.3, device="cuda"); v = torch.rand(out, 1, device="cuda") * 1e-2
    kw = dict(C=C, lr=5e-3, lr_scale=2e-4, rms_beta=0.9, rms_eps=1e-3, seed=7)
    rs, rv = scale.clone(), v.clone()
    ref = counter_update_from_io_hashsr(codes.clone(), rs, rv, x, go, **kw)
    packed = pack_codes(codes).contiguous(); sc2, v2 = scale.clone(), v.clone()
    triton_counter_update_from_io(packed, sc2, v2, x, go, **kw)
    got = unpack_codes(packed, in_)
    gt, gc = decode_state(got, C); rt, rc = decode_state(ref, C)
    frac = (got != ref).float().mean().item()
    quanta = ((gt.float() + gc.float() / C) - (rt.float() + rc.float() / C)).abs().max().item()
    assert frac < 1e-3 and quanta < 1.0 / C + 1e-6
    assert torch.allclose(sc2, rs, atol=1e-5)
