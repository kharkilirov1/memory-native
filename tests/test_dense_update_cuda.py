"""CUDA parity gate for the semi-strict dense update (optimization plan L2)."""
import pytest
import torch

from memory_native.counter import decode_state, encode_state
from memory_native.group_scale_kernels import (
    HAS_TRITON,
    group_counter_update_hashsr,
    triton_group_counter_update_dense,
)
from memory_native.group_scale_packed import PackedGroupScaleCounterLinear
from memory_native.packed import pack_codes, unpack_codes

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAS_TRITON), reason="needs CUDA + Triton"
)


def _case():
    torch.manual_seed(0)
    n, k, group, C = 64, 128, 32, 11
    perm = torch.randperm(k, device="cuda")
    t = torch.randint(-1, 2, (n, k), dtype=torch.int16, device="cuda")
    c = torch.randint(-(C - 1), C, (n, k), dtype=torch.int16, device="cuda")
    codes_perm = encode_state(t[:, perm], c[:, perm], C)
    packed = pack_codes(codes_perm).contiguous()
    scale = torch.rand(n, k // group, device="cuda") * 0.15 + 0.05
    v = torch.rand(n, 1, device="cuda") * 1e-2
    gw = torch.randn(n, k, device="cuda", dtype=torch.float32)
    return n, k, group, C, perm, packed, codes_perm, scale, v, gw


def test_dense_update_quanta_parity_against_reference():
    n, k, group, C, perm, packed, codes_perm, scale, v, gw = _case()
    kw = dict(group=group, C=C, lr=2e-3, lr_scale=2e-4, rms_beta=0.9,
              rms_eps=1e-3, seed=17, residual_alpha=0.35, clip=1.0)
    ref_scale, ref_v = scale.clone(), v.clone()
    ref = group_counter_update_hashsr(codes_perm.clone(), ref_scale, ref_v, gw, perm, **kw)
    got_packed, got_scale, got_v = packed.clone(), scale.clone(), v.clone()
    triton_group_counter_update_dense(got_packed, got_scale, got_v, gw, perm, **kw)
    got = unpack_codes(got_packed, k)
    gt, gc = decode_state(got, C)
    rt, rc = decode_state(ref, C)
    different = (got != ref).float().mean().item()
    latent_quanta = (
        (gt.float() + gc.float() / C) - (rt.float() + rc.float() / C)
    ).abs().max().item()
    assert different < 5e-3, different
    assert latent_quanta <= 1.0 / C + 1e-6, latent_quanta
    assert torch.allclose(got_scale, ref_scale, atol=2e-4, rtol=2e-4)
    assert torch.allclose(got_v, ref_v, atol=2e-4, rtol=2e-4)


def test_gemm_layer_on_cuda_uses_dense_update_and_learns():
    torch.manual_seed(4)
    k, out, group = 64, 16, 32
    perm = torch.randperm(k)
    layer = PackedGroupScaleCounterLinear(
        k, out, group=group, C=11, lr=2e-3, lr_scale=2e-4,
        local_grad_clip=1.0, perm=perm, kernel_mode="gemm",
    ).cuda()
    t = torch.randint(-1, 2, (out, k), dtype=torch.int16)
    c = torch.zeros_like(t)
    scale = torch.full((out, k // group), 0.1)
    layer.load_group_state(scale, t, c, perm)
    before = layer.state.clone()
    x = torch.randn(33, k, device="cuda")
    (layer(x) - torch.randn(33, out, device="cuda")).square().mean().backward()
    assert int(layer.sr_step) == 1
    assert not torch.equal(layer.state, before) or not torch.allclose(
        layer.scale, torch.full_like(layer.scale, 0.1)
    )
