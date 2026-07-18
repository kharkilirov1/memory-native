"""CUDA parity gates for packed group-scale solver-v3 kernels."""
import pytest
import torch

from memory_native.counter import decode_state, encode_state
from memory_native.group_scale_kernels import (
    HAS_TRITON,
    group_counter_update_from_io_hashsr,
    triton_group_counter_update_from_io,
    triton_group_decode_matmul,
    triton_group_grad_x,
)
from memory_native.packed import pack_codes, unpack_codes

CUDA_TRITON = torch.cuda.is_available() and HAS_TRITON
pytestmark = pytest.mark.skipif(not CUDA_TRITON, reason="needs CUDA + Triton")


def _case(dtype=torch.float32):
    torch.manual_seed(0)
    m, n, k, group, C = 96, 64, 128, 32, 11
    perm = torch.randperm(k, device="cuda")
    t = torch.randint(-1, 2, (n, k), dtype=torch.int16, device="cuda")
    c = torch.randint(-(C - 1), C, (n, k), dtype=torch.int16, device="cuda")
    codes_perm = encode_state(t[:, perm], c[:, perm], C)
    packed = pack_codes(codes_perm).contiguous()
    scale = torch.rand(n, k // group, device="cuda") * 0.15 + 0.05
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    go = torch.randn(m, n, device="cuda", dtype=dtype)
    alpha = 0.35
    pos_group = torch.arange(k, device="cuda") // group
    w_perm = scale[:, pos_group] * (
        t[:, perm].float() + alpha * c[:, perm].float() / C
    )
    w = torch.empty_like(w_perm)
    w[:, perm] = w_perm
    return m, n, k, group, C, perm, packed, scale, x, go, alpha, w


@pytest.mark.parametrize("dtype,atol", [(torch.float32, 4e-3), (torch.bfloat16, 4e-2)])
def test_group_forward_and_gradx_kernel(dtype, atol):
    _, _, k, group, C, perm, packed, scale, x, go, alpha, w = _case(dtype)
    got_y = triton_group_decode_matmul(
        x, packed, scale, perm, C=C, group=group, residual_alpha=alpha
    )
    ref_y = x @ w.to(dtype).t()
    got_gx = triton_group_grad_x(
        go, packed, scale, perm, in_features=k, C=C, group=group,
        residual_alpha=alpha,
    )
    ref_gx = go @ w.to(dtype)
    assert torch.allclose(got_y.float(), ref_y.float(), atol=atol, rtol=atol)
    assert torch.allclose(got_gx.float(), ref_gx.float(), atol=atol, rtol=atol)


def test_group_strict_update_bitquantified_against_reference():
    _, n, k, group, C, perm, packed, scale, x, go, alpha, _ = _case(torch.float32)
    codes = unpack_codes(packed, k)
    v = torch.rand(n, 1, device="cuda") * 1e-2
    ref_scale, ref_v = scale.clone(), v.clone()
    ref = group_counter_update_from_io_hashsr(
        codes.clone(), ref_scale, ref_v, x, go, perm,
        group=group, C=C, lr=2e-3, lr_scale=2e-4, rms_beta=0.9,
        rms_eps=1e-3, seed=17, residual_alpha=alpha, clip=1.0,
    )
    got_packed = packed.clone()
    got_scale, got_v = scale.clone(), v.clone()
    triton_group_counter_update_from_io(
        got_packed, got_scale, got_v, x, go, perm,
        group=group, C=C, lr=2e-3, lr_scale=2e-4, rms_beta=0.9,
        rms_eps=1e-3, seed=17, residual_alpha=alpha, clip=1.0,
    )
    got = unpack_codes(got_packed, k)
    gt, gc = decode_state(got, C)
    rt, rc = decode_state(ref, C)
    different = (got != ref).float().mean().item()
    latent_quanta = (
        (gt.float() + gc.float() / C) - (rt.float() + rc.float() / C)
    ).abs().max().item()
    # GEMM/reduction order can move a small SR-boundary fraction by one counter quantum.
    assert different < 5e-3, different
    assert latent_quanta <= 1.0 / C + 1e-6, latent_quanta
    assert torch.allclose(got_scale, ref_scale, atol=2e-4, rtol=2e-4)
    assert torch.allclose(got_v, ref_v, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_salient_correction_on_triton_path(dtype):
    """Salient sparse override on the CUDA path: fp32 COO vs bf16/fp16 activations must not
    dtype-crash (sparse.mm requires matching dtypes) and must match the dense reference."""
    from memory_native.group_scale_packed import PackedGroupScaleCounterLinear

    torch.manual_seed(3)
    k, out, group, C = 64, 16, 32, 11
    perm = torch.randperm(k)
    layer = PackedGroupScaleCounterLinear(
        k, out, group=group, C=C, perm=perm, kernel_mode="triton").cuda()
    t = torch.randint(-1, 2, (out, k), dtype=torch.int16)
    c = torch.zeros_like(t)
    scale = torch.rand(out, k // group) * 0.15 + 0.05
    idx = torch.tensor([5, 70, 131, 200], dtype=torch.int32)
    val = torch.tensor([0.7, -0.4, 0.9, -1.1], dtype=torch.float16)
    layer.load_group_state(scale, t, c, perm, salient_idx=idx, salient_val=val)

    x = torch.randn(12, k, device="cuda", dtype=dtype)
    with torch.no_grad():
        ref_w = layer.visible_weight()
        y = layer(x)                                   # triton base + sparse correction
        assert torch.allclose(y.float(), (x.float() @ ref_w.t()), atol=5e-2)
    go = torch.randn(12, out, device="cuda", dtype=dtype)
    gx = layer._grad_x_2d(go)
    assert torch.allclose(gx.float(), (go.float() @ ref_w), atol=5e-2)
