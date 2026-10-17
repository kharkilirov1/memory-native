"""Correctness tests for the multi-token LUT-GEMM pipeline (branch research/lut-multi-token).

CPU: the multi-token algebra must reproduce the exact dense product (the reference loops
over tokens with the same LUT identity).

CUDA: the three kernels (index build, LUT build, gather) must individually match the torch
reference, and the composed matmul must match the exact fp32 weight product for several
token counts. Strict ternary only (alpha=0, no salient), which is what the module supports.
"""
from __future__ import annotations

import pytest
import torch

from memory_native.counter import decode_state
from memory_native.group_scale_packed import PackedGroupScaleCounterLinear
from memory_native.lut_gemm import build_lut_blocks, lut_group_matmul
from memory_native.lut_scaling import (
    HAS_TRITON,
    lut_build,
    lut_gather_multi,
    lut_group_matmul_reference,
    lut_matmul_multi,
    ternary_index_build,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

GROUP, C = 128, 11


def _make_layer(shape: int, out_features: int | None = None, seed: int = 0,
                device: str = "cpu"):
    torch.manual_seed(seed)
    n = out_features or shape
    layer = PackedGroupScaleCounterLinear(shape, n, group=GROUP, C=C,
                                          kernel_mode="torch", lr=1e-3)
    layer = layer.to(device)
    levels = 3 * (2 * C - 1)
    from memory_native.packed import pack_codes
    layer.state.copy_(pack_codes(torch.randint(0, levels, (n, shape),
                                               dtype=torch.uint8, device=device)))
    layer.scale.copy_(torch.rand(n, shape // GROUP, device=device) * 0.02 + 0.005)
    return layer


@pytest.mark.parametrize("m", [1, 2, 5])
def test_cpu_reference_matches_exact_product(m):
    layer = _make_layer(256)
    x = torch.randn(m, 256)
    want = x @ layer.visible_weight().t()
    got = lut_group_matmul_reference(x, layer.state, layer.scale, layer.perm,
                                     group=GROUP, C=C)
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5), (got - want).abs().max()


def test_cpu_reference_matches_lut_gemm_reference():
    layer = _make_layer(256)
    x = torch.randn(3, 256)
    a = lut_group_matmul_reference(x, layer.state, layer.scale, layer.perm,
                                   group=GROUP, C=C)
    b = lut_group_matmul(x, layer.state, layer.scale, layer.perm, group=GROUP, C=C)
    assert torch.equal(a, b)


def _cuda_gate():
    if not (torch.cuda.is_available() and HAS_TRITON):
        pytest.skip("needs CUDA + Triton")


@pytest.mark.parametrize("m", [1, 2, 8, 33])
def test_cuda_index_build_matches_cpu(m):
    _cuda_gate()
    layer = _make_layer(512).cuda()
    idx_cuda = ternary_index_build(layer.state, 512, C=C)
    from memory_native.lut_scaling import _ternary_index_cpu
    idx_cpu = _ternary_index_cpu(layer.state.cpu(), 512, C)
    assert torch.equal(idx_cuda.cpu(), idx_cpu)
    assert idx_cuda.dtype == torch.int8 and int(idx_cuda.max()) <= 80


@pytest.mark.parametrize("m", [1, 3, 9])
def test_cuda_lut_build_matches_reference(m):
    _cuda_gate()
    torch.manual_seed(0)
    x = torch.randn(m, 512, device="cuda")
    got = lut_build(x)
    want = build_lut_blocks(x)
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5), (got - want).abs().max()


@pytest.mark.parametrize("m", [1, 2, 8, 64])
def test_cuda_multi_matmul_matches_exact_product(m):
    _cuda_gate()
    layer = _make_layer(512).cuda()
    torch.manual_seed(1)
    x = torch.randn(m, 512, device="cuda")
    want = x @ layer.visible_weight().t()
    got = lut_matmul_multi(x, layer.state, layer.scale, layer.perm, group=GROUP, C=C)
    rel = float((got - want).abs().max() / max(float(want.abs().max()), 1e-12))
    assert rel < 1e-5, rel


def test_cuda_amortized_pipeline_matches_total():
    _cuda_gate()
    layer = _make_layer(512).cuda()
    torch.manual_seed(2)
    x = torch.randn(4, 512, device="cuda")
    x_perm = x[:, layer.perm.long()]
    idx = ternary_index_build(layer.state, 512, C=C)
    lut = lut_build(x_perm.contiguous())
    amortized = lut_gather_multi(lut, idx, layer.scale, group=GROUP, block_n=64)
    total = lut_matmul_multi(x, layer.state, layer.scale, layer.perm, group=GROUP, C=C,
                             block_n=64)
    assert torch.equal(amortized, total)
