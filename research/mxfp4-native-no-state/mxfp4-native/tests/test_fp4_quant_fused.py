"""Fused NVFP4 activation quantizer (mx6/fp4_gemm.py: `_quant_act_kernel`, `quantize_nvfp4_fused`, `quantize_act`).

CPU: the kernel's arithmetic is checked through `emulate_quant_kernel` (torch transcription: comparison coder, swizzle
index formula, tensor scale without fp32 copies) against the eager reference `quantize_nvfp4` + `to_blocked`.
GPU (skipped without CUDA + Triton; a skip is not a pass): the real Triton kernel is BIT-EXACT vs the reference.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mx6 import fp4_gemm as fg                                         # noqa: E402

GPU = pytest.mark.skipif(not (torch.cuda.is_available() and fg.HAS_TRITON), reason="needs CUDA + Triton")
SHAPES = [(5, 16), (7, 48), (128, 64), (130, 80), (257, 144), (64, 2048)]


def _data(r, c, seed=0, dtype=torch.float32, outliers=True):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(r, c, generator=g)
    if outliers:
        x[:, ::7] *= 20.0                                   # block-level dynamic range
        x[r // 2, : min(c, 16)] = 0.0                       # an all-zero block
        x[0, 0] = 1e3                                       # a tensor-scale-setting outlier
    return x.to(dtype)


def _ref(x):
    q = fg.quantize_nvfp4(x)
    return q.nibbles, q.scales, fg.to_blocked(q.scales), q.tensor_scale


# ── CPU ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_emulated_kernel_equals_reference_bit_exactly(shape, dtype):
    x = _data(*shape, dtype=dtype)
    nib, sc, sw = fg.emulate_quant_kernel(x)
    rn, rs, rsw, _ = _ref(x)
    assert torch.equal(nib, rn) and torch.equal(sc, rs) and torch.equal(sw, rsw.view(torch.uint8) if rsw.dtype != torch.uint8 else rsw)


@pytest.mark.parametrize("rows,ncols", [(5, 3), (128, 4), (130, 9), (257, 36)])
def test_swizzle_index_matches_to_blocked(rows, ncols):
    v = torch.arange(1, rows * ncols + 1, dtype=torch.int32).reshape(rows, ncols)
    blocked = fg.to_blocked(v)
    idx = fg.swizzle_index(rows, ncols)
    assert torch.equal(blocked[idx.reshape(-1)], v.reshape(-1))                  # every scale lands where cuBLAS reads it
    mask = torch.ones(blocked.numel(), dtype=torch.bool)
    mask[idx.reshape(-1)] = False
    assert bool((blocked[mask] == 0).all())                                     # the rest is zero padding


def test_swizzle_null_twin_row_major_is_wrong():
    v = torch.arange(1, 256 * 8 + 1, dtype=torch.int32).reshape(256, 8)
    assert not torch.equal(fg.to_blocked(v), v.reshape(-1))                     # a plain row-major store would be wrong


def test_comparison_coder_equals_bucketize_incl_ties_and_null_twin():
    mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    a = torch.cat([mids, -mids, mids.nextafter(torch.tensor(9.0)), mids.nextafter(torch.tensor(-9.0)),
                   torch.tensor([0.0, -0.0, 6.0, -6.0, 7.5, -100.0, float("inf"), float("-inf")]),
                   torch.randn(10000) * 3])
    ref = fg._e2m1_code(torch.nan_to_num(a, nan=0.0, posinf=6.0, neginf=-6.0))
    assert torch.equal(fg._code_cmp(a), ref)                                    # ties go toward zero, like bucketize
    wrong = sum((a.abs() >= t).to(torch.int64) for t in mids.tolist())          # null twin: ties away from zero
    assert not torch.equal(wrong, ref & 7)


def test_tensor_scale_matches_reference_and_zero_tensor():
    x = _data(33, 64, dtype=torch.bfloat16)
    assert torch.equal(fg.act_tensor_scale(x), fg.quantize_nvfp4(x).tensor_scale)
    z = torch.zeros(4, 32)
    assert float(fg.act_tensor_scale(z)) == 1.0
    nib, sc, sw = fg.emulate_quant_kernel(z)
    assert torch.equal(nib, torch.zeros_like(nib)) and bool((sc == 1).all())    # scale floor 2^-9 = byte 0x01


def test_quantize_act_falls_back_to_reference_on_cpu_and_error_is_small():
    x = _data(64, 128, outliers=False)
    q = fg.quantize_act(x)
    r = fg.quantize_nvfp4(x)
    assert torch.equal(q.nibbles, r.nibbles) and torch.equal(q.scales, r.scales)
    e = fg.act_quant_error(x)
    assert 0.02 < e["rel_rms"] < 0.2 and e["rel_max"] < 0.5                    # NVFP4 on a Gaussian: ~10% RMS
    assert e["zero_frac"] < 0.2


# ── GPU ──────────────────────────────────────────────────────────────────────

@GPU
@pytest.mark.parametrize("shape", SHAPES + [(4096, 2048), (16384, 512)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
def test_fused_kernel_is_bit_exact_vs_reference(shape, dtype):
    x = _data(*shape, dtype=dtype).cuda()
    q = fg.quantize_nvfp4_fused(x)
    rn, rs, rsw, rts = _ref(x)
    assert torch.equal(q.tensor_scale, rts)
    assert torch.equal(q.scales, rs), "E4M3 block-scale bytes differ"
    assert torch.equal(q.nibbles, rn), "E2M1 nibbles differ"
    assert torch.equal(q.swizzled_scales().view(torch.uint8), rsw.view(torch.uint8)), "swizzled layout differs"


@GPU
def test_fused_kernel_strided_input_and_zero_and_edge_values():
    x = _data(96, 256, dtype=torch.bfloat16).cuda()
    xt = x.t()                                                                # non-contiguous view: rows = 256, cols = 96
    q = fg.quantize_nvfp4_fused(xt)
    rn, rs, rsw, rts = _ref(xt.contiguous())
    assert torch.equal(q.nibbles, rn) and torch.equal(q.scales, rs) and torch.equal(q.tensor_scale, rts)
    z = torch.zeros(16, 64, device="cuda", dtype=torch.bfloat16)
    qz = fg.quantize_nvfp4_fused(z)
    assert torch.equal(qz.nibbles, fg.quantize_nvfp4(z).nibbles) and torch.equal(qz.scales, fg.quantize_nvfp4(z).scales)
    # exact midpoint values (ties) at every block position
    mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0], device="cuda")
    t = torch.cat([mids, -mids]).repeat(8, 4).to(torch.bfloat16)               # [8, 64]
    qt = fg.quantize_nvfp4_fused(t)
    assert torch.equal(qt.nibbles, fg.quantize_nvfp4(t).nibbles)


@GPU
def test_fused_quantizer_feeds_scaled_mm_with_identical_result():
    ok, why = fg.backend_status("torch_scaled_mm")
    if not ok:
        pytest.skip(why)
    g = torch.Generator().manual_seed(0)
    w = fg.quantize_nvfp4(torch.randn(256, 512, generator=g).cuda())
    x = torch.randn(384, 512, generator=g).cuda().to(torch.bfloat16)
    fg.set_fused_quant(True)
    try:
        y1 = fg.fp4_matmul(x, w, "torch_scaled_mm", torch.bfloat16)
        fg.set_fused_quant(False)
        y0 = fg.fp4_matmul(x, w, "torch_scaled_mm", torch.bfloat16)
    finally:
        fg.set_fused_quant(True)
    assert torch.equal(y1, y0)
    assert fg._FUSED["error"] is None, fg._FUSED["error"]
