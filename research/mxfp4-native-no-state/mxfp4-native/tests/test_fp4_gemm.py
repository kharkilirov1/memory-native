"""NVFP4 packing + matmul backends (mx6/fp4_gemm.py).

CPU tests (always run): pack/unpack is exact for every E2M1 value x every E4M3 scale byte, a LatentGridLinear
packs losslessly, the reference matches an independent float64 computation, swizzle layout sanity.

GPU tests (skip without Triton + CUDA; a skip is not a pass): each backend vs the float64 `expected_*` result.
Tolerances (relative to max|expected|):
* triton_bf16, dot_scaled (bf16 x): x is rounded to bf16, the decoded W is exact in bf16, products have <= 10
  significant bits (exact in fp32); only fp32 summation order differs (K up to 8192): 1e-4.
* dot_scaled_fp4 / torch_scaled_mm: compared with the product of the SAME NVFP4-quantized x, so only kernel
  arithmetic differs (hardware block-scaled accumulation order / bf16 output rounding for torch): 1e-2.
  The error vs the true (unquantized) x is PRINTED, only sanity-bounded (< 0.35), because it is the NVFP4
  activation-quantization error, not a kernel property.
* dx via dot_scaled* / torch_scaled_mm uses the re-quantized W^T; compared with the product with that W^T.
A backend that is unavailable on the device (backend_status) skips with its reason.
"""
from __future__ import annotations

import pytest
import torch

from mx6 import fp4_gemm as fg
from mx6.latent_grid import E2M1_SIGNED, LatentGridLinear
from mx6.packed_na import e4m3_decode

needs_gpu = pytest.mark.skipif(not fg.HAS_TRITON or not torch.cuda.is_available(),
                               reason="needs Triton + CUDA")
VALS = torch.tensor(E2M1_SIGNED, dtype=torch.float32)               # 15 signed values


# ------------------------------------------------------------------------------------------------ CPU

def test_pack_unpack_all_values_all_scale_bytes():
    bytes_ = torch.arange(1, 127, dtype=torch.uint8)                  # every positive finite E4M3 byte (<= 448)
    vals16 = torch.cat([VALS, VALS[:1]])                              # 15 values + 1 repeat = one block
    ts = torch.tensor(0.0123, dtype=torch.float32)
    sc = e4m3_decode(bytes_) * ts
    w = vals16[None, :] * sc[:, None]                                 # [126, 16], same op order as the layer
    q = fg.pack_fp4(w, bytes_.reshape(-1, 1), ts)
    assert q.nibbles.dtype == torch.uint8 and q.nibbles.shape == (126, 8)
    assert q.scales.shape == (126, 1) and torch.equal(q.scales.flatten(), bytes_)
    back = fg.unpack_fp4(*q)
    assert torch.equal(back, w)                                       # bit-exact
    n0 = q.nibbles[0].tolist()
    # nibble codes: low nibble = even element, sign = bit 3, magnitude index into (0,.5,1,1.5,2,3,4,6)
    codes = fg._e2m1_code(vals16)
    assert [(c & 15) | ((d & 15) << 4) for c, d in zip(codes[0::2].tolist(), codes[1::2].tolist())] == n0
    assert fg._nibbles_to_values(codes).tolist() == vals16.tolist()


def test_tuple_unpacking_and_shapes():
    nib, sc, ts = fg.pack_fp4(torch.randn(8, 32))
    assert nib.shape == (8, 16) and sc.shape == (8, 2) and ts.ndim == 0 and ts.dtype == torch.float32


def test_latent_grid_layer_packs_losslessly():
    torch.manual_seed(0)
    layer = LatentGridLinear.from_dense(torch.randn(48, 96) * 0.1, k=8)
    q = fg.pack_fp4(layer)
    assert torch.equal(q.dequantize(), layer.weight())
    assert torch.equal(q.scales, layer._scale) and torch.equal(q.tensor_scale, layer._tscale)
    # the E2M1 code of every weight is the deployed rung
    rung = layer.rung_index()
    codes = torch.stack([q.nibbles & 15, q.nibbles >> 4], -1).reshape(48, 96).long()
    assert torch.equal(fg._nibbles_to_values(codes), torch.tensor(E2M1_SIGNED)[rung])


def test_quantize_nvfp4_roundtrip_error_bounded():
    torch.manual_seed(1)
    x = torch.randn(64, 128)
    q = fg.quantize_nvfp4(x)
    err = (q.dequantize() - x).norm() / x.norm()
    assert 0.0 < err < 0.2, err.item()                                # NVFP4 activation quantization ~ 10 %
    q2 = fg.quantize_nvfp4(q.dequantize())                            # idempotent on its own output
    assert torch.equal(q2.dequantize(), q.dequantize())


def test_reference_matches_float64():
    torch.manual_seed(2)
    layer = LatentGridLinear.from_dense(torch.randn(40, 80) * 0.2, k=8)
    q = fg.pack_fp4(layer)
    x = torch.randn(7, 80)
    ref = fg.reference_matmul(x, q)
    w64 = layer.weight().double()
    indep = x.to(torch.bfloat16).double() @ w64.t()
    assert torch.allclose(ref.double(), indep, rtol=1e-6, atol=1e-6)
    dy = torch.randn(7, 40)
    assert torch.allclose(fg.fp4_matmul_dx(dy, q, "reference").double(),
                          dy.to(torch.bfloat16).double() @ w64, rtol=1e-6, atol=1e-6)
    assert fg.fp4_matmul(x, q, "reference").shape == (7, 40)
    assert fg.fp4_matmul(x.reshape(1, 7, 80), q, "reference").shape == (1, 7, 40)


def test_to_blocked_layout():
    sc = torch.arange(128 * 4, dtype=torch.int32).reshape(128, 4).to(torch.uint8) + 1
    b = fg.to_blocked(sc)
    assert b.numel() == 512 and sorted(b.tolist()) == sorted(sc.flatten().tolist())
    assert fg.to_blocked(torch.ones(3, 2, dtype=torch.uint8)).numel() == 512      # padded to 128 x 4


def test_bad_inputs_rejected():
    with pytest.raises(ValueError):
        fg.pack_fp4(torch.randn(4, 20))                               # K not a multiple of 16
    with pytest.raises(ValueError):
        fg.fp4_matmul(torch.randn(2, 32), fg.pack_fp4(torch.randn(4, 32)), "nope")


def test_module_imports_without_triton_flag():
    assert isinstance(fg.HAS_TRITON, bool) and set(fg.BACKENDS) >= {"reference", "triton_bf16"}


# ------------------------------------------------------------------------------------------------ GPU

def _usable(backend):
    ok, why = fg.backend_status(backend)
    if not ok:
        pytest.skip(f"{backend} unavailable: {why}")


def _problem(m, n, k, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    layer = LatentGridLinear.from_dense(torch.randn(n, k, generator=g) * 0.3, k=8)
    q = fg.pack_fp4(layer).to("cuda")
    x = torch.randn(m, k, generator=g).to("cuda")
    dy = torch.randn(m, n, generator=g).to("cuda")
    return q, x, dy


def _rel(a, b):
    return ((a.double() - b.double()).abs().max() / b.double().abs().max()).item()


EXACT = ("triton_bf16", "dot_scaled")
APPROX = ("dot_scaled_fp4", "torch_scaled_mm")
TOL = {"triton_bf16": 1e-4, "dot_scaled": 1e-4, "dot_scaled_fp4": 1e-2, "torch_scaled_mm": 1e-2}


@needs_gpu
@pytest.mark.parametrize("backend", EXACT + APPROX)
@pytest.mark.parametrize("m", [1, 64, 4096, 8192])
def test_fwd_backends(backend, m):
    _usable(backend)
    q, x, _ = _problem(m, 256, 512)
    y = fg.fp4_matmul(x, q, backend)
    assert y.shape == (m, 256) and y.dtype == torch.float32
    exp = fg.expected_matmul(x, q, backend)
    assert _rel(y, exp) < TOL[backend], (backend, m, _rel(y, exp))
    if backend in APPROX:
        err = _rel(y, fg.reference_matmul(x, q))
        print(f"\n[{backend}] M={m} fwd NVFP4-activation error vs exact x: {err:.4f}")
        assert err < 0.35


@needs_gpu
@pytest.mark.parametrize("backend", EXACT + APPROX)
@pytest.mark.parametrize("m", [1, 64, 4096])
def test_dx_backends(backend, m):
    _usable(backend)
    q, _, dy = _problem(m, 256, 512)
    g = fg.fp4_matmul_dx(dy, q, backend)
    assert g.shape == (m, 512)
    exp = fg.expected_dx(dy, q, backend)
    assert _rel(g, exp) < TOL[backend], (backend, m, _rel(g, exp))
    if backend != "triton_bf16":
        err = _rel(g, fg.reference_dx(dy, q))
        print(f"\n[{backend}] M={m} dx error vs exact W/dy (re-quantized W^T{', fp4 dy' if backend in APPROX else ''}): {err:.4f}")
        assert err < 0.5


@needs_gpu
@pytest.mark.parametrize("backend", EXACT)
@pytest.mark.parametrize("shape", [(37, 100, 208), (5, 33, 48), (130, 96, 1040)])
def test_non_multiple_shapes(backend, shape):
    _usable(backend)
    m, n, k = shape
    q, x, dy = _problem(m, n, k)
    assert _rel(fg.fp4_matmul(x, q, backend), fg.expected_matmul(x, q, backend)) < TOL[backend]
    if backend == "triton_bf16" or n % 16 == 0:
        g = fg.fp4_matmul_dx(dy, q, backend)
        assert _rel(g, fg.expected_dx(dy, q, backend)) < TOL[backend]


@needs_gpu
@pytest.mark.parametrize("backend", APPROX)
def test_fp4_act_non_multiple_m(backend):
    _usable(backend)
    q, x, _ = _problem(37, 128, 256)
    y = fg.fp4_matmul(x, q, backend)
    assert _rel(y, fg.expected_matmul(x, q, backend)) < TOL[backend]


@needs_gpu
def test_bf16_and_fp32_out_dtype_and_3d_input():
    q, x, _ = _problem(64, 128, 256)
    y32 = fg.fp4_matmul(x, q, "triton_bf16")
    y16 = fg.fp4_matmul(x.to(torch.bfloat16), q, "triton_bf16", out_dtype=torch.bfloat16)
    assert y16.dtype == torch.bfloat16 and _rel(y16.float(), y32) < 1e-2
    y3 = fg.fp4_matmul(x.reshape(4, 16, 256), q, "triton_bf16")
    assert torch.equal(y3.reshape(64, 128), y32)


@needs_gpu
def test_triton_bf16_vs_dense_bf16_matmul():
    """The deployed weight in dense bf16 (tensor scale folded in -> rounded) agrees to bf16 precision."""
    q, x, _ = _problem(256, 256, 512)
    dense = (x.to(torch.bfloat16) @ q.dense_bf16().t()).float()
    assert _rel(fg.fp4_matmul(x, q, "triton_bf16"), dense) < 2e-2
