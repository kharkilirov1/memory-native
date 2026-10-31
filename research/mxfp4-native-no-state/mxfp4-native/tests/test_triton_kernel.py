"""Parity tests for the fused decode+matmul Triton kernels.

GPU tests skip unless Triton and a CUDA device are present (on the CPU box the
skip is the honest witness, not a pass).  The CPU tests at the bottom check the
host-side ladder derivation and the bf16-exactness argument, and always run.

Tolerances per ``compute_dtype`` (W = decoded weight, always exact in bf16/tf32):

* ``fp32-ieee``: full fp32 FMA dot; only summation order differs from torch
  -> atol = rtol = 1e-4 against the fp32 reference.
* ``bf16``: x is rounded to bf16 in-kernel.  Against a reference that uses the
  same bf16-rounded x the products are exact in fp32, so only summation order
  differs -> 1e-4.  Against the unrounded fp32 reference the bound is
  2^-8 * (|x| @ |W|.T) (bf16 half-ulp = 2^-9 relative per product; 2^-8 slack).
* ``tf32``: x rounded to 10 mantissa bits (half-ulp 2^-11 relative); bound
  2^-10 * (|x| @ |W|.T) + 1e-4.
"""
from __future__ import annotations

import pytest
import torch

from mx6.packed_na import PackedNA5, PackedNA6
from mx6.triton_kernel import (HAS_TRITON, derive_ladder, packed_matmul,
                               packed_matmul_dx)

needs_gpu = pytest.mark.skipif(
    not HAS_TRITON or not torch.cuda.is_available(),
    reason="fused kernel needs Triton + CUDA")

CASES = [(PackedNA5, "PackedNA5"), (PackedNA6, "PackedNA6")]
MODES = ["fp32-ieee", "tf32", "bf16"]
# (N, K): none a multiple of the autotune tile sizes (K multiple of 32 only)
NK = [(50, 96), (130, 224), (256, 160)]
MS = [1, 33, 128]


def _bf16r(t):
    return t.to(torch.bfloat16).to(torch.float32)


def _check(out, a, W, mode):
    """a: [M, *] input, W: reference weight so that ref = a @ W."""
    absbound = a.abs() @ W.abs()
    if mode == "fp32-ieee":
        torch.testing.assert_close(out, a @ W, atol=1e-4, rtol=1e-4)
    elif mode == "bf16":
        # tight: same bf16-rounded input -> only summation order differs
        torch.testing.assert_close(out, _bf16r(a) @ W, atol=1e-4, rtol=1e-4)
        # loose: against the unrounded fp32 reference
        assert ((out - a @ W).abs() <= 2.0 ** -8 * absbound + 1e-4).all()
    else:
        assert ((out - a @ W).abs() <= 2.0 ** -10 * absbound + 1e-4).all()


@needs_gpu
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("fmt,label", CASES)
@pytest.mark.parametrize("N,K", NK)
@pytest.mark.parametrize("M", MS)
def test_matches_dense_decode(fmt, label, M, N, K, mode):
    """out = x @ decode(codes, scale).T must equal the torch reference."""
    torch.manual_seed(0)
    w = torch.randn(N, K, device="cuda") * 0.1
    codes, sb = fmt.quantize(w, headroom=2.0)
    x = torch.randn(M, K, device="cuda")

    W = fmt.dequantize(codes, sb)                        # [N, K] fp32
    out = packed_matmul(x, codes, sb, fmt.code_to_value.to("cuda"), compute_dtype=mode)
    _check(out, x, W.t().contiguous(), mode)


@needs_gpu
@pytest.mark.parametrize("fmt,label", CASES)
def test_default_is_bf16_and_legacy_precision_alias(fmt, label):
    torch.manual_seed(3)
    w = torch.randn(64, 128, device="cuda") * 0.1
    codes, sb = fmt.quantize(w, headroom=2.0)
    x = torch.randn(8, 128, device="cuda")
    lut = fmt.code_to_value.to("cuda")
    default = packed_matmul(x, codes, sb, lut)
    explicit = packed_matmul(x, codes, sb, lut, compute_dtype="bf16")
    torch.testing.assert_close(default, explicit, atol=0.0, rtol=0.0)
    ieee = packed_matmul(x, codes, sb, lut, input_precision="ieee")
    torch.testing.assert_close(ieee, x @ fmt.dequantize(codes, sb).t(), atol=1e-4, rtol=1e-4)
    with pytest.raises(ValueError):
        packed_matmul(x, codes, sb, lut, compute_dtype="fp8")


@needs_gpu
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("fmt,label", CASES)
def test_accumulator_field_does_not_change_the_matmul(fmt, label, mode):
    """The forward must read only the value; the accumulator is invisible."""
    torch.manual_seed(1)
    K, N, M = 128, 64, 16
    w = torch.randn(N, K, device="cuda") * 0.1
    codes, sb = fmt.quantize(w, headroom=2.0)
    x = torch.randn(M, K, device="cuda")
    lut = fmt.code_to_value.to("cuda")

    base = packed_matmul(x, codes, sb, lut, compute_dtype=mode)

    # flip the accumulator residue to a different level, keep the magnitude
    mag = codes.to(torch.long) // fmt.acc_levels
    other_acc = (fmt.C - 1) + 1 if fmt.C > 1 else fmt.C - 1
    shifted = (mag * fmt.acc_levels + other_acc).to(torch.uint8)

    again = packed_matmul(x, shifted, sb, lut, compute_dtype=mode)
    torch.testing.assert_close(again, base, atol=0.0, rtol=0.0)


@needs_gpu
def test_rejects_shape_mismatch():
    fmt = PackedNA6
    x = torch.zeros(4, 96, device="cuda")
    codes = torch.zeros(8, 128, dtype=torch.uint8, device="cuda")
    sb = torch.zeros(8, 4, dtype=torch.uint8, device="cuda")
    with pytest.raises(AssertionError):
        packed_matmul(x, codes, sb, fmt.code_to_value.to("cuda"))


@needs_gpu
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("fmt,label", CASES)
@pytest.mark.parametrize("N,K", NK)
@pytest.mark.parametrize("M", MS)
def test_grad_x_matches_dense_decode(fmt, label, M, N, K, mode):
    """grad_x = go @ decode(codes, scale) must equal the torch reference."""
    torch.manual_seed(2)
    w = torch.randn(N, K, device="cuda") * 0.1
    codes, sb = fmt.quantize(w, headroom=2.0)
    go = torch.randn(M, N, device="cuda")

    W = fmt.dequantize(codes, sb)                        # [N, K]
    out = packed_matmul_dx(go, codes, sb, fmt.code_to_value.to("cuda"), compute_dtype=mode)
    _check(out, go, W, mode)


@needs_gpu
@pytest.mark.parametrize("fmt,label", CASES)
def test_extreme_scale_bytes_decode_exactly(fmt, label):
    """Scale bytes at both ends of the usable range (incl. 2^-127 subnormal)."""
    torch.manual_seed(4)
    N, K = 32, 64
    codes = torch.randint(0, fmt.n_codes, (N, K), device="cuda", dtype=torch.uint8)
    sb = torch.tensor([0, 1, 100, 127, 130, 200, 250, 252], device="cuda",
                      dtype=torch.uint8)[torch.arange(N * (K // 32), device="cuda") % 8]
    sb = sb.reshape(N, K // 32).contiguous()
    # identity-ish probe: one-hot rows of x read the decoded weight columns back
    x = torch.eye(K, device="cuda")
    out = packed_matmul(x, codes, sb, fmt.code_to_value.to("cuda"), compute_dtype="bf16")
    W = fmt.dequantize(codes, sb)
    torch.testing.assert_close(out, W.t().contiguous(), atol=0.0, rtol=0.0)


# ── CPU tests (always run) ───────────────────────────────────────────────────

@pytest.mark.parametrize("fmt,label", CASES)
def test_derive_ladder_roundtrip(fmt, label):
    signed, acc = derive_ladder(fmt.code_to_value)
    assert acc == fmt.acc_levels
    assert signed == [float(v) for v in fmt.signed]
    for c in range(fmt.n_codes):
        assert signed[c // acc] == float(fmt.code_to_value[c])


def test_derive_ladder_rejects_irregular_table():
    with pytest.raises(ValueError):
        derive_ladder(torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]))


@pytest.mark.parametrize("fmt,label", CASES)
def test_ladder_times_pow2_scale_is_exact_in_bf16(fmt, label):
    """The bf16 path's exactness claim: value * 2^(byte-127) survives bf16."""
    vals = fmt.signed.to(torch.float32)
    for byte in range(0, 253):                      # 252: 6*2^125 < 2^128, finite
        scale = torch.tensor(2.0 ** (byte - 127), dtype=torch.float64)
        w64 = vals.to(torch.float64) * scale
        w32 = w64.to(torch.float32)
        assert torch.equal(w32.to(torch.float64), w64), f"fp32 inexact at byte {byte}"
        assert torch.equal(w32.to(torch.bfloat16).to(torch.float32), w32), \
            f"bf16 inexact at byte {byte}"


# ── NVFP4 scale (E4M3 per 16 + fp32 per-tensor scale) ───────────────────────
#
# Exactness (see triton_kernel.py docstring): ladder value x E4M3 scale has <= 6
# significant bits -> exact in bf16/tf32/fp32; the fp32 per-tensor scale is NOT exact in
# bf16, so the kernel applies it to the fp32 accumulator, not to the weight tile.

from mx6.packed_na import PackedNA, e4m3_decode  # noqa: E402

NV_CASES = [
    (PackedNA([0.0, 0.5, 3.0], 3, scale_format="nvfp4"), "NV-wide"),
    (PackedNA([0.0, 0.5, 1.0, 6.0], 5, scale_format="nvfp4"), "NV-e2m1"),
]


def _nv_setup(fmt, N, K, seed):
    torch.manual_seed(seed)
    w = torch.randn(N, K, device="cuda") * 0.1
    codes, sb, ts = fmt.quantize_scaled(w)
    return codes, sb, ts


@needs_gpu
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("fmt,label", NV_CASES)
@pytest.mark.parametrize("N,K", NK)
@pytest.mark.parametrize("M", MS)
def test_nvfp4_matches_dense_decode(fmt, label, M, N, K, mode):
    codes, sb, ts = _nv_setup(fmt, N, K, 0)
    x = torch.randn(M, K, device="cuda")
    W = fmt.dequantize(codes, sb, ts)                    # includes the tensor scale
    out = packed_matmul(x, codes, sb, fmt.code_to_value.to("cuda"), compute_dtype=mode,
                        block=16, scale_format="nvfp4", tensor_scale=ts)
    _check(out, x, W.t().contiguous(), mode)


@needs_gpu
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("fmt,label", NV_CASES)
@pytest.mark.parametrize("N,K", NK)
@pytest.mark.parametrize("M", MS)
def test_nvfp4_grad_x_matches_dense_decode(fmt, label, M, N, K, mode):
    codes, sb, ts = _nv_setup(fmt, N, K, 2)
    go = torch.randn(M, N, device="cuda")
    W = fmt.dequantize(codes, sb, ts)
    out = packed_matmul_dx(go, codes, sb, fmt.code_to_value.to("cuda"), compute_dtype=mode,
                           scale_format="nvfp4", tensor_scale=ts)
    _check(out, go, W, mode)


@needs_gpu
@pytest.mark.parametrize("fmt,label", NV_CASES)
def test_nvfp4_every_e4m3_byte_decodes_exactly(fmt, label):
    """All 126 positive E4M3 bytes through the kernel: one-hot x reads the weight tile back
    (tensor scale 1 so the tile itself is compared, bit-exact in bf16)."""
    N, K = 32, 128
    torch.manual_seed(5)
    codes = torch.randint(0, fmt.n_codes, (N, K), device="cuda", dtype=torch.uint8)
    bytes_ = torch.tensor([b for b in range(128) if b != 0x7F], device="cuda", dtype=torch.uint8)
    sb = bytes_[torch.arange(N * (K // 16), device="cuda") % bytes_.numel()].reshape(N, K // 16).contiguous()
    ts = torch.tensor(1.0, device="cuda")
    out = packed_matmul(torch.eye(K, device="cuda"), codes, sb, fmt.code_to_value.to("cuda"),
                        compute_dtype="bf16", scale_format="nvfp4", tensor_scale=ts)
    torch.testing.assert_close(out, fmt.dequantize(codes, sb, ts).t().contiguous(), atol=0.0, rtol=0.0)


@needs_gpu
@pytest.mark.parametrize("fmt,label", NV_CASES)
def test_nvfp4_tensor_scale_is_applied_in_fp32_on_the_accumulator(fmt, label):
    """out(ts) == ts * out(1) up to ONE fp32 rounding, which a bf16-rounded weight tile
    (ts folded into the tile) could not satisfy."""
    codes, sb, _ = _nv_setup(fmt, 64, 128, 6)
    x = torch.randn(16, 128, device="cuda")
    lut = fmt.code_to_value.to("cuda")
    ts = torch.tensor(0.7391, device="cuda")
    one = torch.tensor(1.0, device="cuda")
    a = packed_matmul(x, codes, sb, lut, scale_format="nvfp4", tensor_scale=ts)
    b = packed_matmul(x, codes, sb, lut, scale_format="nvfp4", tensor_scale=one)
    torch.testing.assert_close(a, b * ts, atol=0.0, rtol=2e-7)


@needs_gpu
def test_nvfp4_requires_tensor_scale_and_block16():
    fmt = NV_CASES[0][0]
    x = torch.zeros(4, 64, device="cuda")
    codes = torch.zeros(8, 64, dtype=torch.uint8, device="cuda")
    sb = torch.full((8, 4), 56, dtype=torch.uint8, device="cuda")
    lut = fmt.code_to_value.to("cuda")
    with pytest.raises(ValueError):
        packed_matmul(x, codes, sb, lut, scale_format="nvfp4")
    with pytest.raises(AssertionError):
        packed_matmul(x, codes, sb, lut, block=32, scale_format="nvfp4",
                      tensor_scale=torch.tensor(1.0))


@pytest.mark.parametrize("ladder", [[0.5, 3.0], [0.5, 1.0, 6.0], [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]])
def test_ladder_times_e4m3_scale_is_exact_in_bf16(ladder):
    """CPU witness of the bf16 claim: value * e4m3(byte) survives bf16 for EVERY byte,
    while value * (e4m3 * arbitrary fp32 tensor scale) does not (hence the epilogue)."""
    vals = torch.tensor([-v for v in ladder[::-1]] + [0.0] + ladder, dtype=torch.float64)
    bytes_ = torch.tensor([b for b in range(128) if b != 0x7F], dtype=torch.uint8)
    sc = e4m3_decode(bytes_).to(torch.float64)
    w64 = vals[:, None] * sc[None, :]
    w32 = w64.to(torch.float32)
    assert torch.equal(w32.to(torch.float64), w64)
    assert torch.equal(w32.to(torch.bfloat16).to(torch.float32), w32)
    folded = (w64 * 0.7391).to(torch.float32)
    assert not torch.equal(folded.to(torch.bfloat16).to(torch.float32), folded)