"""Parity tests for the fused decode+matmul Triton kernel.

The kernel had no tests at all when it was committed. It has no CPU path, so
these skip unless Triton and a CUDA device are present; on the CPU box the
skip is the honest witness (it is not a pass).

Tolerance is tight (1e-4) because ``packed_matmul`` requests
``input_precision="ieee"``: Triton's default is TF32 on Ampere+, which would
silently disagree with the fp32 torch reference by ~1e-3.
"""
from __future__ import annotations

import pytest
import torch

from mx6.packed_na import PackedNA5, PackedNA6
from mx6.triton_kernel import HAS_TRITON, packed_matmul, packed_matmul_dx

pytestmark = pytest.mark.skipif(
    not HAS_TRITON or not torch.cuda.is_available(),
    reason="fused kernel needs Triton + CUDA")

CASES = [(PackedNA5, "PackedNA5"), (PackedNA6, "PackedNA6")]


@pytest.mark.parametrize("fmt,label", CASES)
@pytest.mark.parametrize("M,N,K", [(32, 64, 128), (64, 256, 512), (1, 128, 256)])
def test_matches_dense_decode(fmt, label, M, N, K):
    """out = x @ decode(codes, scale).T must equal the torch reference."""
    torch.manual_seed(0)
    w = torch.randn(N, K, device="cuda") * 0.1
    codes, sb = fmt.quantize(w, headroom=2.0)
    x = torch.randn(M, K, device="cuda")

    ref = x @ fmt.dequantize(codes, sb).t()
    out = packed_matmul(x, codes, sb, fmt.code_to_value.to("cuda"))

    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("fmt,label", CASES)
def test_accumulator_field_does_not_change_the_matmul(fmt, label):
    """The forward must read only the value; the accumulator is invisible."""
    torch.manual_seed(1)
    K, N, M = 128, 64, 16
    w = torch.randn(N, K, device="cuda") * 0.1
    codes, sb = fmt.quantize(w, headroom=2.0)
    x = torch.randn(M, K, device="cuda")

    base = packed_matmul(x, codes, sb, fmt.code_to_value.to("cuda"))

    # flip the accumulator residue to a different level, keep the magnitude
    mag = codes.to(torch.long) // fmt.acc_levels
    other_acc = (fmt.C - 1) + 1 if fmt.C > 1 else fmt.C - 1
    shifted = (mag * fmt.acc_levels + other_acc).to(torch.uint8)

    again = packed_matmul(x, shifted, sb, fmt.code_to_value.to("cuda"))
    torch.testing.assert_close(again, base, atol=0.0, rtol=0.0)


def test_rejects_shape_mismatch():
    fmt = PackedNA6
    x = torch.zeros(4, 96, device="cuda")
    codes = torch.zeros(8, 128, dtype=torch.uint8, device="cuda")
    sb = torch.zeros(8, 4, dtype=torch.uint8, device="cuda")
    with pytest.raises(AssertionError):
        packed_matmul(x, codes, sb, fmt.code_to_value.to("cuda"))


@pytest.mark.parametrize("fmt,label", CASES)
@pytest.mark.parametrize("M,N,K", [(32, 64, 128), (64, 256, 512), (4, 128, 256)])
def test_grad_x_matches_dense_decode(fmt, label, M, N, K):
    """grad_x = go @ decode(codes, scale) must equal the torch reference."""
    torch.manual_seed(2)
    w = torch.randn(N, K, device="cuda") * 0.1
    codes, sb = fmt.quantize(w, headroom=2.0)
    go = torch.randn(M, N, device="cuda")

    ref = go @ fmt.dequantize(codes, sb)
    out = packed_matmul_dx(go, codes, sb, fmt.code_to_value.to("cuda"))

    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
