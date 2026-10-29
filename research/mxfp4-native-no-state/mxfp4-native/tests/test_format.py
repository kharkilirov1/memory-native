"""Invariants of the bit layout (docs/FORMAT.md, section "Invariants")."""
from __future__ import annotations

import math

import pytest
import torch

from mx6 import pack


def slow_quantize(w: torch.Tensor) -> torch.Tensor:
    """Element-by-element MXFP4 rounding, written without any of `pack`'s tricks."""
    out_features, in_features = w.shape
    result = torch.zeros_like(w, dtype=torch.float32)
    for r in range(out_features):
        for b in range(in_features // pack.BLOCK):
            block = w[r, b * pack.BLOCK:(b + 1) * pack.BLOCK]
            amax = max(float(block.abs().max()), 1e-30)
            mantissa, exponent = math.frexp(amax / pack.MAX_LEVEL)
            scale = 2.0 ** (exponent - 1 if mantissa == 0.5 else exponent)
            for i, value in enumerate(block.tolist()):
                magnitude = min(abs(value) / scale, pack.MAX_LEVEL)
                nearest = min(pack.LEVELS.tolist(), key=lambda level: abs(level - magnitude))
                result[r, b * pack.BLOCK + i] = math.copysign(nearest * scale, value)
    return result


@pytest.mark.parametrize("shape", [(16, 32), (8, 64), (5, 96)])
def test_roundtrip_matches_elementwise_reference(shape):
    torch.manual_seed(0)
    w = torch.randn(*shape) * 0.05
    codes, scale = pack.quantize(w)
    assert torch.equal(pack.dequantize(codes, scale), slow_quantize(w))


def test_pack_unpack_is_bit_exact():
    torch.manual_seed(1)
    out_features, in_features = 12, 64
    codes = torch.randint(0, 16, (out_features, in_features), dtype=torch.uint8)
    residual = torch.randint(0, 4, (out_features, in_features), dtype=torch.uint8)
    scale = torch.randint(100, 140, (out_features, in_features // pack.BLOCK), dtype=torch.uint8)

    blob = pack.pack(codes, residual, scale)
    codes2, residual2, scale2 = pack.unpack(blob, out_features, in_features)

    assert torch.equal(codes, codes2)
    assert torch.equal(residual, residual2)
    assert torch.equal(scale, scale2)


def test_size_accounting_matches_the_blob():
    out_features, in_features = 1536, 1536
    layer_bytes = pack.persistent_bytes(out_features, in_features)
    blob = torch.zeros(layer_bytes, dtype=torch.uint8)
    codes, residual, scale = pack.unpack(blob, out_features, in_features)
    assert blob.numel() == out_features * (in_features // pack.BLOCK) * 25
    assert codes.shape == (out_features, in_features)
    assert scale.shape == (out_features, in_features // pack.BLOCK)
    # 6.25 bits per parameter
    assert abs(layer_bytes / (out_features * in_features) - 0.78125) < 1e-12
    assert abs(pack.deployed_bytes(out_features, in_features)
               / (out_features * in_features) - 0.53125) < 1e-12


def test_scale_is_recomputed_exactly_across_a_power_of_two():
    torch.manual_seed(2)
    w = torch.randn(8, 64) * 0.03
    codes, scale = pack.quantize(w)
    codes2, scale2 = pack.quantize(2.0 * w)
    assert torch.equal(codes, codes2), "value codes must not change"
    assert torch.equal(scale2.to(torch.int16), scale.to(torch.int16) + 1), "scale byte must step by 1"
    assert torch.allclose(pack.dequantize(codes2, scale2), 2.0 * pack.dequantize(codes, scale))


def test_no_negative_zero_is_stored():
    w = torch.zeros(4, 32)
    codes, scale = pack.quantize(w)
    assert torch.all(codes == 0)


def test_non_multiple_of_32_is_rejected():
    with pytest.raises(ValueError):
        pack.persistent_bytes(8, 48)

