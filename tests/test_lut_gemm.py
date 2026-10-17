"""LUT-GEMM correctness: the lookup path must reproduce the exact ternary matmul.

The reference (torch) path is CPU-tested here; the Triton kernel is compared against that
reference when a CUDA device is present. Exactness is the point: LUT-GEMM only changes how
the sum is computed, never the arithmetic it represents.
"""
import pytest

torch = pytest.importorskip("torch")

from memory_native.counter import decode_state
from memory_native.lut_gemm import (
    LUT_ENTRIES,
    TERNARY_PER_BLOCK,
    build_lut_blocks,
    lut_group_matmul,
    triton_lut_group_matmul,
)
from memory_native.packed import pack_codes, unpack_codes


def _packed_state(n: int, k: int, C: int = 11, seed: int = 0) -> torch.Tensor:
    """Random but VALID packed codes (encode = (t+1)(2C-1) + (c+C-1))."""
    torch.manual_seed(seed)
    levels = 3 * (2 * C - 1)
    return pack_codes(torch.randint(0, levels, (n, k), dtype=torch.uint8))


def _exact_weight(codes: torch.Tensor, scale: torch.Tensor, k: int, group: int, C: int):
    t, _c = decode_state(unpack_codes(codes, k), C)
    return scale.repeat_interleave(group, dim=1)[:, :k] * t.float()


def test_build_lut_matches_manual_sums():
    x = torch.randn(1, 4 * 3)
    lut = build_lut_blocks(x)
    assert lut.shape == (1, 3, LUT_ENTRIES)
    for p in range(3):
        for idx in range(LUT_ENTRIES):
            digits = [(idx // 3 ** km) % 3 - 1 for km in range(TERNARY_PER_BLOCK)]
            expect = sum(d * float(x[0, 4 * p + km]) for km, d in enumerate(digits))
            assert abs(float(lut[0, p, idx]) - expect) < 1e-6


def test_lut_matmul_matches_exact_ternary_matmul():
    n, k, group, C = 8, 64, 32, 11
    codes = _packed_state(n, k, C)
    scale = torch.full((n, k // group), 0.05)
    perm = torch.arange(k)
    x = torch.randn(2, k)
    got = lut_group_matmul(x, codes, scale, perm, group=group, C=C)
    want = x @ _exact_weight(codes, scale, k, group, C).t()
    assert torch.allclose(got, want, atol=1e-4, rtol=1e-4), (got - want).abs().max()


def test_lut_matmul_honours_the_act_order_permutation():
    n, k, group, C = 6, 64, 32, 11
    torch.manual_seed(3)
    perm = torch.randperm(k)
    codes = _packed_state(n, k, C, seed=5)          # act-order state
    scale = torch.full((n, k // group), 0.03)
    x = torch.randn(2, k)                            # ORIGINAL column order
    got = lut_group_matmul(x, codes, scale, perm, group=group, C=C)
    w_perm = _exact_weight(codes, scale, k, group, C)   # act-order weight
    w = torch.empty_like(w_perm)
    w[:, perm] = w_perm                              # original order, like visible_weight()
    assert torch.allclose(got, x @ w.t(), atol=1e-4, rtol=1e-4)


def test_lut_matmul_is_a_drop_in_for_visible_weight():
    from memory_native.group_scale_packed import PackedGroupScaleCounterLinear

    torch.manual_seed(0)
    layer = PackedGroupScaleCounterLinear(64, 8, group=32, C=11, kernel_mode="gemm")
    x = torch.randn(1, 64)
    got = lut_group_matmul(x, layer.state, layer.scale, layer.perm, group=32, C=11)
    want = layer.visible_weight() @ x[0]
    assert torch.allclose(got[0], want, atol=1e-4, rtol=1e-4)


def test_triton_kernel_matches_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the LUT kernel")
    n, k, group, C = 128, 512, 128, 11
    torch.manual_seed(1)
    codes = _packed_state(n, k, C).cuda()
    scale = torch.full((n, k // group), 0.02, device="cuda")
    perm = torch.randperm(k, device="cuda")
    x = torch.randn(1, k, device="cuda")
    want = lut_group_matmul(x.cpu(), codes.cpu(), scale.cpu(), perm.cpu(), group=group, C=C)[0]
    got = triton_lut_group_matmul(x, codes, scale, perm, group=group, C=C)
    assert torch.allclose(got.cpu(), want, atol=2e-4, rtol=2e-4), (got.cpu() - want).abs().max()
