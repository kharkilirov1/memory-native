"""Regression tests for the acceleration review fixes.

1. proxy RMS used mean over tokens instead of sum -> denominator was ~1/sqrt(M) too small (the
   dynamics drifted with batch/seq length). The fixed proxy must match the exact row second
   moment across M.
2. int8 forward needs a per-TOKEN (row) scale to factor out of X T^T; the per-column scale (right
   for the update correlation) is wrong for forward. int8_forward_ternary must be unbiased.
3. the CUDA fused update mutates packed state directly, bypassing the cache refresh -> the derived
   T cache must be rebuilt so forward never reads a stale T (CUDA-only).
"""
import torch

from memory_native import int8_forward_ternary
from memory_native.counter import decode_state
from memory_native.packed import PackedRMSCounterLinear, unpack_codes
from memory_native.fused_update import HAS_TRITON

CUDA = torch.cuda.is_available()


def test_proxy_rms_scale_matches_exact_across_M():
    """Fixed proxy (sum over tokens) tracks the exact row RMS regardless of M; the old mean drifts
    by ~1/sqrt(M)."""
    torch.manual_seed(0)
    for M in (16, 128, 512):
        D = torch.randn(M, 8)
        X = torch.randn(M, 12)
        G = D.t() @ X
        exact = G.pow(2).mean(dim=1)                       # (1/K)||G_o||^2
        proxy = D.pow(2).sum(dim=0) * X.pow(2).mean()      # fixed proxy
        ratio = (proxy / exact).mean().item()
        assert 0.5 < ratio < 2.0, f"M={M}: proxy/exact={ratio:.3f}"   # O(1), no M-drift
        old = D.pow(2).mean(dim=0) * X.pow(2).mean()       # the buggy version
        assert (old / exact).mean().item() < 0.3           # demonstrably wrong (shrinks ~1/M)


def test_int8_forward_ternary_unbiased():
    torch.manual_seed(0)
    X = torch.randn(64, 32)
    T = torch.randint(-1, 2, (8, 32), dtype=torch.int8)
    exact = X @ T.float().t()
    est = torch.stack([int8_forward_ternary(X, T) for _ in range(400)]).mean(0)
    rel = (est - exact).abs().mean() / exact.abs().mean()
    assert rel.item() < 0.03, f"int8 forward biased: rel {rel.item():.4f}"


@torch.no_grad()
def _decoded_t(lay):
    t, _ = decode_state(unpack_codes(lay.state, lay.in_features), lay.C)
    return t


import pytest


@pytest.mark.skipif(not (CUDA and HAS_TRITON), reason="needs CUDA + triton (fused update)")
def test_cache_consistent_after_fused_update():
    torch.manual_seed(0)
    lay = PackedRMSCounterLinear(64, 48, C=11, cache_mode="int8").cuda().train()
    x = torch.randn(16, 64, device="cuda")
    for _ in range(5):
        lay(x).pow(2).mean().backward()        # fires the CUDA fused update
    assert torch.equal(lay._t_cache.to(torch.int16), _decoded_t(lay))
