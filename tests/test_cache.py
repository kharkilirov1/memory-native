"""Derived visible-weight compute cache (acceleration memo M5).

cache_mode keeps the visible ternary T in a GEMM-friendly dtype (fp16/int8) so forward/grad_x
never pay the 6-bit decode tax. T is the source of truth's *derived view*: forward through the
cache must equal forward through the decode (T = -1/0/1 is exact in fp16 and int8), and the cache
must track the state as visible weights flip during updates.
"""
import math

import pytest
import torch

from memory_native import RMSCounterLinear
from memory_native.counter import decode_state
from memory_native.packed import PackedRMSCounterLinear, unpack_codes


@pytest.mark.parametrize("cache_mode", ["fp16", "int8"])
def test_cache_forward_matches_decode(cache_mode):
    torch.manual_seed(0)
    ref = PackedRMSCounterLinear(64, 48, C=11)
    cached = PackedRMSCounterLinear(64, 48, C=11, cache_mode=cache_mode)
    cached.state.copy_(ref.state)
    cached.scale.copy_(ref.scale)
    cached._build_t_cache()
    x = torch.randn(8, 5, 64)
    with torch.no_grad():
        assert torch.equal(ref(x), cached(x))   # T is exact in fp16/int8 -> bit-exact forward


@pytest.mark.parametrize("cache_mode", ["fp16", "int8"])
def test_cache_tracks_state_through_updates(cache_mode):
    torch.manual_seed(0)
    lay = PackedRMSCounterLinear(64, 48, C=11, cache_mode=cache_mode).train()
    x = torch.randn(16, 64)
    for _ in range(10):
        lay(x).pow(2).mean().backward()         # self-updates flip visible weights
    t, _ = decode_state(unpack_codes(lay.state, 64), 11)   # truth T from the packed state
    assert torch.equal(lay._t_cache.to(torch.int16), t)    # cache stayed in sync


def test_cache_layer_still_recovers_teacher():
    torch.manual_seed(0)
    n, N, C = 24, 256, 11
    ts = 0.25
    base = math.sqrt(3.0 / (2.0 * n))
    tw = torch.randint(-1, 2, (n, n)).float()
    x = torch.randn(N, n)
    y = x @ (ts * tw).t()
    lay = RMSCounterLinear(n, n, C=C, lr=0.02, lr_scale=2e-4, init_gain=ts / base,
                           cache_mode="int8").train()
    for _ in range(500):
        (lay(x) - y).pow(2).mean().backward()
    with torch.no_grad():
        assert (lay(x) - y).pow(2).mean().item() < 0.02
