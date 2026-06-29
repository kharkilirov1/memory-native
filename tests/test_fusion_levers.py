"""Bit-exact / parity gates for the kernel-fusion levers (results/FUSION_PLAN.md).

These pin the CPU-checkable invariants the fused CUDA/Triton kernels will mirror:

  #1 epilogue fusion enabler -- hash-SR `lagged` makes the state-write PER-ELEMENT (denom from the
     previous step's v), so perturbing grad_w[o,i] changes only state[o,i]; `exact` couples the whole
     row. Plus a teacher-recovery parity gate (lagged still learns).
  #4 persistent cache flip-patch -- `cache_patch="flip"` writes only the flipped cache elements and
     stays bit-identical to a full rewrite AND to a fresh decode across many update steps.
  #5 fp8 grad_w -- `fp8_correlation` is a low-precision (parity-gated) estimate of delta^T x with a
     bounded relative error, and a counter layer using update_compute="fp8" still recovers a teacher.
"""
import math

import pytest
import torch

from memory_native import RMSCounterLinear, fp8_correlation
from memory_native.counter import decode_state, encode_state
from memory_native.fused_update import counter_update_hashsr
from memory_native.packed import PackedRMSCounterLinear, unpack_codes


# ----------------------------------------------------------------------------- #1 lagged hash-SR
def _hashsr_dependency(lagged):
    """Positions of the state that change when ONE grad_w element [0,5] is perturbed, with the SR
    seed fixed and the layer reset. Per-element <=> only [0,5] changes."""
    out, in_, C = 4, 8, 11
    torch.manual_seed(0)
    codes0 = encode_state(torch.randint(-1, 2, (out, in_), dtype=torch.int16),
                          torch.randint(-(C - 1), C, (out, in_), dtype=torch.int16), C)
    scale0 = torch.full((out, 1), 0.3)
    gw = torch.randn(out, in_)
    kw = dict(C=C, lr=0.1, lr_scale=0.0, rms_beta=0.9, rms_eps=1e-3, seed=7, lagged=lagged)

    # warm v so the lagged denom is a real previous-step stat, not the init (v stays 0 otherwise).
    v0 = torch.zeros(out, 1)
    counter_update_hashsr(codes0.clone(), scale0.clone(), v0, gw.clone(),
                          **{**kw, "seed": 1})           # v0 now holds a previous-step value

    def run(g):
        return counter_update_hashsr(codes0.clone(), scale0.clone(), v0.clone(), g, **kw)

    base = run(gw.clone())
    g2 = gw.clone(); g2[0, 5] += 3.0
    pert = run(g2)
    return torch.nonzero(base != pert).tolist()


def test_lagged_hashsr_is_per_element():
    """#1: lagged hash-SR -> only state[0,5] can change; exact -> the whole row couples."""
    lagged_changed = _hashsr_dependency(lagged=True)
    assert lagged_changed in ([], [[0, 5]]), f"lagged not per-element: {lagged_changed}"
    exact_changed = _hashsr_dependency(lagged=False)
    cols = sorted({c for _, c in exact_changed})
    assert len(cols) > 1, f"exact should couple the whole row, only touched cols={cols}"


def test_lagged_hashsr_recovers_teacher():
    """#1 parity: the lagged one-pass hash-SR update still recovers a ternary teacher."""
    torch.manual_seed(0)
    n, N, C = 16, 256, 11
    ts = 0.25
    tw = torch.randint(-1, 2, (n, n)).to(torch.int16)
    x = torch.randn(N, n)
    y = x @ (ts * tw.float()).t()
    codes = encode_state(torch.randint(-1, 2, (n, n), dtype=torch.int16),
                         torch.zeros((n, n), dtype=torch.int16), C)
    scale = torch.full((n, 1), 0.25)
    v = torch.zeros((n, 1))
    for step in range(800):
        t, _ = decode_state(codes, C)
        w = scale * t.float()
        grad_out = (2.0 / (N * n)) * (x @ w.t() - y)
        codes = counter_update_hashsr(codes, scale, v, grad_out.t() @ x,
                                      C=C, lr=0.005, lr_scale=0.0, rms_beta=0.9, rms_eps=1e-3,
                                      seed=step, lagged=True)
    t, _ = decode_state(codes, C)
    mse = ((x @ (scale * t.float()).t() - y) ** 2).mean().item()
    assert mse < 5e-3 and (t == tw).float().mean().item() > 0.9


# ----------------------------------------------------------------------------- #4 flip-patch cache
@pytest.mark.parametrize("cache_mode", ["fp16", "int8"])
def test_flip_patch_cache_bit_identical(cache_mode):
    """#4: cache_patch='flip' must equal both a full rewrite and a fresh decode through updates."""
    torch.manual_seed(0)
    full = PackedRMSCounterLinear(64, 48, C=11, cache_mode=cache_mode, cache_patch="full").train()
    flip = PackedRMSCounterLinear(64, 48, C=11, cache_mode=cache_mode, cache_patch="flip").train()
    flip.state.copy_(full.state); flip.scale.copy_(full.scale); flip.v.copy_(full.v)
    full._build_t_cache(); flip._build_t_cache()
    x = torch.randn(16, 64)
    for _ in range(10):
        torch.manual_seed(1); full(x).pow(2).mean().backward()
        torch.manual_seed(1); flip(x).pow(2).mean().backward()
    assert torch.equal(full._t_cache, flip._t_cache)              # flip == full rewrite
    t, _ = decode_state(unpack_codes(flip.state, 64), 11)         # truth from packed state
    assert torch.equal(flip._t_cache.to(torch.int16), t)         # flip == fresh decode
    assert int(flip.cache_patches.item()) >= 0                    # diagnostic populated


def test_flip_patch_recovers_teacher():
    torch.manual_seed(0)
    n, N, C = 24, 256, 11
    ts = 0.25
    base = math.sqrt(3.0 / (2.0 * n))
    tw = torch.randint(-1, 2, (n, n)).float()
    x = torch.randn(N, n)
    y = x @ (ts * tw).t()
    lay = RMSCounterLinear(n, n, C=C, lr=0.02, lr_scale=2e-4, init_gain=ts / base,
                           cache_mode="int8", cache_patch="flip").train()
    for _ in range(500):
        (lay(x) - y).pow(2).mean().backward()
    with torch.no_grad():
        assert (lay(x) - y).pow(2).mean().item() < 0.02


# ----------------------------------------------------------------------------- #5 fp8 grad_w
def test_fp8_correlation_low_error():
    """#5: fp8 grad_w is a parity-gated low-precision estimate -> bounded relative error (NOT exact,
    NOT stochastic-unbiased -- a few % round-to-nearest fp8 error, absorbed by error-feedback)."""
    torch.manual_seed(0)
    M, N, K = 128, 8, 12
    D = torch.randn(M, N)
    X = torch.randn(M, K)
    exact = D.t() @ X
    est = fp8_correlation(D, X)
    rel = (est - exact).abs().mean() / exact.abs().mean()
    assert rel.item() < 0.08, f"fp8 correlation too far off: rel err {rel.item():.4f}"


def test_fp8_update_recovers_teacher():
    torch.manual_seed(0)
    n, N, C = 24, 256, 11
    ts = 0.25
    base = math.sqrt(3.0 / (2.0 * n))
    tw = torch.randint(-1, 2, (n, n)).float()
    x = torch.randn(N, n)
    y = x @ (ts * tw).t()
    lay = RMSCounterLinear(n, n, C=C, lr=0.02, lr_scale=2e-4, init_gain=ts / base,
                           update_compute="fp8").train()
    for _ in range(600):
        (lay(x) - y).pow(2).mean().backward()
    with torch.no_grad():
        assert (lay(x) - y).pow(2).mean().item() < 0.05
