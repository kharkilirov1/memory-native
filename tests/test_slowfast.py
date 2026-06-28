"""Tests for SlowFastCounterLinear (method M3).

(a) parity: rank==0 (no fast path) reduces to the base RMSCounterLinear, step-for-step.
(b) recovery: slow-fast recovers a ternary teacher within ~2x the exact counter's MSE.
(c) stability: a merge step does not increase the loss beyond a small tolerance.
"""
import math

import torch

from memory_native import RMSCounterLinear, SlowFastCounterLinear


def _teacher(n, N, ts=0.25, seed=0):
    g = torch.Generator().manual_seed(seed)
    tw = torch.randint(-1, 2, (n, n), generator=g).float()
    x = torch.randn(N, n, generator=g)
    y = x @ (ts * tw).t()
    return x, y


def test_rank0_parity_with_base():
    """rank=0 -> no A,B, base updates every step -> bit-identical to a standalone RMSCounterLinear."""
    n, N, C = 16, 128, 11
    x, _ = _teacher(n, N)
    target = torch.randn(N, n, generator=torch.Generator().manual_seed(3))

    torch.manual_seed(7)
    ref = RMSCounterLinear(n, n, C=C, lr=0.02, lr_scale=2e-4).train()
    torch.manual_seed(7)
    sf = SlowFastCounterLinear(n, n, rank=0, C=C, lr=0.02, lr_scale=2e-4).train()

    assert sf.fast_parameters() == []
    for _ in range(50):
        torch.manual_seed(100)
        (ref(x) - target).pow(2).mean().backward()
        torch.manual_seed(100)
        (sf(x) - target).pow(2).mean().backward()

    # states and scales must match exactly (same update path, same SR seeds).
    assert torch.equal(ref.state, sf.base.state), "rank=0 state diverged from base"
    assert torch.allclose(ref.scale, sf.base.scale), "rank=0 scale diverged from base"


def test_teacher_recovery_within_2x_of_exact():
    """slow-fast recovers a teacher to MSE within ~2x of the exact every-step counter."""
    n, N, C, steps = 24, 256, 11, 600
    ts = 0.25
    base_scale = math.sqrt(3.0 / (2.0 * n))
    x, y = _teacher(n, N, ts=ts, seed=1)

    torch.manual_seed(0)
    exact = RMSCounterLinear(n, n, C=C, lr=0.02, lr_scale=2e-4, init_gain=ts / base_scale).train()
    for _ in range(steps):
        (exact(x) - y).pow(2).mean().backward()
    with torch.no_grad():
        mse_exact = (exact(x) - y).pow(2).mean().item()

    torch.manual_seed(0)
    sf = SlowFastCounterLinear(n, n, rank=16, merge_every=16, C=C, lr=0.02, lr_scale=2e-4,
                               init_gain=ts / base_scale, fast_init=0.2).train()
    opt = torch.optim.SGD(sf.fast_parameters(), lr=2.0)
    for _ in range(steps):
        opt.zero_grad()
        loss = (sf(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        mse_sf = (sf(x) - y).pow(2).mean().item()

    assert mse_sf < max(2.0 * mse_exact, 5e-3), \
        f"slow-fast mse {mse_sf:.5f} not within 2x of exact {mse_exact:.5f}"


def test_merge_does_not_spike_loss():
    """A merge step must not increase the loss by more than a small tolerance vs just before it."""
    n, N, C = 24, 256, 11
    ts = 0.25
    base_scale = math.sqrt(3.0 / (2.0 * n))
    x, y = _teacher(n, N, ts=ts, seed=2)

    torch.manual_seed(0)
    K = 8
    sf = SlowFastCounterLinear(n, n, rank=16, merge_every=K, C=C, lr=0.02, lr_scale=2e-4,
                               init_gain=ts / base_scale, fast_init=0.2).train()
    opt = torch.optim.SGD(sf.fast_parameters(), lr=2.0)

    worst = 0.0
    for step in range(1, 401):
        opt.zero_grad()
        loss = (sf(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            pre = (sf(x) - y).pow(2).mean().item()      # post-step, pre-merge loss
        if sf.flush_merge():                            # fold the residual into the base now
            with torch.no_grad():
                post = (sf(x) - y).pow(2).mean().item()  # post-merge loss
            worst = max(worst, post - pre)
    # the merge re-encodes s*T+A@B^T into the base; it may perturb slightly but must not blow up.
    assert worst < 0.05, f"a merge spiked the loss by {worst:.4f}"
