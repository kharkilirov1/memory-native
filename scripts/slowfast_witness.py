"""Slow-fast low-rank residual witness (method M3).

Teacher-recovery arms, same steps / data / dims:

    counter_rms (exact every-step baseline)
    counter_slowfast  r in {8,16,32}  x  merge in {8,16,32}

Prints, per arm: final teacher MSE, the val-GAP vs the exact counter (%), how many full base
correlation (Delta^T X) steps were spent vs the exact arm (merge cadence => ~Kx fewer), and
whether merge steps spike the loss (loss right after each merge vs the step just before it).

    python scripts/slowfast_witness.py
"""
from __future__ import annotations

import math

import torch

from memory_native import RMSCounterLinear, SlowFastCounterLinear


def teacher(n, N, ts=0.25, seed=1):
    g = torch.Generator().manual_seed(seed)
    tw = torch.randint(-1, 2, (n, n), generator=g).float()
    x = torch.randn(N, n, generator=g)
    y = x @ (ts * tw).t()
    return x, y


def run_exact(x, y, n, C, steps, init_gain):
    torch.manual_seed(0)
    lay = RMSCounterLinear(n, n, C=C, lr=0.02, lr_scale=2e-4, init_gain=init_gain).train()
    for _ in range(steps):
        (lay(x) - y).pow(2).mean().backward()
    with torch.no_grad():
        mse = (lay(x) - y).pow(2).mean().item()
    # the exact counter runs a full Delta^T X every step.
    return mse, steps


def run_slowfast(x, y, n, C, steps, init_gain, rank, merge_every, fast_lr=0.05):
    torch.manual_seed(0)
    lay = SlowFastCounterLinear(n, n, rank=rank, merge_every=merge_every, C=C, lr=0.02,
                                lr_scale=2e-4, init_gain=init_gain).train()
    opt = torch.optim.SGD(lay.fast_parameters(), lr=fast_lr)
    worst_merge_jump = float("-inf")
    for step in range(1, steps + 1):
        opt.zero_grad()
        loss = (lay(x) - y).pow(2).mean()      # computed on pre-update params
        loss.backward()
        opt.step()                              # SGD on A,B
        with torch.no_grad():
            pre = (lay(x) - y).pow(2).mean().item()      # post-step, pre-merge
        if lay.flush_merge():                            # deterministically fold A@B^T into base
            with torch.no_grad():
                post = (lay(x) - y).pow(2).mean().item()  # post-merge
            worst_merge_jump = max(worst_merge_jump, post - pre)
    with torch.no_grad():
        mse = (lay(x) - y).pow(2).mean().item()
    base_corr = int(lay.merge_count)            # base full-correlation steps == number of merges
    return mse, base_corr, worst_merge_jump


def main():
    n, N, C, steps = 24, 256, 11, 600
    ts = 0.25
    init_gain = ts / math.sqrt(3.0 / (2.0 * n))
    x, y = teacher(n, N, ts=ts)

    print("=== M3 slow-fast low-rank residual: teacher recovery ===")
    print(f"  dims n={n}  tokens N={N}  C={C}  steps={steps}\n")

    mse_exact, exact_corr = run_exact(x, y, n, C, steps, init_gain)
    print(f"  counter_rms (exact)          MSE {mse_exact:.5f}   "
          f"full Delta^T X steps = {exact_corr}\n")

    header = (f"  {'arm':28s} {'MSE':>10s} {'gap%':>8s} "
              f"{'baseCorr':>9s} {'corr-redux':>11s} {'maxMergeJump':>13s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    gates = []
    for rank in (8, 16, 32):
        for merge_every in (8, 16, 32):
            mse, base_corr, jump = run_slowfast(x, y, n, C, steps, init_gain, rank, merge_every)
            gap = 100.0 * (mse - mse_exact) / max(mse_exact, 1e-12)
            redux = exact_corr / max(base_corr, 1)
            name = f"slowfast r={rank} K={merge_every}"
            print(f"  {name:28s} {mse:10.5f} {gap:8.1f} "
                  f"{base_corr:9d} {redux:10.1f}x {jump:13.5f}")
            gates.append((name, rank, merge_every, mse, gap, base_corr, redux, jump))

    print("\n=== GATES ===")
    # focus the verdict on the K=8 arms (the spec's >=8x correlation-reduction target).
    for name, rank, merge_every, mse, gap, base_corr, redux, jump in gates:
        if merge_every != 8:
            continue
        g_gap = "PASS" if gap <= 2.0 else ("WARN" if gap <= 10.0 else "FAIL")
        g_redux = "PASS" if redux >= 8.0 else "FAIL"
        g_stab = "PASS" if jump < 0.05 else "FAIL"
        print(f"  {name:22s} val-gap {gap:6.1f}% [{g_gap}]   "
              f"corr-redux {redux:4.1f}x [{g_redux}]   "
              f"max-merge-jump {jump:.5f} [{g_stab}]")


if __name__ == "__main__":
    main()
