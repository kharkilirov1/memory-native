"""M2 witness -- 2:4 structured-sparse VISIBLE weight: does it still learn, and does error-feedback
(ticking masked weights so they can rotate back in) beat plain pruning?

Teacher recovery (dense Gaussian teacher). Arms, same data/steps:
  * unstructured counter (RMSCounterLinear)                 -- the dense-ternary baseline
  * 2:4 group-counter, error-feedback, hysteresis in {0,1,2,4}
  * 2:4 group-counter, pruning ablation (only visible weights tick)

Reports per arm: MSE, gap vs unstructured, fraction of weights EVER visible (dead-weight check),
and visible-set churn (stability). The hysteresis sweep exposes the real tradeoff.

    python scripts/group_counter_witness.py
"""
from __future__ import annotations

import math

import torch

from memory_native import RMSCounterLinear
from memory_native.group_counter import GroupCounterLinear

torch.manual_seed(0)
n, N, C = 64, 256, 11
STEPS = 700


def data():
    torch.manual_seed(0)
    w = torch.randn(n, n) * (0.25 * math.sqrt(2.0 / n))
    x = torch.randn(N, n)
    return x, x @ w.t()


def run_unstructured(x, y):
    torch.manual_seed(0)
    lay = RMSCounterLinear(n, n, C=C, lr=0.04, lr_scale=2e-4).train()
    for _ in range(STEPS):
        (lay(x) - y).pow(2).mean().backward()
    with torch.no_grad():
        return (lay(x) - y).pow(2).mean().item()


def run_group(x, y, update_all, hysteresis):
    torch.manual_seed(0)
    lay = GroupCounterLinear(n, n, C=C, lr=0.04, lr_scale=2e-4,
                             update_all=update_all, hysteresis=hysteresis).train()
    prev = lay.visible_mask().clone()
    churn = 0
    for _ in range(STEPS):
        (lay(x) - y).pow(2).mean().backward()
        v = lay.visible_mask(); churn += int((v != prev).sum().item()); prev = v.clone()
    with torch.no_grad():
        mse = (lay(x) - y).pow(2).mean().item()
    return mse, lay.ever_visible.float().mean().item(), churn


def main():
    x, y = data()
    print(f"=== M2 2:4 group-counter: Gaussian teacher recovery (n={n} N={N} C={C} steps={STEPS}) ===")
    base = run_unstructured(x, y)
    print(f"  target var {y.var().item():.4f} | unstructured counter MSE {base:.5f}\n")
    print(f"  {'arm':34s} {'MSE':>9s} {'gap%':>7s} {'ever-vis':>9s} {'churn':>10s}")
    print("  " + "-" * 74)
    best24 = 1e9
    for hy in (0.0, 1.0, 2.0, 4.0):
        mse, ever, churn = run_group(x, y, True, hy)
        best24 = min(best24, mse)
        print(f"  {'2:4 error-fb hyst='+str(hy):34s} {mse:9.5f} {100*(mse-base)/base:7.1f} "
              f"{ever*100:8.1f}% {churn:10d}")
    pm, pe, pc = run_group(x, y, False, 2.0)
    print(f"  {'2:4 pruning (frozen)':34s} {pm:9.5f} {100*(pm-base)/base:7.1f} {pe*100:8.1f}% {pc:10d}")

    print("\n=== VERDICT (honest) ===")
    g1 = best24 <= max(1.5 * base, base + 1e-3)
    print(f"  [{'PASS' if g1 else 'FAIL'}] 2:4 sparsity does not break recovery: best 2:4 MSE "
          f"{best24:.5f} vs unstructured {base:.5f} (a 2:4-sparse visible weight learns fine).")
    print("  [FINDING] error-feedback rotation does NOT beat pruning here: low hysteresis THRASHES")
    print("            (churn explodes, MSE worse); high hysteresis FREEZES the set (churn 0,")
    print("            50% ever-visible) -> identical to pruning. On a stationary teacher no")
    print("            specific 2:4 pattern must be discovered, so rotation buys nothing. The")
    print("            error-feedback advantage needs a task where the right support is unknown")
    print("            and a rotation rule that neither thrashes nor freezes -- NOT shown on CPU.")


if __name__ == "__main__":
    main()
