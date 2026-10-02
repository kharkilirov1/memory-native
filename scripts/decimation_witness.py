"""Decimation witness — the FLOP-cut lever on the group-local update (CPU, minutes).

Group-locality makes decimation WELL-DEFINED: updating a subset of groups is exactly the
full math restricted to that subset (no statistic crosses a group boundary), so a
round-robin 1/S schedule cuts update FLOPs by S without changing the update rule at all.
On the fused kernel it maps to launching the grid over the scheduled groups only.

Protocol (the project's standard first gate): ternary-teacher recovery with the recovery
recipe's clip=1.0, group-local reference, arms:

  full          — every group every step (baseline);
  dec4 lr x k   — 1/4 of groups per step (g % 4 == step % 4), lr scaled by k in {1, 2, 4};

compared at EQUAL STEPS (same wall budget) and at EQUAL UPDATE-FLOPs (dec4 gets 4x steps).
The open question this measures: does the 4x-rarer integrated signal per counter cost
convergence, and does lr compensation recover it?

Run: PYTHONPATH=src python scripts/decimation_witness.py
"""
from __future__ import annotations

import torch

from memory_native.counter import decode_state, encode_state
from memory_native.group_scale_kernels import group_counter_update_grouplocal_hashsr

OUT, IN, GROUP, C = 64, 128, 16, 11
GROUPS = IN // GROUP
N_SAMPLES = 512
TS = 0.25
BASE_LR = 0.02
STEPS = 600
S = 4  # decimation factor


def run_arm(schedule: str, lr_mul: float, steps: int, seed_data: int = 0) -> list[float]:
    torch.manual_seed(seed_data)
    tw = torch.randint(-1, 2, (OUT, IN)).to(torch.int16)
    x = torch.randn(N_SAMPLES, IN)
    y = x @ (TS * tw.float()).t()
    perm = torch.arange(IN)
    codes = encode_state(torch.zeros(OUT, IN, dtype=torch.int16),
                         torch.zeros(OUT, IN, dtype=torch.int16), C)
    scale = torch.full((OUT, GROUPS), TS)
    v = torch.zeros(OUT, GROUPS)
    mses = []
    group_ids = torch.arange(GROUPS)
    for step in range(steps):
        t, _ = decode_state(codes, C)
        w = scale[:, torch.arange(IN) // GROUP] * t.float()
        pred = x @ w.t()
        mses.append(((pred - y) ** 2).mean().item())
        go = (2.0 / (N_SAMPLES * OUT)) * (pred - y)
        gw = go.t() @ x
        active = None
        if schedule == "dec":
            active = (group_ids % S) == (step % S)
        codes = group_counter_update_grouplocal_hashsr(
            codes, scale, v, gw, perm,
            group=GROUP, C=C, lr=BASE_LR * lr_mul, lr_scale=2e-4,
            rms_beta=0.9, rms_eps=1e-3, seed=step, clip=1.0,
            active_groups=active,
        )
    t, _ = decode_state(codes, C)
    w = scale[:, torch.arange(IN) // GROUP] * t.float()
    mses.append(((x @ w.t() - y) ** 2).mean().item())
    return mses


def steps_to(mses: list[float], thr: float) -> int | None:
    for i, m in enumerate(mses):
        if m < thr:
            return i
    return None


def main() -> None:
    torch.manual_seed(0)
    y_var = None
    # full lr x2/x4 arms keep the comparison honest: without them "dec4 lr x4 wins"
    # cannot be told apart from "a hotter lr wins" on this problem.
    arms = {
        "full lr x1": run_arm("full", 1.0, STEPS),
        "full lr x2": run_arm("full", 2.0, STEPS),
        "full lr x4": run_arm("full", 4.0, STEPS),
        "dec4 lr x1": run_arm("dec", 1.0, STEPS),
        "dec4 lr x2": run_arm("dec", 2.0, STEPS),
        "dec4 lr x4": run_arm("dec", 4.0, STEPS),
        "dec4 lr x1 (equal-FLOPs)": run_arm("dec", 1.0, STEPS * S),
        "dec4 lr x2 (equal-FLOPs)": run_arm("dec", 2.0, STEPS * S),
    }
    torch.manual_seed(0)
    tw = torch.randint(-1, 2, (OUT, IN)).to(torch.int16)
    x = torch.randn(N_SAMPLES, IN)
    y = x @ (TS * tw.float()).t()
    y_var = y.var().item()
    thr = 0.02
    print(f"teacher-recovery, out={OUT} in={IN} group={GROUP} groups={GROUPS} "
          f"C={C} clip=1.0 base_lr={BASE_LR} y_var={y_var:.3f} thr={thr}")
    print(f"{'arm':<28} {'steps':>6} {'final mse':>10} {'steps<thr':>10} {'upd FLOPs':>10}")
    full_flops = STEPS
    for name, mses in arms.items():
        n = len(mses) - 1
        flops = n if "full" in name else n / S
        rel = flops / full_flops
        st = steps_to(mses, thr)
        print(f"{name:<28} {n:>6} {mses[-1]:>10.5f} {str(st):>10} {rel:>9.2f}x")


if __name__ == "__main__":
    main()
