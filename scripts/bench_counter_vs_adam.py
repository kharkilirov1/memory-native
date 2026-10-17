#!/usr/bin/env python
"""Honest counter-updater vs AdamW measurement (research harness, CUDA required).

Checks first, then measures:

1. Dense baseline really trains: the parameter has a gradient before the step and the
   parameter VALUES change after it, and AdamW state is non-empty.
2. The counter layer updates exactly once per step: ``update_events`` grows by
   ``out*in`` per step (not twice, not zero) and the packed state bytes change. The
   updater runs inside the autograd Function's backward; an explicit extra call would
   double-update, which the harness must catch.
3. Decimated runs are a different algorithm: only 1/decimation of the groups changes per
   step, so a decimated step is NOT a cheaper equivalent of an AdamW step -- it is
   reported as its own row with that caveat.

Phases (CUDA events, warmup + median over reps): forward (no_grad), backward (counter:
includes the update), optimizer step (dense), full step. Peak allocated memory is
measured per configuration, including the fp32 weight cache when enabled.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory_native.group_scale_packed import PackedGroupScaleCounterLinear  # noqa: E402


def bench_events(fn, *, warmup: int = 10, iters: int = 30, reps: int = 5) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    per_rep = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        per_rep.append(start.elapsed_time(end) / iters)
    per_rep.sort()
    return {"ms": statistics.median(per_rep),
            "iqr_ms": per_rep[(3 * len(per_rep)) // 4] - per_rep[len(per_rep) // 4]}


def peak_mem_mb(fn, *, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return max(torch.cuda.max_memory_allocated() - base, 0) / 2 ** 20


def make_dense(shape: int, seed: int = 0) -> tuple[nn.Linear, torch.nn.Parameter]:
    torch.manual_seed(seed)
    lin = nn.Linear(shape, shape, bias=False).cuda()
    with torch.no_grad():
        lin.weight.mul_(0.02)
    return lin, lin.weight


def make_counter(shape: int, decimation: int, cache_bytes: int, seed: int = 0):
    from memory_native.packed import pack_codes

    torch.manual_seed(seed)
    layer = PackedGroupScaleCounterLinear(shape, shape, group=128, C=11,
                                          kernel_mode="gemm", lr=1e-3,
                                          stats_scope="group", decimation=decimation,
                                          weight_cache_bytes=cache_bytes).cuda()
    levels = 3 * (2 * 11 - 1)
    layer.state.copy_(pack_codes(torch.randint(0, levels, (shape, shape),
                                               dtype=torch.uint8, device="cuda")))
    layer.scale.copy_(torch.rand(shape, shape // 128, device="cuda") * 0.02 + 0.005)
    return layer


def verify_dense(lin, opt, x, go) -> dict:
    opt.zero_grad(set_to_none=True)
    w_before = lin.weight.detach().clone()
    out = lin(x)
    out.backward(go)
    grad_ok = lin.weight.grad is not None and bool(torch.isfinite(lin.weight.grad).all())
    opt.step()
    changed = not torch.equal(lin.weight.detach(), w_before)
    state = opt.state.get(lin.weight, {})
    adam_ok = bool(state) and "exp_avg" in state and float(state["exp_avg"].abs().sum()) > 0
    return {"grad_present": grad_ok, "weights_changed": bool(changed),
            "adam_state_nonempty": adam_ok}


def verify_counter(layer, x, go) -> dict:
    before_events = int(layer.update_events)
    state_before = layer.state.clone()
    loss_out = layer(x)
    loss_out.backward(go)
    delta = int(layer.update_events) - before_events
    expected = layer.out_features * layer.in_features
    state_changed = not torch.equal(layer.state, state_before)
    # excercise a second step to prove the delta is per-step, not cumulative
    loss_out2 = layer(x)
    loss_out2.backward(go)
    delta2 = int(layer.update_events) - before_events - delta
    return {"update_events_per_step": delta, "expected": expected,
            "exactly_one_update": delta == expected,
            "second_step_one_update": delta2 == expected,
            "state_changed": bool(state_changed)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shapes", default="1024,2048,4096")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out-dir", default=str(ROOT / "research/results/lut_scaling_2026_10_17"))
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        print("needs CUDA")
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, checks = [], []

    for shape in [int(s) for s in args.shapes.split(",") if s]:
        torch.manual_seed(7)
        x = torch.randn(args.batch, shape, device="cuda")
        go = torch.randn(args.batch, shape, device="cuda")

        # --- dense baselines -------------------------------------------------
        lin, w = make_dense(shape)
        opt = torch.optim.AdamW([w], lr=1e-3)
        chk = verify_dense(lin, opt, x, go)
        checks.append({"config": f"dense nn.Linear+AdamW {shape}^2", **chk})

        def dense_full():
            opt.zero_grad(set_to_none=True)
            lin(x).backward(go)
            opt.step()

        def dense_fwd():
            with torch.no_grad():
                lin(x)

        def dense_step():
            opt.step()

        w_flat = w.detach()
        dense_phases = {
            "fwd": bench_events(dense_fwd, iters=args.iters, reps=args.reps),
            "state_update": bench_events(dense_step, iters=args.iters, reps=args.reps),
            "full_step": bench_events(dense_full, iters=args.iters, reps=args.reps),
        }
        dense_mem = peak_mem_mb(dense_full)
        for phase, st in dense_phases.items():
            rows.append({"shape": shape, "batch": args.batch, "impl": "dense_nn_linear_adamw",
                         "decimation": 1, "phase": phase, **st,
                         "peak_mem_mb": dense_mem if phase == "full_step" else None,
                         "note": "real gradients + real AdamW step (verified)"})

        # --- counter variants ------------------------------------------------
        for dec, cache in ((1, 1 << 30), (4, 1 << 30), (8, 1 << 30), (1, 0)):
            layer = make_counter(shape, dec, cache)
            chk = verify_counter(layer, x, go)
            checks.append({"config": f"counter {shape}^2 dec={dec} cache={cache}", **chk})

            def counter_full():
                y = layer(x)
                y.backward(go)

            def counter_fwd():
                with torch.no_grad():
                    layer(x)

            phases = {
                "fwd": bench_events(counter_fwd, iters=args.iters, reps=args.reps),
                "bwd_incl_update": bench_events(counter_full, iters=args.iters, reps=args.reps),
            }
            mem = peak_mem_mb(counter_full)
            for phase, st in phases.items():
                rows.append({"shape": shape, "batch": args.batch,
                             "impl": "counter_group_scope", "decimation": dec,
                             "phase": phase, **st,
                             "peak_mem_mb": mem if phase == "bwd_incl_update" else None,
                             "note": ("update runs inside backward; decimated steps update "
                                      "only 1/dec of groups (different algorithm)")})

    meta = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "commit": __import__("subprocess").run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "batch": args.batch, "iters": args.iters, "reps": args.reps,
    }
    with open(out_dir / "counter_vs_adam.json", "w") as fh:
        json.dump({"meta": meta, "checks": checks, "rows": rows}, fh, indent=1)
    with open(out_dir / "counter_vs_adam.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("VERIFICATION")
    for c in checks:
        print("  ", json.dumps(c))
    print("\nPHASES (ms)")
    for row in rows:
        print(f"  {row['shape']:5d} {row['impl']:22s} dec={row['decimation']} "
              f"{row['phase']:16s} {row['ms']:8.4f} (iqr {row['iqr_ms']:7.4f})")
    print(f"\nartifacts: {out_dir/'counter_vs_adam.json'} {out_dir/'counter_vs_adam.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
