#!/usr/bin/env python3
"""Eight predeclared paired input-metric counter BLAST regression diagnostics.

This pilot asks whether right-preconditioning small-factor correlations helps
counter learning, and whether nearly fixed factor scales confound that answer.
It does not select a learning rate or validate language-model generalization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from memory_native.counter import RMSCounterLinear
from memory_native.research.covariance import CovarianceRMSCounterLinear, precondition_counter_blast
from memory_native.research.structured import CounterBLASTLinear
from new_math_capacity_ablation import (
    BATCH, BLOCK_SIZE, DIMENSION, TRAIN_ROWS, counter_report, dataset,
    spectrum_report, tensor_sha256, write_json,
)


ARMS = (
    {"name": "baseline_C8", "C": 8, "ridge": None, "lr_scale": 2e-4},
    {"name": "cov_C8_ridge01", "C": 8, "ridge": .1, "lr_scale": 2e-4},
    {"name": "cov_C8_ridge001", "C": 8, "ridge": .01, "lr_scale": 2e-4},
    {"name": "baseline_C2", "C": 2, "ridge": None, "lr_scale": 2e-4},
    {"name": "cov_C2_ridge01", "C": 2, "ridge": .1, "lr_scale": 2e-4},
    {"name": "cov_C2_ridge001", "C": 2, "ridge": .01, "lr_scale": 2e-4},
    {"name": "baseline_C8_scale001", "C": 8, "ridge": None, "lr_scale": .01},
    {"name": "cov_C8_ridge01_scale001", "C": 8, "ridge": .1, "lr_scale": .01},
)
BODY_LR, COV_BETA, COV_EPS = .03, .95, 1e-6


def state_hash(state):
    payload = {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                     "sha256": tensor_sha256(value)} for key, value in sorted(state.items())}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@torch.no_grad()
def scale_report(model, initial_scales):
    deltas, ratios, per_factor = [], [], {}
    for name, module in model.named_modules():
        if isinstance(module, RMSCounterLinear):
            initial = initial_scales[name]
            delta = float((module.scale-initial).norm()/initial.norm())
            ratio = (module.scale/initial).flatten()
            deltas.append((module.scale-initial).flatten())
            ratios.append(ratio)
            per_factor[name] = {"relative_l2_change": delta, "min_scale_ratio": float(ratio.min()),
                                "max_scale_ratio": float(ratio.max())}
    initial = torch.cat([value.flatten() for value in initial_scales.values()])
    return {"relative_l2_change": float(torch.cat(deltas).norm()/initial.norm()),
            "min_scale_ratio": float(torch.cat(ratios).min()),
            "max_scale_ratio": float(torch.cat(ratios).max()), "factors": per_factor}


def run_arm(arm, args, x, y, data_info):
    torch.manual_seed(args.seed)
    student = CounterBLASTLinear(DIMENSION, DIMENSION, block_size=BLOCK_SIZE, rank=4,
        C=arm["C"], counter_lr=BODY_LR, lr_scale=arm["lr_scale"])
    original = {name: value.clone() for name, value in student.state_dict().items()}
    baseline_hash = state_hash(original)
    initial = spectrum_report(student)
    if arm["ridge"] is not None:
        precondition_counter_blast(student, cov_beta=COV_BETA, cov_ridge=arm["ridge"], cov_eps=COV_EPS)
        if any(not torch.equal(value, student.state_dict()[name]) for name, value in original.items()):
            raise AssertionError("covariance conversion changed initial counter state")
        if spectrum_report(student)["visible_weight_sha256"] != initial["visible_weight_sha256"]:
            raise AssertionError("covariance conversion changed the initial visible operator")
    initial_scales = {name: module.scale.clone() for name, module in student.named_modules()
                      if isinstance(module, RMSCounterLinear)}
    torch.manual_seed(100_000+args.seed)
    stream = torch.Generator().manual_seed(10_000+args.seed)
    rows_hash = hashlib.sha256()
    curve, elapsed = [], 0.
    for step in range(args.steps+1):
        if step % args.eval_every == 0 or step == args.steps:
            with torch.no_grad():
                curve.append({"step": step,
                    "train_relative_mse": float((student(x[:TRAIN_ROWS])-y[:TRAIN_ROWS]).square().mean()
                                                / y[:TRAIN_ROWS].square().mean()),
                    "heldout_relative_mse": float((student(x[TRAIN_ROWS:])-y[TRAIN_ROWS:]).square().mean()
                                                  / y[TRAIN_ROWS:].square().mean()),
                    "train_seconds": elapsed})
        if step == args.steps:
            break
        rows = torch.randint(TRAIN_ROWS, (BATCH,), generator=stream)
        rows_hash.update(rows.numpy().tobytes())
        start = time.perf_counter()
        loss = (student(x[rows])-y[rows]).square().mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"nonfinite training loss in {arm['name']} at step {step+1}")
        loss.backward()
        elapsed += time.perf_counter()-start
    cov_factors = {name: factor for name, factor in student.named_modules()
                   if isinstance(factor, CovarianceRMSCounterLinear)}
    covariance_bytes = sum(f.input_gram.numel()*f.input_gram.element_size()
                           for f in cov_factors.values())
    covariance_state_bytes = sum((f.input_gram.numel()*f.input_gram.element_size()
                                 + f._cov_config.numel()*f._cov_config.element_size()
                                 + f.gram_updates.numel()*f.gram_updates.element_size())
                                for f in cov_factors.values())
    return {"task": "covariance_ablation", "status": "completed", "arm": arm, "data": data_info,
        "seed": args.seed, "steps": args.steps, "batch": BATCH, "body_lr": BODY_LR,
        "cov_beta": COV_BETA if cov_factors else None, "cov_eps": COV_EPS if cov_factors else None,
        "coefficient_count": student.coefficient_count, "global_rank_bound": student.rank_bound,
        "fp_parameter_count": sum(p.numel() for p in student.parameters()),
        "persistent_model_bytes": sum(t.numel()*t.element_size() for t in student.state_dict().values()),
        "covariance_gram_bytes": covariance_bytes, "covariance_total_added_bytes": covariance_state_bytes,
        "initial_counter_state_sha256": baseline_hash, "initial": initial,
        "final": spectrum_report(student), "counter_statistics": counter_report(student),
        "scale_change": scale_report(student, initial_scales),
        "gram_update_counts": {name: int(f.gram_updates) for name, f in cov_factors.items()},
        "training_stream_seed": 10_000+args.seed, "rounding_seed": 100_000+args.seed,
        "sampled_batch_indices_sha256": rows_hash.hexdigest(), "sampled_training_rows": args.steps*BATCH,
        "train_relative_mse": curve[-1]["train_relative_mse"],
        "heldout_relative_mse": curve[-1]["heldout_relative_mse"], "train_seconds": elapsed, "curve": curve}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=50)
    args = parser.parse_args()
    if args.steps <= 0 or args.eval_every <= 0 or args.seed < 0:
        parser.error("steps and eval-every must be positive; seed must be nonnegative")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError("output directory must be empty to preserve existing evidence")
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), root/"scripts/new_math_capacity_ablation.py",
        root/"scripts/new_math_experiments.py", root/"src/memory_native/counter.py",
        root/"src/memory_native/research/structured.py", root/"src/memory_native/research/pam.py",
        root/"src/memory_native/research/covariance.py", root/"tests/test_research_covariance.py"]
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    x, y, data_info = dataset("blast")
    manifest = {"protocol_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_sha256": hashes, "arms_predeclared": list(ARMS), "data": data_info,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
        "device": "cpu", "dtype": "float32", "torch_threads": 1, "torch_interop_threads": 1,
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "formula": "G=.95*G+.05*X.T@X/n; Htilde=H@(G+(ridge*max(trace(G)/d,1e-6)+1e-6)*I)^-1",
        "initial_gram": "identity", "input_statistic": "uncentered, observed current training forward",
        "scope": "small input Gram on every U/V/S; forward and old-weight input derivative unchanged",
        "update_effect": "both RMS/ticks and scale learning consume the preconditioned correlation",
        "protocol": "8 arms predeclared before execution; fixed horizon and shared data/batches/initial state; no test-based selection",
        "limitations": ["single seed, 32x32 linear regression; no LLM or hardware speed evidence",
            "known input-metric preconditioning; combination with discrete factor learning is a hypothesis",
            "extra fp32 O(d_local^2) statistics and Gram/Cholesky work; not six-bit total training state",
            "baseline nearly fixed scales probed separately with lr_scale=.01, not tuned after evaluation",
            "Gram EMA includes internal activations; factor gauges evolve during learning",
            "timings exclude evaluation and are CPU reference timings, without isolated hardware guarantees"]}
    write_json(args.output/"manifest.json", manifest)
    results = []
    for arm in ARMS:
        result = run_arm(arm, args, x, y, data_info)
        filename = f"blast_{arm['name']}_s{args.seed}.json"
        write_json(args.output/filename, result)
        results.append(result)
        print(json.dumps({"file": filename, "heldout_relative_mse": result["heldout_relative_mse"],
            "train_relative_mse": result["train_relative_mse"], "seconds": result["train_seconds"],
            "scale_relative_l2_change": result["scale_change"]["relative_l2_change"],
            "persistent_bytes": result["persistent_model_bytes"]}), flush=True)
    for C, scale in {(arm["C"], arm["lr_scale"]) for arm in ARMS}:
        group = [r for r in results if r["arm"]["C"] == C and r["arm"]["lr_scale"] == scale]
        if len({r["initial_counter_state_sha256"] for r in group}) != 1 \
                or len({r["sampled_batch_indices_sha256"] for r in group}) != 1:
            raise AssertionError("paired initial counter states or sampled batches differ")
    current = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    if hashes != current:
        raise RuntimeError("source changed during the covariance experiment")
    write_json(args.output/"completion.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "all_pair_checks_passed": True, "source_hashes_unchanged": True,
        "results": [{"arm": r["arm"]["name"], "heldout_relative_mse": r["heldout_relative_mse"],
                     "scale_relative_l2_change": r["scale_change"]["relative_l2_change"]} for r in results]})


if __name__ == "__main__":
    main()
