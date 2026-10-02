#!/usr/bin/env python3
"""Predeclared single-seed capacity/optimization diagnostic for counter BLAST.

This auxiliary regression does not change the primary experiment or select an LR.
All eight arms receive identical Gaussian training data and sampled batches. The
held-out data are diagnostics, evaluated at fixed steps with no early stopping.
Only small CPU reference operators are measured; no GPU or LLM claim is made.
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
from memory_native.research.structured import BLASTLinear, CounterBLASTLinear


ARMS = (
    {"name": "fp_r4", "kind": "blast", "rank": 4, "C": None, "lr": .01},
    {"name": "fp_r8", "kind": "blast", "rank": 8, "C": None, "lr": .01},
    {"name": "counter_r4_C8_lr003", "kind": "blast_counter", "rank": 4, "C": 8, "lr": .03},
    {"name": "counter_r4_C8_lr01", "kind": "blast_counter", "rank": 4, "C": 8, "lr": .1},
    {"name": "counter_r4_C4_lr003", "kind": "blast_counter", "rank": 4, "C": 4, "lr": .03},
    {"name": "counter_r4_C2_lr003", "kind": "blast_counter", "rank": 4, "C": 2, "lr": .03},
    {"name": "counter_r4_C2_lr01", "kind": "blast_counter", "rank": 4, "C": 2, "lr": .1},
    {"name": "counter_r8_C2_lr003", "kind": "blast_counter", "rank": 8, "C": 2, "lr": .03},
)
DIMENSION, BLOCK_SIZE, TRAIN_ROWS, TEST_ROWS, BATCH = 32, 8, 1536, 512, 32
SCALE_LR = 2e-4


def tensor_sha256(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def write_json(path, payload):
    if path.exists():
        raise FileExistsError(f"refusing to overwrite research evidence: {path}")
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@torch.no_grad()
def visible_weight(model):
    # Dense reconstruction is analysis of a 32x32 operator, never a training path.
    return model(torch.eye(model.in_features)).T.contiguous()


@torch.no_grad()
def spectrum_report(model):
    weight = visible_weight(model)
    singular_values = torch.linalg.svdvals(weight)
    return {
        "visible_weight_sha256": tensor_sha256(weight),
        "visible_weight_rank": int(torch.linalg.matrix_rank(weight)),
        "visible_weight_singular_values": singular_values.tolist(),
        "factor_ranks": {name: [int(torch.linalg.matrix_rank(visible_weight(f))) for f in getattr(model, name)]
                         for name in ("U", "V", "S")},
    }


@torch.no_grad()
def counter_report(model):
    factors = [(name, m) for name, m in model.named_modules() if isinstance(m, RMSCounterLinear)]
    if not factors:
        return None
    return {
        "weight_flips": sum(int(m.weight_flips) for _, m in factors),
        "update_events": sum(int(m.update_events) for _, m in factors),
        "factors": {name: {**m.state_statistics(), "scale_min": float(m.scale.min()),
                          "scale_max": float(m.scale.max()), "v_mean": float(m.v.mean())}
                    for name, m in factors},
    }


def dataset(teacher_kind):
    generator = torch.Generator().manual_seed(12345)
    x = torch.randn(TRAIN_ROWS + TEST_ROWS, DIMENSION, generator=generator)
    torch.manual_seed(123)
    if teacher_kind == "blast":
        teacher = BLASTLinear(DIMENSION, DIMENSION, block_size=BLOCK_SIZE, rank=4)
    else:
        teacher = torch.nn.Linear(DIMENSION, DIMENSION, bias=False)
        torch.nn.init.normal_(teacher.weight, std=1 / DIMENSION ** .5)
    with torch.no_grad():
        y = teacher(x)
        weight = teacher(torch.eye(DIMENSION)).T.contiguous()
        sv = torch.linalg.svdvals(weight)
    return x, y, {
        "teacher": teacher_kind, "teacher_seed": 123, "data_seed": 12345,
        "teacher_weight_sha256": tensor_sha256(weight), "teacher_rank": int(torch.linalg.matrix_rank(weight)),
        "teacher_singular_values": sv.tolist(), "input_sha256": tensor_sha256(x),
        "targets_sha256": tensor_sha256(y), "rank16_population_relative_mse_lower_bound":
            float(sv[16:].square().sum() / sv.square().sum()),
        "dimension": DIMENSION, "teacher_block_size": BLOCK_SIZE if teacher_kind == "blast" else None,
        "teacher_factor_rank": 4 if teacher_kind == "blast" else None,
        "train_rows": TRAIN_ROWS, "heldout_rows": TEST_ROWS,
    }


def run_arm(arm, seed, steps, eval_every, x, y, data_info):
    torch.manual_seed(seed)
    if arm["kind"] == "blast":
        student = BLASTLinear(DIMENSION, DIMENSION, block_size=BLOCK_SIZE, rank=arm["rank"])
    else:
        student = CounterBLASTLinear(DIMENSION, DIMENSION, block_size=BLOCK_SIZE, rank=arm["rank"],
                                    C=arm["C"], counter_lr=arm["lr"], lr_scale=SCALE_LR)
    initial = spectrum_report(student)
    torch.manual_seed(100_000 + seed)
    parameters = list(student.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=arm["lr"], weight_decay=0.) if parameters else None
    stream = torch.Generator().manual_seed(10_000 + seed)
    curve, elapsed = [], 0.
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            with torch.no_grad():
                curve.append({"step": step,
                              "train_relative_mse": float((student(x[:TRAIN_ROWS]) - y[:TRAIN_ROWS]).square().mean()
                                                          / y[:TRAIN_ROWS].square().mean()),
                              "heldout_relative_mse": float((student(x[TRAIN_ROWS:]) - y[TRAIN_ROWS:]).square().mean()
                                                            / y[TRAIN_ROWS:].square().mean()),
                              "train_seconds": elapsed})
        if step == steps:
            break
        rows = torch.randint(TRAIN_ROWS, (BATCH,), generator=stream)
        before = time.perf_counter()
        if optimizer:
            optimizer.zero_grad(set_to_none=True)
        mse = (student(x[rows]) - y[rows]).square().mean()
        if not torch.isfinite(mse):
            raise FloatingPointError(f"nonfinite training MSE in {arm['name']} at step {step + 1}")
        mse.backward()
        if optimizer:
            optimizer.step()
        elapsed += time.perf_counter() - before
    final = spectrum_report(student)
    return {
        "task": "capacity_ablation", "status": "completed", "arm": arm, "data": data_info,
        "seed": seed, "steps": steps, "batch": BATCH, "scale_lr": SCALE_LR if arm["C"] else None,
        "sampled_training_rows": steps * BATCH, "training_stream_seed": 10_000 + seed,
        "rounding_seed": 100_000 + seed, "weight_decay": 0., "gradient_clipping": None,
        "block_size": BLOCK_SIZE, "coefficient_count": student.coefficient_count,
        "global_rank_bound": student.rank_bound, "fp_parameter_count": sum(p.numel() for p in parameters),
        "persistent_model_bytes": sum(t.numel() * t.element_size() for t in student.state_dict().values()),
        "adam_state_bytes": sum(t.numel() * t.element_size() for state in optimizer.state.values()
                                 for t in state.values() if isinstance(t, torch.Tensor)) if optimizer else 0,
        "initial": initial, "final": final, "counter_statistics": counter_report(student),
        "train_relative_mse": curve[-1]["train_relative_mse"],
        "heldout_relative_mse": curve[-1]["heldout_relative_mse"],
        "train_seconds": elapsed, "curve": curve,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--teacher", choices=("blast", "dense"), default="blast")
    args = parser.parse_args()
    if args.steps <= 0 or args.eval_every <= 0 or args.seed < 0:
        parser.error("steps/eval-every must be positive and seed nonnegative")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError("output directory must be empty to preserve prior evidence")
    root = Path(__file__).resolve().parents[1]
    source_files = [Path(__file__).resolve(), root / "src/memory_native/research/structured.py",
                    root / "src/memory_native/counter.py"]
    x, y, data_info = dataset(args.teacher)
    manifest = {
        "protocol_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "arms_predeclared": list(ARMS), "data": data_info,
        "python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
        "device": "cpu", "dtype": "float32", "torch_threads": 1, "torch_interop_threads": 1,
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cuda_available": torch.cuda.is_available(),
        "protocol": "All arms run the same fixed horizon; no LR selection, early stopping or test-based winner.",
        "limitations": ["single seed, small linear regression; diagnostic only",
                        "rank bound is not a counter-family representability guarantee",
                        "rank8 increases coefficient count and changes the operator family",
                        "C changes hidden counter resolution, not the visible ternary factor values",
                        "capacity and optimization limits cannot be uniquely separated by this ablation",
                        "CPU PyTorch reference timings; memory excludes activations/transient peak",
                        "counter coefficients use uint8 storage, not six-bit packing"],
    }
    write_json(args.output / "manifest.json", manifest)
    for arm in ARMS:
        result = run_arm(arm, args.seed, args.steps, args.eval_every, x, y, data_info)
        filename = f"{args.teacher}_{arm['name']}_s{args.seed}.json"
        write_json(args.output / filename, result)
        print(json.dumps({"file": filename, "heldout_relative_mse": result["heldout_relative_mse"],
                          "train_relative_mse": result["train_relative_mse"], "seconds": result["train_seconds"],
                          "initial_rank": result["initial"]["visible_weight_rank"],
                          "final_rank": result["final"]["visible_weight_rank"]}), flush=True)


if __name__ == "__main__":
    main()
