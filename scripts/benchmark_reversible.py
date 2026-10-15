"""Compare inverse backward with a historical checkout; emit a CPU-only JSON witness.

Run from the repository root with PYTHONPATH=src. The baseline source comes from the
local git history. No downloads, model data, counter-state updates, or CUDA are involved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import statistics
import subprocess
import time
import types

import torch
from torch import nn

import memory_native.reversible as current


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="851c91e")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    if min(args.depth, args.width, args.batch, args.repeats) < 1 or args.width % 2:
        parser.error("dimensions/repeats must be positive and width must be even")
    root = Path(__file__).resolve().parents[1]
    baseline_sha = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{args.baseline_ref}^{{commit}}"], cwd=root, text=True).strip()
    source = subprocess.check_output(
        ["git", "show", f"{baseline_sha}:src/memory_native/reversible.py"], cwd=root, text=True)
    legacy = types.ModuleType("legacy_reversible")
    exec(compile(source, f"{baseline_sha}:reversible.py", "exec"), legacy.__dict__)
    torch.set_num_threads(1)

    def build(module):
        torch.manual_seed(2026)
        half = args.width // 2
        def branch():
            return nn.Sequential(nn.Linear(half, half), nn.Tanh(), nn.Linear(half, half))
        return module.ReversibleSequence([
            module.ReversibleCouplingBlock(args.width, branch(), branch()) for _ in range(args.depth)])

    models = {"baseline": build(legacy), "single_pass": build(current)}
    torch.manual_seed(42)
    data, upstream = torch.randn(args.batch, args.width), torch.randn(args.batch, args.width)
    inputs = {key: data.clone().requires_grad_(True) for key in models}
    def step(key):
        models[key].zero_grad(set_to_none=True)
        inputs[key].grad = None
        models[key](inputs[key]).backward(upstream)

    calls, gradients = {}, {}
    for key, model in models.items():
        count = [0]
        def counted(*_):
            count[0] += 1
        hooks = [branch.register_forward_hook(counted)
                 for block in model.blocks for branch in (block.F, block.G)]
        step(key)
        calls[key] = count[0]
        gradients[key] = [inputs[key].grad.clone()] + [p.grad.clone() for p in model.parameters()]
        for hook in hooks:
            hook.remove()
    max_abs_error = max((a - b).abs().max().item()
                        for a, b in zip(gradients["baseline"], gradients["single_pass"]))
    for _ in range(5):
        for key in models:
            step(key)
    timings = {key: [] for key in models}
    # Alternate measurement order to reduce systematic warm-up / scheduling bias.
    for i in range(args.repeats):
        keys = list(models) if i % 2 == 0 else list(reversed(models))
        for key in keys:
            start = time.perf_counter()
            step(key)
            timings[key].append((time.perf_counter() - start) * 1000)
    medians = {key: statistics.median(values) for key, values in timings.items()}
    print(json.dumps({
        "scope": "CPU float32 nonlinear coupling stack, forward + backward; no optimizer",
        "baseline_commit": baseline_sha,
        "candidate_reversible_sha256": hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(), "torch": torch.__version__,
        "platform": platform.platform(), "threads": torch.get_num_threads(),
        "config": vars(args), "branch_calls_per_step": calls,
        "max_abs_gradient_difference_vs_baseline": max_abs_error,
        "median_step_ms": medians,
        "baseline_over_single_pass": medians["baseline"] / medians["single_pass"],
        "samples_ms": timings,
        "limitations": "One CPU workload; not a GPU or end-to-end LLM speedup. Float inverse remains approximate.",
    }, indent=2))


if __name__ == "__main__":
    main()
