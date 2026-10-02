#!/usr/bin/env python3
"""Controlled CPU pilots for structural counter training and PAM arithmetic.

This runner requires an explicit real corpus. It never falls back to synthetic text.
All variants see identical batches and fixed held-out windows. Test text is evaluated
only at the end; LR selection uses a separate validation segment and tuning seed.
Timings describe this PyTorch CPU implementation, not hardware potential.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from memory_native.counter import RMSCounterLinear
from memory_native.models import GPT, GPTConfig
from memory_native.research.pam import PAMLinear, pam
from memory_native.research.structured import BLASTLinear, CounterBLASTLinear, LowRankLinear


VARIANTS = ("dense", "dense_narrow", "counter", "blast", "blast_counter", "lowrank", "pam_exact", "pam_surrogate")


def layer(kind, fin, fout, gain, lr, block_size, rank):
    if kind == "lowrank":
        blast_count = rank * (fin + fout + math.ceil(fin / block_size) * math.ceil(fout / block_size))
        matched_rank = max(1, blast_count // (fin + fout))
        return LowRankLinear(fin, fout, rank=matched_rank, init_gain=gain)
    if kind == "counter":
        return RMSCounterLinear(fin, fout, init_gain=gain, lr=lr, lr_scale=2e-4, C=8)
    if kind in {"blast", "blast_counter"}:
        cls = BLASTLinear if kind == "blast" else CounterBLASTLinear
        return cls(fin, fout, init_gain=gain, block_size=block_size, rank=rank,
                   counter_lr=lr, lr_scale=2e-4, C=8)
    if kind.startswith("pam_"):
        return PAMLinear(fin, fout, init_gain=gain, backward=kind[4:], chunk_size=16)
    result = torch.nn.Linear(fin, fout, bias=False)
    torch.nn.init.normal_(result.weight, std=gain / math.sqrt(fin))
    return result


def make_model(kind, seed, lr, args, vocab):
    torch.manual_seed(seed)
    dim = args.dim // 2 if kind == "dense_narrow" else args.dim
    cfg = GPTConfig(vocab, args.context, args.layers, args.heads, dim)
    model = GPT("dense", cfg)
    # Construct the same dense shell first: embeddings, norms, and PAM weights are
    # identical at equal width/seed. Structured and counter bodies have different
    # random initial operators; their distributions have matched output variance.
    if kind not in {"dense", "dense_narrow"}:
        for block in model.blocks:
            for name in ("q", "k", "v", "proj", "fc", "fc2"):
                original = getattr(block, name)
                gain = 1 / math.sqrt(2 * args.layers) if name in {"proj", "fc2"} else 1.
                replacement = layer(kind, original.in_features, original.out_features,
                                    gain, lr, args.block_size, args.rank)
                if kind.startswith("pam_"):
                    with torch.no_grad():
                        replacement.weight.copy_(original.weight)
                setattr(block, name, replacement)
    # Counter SR draws must not change the data stream. Training batches use a
    # dedicated Generator; this generator governs operator/optimizer randomness.
    torch.manual_seed(100_000 + seed)
    return model


def parameter_report(model, tokens):
    def unique_bytes(tensors):
        seen, total = set(), 0
        for t in tensors:
            key = (t.device, t.untyped_storage().data_ptr())
            if key not in seen:
                seen.add(key)
                total += t.untyped_storage().nbytes()
        return total
    body_coefficients = forward_macs = 0
    rank_bounds = []
    for block in model.blocks:
        for name in ("q", "k", "v", "proj", "fc", "fc2"):
            m = getattr(block, name)
            p = getattr(m, "coefficient_count", m.in_features * m.out_features)
            body_coefficients += p
            forward_macs += tokens * p
            rank_bounds.append(getattr(m, "rank_bound", min(m.in_features, m.out_features)))
    counters = [m for m in model.modules() if isinstance(m, RMSCounterLinear)]
    return {
        "fp_trainable_parameters": sum(p.numel() for p in model.parameters()),
        "body_coefficients": body_coefficients,
        "body_matrix_rank_bounds": rank_bounds,
        "counter_coefficients": sum(m.in_features * m.out_features for m in counters),
        "persistent_model_bytes": unique_bytes(model.state_dict().values()),
        "body_forward_pair_products_per_step": forward_macs,
        "body_forward_backward_pair_evaluations_per_step": 3 * forward_macs,
        # Arithmetic elsewhere remains conventional, including attention and head.
        "unchanged_attention_and_head_forward_macs_per_step": tokens * (
            2 * model.cfg.n_layer * model.cfg.block_size * model.cfg.n_embd
            + model.cfg.n_embd * model.cfg.vocab_size),
        "forward_pair_kind": "PAM" if model.kind.startswith("pam_") else "ordinary MAC",
        "backward_pair_kind": ("exact PAM slope and exponent shift" if model.kind == "pam_exact"
            else "PAM" if model.kind == "pam_surrogate" else "ordinary MAC"),
    }


def corpus(args):
    raw = Path(args.data).read_bytes()
    text = raw.decode("utf-8")
    vocabulary = sorted(set(text))
    lookup = {c: i for i, c in enumerate(vocabulary)}
    ids = torch.tensor([lookup[c] for c in text], dtype=torch.long)
    cut1, cut2 = int(.8 * len(ids)), int(.9 * len(ids))
    splits = [ids[:cut1], ids[cut1:cut2], ids[cut2:]]
    if any(len(s) <= args.context for s in splits):
        raise ValueError("corpus splits too short")
    return splits, len(vocabulary), {
        "sha256": hashlib.sha256(raw).hexdigest(), "characters": len(ids),
        "vocab_size": len(vocabulary), "vocabulary": "".join(vocabulary),
        "split_characters": [len(s) for s in splits],
        "local_filename": Path(args.data).name,
    }


def batch_at(data, starts, context):
    offsets = torch.arange(context)
    indices = starts[:, None] + offsets[None]
    return data[indices], data[indices + 1]


def fixed_eval(data, args, seed):
    gen = torch.Generator().manual_seed(seed)
    return [batch_at(data, torch.randint(len(data) - args.context, (args.batch,), generator=gen),
                     args.context) for _ in range(args.eval_batches)]


@torch.no_grad()
def evaluate(model, batches):
    model.eval()
    losses = [float(model(x, y)[1]) for x, y in batches]
    model.train()
    return float(np.mean(losses))


def optimizer_bytes(optimizer):
    return sum(t.numel() * t.element_size() for state in optimizer.state.values()
               for t in state.values() if isinstance(t, torch.Tensor))


def run_lm(kind, seed, lr, steps, args, data, vocab, validation, test, phase):
    model = make_model(kind, seed, lr, args, vocab)
    model.kind = kind
    # FP shell learning rate stays fixed in both counter arms. Counter factors
    # self-update in backward; conventional gradient clipping cannot clip them.
    # No arm uses gradient clipping, dropout, a pretrained teacher, or checkpointing.
    fp_lr = .003
    body_parameter_ids = {id(p) for block in model.blocks
        for name in ("q", "k", "v", "proj", "fc", "fc2")
        for p in getattr(block, name).parameters()}
    shell = [p for p in model.parameters() if id(p) not in body_parameter_ids]
    body = [p for p in model.parameters() if id(p) in body_parameter_ids]
    optimizer = torch.optim.AdamW([{"params": shell, "lr": fp_lr},
                                   {"params": body, "lr": lr}], weight_decay=0.)
    stream = torch.Generator().manual_seed(seed + 10_000)
    record = {"task": "charlm", "phase": phase, "variant": kind, "seed": seed,
              "body_lr": lr, "fp_lr": fp_lr, "steps": steps,
              "training_tokens": steps * args.batch * args.context,
              "shape": {"dim": model.cfg.n_embd, "layers": args.layers,
                        "context": args.context, "batch": args.batch,
                        "block_size": args.block_size, "rank": args.rank},
              **parameter_report(model, args.batch * args.context), "curve": []}
    initial = evaluate(model, validation)
    record["curve"].append({"step": 0, "train_seconds": 0., "validation_loss": initial})
    elapsed = 0.
    for step in range(1, steps + 1):
        starts = torch.randint(len(data) - args.context, (args.batch,), generator=stream)
        x, y = batch_at(data, starts, args.context)
        before = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{kind}, seed {seed}, step {step}: nonfinite loss")
        loss.backward()
        optimizer.step()
        elapsed += time.perf_counter() - before
        if step % args.eval_every == 0 or step == steps:
            entry = {"step": step, "train_seconds": elapsed, "train_loss": float(loss.detach()),
                     "validation_loss": evaluate(model, validation)}
            record["curve"].append(entry)
    counters = [m for m in model.modules() if isinstance(m, RMSCounterLinear)]
    record.update({"validation_loss": record["curve"][-1]["validation_loss"],
                   "train_seconds": elapsed, "training_tokens_per_second": record["training_tokens"] / elapsed,
                   "adam_state_bytes": optimizer_bytes(optimizer),
                   "retained_fp_gradient_bytes": sum(p.grad.numel() * p.grad.element_size()
                       for p in model.parameters() if p.grad is not None),
                   "counter_weight_flips": sum(int(m.weight_flips) for m in counters)})
    if test is not None:
        record["test_loss"] = evaluate(model, test)
        record["test_perplexity"] = math.exp(record["test_loss"])
    return record


def regression(kind, seed, teacher_kind, lr, args):
    dim = 32
    gen = torch.Generator().manual_seed(12345)
    x = torch.randn(2048, dim, generator=gen)
    torch.manual_seed(123)
    if teacher_kind == "dense":
        teacher = layer("dense", dim, dim, 1., lr, 8, 4)
    else:
        teacher = BLASTLinear(dim, dim, block_size=8, rank=4)
    with torch.no_grad():
        y = teacher(x)
        # Reconstruction is analysis of the small teacher, never a student path.
        teacher_weight = teacher(torch.eye(dim)).T
        singular_values = torch.linalg.svdvals(teacher_weight)
        # BLAST has a br-dimensional bottleneck here (4 blocks x rank 4 =16),
        # so even unrestricted rank-16 regression cannot beat this population bound.
        rank16_population_mse_lower_bound = float(singular_values[16:].square().sum()
            / singular_values.square().sum())
    # All models see the same teacher, Gaussian data, and sampled batches.
    torch.manual_seed(seed)
    student = layer(kind, dim, dim, 1., lr, 8, 4)
    torch.manual_seed(100_000 + seed)
    params = list(student.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.) if params else None
    stream = torch.Generator().manual_seed(10_000 + seed)
    curve, elapsed = [], 0.
    for step in range(args.regression_steps + 1):
        if step % 50 == 0 or step == args.regression_steps:
            with torch.no_grad():
                relative_mse = float((student(x[1536:]) - y[1536:]).square().mean() / y[1536:].square().mean())
            curve.append({"step": step, "relative_mse": relative_mse})
        if step == args.regression_steps:
            break
        ix = torch.randint(1536, (32,), generator=stream)
        before = time.perf_counter()
        if optimizer:
            optimizer.zero_grad(set_to_none=True)
        mse = (student(x[ix]) - y[ix]).square().mean()
        if not torch.isfinite(mse):
            raise FloatingPointError(f"nonfinite regression {kind}")
        mse.backward()
        if optimizer:
            optimizer.step()
        elapsed += time.perf_counter() - before
    return {"task": "regression", "variant": kind, "teacher": teacher_kind, "seed": seed,
            "lr": lr, "steps": args.regression_steps, "train_examples": 1536, "test_examples": 512,
            "dimension": dim, "block_size": 8, "rank": 4,
            "rank16_population_mse_lower_bound": rank16_population_mse_lower_bound,
            "teacher_sha256": hashlib.sha256(teacher_weight.numpy().tobytes()).hexdigest(),
            "relative_mse": curve[-1]["relative_mse"], "train_seconds": elapsed, "curve": curve}


def write_record(root, record):
    task = record["task"]
    suffix = record.get("phase", record.get("teacher", ""))
    name = f"{task}_{suffix}_{record['variant']}_s{record['seed']}_lr{record.get('body_lr', record.get('lr'))}.json"
    path = root / name
    if path.exists():
        raise FileExistsError(f"refusing to overwrite research evidence: {path}")
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: record[k] for k in ("task", "variant", "seed", "train_seconds")}
                     | {"file": name, "loss": record.get("test_loss", record.get("validation_loss", record.get("relative_mse")))},
                     allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("pilot", "tune", "main", "regression"), required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--tune-steps", type=int, default=100)
    parser.add_argument("--regression-steps", type=int, default=400)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    for name in ("steps", "tune_steps", "regression_steps", "layers", "heads", "context",
                 "batch", "block_size", "rank", "eval_batches", "eval_every", "threads"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.dim % (2 * args.heads):
        parser.error("dim must be divisible by 2*heads for the narrow control")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    splits, vocab, data_info = corpus(args)
    validation, test = fixed_eval(splits[1], args, 2026), fixed_eval(splits[2], args, 2027)
    source_files = [Path(__file__), Path(__file__).resolve().parents[1] / "src/memory_native/research/pam.py",
                    Path(__file__).resolve().parents[1] / "src/memory_native/research/structured.py"]
    manifest = {"protocol_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                "data": data_info, "python": platform.python_version(), "torch": torch.__version__,
                "numpy": np.__version__, "device": "cpu", "cuda_available": torch.cuda.is_available(),
                "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                "validation_windows_sha256": hashlib.sha256(torch.cat([x.flatten() for x, _ in validation]).numpy().tobytes()).hexdigest(),
                "test_windows_sha256": hashlib.sha256(torch.cat([x.flatten() for x, _ in test]).numpy().tobytes()).hexdigest(),
                "limitations": ["small character model, not an LLM scale result", "CPU PyTorch reference timings",
                    "PAM replaces body linears only", "counter state stored in uint8, not packed six bits",
                    "memory accounting excludes activations and allocator/transient peak", "different operator families cannot share initial body weights"]}
    (args.output / f"manifest_{args.mode}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.mode == "regression":
        for teacher in ("dense", "blast"):
            for variant in args.variants:
                if variant == "dense_narrow":
                    continue
                for seed in args.seeds:
                    # A deliberately independent regression recipe, not LM LR tuning.
                    lr = .03 if variant in {"counter", "blast_counter"} else .01
                    write_record(args.output, regression(variant, seed, teacher, lr, args))
        return
    if args.mode == "pilot":
        for variant in args.variants:
            lr = .01 if variant in {"counter", "blast_counter"} else .003
            write_record(args.output, run_lm(variant, 90, lr, args.steps, args, splits[0], vocab, validation, None, "pilot"))
        return
    if args.mode == "tune":
        choices = {}
        for variant in args.variants:
            rates = [.003, .01, .03] if variant in {"counter", "blast_counter"} else [.001, .003, .01]
            records = []
            for lr in rates:
                record = run_lm(variant, 90, lr, args.tune_steps, args, splits[0], vocab, validation, None, "tune")
                records.append(record)
                write_record(args.output, record)
            best = min(records, key=lambda r: r["validation_loss"])
            choices[variant] = best["body_lr"]
        (args.output / "selected_lrs.json").write_text(json.dumps(choices, indent=2) + "\n")
        return
    choices = json.loads((args.output / "selected_lrs.json").read_text())
    for variant in args.variants:
        for seed in args.seeds:
            write_record(args.output, run_lm(variant, seed, choices[variant], args.steps, args,
                                           splits[0], vocab, validation, test, "main"))


if __name__ == "__main__":
    main()
