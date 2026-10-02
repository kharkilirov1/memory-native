#!/usr/bin/env python3
"""Summarize committed raw CPU BLAST/counter/PAM experiments without mixing phases.

Usage: python scripts/summarize_new_math.py --input results/new_math --output results/new_math
Requires NumPy and matplotlib. Only charlm phase=main and regression JSON records enter
statistics. Sample standard deviations describe seed variation, not confidence bounds.
State accounting is a lower bound, not measured training peak memory.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

VARIANTS = ("dense", "dense_narrow", "counter", "blast", "blast_counter", "lowrank", "pam_exact", "pam_surrogate")
COLORS = dict(zip(VARIANTS, ("#303841", "#7c8797", "#dc7744", "#2973a4", "#329665", "#b89a30", "#9256b8", "#ce537d")))
TINY_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
TINY_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
METRICS = (
    "test_loss", "test_perplexity", "validation_loss", "training_tokens_per_second", "train_seconds",
    "persistent_model_bytes", "adam_state_bytes", "retained_fp_gradient_bytes",
    "accounted_training_state_bytes", "fp_trainable_parameters", "body_coefficients", "counter_coefficients",
    "body_forward_pair_products_per_step", "body_forward_backward_pair_evaluations_per_step",
    "unchanged_attention_and_head_forward_macs_per_step", "training_tokens", "counter_weight_flips",
)


def stats(values):
    values = [float(value) for value in values]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("nonfinite metric in raw records")
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def read_inputs(root):
    mains, regressions, manifests, sources, selections = [], [], {}, [], []
    for path in sorted(root.rglob("*.json")):
        name = path.name
        if not (name.startswith("charlm_main_") or name.startswith("regression_")
                or name in {"manifest_main.json", "manifest_tune.json", "manifest_regression.json", "selected_lrs.json"}):
            continue
        raw = path.read_bytes()
        record = json.loads(raw)
        source = {"path": str(path.relative_to(root)), "sha256": hashlib.sha256(raw).hexdigest()}
        if name.startswith("manifest_"):
            manifests.setdefault(name, []).append({"source": source, "manifest": record})
        elif name == "selected_lrs.json":
            selections.append({"source": source, "selected_lrs": record})
        elif record.get("task") == "charlm" and record.get("phase") == "main":
            record["source"] = source
            mains.append(record)
        elif record.get("task") == "regression":
            record["source"] = source
            regressions.append(record)
        else:
            raise ValueError(f"unexpected record task/phase in {path}")
        sources.append(source)
    if not mains and not regressions:
        raise ValueError("no charlm main or regression records found; pilot/tune records are deliberately excluded")
    if mains and not manifests.get("manifest_main.json"):
        raise ValueError("charlm main records require manifest_main.json for data/evaluation provenance")
    if regressions and not manifests.get("manifest_regression.json"):
        raise ValueError("regression records require manifest_regression.json for runtime provenance")
    for entries in manifests.values():
        # Prefer an explicit top-level combined manifest over a family-specific one.
        entries.sort(key=lambda item: (item["source"]["path"].count("/"), item["source"]["path"]))
    # Refuse to pool corpus/protocol variants just because filenames happen to match.
    main_protocols = []
    for item in manifests.get("manifest_main.json", []):
        m = item["manifest"]
        a = m.get("args", {})
        main_protocols.append((m.get("data", {}).get("sha256"), m.get("validation_windows_sha256"),
            m.get("test_windows_sha256"), m.get("device"), m.get("torch"), m.get("python"),
            tuple(a.get(k) for k in ("dim", "layers", "heads", "context", "batch", "block_size", "rank",
                                    "steps", "eval_batches", "threads")),
            json.dumps(m.get("source_sha256"), sort_keys=True)))
    if len(set(main_protocols)) > 1:
        raise ValueError("multiple main manifests disagree on corpus/evaluation/runtime; summarize separately")
    seen = set()
    for r in mains:
        key = (r["variant"], r["seed"])
        if key in seen:
            raise ValueError(f"duplicate primary variant/seed {key}; do not mix independent reruns")
        seen.add(key)
        if r["variant"] not in VARIANTS:
            raise ValueError(f"unknown primary variant {r['variant']}")
        r["accounted_training_state_bytes"] = sum(r[k] for k in (
            "persistent_model_bytes", "adam_state_bytes", "retained_fp_gradient_bytes"))
        if not math.isclose(r["test_perplexity"], math.exp(r["test_loss"]), rel_tol=1e-6):
            raise ValueError(f"test perplexity disagrees with exp(test loss): {key}")
        m = manifests["manifest_main.json"][0]["manifest"]
        a = m.get("args", {})
        if r["steps"] != a.get("steps"):
            raise ValueError(f"primary run budget disagrees with manifest: {key}")
        expected_dim = a["dim"] // 2 if r["variant"] == "dense_narrow" else a["dim"]
        if any(r["shape"].get(k) != v for k, v in {"dim": expected_dim,
                **{k: a[k] for k in ("layers", "context", "batch", "block_size", "rank")}}.items()):
            raise ValueError(f"primary run shape disagrees with manifest: {key}")
        for c in r["curve"]:
            if not math.isfinite(float(c["validation_loss"])) or not math.isfinite(float(c["train_seconds"])):
                raise ValueError(f"nonfinite curve in primary record: {key}")
        if not r["curve"] or r["curve"][-1]["step"] != r["steps"]:
            raise ValueError(f"primary record is not a completed run: {key}")
        for selection in selections:
            chosen = selection["selected_lrs"].get(r["variant"])
            if chosen is not None and not math.isclose(r["body_lr"], chosen):
                raise ValueError(f"record disagrees with selected LR: {key}")
    regression_seen, teacher_signatures = set(), defaultdict(set)
    for r in regressions:
        key = (r["teacher"], r["variant"], r["seed"])
        if key in regression_seen:
            raise ValueError(f"duplicate regression teacher/variant/seed {key}")
        regression_seen.add(key)
        teacher_signatures[r["teacher"]].add((r["teacher_sha256"], r["dimension"],
                                              r["train_examples"], r["test_examples"], r["steps"]))
        stats([r["relative_mse"], r["train_seconds"], *[p["relative_mse"] for p in r["curve"]]])
    if any(len(s) > 1 for s in teacher_signatures.values()):
        raise ValueError("regression teacher identities or budgets disagree; summarize separately")
    return mains, regressions, manifests, sources, selections


def aggregate(mains, regressions, manifests, sources, selections):
    groups, regression_groups = defaultdict(list), defaultdict(list)
    for r in mains:
        groups[r["variant"]].append(r)
    for r in regressions:
        regression_groups[(r["teacher"], r["variant"])].append(r)
    summary = {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "statistic": "arithmetic mean and sample standard deviation over seeds (ddof=1); std=null for n<2",
        "memory_definition": "sum of model state tensors, Adam state tensors and retained FP parameter gradients; not measured peak",
        "timing_definition": "CPU optimization-loop wall time, excluding batch sampling, evaluation, setup and I/O",
        "manifests": manifests, "selected_lrs": selections, "sources": sources,
        "primary": {}, "paired_test_ce_vs_dense": {}, "regression": {}, "coverage": {}}
    for variant in VARIANTS:
        records = groups.get(variant, [])
        if not records:
            continue
        summary["primary"][variant] = {"seeds": sorted(r["seed"] for r in records),
            "body_lrs": sorted({r["body_lr"] for r in records}),
            "forward_pair_kinds": sorted({r.get("forward_pair_kind", "unrecorded") for r in records}),
            "backward_pair_kinds": sorted({r.get("backward_pair_kind", "unrecorded") for r in records}),
            "body_matrix_rank_bounds": records[0].get("body_matrix_rank_bounds"),
            "metrics": {key: stats([r[key] for r in records]) for key in METRICS},
            "records": sorted(records, key=lambda r: r["seed"])}
    dense = {r["seed"]: r for r in groups.get("dense", [])}
    for variant in VARIANTS:
        if variant == "dense" or variant not in groups:
            continue
        pairs = [{"seed": r["seed"], "ce_difference": r["test_loss"] - dense[r["seed"]]["test_loss"],
                  "perplexity_ratio": math.exp(r["test_loss"] - dense[r["seed"]]["test_loss"])}
                 for r in sorted(groups[variant], key=lambda r: r["seed"]) if r["seed"] in dense]
        summary["paired_test_ce_vs_dense"][variant] = {"pairs": pairs,
            "ce_difference": stats([p["ce_difference"] for p in pairs]),
            "perplexity_ratio": stats([p["perplexity_ratio"] for p in pairs])}
    for (teacher, variant), records in sorted(regression_groups.items()):
        summary["regression"].setdefault(teacher, {})[variant] = {
            "seeds": sorted(r["seed"] for r in records),
            "relative_mse": stats([r["relative_mse"] for r in records]),
            "train_seconds": stats([r["train_seconds"] for r in records]),
            "rank16_population_mse_lower_bound": stats([r["rank16_population_mse_lower_bound"] for r in records]),
            "records": sorted(records, key=lambda r: r["seed"])}
    for filename, observed in (("manifest_main.json", [(r["variant"], r["seed"]) for r in mains]),
                               ("manifest_regression.json", [(r["teacher"], r["variant"], r["seed"]) for r in regressions])):
        if filename not in manifests:
            continue
        expected = []
        expected_manifests = list(manifests[filename])
        if filename == "manifest_main.json":
            # Tuning manifests also identify planned families whose main runs may
            # not have started yet. Do not call a fast-only subset complete.
            expected_manifests += manifests.get("manifest_tune.json", [])
        for item in expected_manifests:
            a = item["manifest"]["args"]
            variants, seeds = a.get("variants", VARIANTS), a.get("seeds", [0, 1, 2])
            expected.extend([(v, s) for v in variants for s in seeds] if filename == "manifest_main.json" else [
                (t, v, s) for t in ("dense", "blast") for v in variants if v != "dense_narrow" for s in seeds])
        expected = set(expected)
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        summary["coverage"][filename] = {"expected_runs": len(expected), "observed_runs": len(observed),
                                          "missing": missing, "extra": extra, "complete": not missing and not extra}
    return summary


def figure_outputs(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 130, "svg.fonttype": "none"})
    outputs = []

    def save(fig, stem):
        fig.tight_layout()
        for extension in ("png", "svg"):
            name = f"{stem}.{extension}"
            fig.savefig(output / name, bbox_inches="tight")
            outputs.append(name)
        plt.close(fig)

    if summary["primary"]:
        for xfield, xlabel, stem in (("step", "Optimization steps", "charlm_validation_steps"),
                ("train_seconds", "CPU optimization-loop seconds (evaluation excluded)", "charlm_validation_time")):
            fig, ax = plt.subplots(figsize=(9, 5))
            for variant in VARIANTS:
                if variant not in summary["primary"]:
                    continue
                records = summary["primary"][variant]["records"]
                common_steps = sorted(set.intersection(*[set(c["step"] for c in r["curve"]) for r in records]))
                curves = [{c["step"]: c for c in r["curve"]} for r in records]
                x = [statistics.mean(c[step][xfield] for c in curves) for step in common_steps]
                y = [statistics.mean(c[step]["validation_loss"] for c in curves) for step in common_steps]
                for r in records:
                    ax.plot([c[xfield] for c in r["curve"]], [c["validation_loss"] for c in r["curve"]],
                            color=COLORS[variant], alpha=.17, linewidth=.8)
                ax.plot(x, y, label=f"{variant} (n={len(records)})", color=COLORS[variant], linewidth=1.8)
            ax.set(xlabel=xlabel, ylabel="Held-out validation CE (nats/character)",
                   title="TinyShakespeare: seed traces and means, fixed validation windows")
            ax.grid(alpha=.18)
            ax.legend(loc="best", fontsize=8, ncols=2)
            save(fig, stem)
        variants = [v for v in VARIANTS if v in summary["primary"]]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        values = [summary["primary"][v]["metrics"]["test_loss"] for v in variants]
        ax.bar(variants, [s["mean"] for s in values], color=[COLORS[v] for v in variants],
               yerr=[s["std"] or 0. for s in values], capsize=3)
        for i, v in enumerate(variants):
            ax.scatter([i] * len(summary["primary"][v]["records"]),
                       [r["test_loss"] for r in summary["primary"][v]["records"]], color="black", s=12, zorder=3)
        ax.set(ylabel="Test CE (nats/character)", title="Final held-out test: mean ± sample SD, dots are seeds")
        ax.tick_params(axis="x", rotation=20)
        save(fig, "charlm_test_ce")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        bottom = np.zeros(len(variants))
        for key, label, color in (("persistent_model_bytes", "Model tensors", "#4c78a8"),
                ("adam_state_bytes", "Adam tensors", "#f2a154"),
                ("retained_fp_gradient_bytes", "Retained FP gradients", "#74ae89")):
            values = np.array([summary["primary"][v]["metrics"][key]["mean"] / 2**20 for v in variants])
            ax.bar(variants, values, bottom=bottom, label=label, color=color)
            bottom += values
        ax.set(ylabel="Accounted state (MiB)", title="Tensor-state accounting: activations and transient peak excluded")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(fontsize=8)
        save(fig, "charlm_accounted_state")
    if summary["regression"]:
        teachers = sorted(summary["regression"])
        fig, axes = plt.subplots(1, len(teachers), figsize=(6 * len(teachers), 4.5), squeeze=False)
        for ax, teacher in zip(axes[0], teachers):
            for variant in VARIANTS:
                if variant not in summary["regression"][teacher]:
                    continue
                records = summary["regression"][teacher][variant]["records"]
                common = sorted(set.intersection(*[set(c["step"] for c in r["curve"]) for r in records]))
                curves = [{c["step"]: c["relative_mse"] for c in r["curve"]} for r in records]
                for r in records:
                    ax.plot([c["step"] for c in r["curve"]], [max(c["relative_mse"], 1e-12) for c in r["curve"]],
                            color=COLORS[variant], alpha=.15, linewidth=.8)
                ax.plot(common, [max(statistics.mean(c[s] for c in curves), 1e-12) for s in common],
                        color=COLORS[variant], label=variant, linewidth=1.6)
            if teacher == "dense":
                first = next(iter(summary["regression"][teacher].values()))
                bound = first["rank16_population_mse_lower_bound"]["mean"]
                ax.axhline(bound, color="#666666", linestyle=":", label="rank-16 population bound")
            ax.set(yscale="log", xlabel="Optimization steps", ylabel="Held-out relative MSE",
                   title=f"{teacher} teacher (32×32)")
            ax.grid(alpha=.18)
            ax.legend(fontsize=8)
        save(fig, "regression_teacher_comparison")
    return outputs


def format_stat(s, digits=3, divisor=1.):
    if s["mean"] is None:
        return "—"
    mean = s["mean"] / divisor
    if s["std"] is None:
        return f"{mean:.{digits}f} (n=1)"
    return f"{mean:.{digits}f} ± {s['std'] / divisor:.{digits}f}"


def commands(manifests):
    result = []
    for filename in ("manifest_tune.json", "manifest_main.json", "manifest_regression.json"):
        for item in manifests.get(filename, []):
            a = item["manifest"]["args"]
            result.append(command_for_args(a))
    return result


def command_for_args(a):
    args = ["python", "scripts/new_math_experiments.py"]
    # shlex.quote preserves paths without pretending argv recorded an original command string.
    import shlex
    for key, value in a.items():
        if value is None:
            continue
        args.append("--" + key.replace("_", "-"))
        if isinstance(value, list):
            args.extend(str(v) for v in value)
        else:
            args.append(str(value))
    return " ".join(shlex.quote(v) for v in args)


def markdown_report(summary, figures):
    lines = ["# Structured counter and PAM CPU experiment", "",
        "Generated from the raw JSON records indexed in [SUMMARY.json](SUMMARY.json). "
        "Reported ± values are sample standard deviations across seeds; they are not confidence intervals.", ""]
    for filename, coverage in summary["coverage"].items():
        status = "complete" if coverage["complete"] else "**partial**"
        lines += [f"`{filename}`: {coverage['observed_runs']}/{coverage['expected_runs']} runs, {status}.", ""]
        if coverage["missing"]:
            lines += ["Missing variant/seed tuples: `" + json.dumps(coverage["missing"]) + "`.", ""]
    manifest = next(iter(summary["manifests"].get("manifest_main.json", [])), {}).get("manifest")
    if manifest:
        a, data = manifest["args"], manifest["data"]
        corpus_name = "TinyShakespeare" if data["sha256"] == TINY_SHA256 else data.get("local_filename", "local text corpus")
        lines += [f"Corpus: {corpus_name}, {data['characters']:,} characters, {data['vocab_size']} character vocabulary; "
                  f"chronological 80/10/10 split ({data['split_characters']}). Corpus SHA-256: `{data['sha256']}`.", "",
            f"Primary budget: {a['steps']} steps, batch {a['batch']} × context {a['context']}, {a['layers']} blocks, "
            f"width {a['dim']} (dense_narrow: {a['dim']//2}), {a['heads']} heads, {a['threads']} CPU thread(s). "
            f"Seeds: `{a['seeds']}`. Fixed validation/test windows are identified by hashes in SUMMARY.json.", "",
            "The harness selects each body LR on validation with a separate seed-90 tuning stage; "
            "the actual tuning budgets and selected-LR files are retained in provenance when supplied. "
            "The FP shell uses LR 0.003, and test windows are evaluated once at the end of each primary run. "
            "No gradient clipping, dropout, teacher distillation or activation checkpointing is used in the LM pilot.", "",
            f"Runtime: Python {manifest['python']}, PyTorch {manifest['torch']}, NumPy {manifest['numpy']}; "
            "CPU reference implementation.", ""]
        if data["sha256"] == TINY_SHA256:
            lines += [f"Verified corpus bytes match the [Karpathy TinyShakespeare source]({TINY_URL}) "
                      "downloaded for this study; the SHA-256 pins the content independently of the live URL.", ""]
        tune_budgets = sorted({item["manifest"]["args"]["tune_steps"]
                               for item in summary["manifests"].get("manifest_tune.json", [])})
        if tune_budgets:
            lines += [f"Recorded tuning budget(s): `{tune_budgets}` steps per body LR, seed 90; "
                      "validation selects the LR and test CE is not used for selection.", ""]
    if summary["primary"]:
        lines += ["## Held-out character-model results", "",
            "| Variant | Seeds | Body LR | Test CE, nats/char | Test PPL | Body coefficients | Accounted state, MiB | Training tok/s |",
            "|---|---:|---|---:|---:|---:|---:|---:|"]
        for v in VARIANTS:
            if v not in summary["primary"]:
                continue
            g = summary["primary"][v]
            m = g["metrics"]
            lines.append(f"| {v} | {len(g['seeds'])} | {', '.join(str(lr) for lr in g['body_lrs'])} | "
                f"{format_stat(m['test_loss'], 4)} | {format_stat(m['test_perplexity'], 2)} | "
                f"{m['body_coefficients']['mean']:,.0f} | {format_stat(m['accounted_training_state_bytes'], 3, 2**20)} | "
                f"{format_stat(m['training_tokens_per_second'], 0)} |")
        lines += ["", "PPL is exp(CE) within each seed; mean PPL is not exp(mean CE). "
            "Each variant has a different initial operator except the matched dense/PAM weights. "
            "Matching the random seed does not make operator families identical at initialization.", "",
            "### Paired differences from dense", "",
            "Negative CE differences favor the variant. Pairing uses the same primary seed and held-out "
            "windows; only seeds available for both variants enter each row. Three seeds cannot establish "
            "robust significance or long-run convergence parity.", "",
            "| Variant | Paired seeds | Test CE difference, mean ± SD | Per-seed CE differences |",
            "|---|---:|---:|---|"]
        for v, p in summary["paired_test_ce_vs_dense"].items():
            differences = "; ".join(f"s{x['seed']}: {x['ce_difference']:+.4f}" for x in p["pairs"])
            lines.append(f"| {v} | {len(p['pairs'])} | {format_stat(p['ce_difference'], 4)} | {differences or '—'} |")
        lines += ["", "### State components", "",
            "These are tensor bytes after training: model buffers/parameters, Adam state and retained FP "
            "parameter gradients. Their sum excludes activations, decoded factor weights, factor intermediates, "
            "temporary correlation/gradient buffers, Python/runtime overhead and allocator behavior. "
            "**Training peak memory was not measured.** Counter codes here are uint8, not six-bit packed. "
            "PAM keeps full FP weights and Adam state; changing the product does not compress them.", "",
            "| Variant | Model, MiB | Adam, MiB | FP gradients, MiB | Body forward pair products/step |",
            "|---|---:|---:|---:|---:|"]
        for v in VARIANTS:
            if v not in summary["primary"]:
                continue
            m = summary["primary"][v]["metrics"]
            lines.append(f"| {v} | {format_stat(m['persistent_model_bytes'], 3, 2**20)} | "
                f"{format_stat(m['adam_state_bytes'], 3, 2**20)} | {format_stat(m['retained_fp_gradient_bytes'], 3, 2**20)} | "
                f"{m['body_forward_pair_products_per_step']['mean']:,.0f} |")
        lines += ["", "Pair-product counts are arithmetic accounting, not measured FLOPs or execution speed. "
            "Exact PAM backward evaluates slopes and exponent shifts, while surrogate PAM backward uses "
            "PAM products; three pair evaluations do not mean three ordinary MACs or three PAM products. "
            "The raw forward/backward kinds are preserved in SUMMARY.json. "
            "BLAST applies U/V/S factors directly without reconstructing full W; factor intermediate tensors "
            "still exist. Its p=16, rank=8 square-layer global rank is bounded by d/2, even though individual "
            "blocks use shared rank-8 factors. Rectangular layers have the corresponding input/output block "
            "bottleneck. The dense_narrow control reduces the complete model width rather than imposing this "
            "factorization. The lowrank arm approximates the BLAST coefficient budget with a lower plain "
            "matrix rank. Actual body coefficients and rank bounds "
            "are preserved in SUMMARY.json. "
            "Attention, head, norms, loss and optimizer remain ordinary PyTorch arithmetic; "
            "PAM changes only body-linear pair products.", ""]
        if "blast" in summary["primary"] and "lowrank" in summary["primary"]:
            blast = summary["primary"]["blast"]
            lowrank = summary["primary"]["lowrank"]
            p_blast, p_lowrank = (g["metrics"]["body_coefficients"]["mean"] for g in (blast, lowrank))
            bounds_blast = blast["body_matrix_rank_bounds"]
            bounds_lowrank = lowrank["body_matrix_rank_bounds"]
            lines += [f"Measured coefficient budgets: BLAST {p_blast:,.0f}; lowrank {p_lowrank:,.0f} "
                      f"({100*(p_lowrank/p_blast-1):+.2f}%). Recorded body rank-bound values: "
                      f"BLAST `{sorted(set(bounds_blast)) if bounds_blast else 'unrecorded'}`; "
                      f"lowrank `{sorted(set(bounds_lowrank)) if bounds_lowrank else 'unrecorded'}`.", ""]
        for stem in ("charlm_validation_steps", "charlm_validation_time", "charlm_test_ce", "charlm_accounted_state"):
            if f"{stem}.png" in figures:
                lines += [f"![{stem.replace('_', ' ')}]({stem}.png)", ""]
    if summary["regression"]:
        lines += ["## Two-teacher regression control", "",
            "Independent 32×32 teacher regression with 1536 training and 512 held-out Gaussian examples. "
            "Teachers and samples are shared across variants. This uses an independent fixed regression LR "
            "recipe, not the LM-selected LRs, so it does not isolate only architecture capacity.", "",
            "| Teacher | Variant | Seeds | Final relative MSE, mean ± SD | Training seconds |",
            "|---|---|---:|---:|---:|"]
        for teacher, groups in summary["regression"].items():
            for v in VARIANTS:
                if v in groups:
                    g = groups[v]
                    lines.append(f"| {teacher} | {v} | {len(g['seeds'])} | {format_stat(g['relative_mse'], 5)} | "
                                 f"{format_stat(g['train_seconds'], 2)} |")
        lines += ["", "The dense teacher includes components outside the BLAST rank-16 bottleneck. "
            "The SVD line is a population relative-MSE lower bound for unrestricted rank-16 linear "
            "regression under isotropic Gaussian inputs, not a finite held-out-sample guarantee. "
            "The BLAST teacher is representable by floating-point BLAST; ternary counter factors impose "
            "additional restrictions and are not guaranteed to represent the same target exactly. "
            "Final MSE reflects optimization and state dynamics as well as model capacity. "
            "The log-scale figure clips values below 1e-12 for display; raw values remain in SUMMARY.json.", ""]
        if "regression_teacher_comparison.png" in figures:
            lines += ["![Regression teacher comparison](regression_teacher_comparison.png)", ""]
    lines += ["## What this experiment establishes", "",
        "The tables are small controlled CPU pilots of these implemented operators and training recipes. "
        "They do not establish LLM/pretraining convergence, donor-quality retention, GPU throughput, energy "
        "savings or production memory fit. The primary families ran as concurrent single-thread processes "
        "on separate pinned CPU cores sharing a host/cgroup quota; wall timings can include contention. "
        "CPU timing depends on this Python/PyTorch implementation and "
        "sequential many-small-matmul/frexp/ldexp reference paths; it is not hardware performance proof. "
        "Timings exclude setup, sampling, evaluation and I/O. Recorded seeds and a fixed held-out slice "
        "do not capture dataset or hardware uncertainty.", "",
        "BLAST and PAM are prior art: [BLAST, arXiv:2410.21262](https://arxiv.org/abs/2410.21262) "
        "and [Multiplication-Free Transformer Training via Piecewise Affine Operations, "
        "arXiv:2305.17190](https://arxiv.org/abs/2305.17190). The new experiment here combines BLAST "
        "factors with memory-native counter updates and compares exact versus surrogate PAM backward. "
        "It does not claim invention of either operator.", "",
        "## Reproduction", "",
        "These commands are reconstructed from recorded manifest arguments; they are not an independently "
        "verified capture of the original shell command. Run the tuning stage first in the same output "
        "directory so selected_lrs.json exists, and preserve its raw records. Corpus bytes must match the "
        "recorded SHA-256. Source JSON SHA-256 values and manifest contents are retained in SUMMARY.json.", "",
        "```bash", *commands(summary["manifests"]),
        "python scripts/summarize_new_math.py --input results/new_math --output results/new_math", "```", "",
        "Plots are available as both PNG and SVG. No unpublished GPU result or missing historical artifact "
        "has been filled in by this report.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        mains, regressions, manifests, sources, selections = read_inputs(args.input)
        summary = aggregate(mains, regressions, manifests, sources, selections)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    figures = figure_outputs(summary, args.output)
    summary["figures"] = figures
    (args.output / "SUMMARY.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (args.output / "REPORT.md").write_text(markdown_report(summary, figures))
    print(json.dumps({"primary_runs": len(mains), "regression_runs": len(regressions),
                      "coverage": summary["coverage"], "output": str(args.output), "figures": figures}))


if __name__ == "__main__":
    main()
