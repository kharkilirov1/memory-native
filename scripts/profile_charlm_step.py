#!/usr/bin/env python
"""Root-cause profiler for the char-LM gate: dense vs the counter layer kinds.

Reproduces the char-LM step exactly (same model factory, same data path, same optimizer
split) and answers two questions with one run:

* how many tokens/s each layer kind delivers at the *full-model* level, and
* where the time goes: top CUDA kernels and top CPU-side ops for one training step,
  measured with torch.profiler (CPU + CUDA activities) on warmed-up steps.

No overlapping profiler intervals are summed as independent costs: the tables report
self/device time per op, and the step wall time is measured separately with CUDA events.

Usage (on a GPU box):
    python scripts/profile_charlm_step.py --kinds dense,counter_rms,counter_packed \
        --steps 20 --batch 32 --out-dir /tmp/prof
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory_native.data import get_batch, load_corpus  # noqa: E402
from memory_native.memory import memory_report  # noqa: E402
from memory_native.models import CONFIGS, GPT  # noqa: E402
from memory_native.optimizers import build_optimizer  # noqa: E402

COUNTER_KINDS = {"counter", "counter_rms", "counter_packed", "counter_triton",
                 "group_scale_packed", "group_packed", "group", "group_scale"}


def build(kind: str, cfg, device, *, lr: float, lr_scale: float, C: int):
    """Same construction as memory_native.cli._train_one, but tolerant of per-kind kwargs."""
    kw = {"lr": lr, "lr_scale": lr_scale, "C": C, "tile_rows": 0} if kind in COUNTER_KINDS else {}
    while True:
        try:
            model = GPT(kind, cfg, **kw).to(device).train()
            return model
        except TypeError as exc:
            msg = str(exc)
            dropped = None
            for key in ("tile_rows", "lr_scale", "C", "lr"):
                if key in msg and key in kw:
                    dropped = key
                    break
            if dropped is None:
                raise
            del kw[dropped]


def counter_layers(model):
    out = []
    for mod in model.modules():
        if hasattr(mod, "update_enabled") and hasattr(mod, "in_features"):
            out.append(mod)
    return out


def one_step(model, opt, x, y):
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()          # counter layers self-update inside backward
    opt.step()
    return loss


def timed_steps(model, opt, x, y, steps: int) -> dict:
    for _ in range(3):
        one_step(model, opt, x, y)
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(steps):
        one_step(model, opt, x, y)
    ev1.record()
    torch.cuda.synchronize()
    ms = ev0.elapsed_time(ev1) / steps
    tokens = x.numel()
    return {"step_ms": ms, "tokens_per_step": tokens,
            "tok_s": tokens / (ms / 1e3)}


def profile_steps(model, opt, x, y, steps: int, out_dir: Path, kind: str) -> dict:
    from torch.profiler import ProfilerActivity, profile
    for _ in range(2):
        one_step(model, opt, x, y)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        for _ in range(steps):
            one_step(model, opt, x, y)
        torch.cuda.synchronize()
    trace = out_dir / f"trace_{kind.replace('.', '_')}.json"
    prof.export_chrome_trace(str(trace))

    # CPU-vs-GPU attribution per step: if CPU total > GPU total the step is dispatch-bound and
    # kernel micro-optimisation cannot help (this decides the next lever).
    cpu_total = 0.0
    cuda_total = 0.0
    launches = 0
    for evt in prof.key_averages():
        cpu_total += getattr(evt, "self_cpu_time_total", 0.0) or 0.0
        cuda_total += getattr(evt, "self_device_time_total", 0.0) or 0.0
        if evt.key == "cudaLaunchKernel":
            launches = evt.count
    totals = {"cpu_ms_per_step": cpu_total / 1e3 / steps,
              "cuda_ms_per_step": cuda_total / 1e3 / steps,
              "launches_per_step": launches / steps,
              "bound": ("cpu" if cpu_total > cuda_total else "gpu")}

    def table(sort_key: str, n: int = 20):
        rows = []
        for evt in prof.key_averages():
            cuda = getattr(evt, "self_device_time_total", 0.0) or 0.0
            cpu = getattr(evt, "self_cpu_time_total", 0.0) or 0.0
            count = evt.count
            if cuda <= 0 and cpu <= 0:
                continue
            rows.append({"op": evt.key, "calls": count,
                         "cuda_ms_total": cuda / 1e3, "cpu_ms_total": cpu / 1e3,
                         "cuda_ms_per_step": cuda / 1e3 / steps,
                         "cpu_ms_per_step": cpu / 1e3 / steps})
        key = "cuda_ms_total" if sort_key == "cuda" else "cpu_ms_total"
        rows.sort(key=lambda r: r[key], reverse=True)
        return rows[:n]

    out = {"cuda_top20": table("cuda"), "cpu_top20": table("cpu"),
           "trace": str(trace), **totals}
    print(f"  attribution: CPU {totals['cpu_ms_per_step']:.2f} ms/step | GPU "
          f"{totals['cuda_ms_per_step']:.2f} ms/step | {totals['launches_per_step']:.0f} "
          f"launches/step -> {totals['bound'].upper()}-bound")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="tiny")
    ap.add_argument("--kinds", default="dense,counter_rms,counter_packed,counter_triton,group_scale_packed")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-3, help="FP optimizer lr")
    ap.add_argument("--fs-lr-rms", type=float, default=3e-3)
    ap.add_argument("--fs-lr-scale", type=float, default=2e-4)
    ap.add_argument("--C", type=int, default=8)
    ap.add_argument("--no-update", action="store_true",
                    help="also time the model with every counter layer's update disabled")
    ap.add_argument("--out-dir", default=str(ROOT / "research/results/speed_recovery_2026_10_18"))
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available")
        return 0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg0 = CONFIGS[args.config]

    torch.manual_seed(args.seed)
    train_data, val_data, vocab = load_corpus(device, None, download=True)
    # GPTConfig is a frozen dataclass: rebuild it with the corpus vocab like cli.py does.
    cfg = type(cfg0)(vocab, cfg0.block_size, cfg0.n_layer, cfg0.n_head, cfg0.n_embd)
    print(f"corpus vocab={vocab} config={args.config} device={device}")
    g = torch.Generator().manual_seed(args.seed)
    x, y = get_batch(train_data, cfg.block_size, args.batch, device, g)

    results = {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
               "torch": torch.__version__, "config": args.config, "batch": args.batch,
               "batch_tokens": int(x.numel()),
               "commit": __import__("subprocess").run(
                   ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                   capture_output=True, text=True).stdout.strip(),
               "variants": {}}

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    for kind in kinds:
        print(f"\n=== {kind} ===")
        model = build(kind, cfg, device, lr=args.fs_lr_rms, lr_scale=args.fs_lr_scale, C=args.C)
        opt = build_optimizer("adamw", model.trainable_parameters(), args.lr) \
            if list(model.trainable_parameters()) else None
        n_counter = len(counter_layers(model))
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        entry = {"counter_layers": n_counter, "kind": kind}
        timing = timed_steps(model, opt, x, y, args.steps)
        entry.update(timing)
        report = memory_report(model)
        entry["persistent_bytes"] = report["persistent_bytes"]
        entry["persistent_mib"] = report["persistent_bytes"] / 2 ** 20
        entry["peak_alloc_mib"] = (torch.cuda.max_memory_allocated() / 2 ** 20
                                   if device.type == "cuda" else 0.0)
        if device.type == "cuda":
            peak2 = torch.cuda.memory_stats().get("allocated_bytes.all.peak", 0) / 2 ** 20
            entry["peak_reserved_mib"] = (
                torch.cuda.memory_stats().get("reserved_bytes.all.peak", 0) / 2 ** 20)
            entry["peak_alloc_meta_mib"] = peak2
        print(f"  {n_counter} counter layers | step {timing['step_ms']:.2f} ms | "
              f"{timing['tok_s']:,.0f} tok/s | persistent {entry['persistent_mib']:.2f} MiB | "
              f"peak {entry['peak_alloc_mib']:.1f} MiB")
        prof = profile_steps(model, opt, x, y, min(args.steps, 15), out_dir, kind)
        entry["profile"] = prof
        print("  top CUDA ops (ms/step):")
        for row in prof["cuda_top20"][:12]:
            print(f"    {row['cuda_ms_per_step']:8.3f}  x{row['calls']:4d}  {row['op'][:70]}")
        print("  top CPU ops (ms/step):")
        for row in prof["cpu_top20"][:12]:
            print(f"    {row['cpu_ms_per_step']:8.3f}  x{row['calls']:4d}  {row['op'][:70]}")

        if args.no_update and entry["counter_layers"]:
            for layer in counter_layers(model):
                layer.update_enabled = False
            timing2 = timed_steps(model, opt, x, y, args.steps)
            entry["no_update"] = timing2
            print(f"  update disabled: step {timing2['step_ms']:.2f} ms | "
                  f"{timing2['tok_s']:,.0f} tok/s")
            for layer in counter_layers(model):
                layer.update_enabled = True

        results["variants"][kind] = entry
        with open(out_dir / "profiling.json", "w") as fh:
            json.dump(results, fh, indent=1)
        del model, opt
        torch.cuda.empty_cache()

    print(f"\nartifacts: {out_dir/'profiling.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
