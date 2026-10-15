"""CPU ablations of opt-in counter dynamics; synthetic data, not an LLM quality claim."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
from pathlib import Path

import torch

from memory_native import PackedRMSCounterLinear, RMSCounterLinear, configure_rounding_streams


def recovery(seed, beta, correction, rebase, steps):
    gen = torch.Generator().manual_seed(seed)
    donor = torch.randn(32, 32, generator=gen) / 32 ** 0.5
    train_x = torch.randn(512, 32, generator=gen)
    valid_x = torch.randn(256, 32, generator=gen)
    train_y, valid_y = train_x @ donor.T, valid_x @ donor.T
    torch.manual_seed(seed)
    layer = PackedRMSCounterLinear.from_dense(
        donor, C=11, lr=0.002, lr_scale=0.0002, rms_beta=beta,
        rms_bias_correction=correction, scale_rebase=rebase)
    configure_rounding_streams(layer, seed)

    def mse():
        with torch.no_grad():
            return (layer(valid_x) - valid_y).square().mean().item()

    initial = mse()
    early = []
    for step in range(steps):
        start = (step % 4) * 128
        loss = (layer(train_x[start:start + 128]) - train_y[start:start + 128]).square().mean()
        loss.backward()
        if step < 10:
            early.append(mse())
    return dict(seed=seed, beta=beta, correction=correction, rebase=rebase,
                initial_mse=initial, first_step_mse=early[0], first_10_max_mse=max(early),
                final_mse=mse(), weight_flips=int(layer.weight_flips),
                rms_buffer_bytes=layer.v.numel() * layer.v.element_size(),
                extra_count_bytes=(layer.rms_updates.numel() * layer.rms_updates.element_size()
                                   if correction else 0))


def rebase_probe(mode, cumulative=False):
    layer = RMSCounterLinear(64, 64, C=11, lr=0, lr_scale=0.01,
                             use_rms=False, scale_rebase=mode)
    t = torch.zeros(64, 64, dtype=torch.int16)
    c = torch.full_like(t, 5)
    t[:, 0], c[:, 0] = 1, 0
    layer.load_counter_state(torch.ones(64, 1), t, c)
    configure_rounding_streams(layer, 91)
    grad = torch.zeros(64, 64)
    touched = torch.zeros_like(t, dtype=torch.bool)
    rebases = 0
    max_relative_staleness = 0
    for step in range(1, 1001):
        # One driver synapse per row moves the shared scale; the other 63 receive
        # zero gradient. lr=0 isolates scale rebasing from gradient counter ticks.
        target_scale = 1.002 ** step if cumulative else 1 + (-1) ** step * 0.002
        grad[:, 0] = (layer.scale[:, 0] - target_scale) * 64 ** 0.5 / layer.lr_scale
        previous_base = layer.s_base.clone()
        layer.apply_update_from_grad_w(grad)
        touched |= layer._decode_rows(0, 64)[1] != c
        rebases += int(not torch.equal(previous_base, layer.s_base))
        if mode == "threshold":
            max_relative_staleness = max(max_relative_staleness,
                                        float((layer.s_base / layer.scale - 1).abs().max()))
    return dict(mode=mode, cumulative=cumulative, changed_fraction=float(touched[:, 1:].float().mean()),
                rebase_steps=rebases if mode == "threshold" else 1000,
                final_scale=float(layer.scale[0, 0]),
                max_relative_staleness=max_relative_staleness if mode == "threshold" else None)


def stream_probe():
    model = torch.nn.Sequential(RMSCounterLinear(256, 256), RMSCounterLinear(256, 256))
    ids = configure_rounding_streams(model, 42)
    value = torch.full((256, 256), 0.5)
    a, b = [layer._round_counter(value) for layer in model]
    return dict(stream_ids=ids, samples=a.numel(), mismatch_fraction=float((a != b).float().mean()),
                round_up_fractions=[float(a.mean()), float(b.mean())])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 10 or args.seeds < 1:
        parser.error("steps >= 10 and seeds >= 1 required")
    torch.set_num_threads(1)
    rows = [recovery(seed, beta, correction, rebase, args.steps)
            for beta in (0.9, 0.99)
            for correction, rebase in ((False, "eager"), (True, "eager"),
                                       (False, "threshold"), (True, "threshold"))
            for seed in range(args.seeds)]
    summaries = []
    for beta in (0.9, 0.99):
        for correction, rebase in ((False, "eager"), (True, "eager"),
                                   (False, "threshold"), (True, "threshold")):
            subset = [r for r in rows if (r["beta"], r["correction"], r["rebase"]) ==
                      (beta, correction, rebase)]
            summaries.append(dict(beta=beta, correction=correction, rebase=rebase,
                                  final_mse_mean=statistics.mean(r["final_mse"] for r in subset),
                                  final_mse_min=min(r["final_mse"] for r in subset),
                                  final_mse_max=max(r["final_mse"] for r in subset),
                                  first_step_mse_mean=statistics.mean(r["first_step_mse"] for r in subset)))
    root = Path(__file__).resolve().parents[1]
    sources = ["src/memory_native/counter.py", "src/memory_native/rounding.py",
               "src/memory_native/packed.py", "scripts/benchmark_counter_stability.py"]
    result = dict(task="synthetic dense-donor linear recovery; held-out inputs, same teacher",
                  runtime=dict(python=platform.python_version(), torch=torch.__version__,
                               device="cpu", threads=1), steps=args.steps, seeds=args.seeds,
                  source_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources},
                  runs=rows, summaries=summaries,
                  rebase_probes=[rebase_probe("eager"), rebase_probe("threshold"),
                                 rebase_probe("threshold", cumulative=True)],
                  stream_probe=stream_probe(),
                  limits="All recovery variants use named hash SR. Fixed hyperparameters, no tuning. "
                         "No real corpus, GPU timing, peak-memory, or multi-process DDP validation.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("summaries", "rebase_probes", "stream_probe")}, indent=2))


if __name__ == "__main__":
    main()
