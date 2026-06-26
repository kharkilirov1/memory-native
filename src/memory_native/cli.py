"""Console entry points: a char-LM parity gate and a training-peak memory gate.

  memory-native-charlm  --kinds dense,qat,counter,counter_rms [--data-path FILE] ...
  memory-native-memgate --config tiny [--device cuda]

Both run on stock PyTorch (CPU/CUDA), no engine, and fall back to a synthetic corpus offline.
"""
from __future__ import annotations

import argparse
import time

import torch

from .data import get_batch, load_corpus
from .memory import fmt_bytes, memory_report, peak_training_memory
from .models import CONFIGS, GPT
from .optimizers import available_optimizers, build_optimizer


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        print("cuda requested but not available; using cpu")
        return torch.device("cpu")
    return torch.device(name)


def _train_one(kind: str, args, vocab, train_data, val_data, device) -> dict:
    cfg = CONFIGS[args.config]
    counter_kw = dict(lr=args.fs_lr if kind == "counter" else args.fs_lr_rms,
                      lr_scale=args.fs_lr_scale, C=args.C, tile_rows=args.tile_rows)
    kw = counter_kw if kind in ("counter", "counter_rms") else {}
    model = GPT(kind, cfg, **kw).to(device)
    model.train()

    lr = args.base_lr if kind in ("dense", "qat") else args.fp_lr
    opt = build_optimizer(args.optimizer, model.trainable_parameters(), lr)
    g = torch.Generator().manual_seed(args.seed)

    @torch.no_grad()
    def evaluate():
        model.eval()
        tot = 0.0
        for _ in range(args.eval_iters):
            x, y = get_batch(val_data, cfg.block_size, args.batch, device, g)
            _, loss = model(x, y)
            tot += loss.item()
        model.train()
        return tot / args.eval_iters

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    for step in range(args.steps + 1):
        x, y = get_batch(train_data, cfg.block_size, args.batch, device, g)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()  # counter layers self-update during backward
        opt.step()
        if step % max(1, args.steps // 6) == 0:
            print(f"  [{kind:12s}] step {step:5d}  train {loss.item():.4f}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.time() - started
    final_val = evaluate()
    rep = memory_report(model)
    tokens = (args.steps + 1) * args.batch * cfg.block_size
    tok_s = tokens / max(elapsed, 1e-9)
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    peak_str = f"  peak {fmt_bytes(peak)}" if peak else ""
    print(f"  [{kind:12s}] final val {final_val:.4f}  persistent {fmt_bytes(rep['persistent_bytes'])}"
          f"  {tok_s:,.0f} tok/s{peak_str}  ({elapsed:.0f}s)")
    return {"final_val": final_val, "report": rep, "tok_s": tok_s, "peak_bytes": peak}


def charlm_main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="char-LM parity gate (counter vs AdamW baselines)")
    ap.add_argument("--config", choices=list(CONFIGS), default="tiny")
    ap.add_argument("--kinds", default="dense,qat,counter,counter_rms")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--C", type=int, default=8)
    ap.add_argument("--fs-lr", type=float, default=0.04)
    ap.add_argument("--fs-lr-rms", type=float, default=3e-3)
    ap.add_argument("--fs-lr-scale", type=float, default=2e-4)
    ap.add_argument("--fp-lr", type=float, default=2e-3)
    ap.add_argument("--base-lr", type=float, default=3e-3)
    ap.add_argument("--tile-rows", type=int, default=0,
                    help="0=untiled update (fast, default); >0 tiles grad_w to never "
                         "materialize the full [out,in] gradient (strict, ~3x slower)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--optimizer", default="adamw", choices=available_optimizers(),
                    help="optimizer for the trainable (non-counter) params: "
                         "adamw | bnb8 | galore | lomo")
    args = ap.parse_args(argv)

    device = _device(args.device)
    train_data, val_data, vocab = load_corpus(device, args.data_path)
    # vocab from the corpus overrides the config's placeholder vocab_size
    cfg = CONFIGS[args.config]
    CONFIGS[args.config] = cfg.__class__(vocab, cfg.block_size, cfg.n_layer, cfg.n_head, cfg.n_embd)
    print(f"corpus vocab={vocab}  config={args.config}  device={device}")
    print("=" * 64)

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    res = {k: _train_one(k, args, vocab, train_data, val_data, device) for k in kinds}
    print("=" * 64)
    ref_key = "dense" if "dense" in res else next(iter(res))
    ref = res[ref_key]["final_val"]
    print("FINAL val loss:")
    for k in kinds:
        v = res[k]["final_val"]
        print(f"  {k:14s} {v:.4f}   gap {v-ref:+.4f} ({(v-ref)/ref*100:+.1f}% vs {ref_key})")
    if "qat" in res:
        qv = res["qat"]["final_val"]
        for k in ("counter", "counter_rms"):
            if k in res:
                iso = res[k]["final_val"] - qv
                print(f"  ISOLATION {k} - qat = {iso:+.4f} ({iso/qv*100:+.1f}%) = counter-optimizer cost")


def memgate_main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="training-peak memory gate: counter_rms vs dense across memory-efficient optimizers")
    ap.add_argument("--config", choices=list(CONFIGS), default="tiny")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--optimizers", default="adamw,galore,lomo",
                    help="comma list of dense baselines to compare against: "
                         f"{','.join(available_optimizers())}")
    args = ap.parse_args(argv)
    device = _device(args.device)
    cfg = CONFIGS[args.config]

    import torch.nn.functional as F
    x = torch.randint(0, cfg.vocab_size, (args.batch, cfg.block_size), device=device)
    y = torch.randint(0, cfg.vocab_size, (args.batch, cfg.block_size), device=device)

    def step(model, opt=None):
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        if opt is not None:
            opt.step()

    # counter_rms: self-updating, no optimizer over its ternary weights.
    counter = GPT("counter_rms", cfg).to(device).train()
    cpeak = peak_training_memory(lambda: step(counter), device)
    cr = memory_report(counter)

    print(f"device={device}  config={args.config}  batch={args.batch}")
    print(f"persistent state: counter_rms={fmt_bytes(cr['persistent_bytes'])}  "
          f"(counter weights packed-to-6bit would be {fmt_bytes(cr['counter_packed_6bit_bytes'])})")

    rows = []
    for oname in [o.strip() for o in args.optimizers.split(",") if o.strip()]:
        dense = GPT("dense", cfg).to(device).train()
        try:
            dopt = build_optimizer(oname, dense.trainable_parameters(), lr=3e-3)
        except RuntimeError as exc:
            print(f"  dense+{oname:7s}: SKIP ({exc})")
            continue
        dpeak = peak_training_memory(lambda: step(dense, dopt), device)
        rows.append((oname, dpeak, memory_report(dense)))

    if device.type == "cuda":
        print(f"TRAINING PEAK (max_memory_allocated):  counter_rms = {fmt_bytes(cpeak)}")
        for oname, dpeak, _ in rows:
            print(f"  dense+{oname:7s} = {fmt_bytes(dpeak):>12s}   "
                  f"{dpeak/max(cpeak,1):.2f}x of counter_rms")
    else:
        print("training peak: CPU has no allocator-peak API; run with --device cuda for the "
              "real peak. Persistent-state comparison:")
        for oname, _, dr in rows:
            print(f"  dense+{oname:7s} persistent {fmt_bytes(dr['persistent_bytes'])}")


if __name__ == "__main__":
    charlm_main()
