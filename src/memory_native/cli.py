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

    opt = torch.optim.AdamW(model.trainable_parameters(),
                            lr=args.base_lr if kind in ("dense", "qat") else args.fp_lr)
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

    started = time.time()
    for step in range(args.steps + 1):
        x, y = get_batch(train_data, cfg.block_size, args.batch, device, g)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()  # counter layers self-update during backward
        opt.step()
        if step % max(1, args.steps // 6) == 0:
            print(f"  [{kind:12s}] step {step:5d}  train {loss.item():.4f}")
    final_val = evaluate()
    rep = memory_report(model)
    print(f"  [{kind:12s}] final val {final_val:.4f}  "
          f"persistent {fmt_bytes(rep['persistent_bytes'])}  "
          f"({time.time()-started:.0f}s)")
    return {"final_val": final_val, "report": rep}


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
    ap.add_argument("--tile-rows", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-path", default=None)
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
    ap = argparse.ArgumentParser(description="training-peak memory gate (counter vs dense+AdamW)")
    ap.add_argument("--config", choices=list(CONFIGS), default="tiny")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    device = _device(args.device)
    cfg = CONFIGS[args.config]

    def batch():
        x = torch.randint(0, cfg.vocab_size, (args.batch, cfg.block_size), device=device)
        y = torch.randint(0, cfg.vocab_size, (args.batch, cfg.block_size), device=device)
        return x, y

    import torch.nn.functional as F
    x, y = batch()

    def step(model, opt=None):
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        if opt is not None:
            opt.step()

    counter = GPT("counter_rms", cfg).to(device).train()
    dense = GPT("dense", cfg).to(device).train()
    dopt = torch.optim.AdamW(dense.trainable_parameters(), lr=3e-3)

    cpeak = peak_training_memory(lambda: step(counter), device)
    dpeak = peak_training_memory(lambda: step(dense, dopt), device)

    cr, dr = memory_report(counter), memory_report(dense)
    print(f"device={device}  config={args.config}")
    print(f"persistent state: counter_rms={fmt_bytes(cr['persistent_bytes'])}  "
          f"dense={fmt_bytes(dr['persistent_bytes'])}")
    print(f"counter weights packed-to-6bit would be {fmt_bytes(cr['counter_packed_6bit_bytes'])}")
    if device.type == "cuda":
        print(f"TRAINING PEAK (max_memory_allocated): counter_rms={fmt_bytes(cpeak)}  "
              f"dense+AdamW={fmt_bytes(dpeak)}  ratio={dpeak/max(cpeak,1):.2f}x")
    else:
        print("training peak: CPU has no allocator-peak API; run with --device cuda for the "
              "real peak. (persistent-state comparison above still holds.)")


if __name__ == "__main__":
    charlm_main()
