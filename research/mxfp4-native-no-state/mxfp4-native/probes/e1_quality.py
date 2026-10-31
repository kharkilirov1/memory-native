#!/usr/bin/env python
"""E1 — quality at *measured* bits, on a causal transformer (not a per-position MLP).

The first version of this probe reused the smoke-test architecture from
tests/test_packed_na.py, which has no context mixing: every arm converged to the
bigram floor (3.49 bits/char vs 3.47 for an ideal bigram) and the comparison was
meaningless. This version uses the same attention block as probes/charlm_phase2.py,
which is what the project's historical quality tables were measured with.

Loss is reported in bits/char (divided by ln 2), matching those tables.

Arms (measured bits/param at d=128):
    dense_bf16   96.00   fp32 master + bf16 autocast + AdamW
    s9_nibble     4.41   MXFP4 E2M1 ladder, stateless SR, nibble-packed
    na5           5.41   PackedNA5
    na6           6.41   PackedNA6

Usage:
    python probes/e1_quality.py --d 128 --steps 1500 --seeds 0,1
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import urllib.request
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mx6.layer import MX6Linear, MXNLinear          # noqa: E402
from mx6.packed_na import PackedNA, PackedNA5, PackedNA6  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SHAKESPEARE = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# Magnitude ladders, so ladder and counter depth can be varied independently.
LADDERS = {
    "wide": [0.0, 0.5, 3.0],            # NA5's ladder
    "e2m1": [0.0, 0.5, 1.0, 6.0],       # NA6's ladder
}

# Code bits only; the E8M0 scale (+0.25) and the per-row RMS (32/in) are added per width.
CODE_BITS = {"dense_bf16": 96.0, "s9_nibble": 4.0}

# Set from --rescale / --scale-mode / --clip.
RESCALE_EVERY = 0
SCALE_MODE = "headroom"
CLIP_FRAC = 0.0


def parse_na(arm: str):
    """'na_wide_c3' -> (ladder name, C)."""
    _, ladder, c = arm.split("_")
    return ladder, int(c[1:])


def load_text(path: Path, chars: int = 200_000) -> str:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SHAKESPEARE, path)
    return path.read_text(encoding="utf-8")[:chars]


def make_linear(arm: str, i: int, o: int, lr: float) -> nn.Module:
    if arm == "dense_bf16":
        return nn.Linear(i, o)
    if arm == "s9_nibble":
        return MX6Linear(i, o, integrator=0, compact_state=True, lr=lr)
    if arm == "na5":
        return MXNLinear(i, o, PackedNA5, lr=lr)
    if arm == "na6":
        return MXNLinear(i, o, PackedNA6, lr=lr)
    if arm.startswith("na_"):
        ladder, c = parse_na(arm)
        return MXNLinear(i, o, PackedNA(mags_unsigned=LADDERS[ladder], C=c), lr=lr,
                         rescale_every=RESCALE_EVERY, scale_mode=SCALE_MODE,
                         clip_frac=CLIP_FRAC)
    raise ValueError(arm)


class Block(nn.Module):
    """Causal self-attention + MLP, same shape as probes/charlm_phase2.py."""

    def __init__(self, arm: str, d: int, lr: float):
        super().__init__()
        self.d = d
        self.ln1 = nn.LayerNorm(d)
        self.qkv = make_linear(arm, d, 3 * d, lr)
        self.proj = make_linear(arm, d, d, lr)
        self.ln2 = nn.LayerNorm(d)
        self.fc = make_linear(arm, d, 4 * d, lr)
        self.out = make_linear(arm, 4 * d, d, lr)

    def forward(self, x):
        t = x.shape[1]
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        att = att.masked_fill(mask, float("-inf")).softmax(dim=-1)
        x = x + self.proj(att @ v)
        return x + self.out(torch.nn.functional.gelu(self.fc(self.ln2(x))))


class CharLM(nn.Module):
    def __init__(self, arm: str, vocab: int, d: int = 128, blocks: int = 2, lr: float = 0.01):
        super().__init__()
        self.arm = arm
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(512, d)
        self.blocks = nn.ModuleList([Block(arm, d, lr) for _ in range(blocks)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        t = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))[None]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))


def measured_bits(model: nn.Module, arm: str) -> float:
    """Bits per counter parameter actually held; embedding/head stay dense on both sides."""
    if arm == "dense_bf16":
        return 96.0
    total = 0
    params = 0
    for m in model.modules():
        if hasattr(m, "persistent_bytes") and hasattr(m, "in_features"):
            total += m.persistent_bytes()
            params += m.in_features * m.out_features
    return total * 8 / params


def expected_bits(model: nn.Module, arm: str) -> float:
    if arm == "dense_bf16":
        return 96.0
    total = 0.0
    params = 0
    for m in model.modules():
        if hasattr(m, "persistent_bytes") and hasattr(m, "in_features"):
            p = m.in_features * m.out_features
            cb = CODE_BITS.get(arm)
            if cb is None:                      # a PackedNA variant: ask the format
                cb = m.fmt.bits
            total += (cb + 0.25 + 32.0 / m.in_features) * p
            params += p
    return total / params


def magnitude_histogram(model: nn.Module):
    """Fraction of codes at each magnitude index, summed over all counter layers.

    Shows whether the outlier slot of a ladder is actually reachable during
    training, or whether the bulk sits on a coarse sub-grid.
    """
    counts = None
    fmt = None
    for m in model.modules():
        if hasattr(m, "fmt") and hasattr(m, "codes"):
            idx = (m.codes.to(torch.long) // m.fmt.acc_levels).flatten()
            c = torch.bincount(idx, minlength=m.fmt.n_vals).float()
            counts = c if counts is None else counts + c
            fmt = m.fmt
    if counts is None:
        return None
    counts = counts / counts.sum()
    return fmt.signed.tolist(), counts.tolist()


def format_histogram(pair) -> str:
    if pair is None:
        return "(no counter layers)"
    values, fracs = pair
    return "  ".join(f"{v:+g}:{f*100:.1f}%" for v, f in zip(values, fracs) if f > 0.0005)


@torch.no_grad()
def evaluate(model, data, seq=128, iters=10) -> float:
    model.eval()
    gen = torch.Generator(device="cpu").manual_seed(1234)
    total = 0.0
    for _ in range(iters):
        starts = torch.randint(0, data.numel() - seq - 1, (32,), generator=gen)
        x = torch.stack([data[s:s + seq] for s in starts]).to(DEV)
        y = torch.stack([data[s + 1:s + seq + 1] for s in starts]).to(DEV)
        total += float(nn.functional.cross_entropy(
            model(x).reshape(-1, model.head.out_features), y.reshape(-1)))
    model.train()
    return total / iters / math.log(2)   # bits/char


def run_arm(arm: str, seed: int, d: int, steps: int, batch: int, seq: int,
            ids: torch.Tensor, vocab: int, lr: float, adam_lr: float) -> dict:
    torch.manual_seed(seed)
    model = CharLM(arm, vocab, d=d, lr=lr).to(DEV)
    bits = measured_bits(model, arm)
    want = expected_bits(model, arm)
    if abs(bits - want) > 0.02:
        raise AssertionError(f"{arm}: measured {bits:.3f} bits != expected {want:.3f}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=adam_lr)
    model.train()
    t0 = time.time()
    for _ in range(steps):
        idx = torch.randint(0, len(ids) - seq - 1, (batch,), device=DEV)
        x = ids[idx.unsqueeze(1) + torch.arange(seq, device=DEV)]
        y = ids[idx.unsqueeze(1) + torch.arange(1, seq + 1, device=DEV)]
        loss = nn.functional.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    elapsed = time.time() - t0

    val = evaluate(model, ids, seq=seq)
    return {"arm": arm, "seed": seed, "bits": bits, "val": val, "sec": elapsed,
            "hist": magnitude_histogram(model)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--adam-lr", type=float, default=3e-3)
    ap.add_argument("--arms", default="na_wide_c3,na_wide_c5,na_e2m1_c3,na_e2m1_c5")
    ap.add_argument("--hist", action="store_true", help="print magnitude histograms")
    ap.add_argument("--scale-mode", default="headroom", choices=["headroom", "mse"])
    ap.add_argument("--clip", type=float, default=0.0, help="clip fraction for the MSE scale")
    ap.add_argument("--rescale", type=int, default=0,
                    help="refresh the E8M0 block scale every N updates (0 = frozen)")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data" / "tinyshakespeare.txt"))
    args = ap.parse_args(argv)

    global RESCALE_EVERY, SCALE_MODE, CLIP_FRAC
    RESCALE_EVERY = args.rescale
    SCALE_MODE = args.scale_mode
    CLIP_FRAC = args.clip

    text = load_text(Path(args.data))
    chars = sorted(set(text))
    vocab_of = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([vocab_of[c] for c in text], dtype=torch.long, device=DEV)

    print(f"device={torch.cuda.get_device_name(0) if DEV == 'cuda' else 'cpu'} "
          f"d={args.d} steps={args.steps} batch={args.batch} seq={args.seq} "
          f"vocab={len(chars)} tokens={len(ids)} (loss in bits/char)", flush=True)

    results = []
    for arm in args.arms.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            r = run_arm(arm.strip(), seed, args.d, args.steps, args.batch, args.seq,
                        ids, len(chars), args.lr, args.adam_lr)
            results.append(r)
            print(f"  {r['arm']:11} seed={r['seed']} bits={r['bits']:.3f} "
                  f"val={r['val']:.4f} ({r['sec']:.0f}s)", flush=True)
            if args.hist:
                print(f"      magnitudes: {format_histogram(r['hist'])}", flush=True)

    print(f"\n{'arm':11} {'bits':>6} {'mean val':>9} {'sd':>7}  seeds")
    for arm in args.arms.split(","):
        arm = arm.strip()
        rows = [r for r in results if r["arm"] == arm]
        vals = [r["val"] for r in rows]
        mean = sum(vals) / len(vals)
        sd = (sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
        print(f"{arm:11} {rows[0]['bits']:6.3f} {mean:9.4f} {sd:7.4f}  {[round(v, 4) for v in vals]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
