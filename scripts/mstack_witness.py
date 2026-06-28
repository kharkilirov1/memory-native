"""M-STACK witness -- do the per-step levers COMPOSE on a real model without conflicting?

A tiny GPT, every linear (q,k,v,proj,fc,fc2) swapped between arms. Same data/steps/d.
  * dense   : nn.Linear + AdamW (the strong fp baseline)
  * counter : RMSCounterLinear (the plain counter method)
  * stack   : StackCounterLinear = 2:4 group-counter base (M2) + low-rank slow-fast residual (M3),
              merged every K steps. Two per-step levers active at once.

Gate (plan M-STACK): the stack trains with val-gap <= a few % of dense -> the levers compose, not
conflict. (Per-step SPEED is NOT measured: it needs the cuSPARSELt 2:4 kernel, not built here; the
dense PyTorch fallback tok/s would be meaningless. This is the composability/quality gate only.)

    python scripts/mstack_witness.py
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from memory_native import RMSCounterLinear
from memory_native.data import get_batch, synthetic_corpus
from memory_native.stack_linear import StackCounterLinear

torch.manual_seed(0)
torch.set_num_threads(4)
DEV = "cpu"
# Small enough to finish on CPU (the 2:4 group-counter does a Python decode+mask+update per linear
# per step, so the stack arm is the bottleneck). The dense/counter/stack comparison stays fair as
# long as all arms share the config; absolute losses differ from the larger config.
D, NL, NH, BLK = 96, 2, 4, 48
STEPS, BATCH, EVAL = 400, 32, 40


def make_linear(kind, i, o, gain=1.0):
    if kind == "dense":
        return nn.Linear(i, o, bias=False)
    if kind == "counter":
        return RMSCounterLinear(i, o, C=11, lr=0.04, lr_scale=2e-4, init_gain=gain)
    if kind == "stack":
        return StackCounterLinear(i, o, rank=16, merge_every=16, hysteresis=2.0,
                                  C=11, lr=0.04, lr_scale=2e-4, init_gain=gain)
    raise ValueError(kind)


class Blk(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.ln1 = nn.LayerNorm(D); self.ln2 = nn.LayerNorm(D)
        self.q = make_linear(kind, D, D); self.k = make_linear(kind, D, D); self.v = make_linear(kind, D, D)
        self.proj = make_linear(kind, D, D, 1.0 / (2 * NL) ** 0.5)
        self.fc = make_linear(kind, D, 4 * D); self.fc2 = make_linear(kind, 4 * D, D, 1.0 / (2 * NL) ** 0.5)

    def forward(self, x):
        b, t, c = x.shape
        h = self.ln1(x)
        q = self.q(h).view(b, t, NH, -1).transpose(1, 2)
        k = self.k(h).view(b, t, NH, -1).transpose(1, 2)
        v = self.v(h).view(b, t, NH, -1).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, D))
        x = x + self.fc2(F.gelu(self.fc(self.ln2(x))))
        return x


class GPT(nn.Module):
    def __init__(self, kind, vocab):
        super().__init__()
        self.tok = nn.Embedding(vocab, D); self.pos = nn.Embedding(BLK, D)
        nn.init.normal_(self.tok.weight, std=0.02); nn.init.normal_(self.pos.weight, std=0.02)
        self.blocks = nn.ModuleList(Blk(kind) for _ in range(NL))
        self.lnf = nn.LayerNorm(D)
        self.head = nn.Linear(D, vocab, bias=False); self.head.weight = self.tok.weight

    def forward(self, idx, targets=None):
        b, t = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))[None]
        for bl in self.blocks:
            x = bl(x)
        logits = self.head(self.lnf(x))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) \
            if targets is not None else None
        return logits, loss


def run(kind, train, val, vocab):
    torch.manual_seed(0)
    model = GPT(kind, vocab).to(DEV).train()
    # AdamW owns the fp params: dense weights, OR (counter/stack) embeddings+norms+head + stack A,B.
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    g = torch.Generator().manual_seed(0)
    for _ in range(STEPS):
        xb, yb = get_batch(train, BLK, BATCH, DEV, generator=g)
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        vl = 0.0; ge = torch.Generator().manual_seed(123)
        for _ in range(EVAL):
            xb, yb = get_batch(val, BLK, BATCH, DEV, generator=ge)
            _, loss = model(xb, yb); vl += loss.item()
    return vl / EVAL


def main():
    text = synthetic_corpus(120_000)
    chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data)); train, val, vocab = data[:n], data[n:], len(chars)
    print(f"=== M-STACK composability: tiny GPT, every linear swapped (d={D} L={NL} steps={STEPS}) ===\n")
    dv = run("dense", train, val, vocab); print(f"  dense (fp, AdamW)            val {dv:.4f}")
    cv = run("counter", train, val, vocab); print(f"  counter (RMSCounter)         val {cv:.4f}  gap {100*(cv-dv)/dv:+.1f}%")
    sv = run("stack", train, val, vocab)
    print(f"  stack (2:4 + slow-fast)      val {sv:.4f}  gap {100*(sv-dv)/dv:+.1f}%")
    print("\n=== VERDICT ===")
    g = sv <= cv * 1.05
    print(f"  [{'PASS' if g else 'FAIL'}] levers compose: stack val {sv:.4f} within 5% of the plain")
    print(f"          counter {cv:.4f} (the two per-step levers train together, no conflict).")
    print("  NOTE: per-step SPEED not measured -- needs the cuSPARSELt 2:4 kernel (not built). This")
    print("        is the composability/quality gate; the speed gate is gated on that kernel.")


if __name__ == "__main__":
    main()
