"""M10 witness -- Mixture-of-Depths (MoD): does per-block token routing cut block FLOPs/token
without hurting val-loss much?

A tiny GPT (same scaffold as memory_ffn_witness.py: same TinyGPT, synthetic corpus, run() loop).
Each transformer block (attn+FFN) is wrapped in a MoDBlock. Arms:
  * full       -- capacity=1.0  (every token through every block == plain TinyGPT)
  * MoD cap=0.5 -- ~half the tokens skip each block (residual bypass)
  * MoD cap=0.25 -- ~a quarter processed

Same steps/data across arms. Per arm we print: val-loss, the block-FLOP fraction (~capacity),
the realized processed-fraction (router not degenerate), and the val-gap vs full.

GATE (plan M10): at capacity=0.5 the val-gap vs full is small (<= a few %) while ~half the block
compute is skipped, and routing stays predictable (realized processed-fraction ~= capacity, no
all-skip / all-keep collapse). External basis: DeepMind 2024 Mixture-of-Depths.

    python scripts/mod_witness.py
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from memory_native.data import get_batch, synthetic_corpus
from memory_native.mod import MoDBlock

torch.manual_seed(0)
DEV = "cpu"
D, NL, NH, BLK = 128, 3, 4, 64
STEPS, BATCH, EVAL = 1500, 32, 80


class Block(nn.Module):
    """A standard residual transformer block: x -> x + attn(x) + ffn(x). Returns (B,T,D)."""

    def __init__(self, d, n_head):
        super().__init__()
        self.n_embd = d
        self.n_head = n_head
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x):
        b, t, c = x.shape
        h = self.ln1(x)
        q, k, v = self.attn(h).split(c, dim=-1)
        q = q.view(b, t, self.n_head, -1).transpose(1, 2)
        k = k.view(b, t, self.n_head, -1).transpose(1, 2)
        v = v.view(b, t, self.n_head, -1).transpose(1, 2)
        # NOTE: when wrapped by MoD the selected tokens are gathered into a (1,k,D) row, so
        # causal masking across the original time order no longer applies -- see MoDBlock's
        # docstring caveat. We keep is_causal=False there (handled by capacity branch below).
        a = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, c))
        x = x + self.fc2(F.gelu(self.fc(self.ln2(x))))
        return x

    is_causal = True


class TinyGPT(nn.Module):
    """Same TinyGPT scaffold; every block optionally wrapped in MoDBlock(capacity)."""

    def __init__(self, vocab, capacity):
        super().__init__()
        self.tok = nn.Embedding(vocab, D)
        self.pos = nn.Embedding(BLK, D)
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.capacity = capacity
        self.blocks = nn.ModuleList()
        for _ in range(NL):
            blk = Block(D, NH)
            if capacity < 1.0:
                # routed tokens are gathered out of time order -> drop causal mask on the block.
                blk.is_causal = False
                self.blocks.append(MoDBlock(blk, capacity=capacity))
            else:
                self.blocks.append(blk)
        self.lnf = nn.LayerNorm(D)
        self.head = nn.Linear(D, vocab, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, idx, targets=None):
        b, t = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))[None]
        for bl in self.blocks:
            x = bl(x)
        logits = self.head(self.lnf(x))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) \
            if targets is not None else None
        return logits, loss

    def realized_fraction(self):
        fracs = [bl.realized_fraction() for bl in self.blocks if isinstance(bl, MoDBlock)]
        return sum(fracs) / len(fracs) if fracs else 1.0


def run(name, capacity, train, val, vocab):
    torch.manual_seed(0)
    model = TinyGPT(vocab, capacity).to(DEV).train()
    # AdamW owns everything trainable, including the MoD routers.
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    for _ in range(STEPS):
        xb, yb = get_batch(train, BLK, BATCH, DEV, generator=g)
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    dt = time.time() - t0
    model.eval()
    with torch.no_grad():
        vl = 0.0
        ge = torch.Generator().manual_seed(123)
        for _ in range(EVAL):
            xb, yb = get_batch(val, BLK, BATCH, DEV, generator=ge)
            _, loss = model(xb, yb)
            vl += loss.item()
        vl /= EVAL
    realized = model.realized_fraction()
    toks = STEPS * BATCH * BLK
    print(f"  {name:22s} val {vl:.4f}  | block-FLOP frac {capacity:4.2f}"
          f"  | realized proc-frac {realized:5.3f}  | {toks/dt:6.0f} tok/s")
    return vl, realized


def main():
    text = synthetic_corpus(120_000)
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train, val, vocab = data[:n], data[n:], len(chars)
    print(f"corpus synthetic | vocab {vocab} | d={D} layers={NL} block={BLK} steps={STEPS}")
    print("MoD: a per-block router keeps top-`capacity` tokens; the rest bypass via residual.\n")

    full_vl, _ = run("full (capacity=1.0)", 1.0, train, val, vocab)
    print()
    results = {}
    for cap in (0.5, 0.25):
        vl, realized = run(f"MoD capacity={cap}", cap, train, val, vocab)
        gap = (vl - full_vl) / full_vl * 100.0
        results[cap] = (vl, realized, gap)
        print(f"  {'':22s} -> val-gap vs full {gap:+5.2f}%  (skips ~{(1-cap)*100:.0f}% of block compute)\n")

    vl50, real50, gap50 = results[0.5]
    ok_gap = gap50 <= 5.0
    ok_route = abs(real50 - 0.5) <= 0.02
    print("GATE (M10): at capacity=0.5, val-gap vs full <= a few % while ~half the block compute")
    print("            is skipped, and routing is predictable (realized proc-frac ~= capacity).")
    print(f"  cap=0.5: val-gap {gap50:+.2f}% (<=5%? {ok_gap}) | "
          f"realized {real50:.3f} ~= 0.5? {ok_route}")
    verdict = "PASS" if (ok_gap and ok_route) else "NEGATIVE (MoD hurts quality at this toy scale)"
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
