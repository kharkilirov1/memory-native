"""M9 witness -- does Multi-Token Prediction give more tokens-to-loss without hurting the
next-token head?

A tiny GPT trunk (dense attention + dense FFN), forwarded ONCE per step. Arms differ only in the
number of output heads: baseline (n_pred=1, the ordinary next-token head) vs MTP (n_pred in
{2, 4}, which additionally predicts t+2 .. t+n_pred from the same final hidden state). SAME
steps/data/optimizer across arms. The trunk FLOPs are identical in every arm -- MTP adds only
cheap output heads, so any difference is the auxiliary signal, not extra trunk compute.

We report, per arm, the PRIMARY-head (j=0) val-loss -- i.e. the usual next-token loss -- and the
gap vs baseline.

GATE (M9): MTP's primary-head val-loss <= baseline's at the same step budget (the extra heads'
auxiliary signal should help, or at least not hurt, the next-token head).

    python scripts/mtp_witness.py
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from memory_native.data import get_batch, synthetic_corpus
from memory_native.mtp import MultiTokenHead

torch.manual_seed(0)
DEV = "cpu"
D, NL, NH, BLK = 128, 3, 4, 64
STEPS, BATCH, EVAL = 1500, 32, 80


class TinyGPTTrunk(nn.Module):
    """Dense TinyGPT trunk. `trunk_hidden` returns the post-lnf hidden state (b, t, D); the output
    head(s) live outside, in the MTP module. The token embedding is exposed for head tying."""

    def __init__(self, vocab):
        super().__init__()
        self.tok = nn.Embedding(vocab, D)
        self.pos = nn.Embedding(BLK, D)
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.blocks = nn.ModuleList()
        for _ in range(NL):
            self.blocks.append(nn.ModuleDict(dict(
                ln1=nn.LayerNorm(D), attn=nn.Linear(D, 3 * D), proj=nn.Linear(D, D),
                ln2=nn.LayerNorm(D),
                fc=nn.Linear(D, 4 * D), fc2=nn.Linear(4 * D, D))))
        self.lnf = nn.LayerNorm(D)

    def trunk_hidden(self, idx):
        b, t = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))[None]
        for bl in self.blocks:
            h = bl["ln1"](x)
            q, k, v = bl["attn"](h).split(D, dim=-1)
            q = q.view(b, t, NH, -1).transpose(1, 2)
            k = k.view(b, t, NH, -1).transpose(1, 2)
            v = v.view(b, t, NH, -1).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + bl["proj"](a.transpose(1, 2).reshape(b, t, D))
            x = x + bl["fc2"](F.gelu(bl["fc"](bl["ln2"](x))))
        return self.lnf(x)


class MTPModel(nn.Module):
    """TinyGPT trunk + MultiTokenHead. Primary head is tied to the token embedding (exactly the
    ordinary tied next-token head); auxiliary heads are also tied, so MTP adds zero new params."""

    def __init__(self, vocab, n_pred):
        super().__init__()
        self.trunk = TinyGPTTrunk(vocab)
        primary = nn.Linear(D, vocab, bias=False)
        primary.weight = self.trunk.tok.weight                 # tie primary head to embedding
        self.head = MultiTokenHead(D, vocab, n_pred=n_pred, primary_head=primary,
                                   embedding=self.trunk.tok, tie_embedding=True)

    def forward(self, idx, targets=None):
        h = self.trunk.trunk_hidden(idx)
        logits_list = self.head(h)
        loss = per_head = None
        if targets is not None:
            loss, per_head = self.head.loss(h, targets)
        return logits_list, loss, per_head


def run(name, n_pred, train, val, vocab):
    torch.manual_seed(0)
    model = MTPModel(vocab, n_pred).to(DEV).train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    for _ in range(STEPS):
        xb, yb = get_batch(train, BLK, BATCH, DEV, generator=g)
        _, loss, _ = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    dt = time.time() - t0
    model.eval()
    with torch.no_grad():
        primary_vl = 0.0
        ge = torch.Generator().manual_seed(123)
        for _ in range(EVAL):
            xb, yb = get_batch(val, BLK, BATCH, DEV, generator=ge)
            logits_list, _, _ = model(xb, yb)
            # primary-head (j=0) val-loss == the ordinary next-token loss
            primary_vl += F.cross_entropy(
                logits_list[0].reshape(-1, vocab), yb.reshape(-1)).item()
        primary_vl /= EVAL
    toks = STEPS * BATCH * BLK
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {name:24s} primary val {primary_vl:.4f}  | n_pred={n_pred}"
          f"  | trainable params {n_params:>9d}  | {toks/dt:6.0f} tok/s")
    return primary_vl


def main():
    text = synthetic_corpus(120_000)
    chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data)); train, val, vocab = data[:n], data[n:], len(chars)
    print(f"corpus synthetic | vocab {vocab} | d={D} layers={NL} block={BLK} steps={STEPS}")
    print("trunk FLOPs identical in every arm; MTP adds only cheap (tied) output heads.\n")

    base_vl = run("baseline (next-token)", 1, train, val, vocab)
    mtp2_vl = run("MTP n_pred=2", 2, train, val, vocab)
    mtp4_vl = run("MTP n_pred=4", 4, train, val, vocab)

    print()
    print(f"  primary val-loss gap  MTP(2) - baseline = {mtp2_vl - base_vl:+.4f}")
    print(f"  primary val-loss gap  MTP(4) - baseline = {mtp4_vl - base_vl:+.4f}")
    best = min(mtp2_vl, mtp4_vl)
    verdict = "PASS" if best <= base_vl else "FAIL"
    print(f"\nGATE (M9): MTP primary-head val <= baseline at the same step budget.")
    print(f"           baseline={base_vl:.4f}  best-MTP={best:.4f}  -> {verdict}")


if __name__ == "__main__":
    main()
