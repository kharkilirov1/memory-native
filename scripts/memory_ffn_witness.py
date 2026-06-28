"""M1 witness -- does a retrieval memory match/beat a dense FFN at EQUAL active compute?

A tiny GPT, FFN sublayer swapped between arms (attention is dense fp in all arms, to isolate the
FFN variable). Same data/steps/d. We report, per arm: val-loss, active MACs/token in the FFN,
persistent bytes of the FFN, tok/s. The gate (plan M1): at <= dense's active compute the memory
FFN reaches <= dense's val-loss, AND its capacity (E) grows persistent memory without growing
active FLOPs (shown by the E-sweep: bigger E, ~same active MACs, lower loss).

    python scripts/memory_ffn_witness.py
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from memory_native import RMSCounterLinear
from memory_native.data import get_batch, synthetic_corpus
from memory_native.memory_ffn import CounterMemoryFFN

torch.manual_seed(0)
DEV = "cpu"
D, NL, NH, BLK = 128, 3, 4, 64
STEPS, BATCH, EVAL = 1500, 32, 80


class DenseFFN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 4 * d); self.fc2 = nn.Linear(4 * d, d)
    def forward(self, x):
        return self.fc2(F.gelu(self.fc(x)))
    def active_macs_per_token(self):
        d = self.fc.in_features
        return 2 * d * 4 * d
    def persistent_bytes(self):
        return sum(p.numel() for p in self.parameters()) * 4


class CounterDenseFFN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = RMSCounterLinear(d, 4 * d, C=11, lr=0.04, lr_scale=2e-4)
        self.fc2 = RMSCounterLinear(4 * d, d, C=11, lr=0.04, lr_scale=2e-4)
    def forward(self, x):
        return self.fc2(F.gelu(self.fc(x)))
    def active_macs_per_token(self):
        return 2 * self.fc.in_features * self.fc.out_features
    def persistent_bytes(self):
        b = 0
        for m in (self.fc, self.fc2):
            b += m.state.numel() + m.scale.numel() * 4
        return b


class TinyGPT(nn.Module):
    """Dense attention everywhere; only the FFN sublayer differs between arms."""
    def __init__(self, vocab, make_ffn):
        super().__init__()
        self.tok = nn.Embedding(vocab, D); self.pos = nn.Embedding(BLK, D)
        nn.init.normal_(self.tok.weight, std=0.02); nn.init.normal_(self.pos.weight, std=0.02)
        self.blocks = nn.ModuleList()
        for _ in range(NL):
            self.blocks.append(nn.ModuleDict(dict(
                ln1=nn.LayerNorm(D), attn=nn.Linear(D, 3 * D), proj=nn.Linear(D, D),
                ln2=nn.LayerNorm(D), ffn=make_ffn())))
        self.lnf = nn.LayerNorm(D)
        self.head = nn.Linear(D, vocab, bias=False); self.head.weight = self.tok.weight

    def forward(self, idx, targets=None):
        b, t = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))[None]
        for bl in self.blocks:
            h = bl["ln1"](x)
            q, k, v = bl["attn"](h).split(D, dim=-1)
            q = q.view(b, t, NH, -1).transpose(1, 2); k = k.view(b, t, NH, -1).transpose(1, 2)
            v = v.view(b, t, NH, -1).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + bl["proj"](a.transpose(1, 2).reshape(b, t, D))
            x = x + bl["ffn"](bl["ln2"](x))
        logits = self.head(self.lnf(x))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) \
            if targets is not None else None
        return logits, loss

    def ffn_active_macs(self):
        return self.blocks[0]["ffn"].active_macs_per_token()
    def ffn_persistent_bytes(self):
        return sum(bl["ffn"].persistent_bytes() for bl in self.blocks)


def run(name, make_ffn, train, val, vocab):
    torch.manual_seed(0)
    model = TinyGPT(vocab, make_ffn).to(DEV).train()
    # counter value tables / counter linears self-update; AdamW owns the rest (incl. the router)
    adam_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(adam_params, lr=3e-4)
    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    for step in range(STEPS):
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
    toks = STEPS * BATCH * BLK
    print(f"  {name:30s} val {vl:.4f}  | FFN active MACs/tok {model.ffn_active_macs():>8d}"
          f"  | FFN persist {model.ffn_persistent_bytes()/1024:7.1f} KiB"
          f"  | {toks/dt:6.0f} tok/s")
    return vl, model.ffn_active_macs()


def main():
    text = synthetic_corpus(120_000)
    chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data)); train, val, vocab = data[:n], data[n:], len(chars)
    print(f"corpus synthetic | vocab {vocab} | d={D} layers={NL} block={BLK} steps={STEPS}")
    print(f"dense FFN active MACs/tok (reference) = {2*D*4*D}\n")

    dense_vl, dense_macs = run("dense FFN (fp, AdamW)", lambda: DenseFFN(D), train, val, vocab)
    run("counter-dense FFN", lambda: CounterDenseFFN(D), train, val, vocab)
    print()
    # memory FFN E-sweep: capacity grows (E up) at ~constant active MACs -> loss should fall
    for E, k, dk in [(4096, 16, 48), (16384, 16, 48), (65536, 16, 48)]:
        run(f"counter-memory FFN E={E} k={k}",
            lambda E=E, k=k, dk=dk: CounterMemoryFFN(D, n_cells=E, k=k, key_dim=dk,
                                                     C=11, lr=0.04, lr_scale=2e-4),
            train, val, vocab)

    print("\nGATE (M1): memory-FFN reaches <= dense val-loss at <= dense active MACs,")
    print(f"           and capacity (E) lowers loss without raising active MACs. dense val={dense_vl:.4f}")


if __name__ == "__main__":
    main()
