"""M4 witness -- does a counter-state MoE FFN match/beat a dense FFN at EQUAL active compute?

A tiny GPT, FFN sublayer swapped between arms (attention is dense fp in all arms, to isolate the
FFN variable). Same data/steps/d. We report, per arm: val-loss, active MACs/token in the FFN,
persistent bytes of the FFN, tok/s, and -- for the MoE arms -- a routing-collapse check (per-expert
token fraction; flag if any expert gets < 1% or one gets > 90%).

The gate (plan M4): at <= dense's active compute the counter-MoE reaches <= dense's val-loss, AND
its capacity (E) grows persistent memory while a token still visits only top_k experts (equal active
compute). Each expert is sized h = 4d/top_k so top_k experts ~ the dense 2*d*4d active MACs.

    python scripts/moe_ffn_witness.py
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from memory_native import RMSCounterLinear
from memory_native.data import get_batch, synthetic_corpus
from memory_native.moe_ffn import CounterMoEFFN

torch.manual_seed(0)
DEV = "cpu"
D, NL, NH, BLK = 128, 3, 4, 64
STEPS, BATCH, EVAL = 1500, 32, 80


class DenseFFN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 4 * d); self.fc2 = nn.Linear(4 * d, d)
        self.last_aux_loss = torch.zeros(())
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
        self.last_aux_loss = torch.zeros(())
    def forward(self, x):
        return self.fc2(F.gelu(self.fc(x)))
    def active_macs_per_token(self):
        return 2 * self.fc.in_features * self.fc.out_features
    def persistent_bytes(self):
        b = 0
        for m in (self.fc, self.fc2):
            b += m.state.numel() + m.scale.numel() * 4 + m.v.numel() * 4
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
        aux = x.new_zeros(())
        for bl in self.blocks:
            h = bl["ln1"](x)
            q, k, v = bl["attn"](h).split(D, dim=-1)
            q = q.view(b, t, NH, -1).transpose(1, 2); k = k.view(b, t, NH, -1).transpose(1, 2)
            v = v.view(b, t, NH, -1).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + bl["proj"](a.transpose(1, 2).reshape(b, t, D))
            x = x + bl["ffn"](bl["ln2"](x))
            aux = aux + bl["ffn"].last_aux_loss
        logits = self.head(self.lnf(x))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) \
            if targets is not None else None
        return logits, loss, aux

    def ffn_active_macs(self):
        return self.blocks[0]["ffn"].active_macs_per_token()
    def ffn_persistent_bytes(self):
        return sum(bl["ffn"].persistent_bytes() for bl in self.blocks)


def _routing_line(model):
    """Aggregate the per-expert token fractions across all MoE blocks; flag collapse."""
    ffns = [bl["ffn"] for bl in model.blocks if isinstance(bl["ffn"], CounterMoEFFN)]
    if not ffns:
        return None
    counts = torch.zeros(ffns[0].E, dtype=torch.float64)
    for f in ffns:
        counts += f.token_count
    frac = (counts / counts.sum().clamp_min(1)).tolist()
    starved = any(x < 0.01 for x in frac)
    dominant = any(x > 0.90 for x in frac)
    flag = "COLLAPSE" if (starved or dominant) else "ok"
    fr = " ".join(f"{x*100:4.1f}" for x in frac)
    return f"      routing [{flag}] min {min(frac)*100:4.1f}% max {max(frac)*100:4.1f}%  per-expert%: {fr}"


def run(name, make_ffn, train, val, vocab, aux_weight=0.0):
    torch.manual_seed(0)
    model = TinyGPT(vocab, make_ffn).to(DEV).train()
    # counter linears self-update; AdamW owns only the fp params (incl. the MoE router).
    adam_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(adam_params, lr=3e-4)
    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    for step in range(STEPS):
        xb, yb = get_batch(train, BLK, BATCH, DEV, generator=g)
        _, loss, aux = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        (loss + aux_weight * aux).backward()
        opt.step()
    dt = time.time() - t0
    model.eval()
    with torch.no_grad():
        vl = 0.0
        ge = torch.Generator().manual_seed(123)
        for _ in range(EVAL):
            xb, yb = get_batch(val, BLK, BATCH, DEV, generator=ge)
            _, loss, _ = model(xb, yb)
            vl += loss.item()
        vl /= EVAL
    toks = STEPS * BATCH * BLK
    print(f"  {name:30s} val {vl:.4f}  | FFN active MACs/tok {model.ffn_active_macs():>8d}"
          f"  | FFN persist {model.ffn_persistent_bytes()/1024:7.1f} KiB"
          f"  | {toks/dt:6.0f} tok/s")
    rl = _routing_line(model)
    if rl is not None:
        print(rl)
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

    # counter-MoE sweep: E (capacity) grows; a token still visits only top_k experts so the expert
    # active MACs stay ~ the dense reference (h = 4d/top_k). aux loss on to avoid routing collapse.
    AUX = 1e-2
    for E in (4, 8, 16):
        for k in (1, 2):
            run(f"counter-MoE E={E} k={k}",
                lambda E=E, k=k: CounterMoEFFN(D, n_experts=E, top_k=k, C=11, lr=0.04,
                                               lr_scale=2e-4, aux_loss_weight=AUX),
                train, val, vocab, aux_weight=AUX)
        print()

    print("GATE (M4): counter-MoE reaches <= dense val-loss at <= dense active MACs, capacity (E)")
    print(f"           grows persistent bytes without raising expert active compute, and routing")
    print(f"           does not collapse. dense val={dense_vl:.4f}, dense active MACs/tok={dense_macs}")


if __name__ == "__main__":
    main()
