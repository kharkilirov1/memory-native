"""Real-data scaling gate (Phase-2) -- the experiment the toy synthetic corpus could NOT be.

The toy witnesses overfit (dense val 0.96 -> 1.80 as steps grow) before bigger capacity pays off,
so the FLOPs-to-loss claim of the architecture levers (M1 memory-FFN, M4 Counter-MoE) was
inconclusive. This runs the SAME swap on a REAL corpus (tinyshakespeare via load_corpus, which
downloads on a networked box) with a held-out val split, a bigger model, and a GPU step budget.

Question: on real data, does counter-MoE / memory-FFN (a) match/beat dense at <= dense active
compute, and (b) keep improving as capacity E grows (monotonic), WITHOUT the dense baseline winning
by overfitting? Built to run on the user's new Kaggle GPU account (device auto-detects CUDA).

    python scripts/realdata_scaling.py
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from memory_native import CounterMemoryFFN, CounterMoEFFN, RMSCounterLinear
from memory_native.data import get_batch, load_corpus

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
D, NL, NH, BLK = 384, 6, 6, 128           # bigger than the toy witness (real GPU budget)
STEPS, BATCH, EVAL = 6000, 32, 100


class DenseFFN(nn.Module):
    def __init__(s, d): super().__init__(); s.fc = nn.Linear(d, 4 * d); s.fc2 = nn.Linear(4 * d, d); s.last_aux_loss = torch.zeros(())
    def forward(s, x): return s.fc2(F.gelu(s.fc(x)))
    def active_macs_per_token(s): return 2 * s.fc.in_features * 4 * s.fc.in_features
    def persistent_bytes(s): return sum(p.numel() for p in s.parameters()) * 4


class CounterDenseFFN(nn.Module):
    def __init__(s, d):
        super().__init__(); s.fc = RMSCounterLinear(d, 4 * d, C=11, lr=0.04, lr_scale=2e-4)
        s.fc2 = RMSCounterLinear(4 * d, d, C=11, lr=0.04, lr_scale=2e-4); s.last_aux_loss = torch.zeros(())
    def forward(s, x): return s.fc2(F.gelu(s.fc(x)))
    def active_macs_per_token(s): return 2 * s.fc.in_features * s.fc.out_features
    def persistent_bytes(s): return sum(m.state.numel() + m.scale.numel() * 4 for m in (s.fc, s.fc2))


class MoEArm(nn.Module):
    def __init__(s, d, E): super().__init__(); s.f = CounterMoEFFN(d, n_experts=E, top_k=2, C=11, lr=0.04, lr_scale=2e-4, aux_loss_weight=1e-2)
    def forward(s, x): return s.f(x)
    @property
    def last_aux_loss(s): return s.f.last_aux_loss
    def active_macs_per_token(s): return s.f.active_macs_per_token()
    def persistent_bytes(s): return s.f.persistent_bytes()


class MemArm(nn.Module):
    def __init__(s, d, E): super().__init__(); s.f = CounterMemoryFFN(d, n_cells=E, k=16, key_dim=48, C=11, lr=0.04, lr_scale=2e-4); s.last_aux_loss = torch.zeros(())
    def forward(s, x): return s.f(x)
    def active_macs_per_token(s): return s.f.active_macs_per_token()
    def persistent_bytes(s): return s.f.persistent_bytes()


class GPT(nn.Module):
    def __init__(s, vocab, make_ffn):
        super().__init__()
        s.tok = nn.Embedding(vocab, D); s.pos = nn.Embedding(BLK, D)
        nn.init.normal_(s.tok.weight, std=0.02); nn.init.normal_(s.pos.weight, std=0.02)
        s.blocks = nn.ModuleList(nn.ModuleDict(dict(
            ln1=nn.LayerNorm(D), attn=nn.Linear(D, 3 * D), proj=nn.Linear(D, D),
            ln2=nn.LayerNorm(D), ffn=make_ffn())) for _ in range(NL))
        s.lnf = nn.LayerNorm(D); s.head = nn.Linear(D, vocab, bias=False); s.head.weight = s.tok.weight
    def forward(s, idx, targets=None):
        b, t = idx.shape
        x = s.tok(idx) + s.pos(torch.arange(t, device=idx.device))[None]
        aux = x.new_zeros(())
        for bl in s.blocks:
            h = bl["ln1"](x)
            q, k, v = bl["attn"](h).split(D, dim=-1)
            q = q.view(b, t, NH, -1).transpose(1, 2); k = k.view(b, t, NH, -1).transpose(1, 2); v = v.view(b, t, NH, -1).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + bl["proj"](a.transpose(1, 2).reshape(b, t, D))
            x = x + bl["ffn"](bl["ln2"](x)); aux = aux + getattr(bl["ffn"], "last_aux_loss", 0.0)
        logits = s.head(s.lnf(x))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) if targets is not None else None
        return logits, loss, aux
    def ffn_macs(s): return s.blocks[0]["ffn"].active_macs_per_token()
    def ffn_bytes(s): return sum(bl["ffn"].persistent_bytes() for bl in s.blocks)


def run(name, make_ffn, train, val, vocab, aux_w=0.0):
    torch.manual_seed(0)
    m = GPT(vocab, make_ffn).to(DEV).train()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=3e-4)
    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    tr_last = 0.0
    for step in range(STEPS):
        xb, yb = get_batch(train, BLK, BATCH, DEV, generator=g)
        _, loss, aux = m(xb, yb); opt.zero_grad(set_to_none=True); (loss + aux_w * aux).backward(); opt.step()
        tr_last = loss.item()
    dt = time.time() - t0
    m.eval()
    with torch.no_grad():
        vl = 0.0; ge = torch.Generator().manual_seed(123)
        for _ in range(EVAL):
            xb, yb = get_batch(val, BLK, BATCH, DEV, generator=ge); _, loss, _ = m(xb, yb); vl += loss.item()
    vl /= EVAL
    print(f"  {name:26s} val {vl:.4f}  train {tr_last:.4f}  | FFN MACs/tok {m.ffn_macs():>8d}"
          f"  | persist {m.ffn_bytes()/1024:8.0f} KiB  | {STEPS*BATCH*BLK/dt:7.0f} tok/s", flush=True)
    return vl


def main():
    print(f"=== REAL-DATA scaling gate | device {DEV} | d={D} L={NL} block={BLK} steps={STEPS} ===", flush=True)
    train, val, vocab = load_corpus(DEV)            # tinyshakespeare (downloads), held-out val
    print(f"vocab {vocab} | train {len(train)} val {len(val)} tokens | dense FFN MACs/tok {2*D*4*D}\n", flush=True)
    dv = run("dense FFN (fp)", lambda: DenseFFN(D), train, val, vocab)
    run("counter-dense FFN", lambda: CounterDenseFFN(D), train, val, vocab)
    print("  -- counter-MoE capacity sweep (FLOPs-to-loss lever) --", flush=True)
    for E in (8, 16, 32):
        run(f"counter-MoE E={E}", lambda E=E: MoEArm(D, E), train, val, vocab, aux_w=1e-2)
    print("  -- counter-memory-FFN capacity sweep --", flush=True)
    for E in (16384, 65536):
        run(f"counter-mem E={E}", lambda E=E: MemArm(D, E), train, val, vocab)
    print(f"\nGATE: on REAL data, do MoE/memory match/beat dense (val {dv:.4f}) at <= its active MACs,")
    print("      and keep improving as E grows (monotonic), without dense winning by overfitting?")


if __name__ == "__main__":
    main()
