"""Scaling re-test (M1) -- does the memory-FFN E-sweep become MONOTONIC with more training?

Context. The M1 witness (scripts/memory_ffn_witness.py) ran a capacity sweep E in {4096, 16384,
65536} at k=16 for STEPS=1500 and observed a NON-MONOTONIC result: E=65536 was WORSE than
E=16384. The suspected cause is UNDERTRAINING -- the product-key router has to learn to address
~65k cells, and 1500 steps may be too few for the largest table. M4's MoE (a handful of experts)
was cleanly monotonic, consistent with "few addresses are easy to learn, many are not".

This script re-runs the SAME M1 memory-FFN E-sweep at THREE step budgets and asks, for each
budget, whether val-loss is monotonically DECREASING in E (bigger table -> lower loss). It reuses
the witness's TinyGPT scaffold and synthetic char corpus verbatim (dense attention everywhere,
only the FFN sublayer swapped), so the only moving parts are E and the step budget.

    python scripts/scaling_retest.py

Reports a (E x steps) table of val-loss + active MACs/tok, the dense FFN baseline at each budget,
and a per-budget monotonicity verdict. A negative result (still non-monotonic at the largest
budget) is a legitimate outcome and is stated as such.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# conftest pins 1 thread for the test suite; this is a standalone script, so use a few threads for
# speed (the memory-FFN backward + AdamW dominate, not the gather). Stay modest to stay deterministic.
torch.set_num_threads(4)

from memory_native.data import get_batch, synthetic_corpus
from memory_native.memory_ffn import CounterMemoryFFN

torch.manual_seed(0)
DEV = "cpu"
D, NL, NH, BLK = 128, 3, 4, 64          # same model as memory_ffn_witness.py
BATCH, EVAL = 32, 80

# The key question lives here: does the E-sweep flip from non-monotonic -> monotonic as steps grow?
STEP_BUDGETS = (1500, 4000, 8000)
E_SWEEP = (4096, 16384, 65536)          # k=16 capacity sweep (the M1 witness's E values)
K, KEY_DIM = 16, 48


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


class TinyGPT(nn.Module):
    """Dense attention everywhere; only the FFN sublayer differs between arms (verbatim from M1)."""
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


def run(make_ffn, steps, train, val, vocab):
    """Train one arm for `steps` steps, return (val_loss, active_macs_per_tok, tok/s). Same
    optimization recipe as the M1 witness: AdamW(3e-4) owns the fp params (incl. the router);
    the counter value table self-updates in backward."""
    torch.manual_seed(0)
    model = TinyGPT(vocab, make_ffn).to(DEV).train()
    adam_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(adam_params, lr=3e-4)
    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    for _ in range(steps):
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
    toks = steps * BATCH * BLK
    return vl, model.ffn_active_macs(), toks / dt


def is_monotonic_decreasing(vals):
    """True iff each successive val-loss is strictly lower than the last (bigger E -> lower loss)."""
    return all(b < a for a, b in zip(vals, vals[1:]))


def main():
    text = synthetic_corpus(120_000)
    chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data)); train, val, vocab = data[:n], data[n:], len(chars)
    print(f"corpus synthetic | vocab {vocab} | d={D} layers={NL} block={BLK} | k={K} key_dim={KEY_DIM}")
    print(f"dense FFN active MACs/tok (reference) = {2*D*4*D}")
    print(f"E-sweep {E_SWEEP} x step budgets {STEP_BUDGETS}  (threads={torch.get_num_threads()})\n")

    # results[steps] = {"dense": vl, E: vl, ...}; macs[E] is constant across steps.
    results = {s: {} for s in STEP_BUDGETS}
    macs = {}
    wall0 = time.time()

    for steps in STEP_BUDGETS:
        # dense baseline at this budget (reference line for the gate)
        vl, m, tps = run(lambda: DenseFFN(D), steps, train, val, vocab)
        results[steps]["dense"] = vl; macs["dense"] = m
        print(f"  steps={steps:5d}  dense FFN          val {vl:.4f}  "
              f"active MACs/tok {m:>8d}  {tps:6.0f} tok/s")
        for E in E_SWEEP:
            vl, m, tps = run(
                lambda E=E: CounterMemoryFFN(D, n_cells=E, k=K, key_dim=KEY_DIM,
                                             C=11, lr=0.04, lr_scale=2e-4),
                steps, train, val, vocab)
            results[steps][E] = vl; macs[E] = m
            print(f"  steps={steps:5d}  memory FFN E={E:<6d}  val {vl:.4f}  "
                  f"active MACs/tok {m:>8d}  {tps:6.0f} tok/s")
        print(f"    [elapsed {(time.time()-wall0)/60:.1f} min]")
    print()

    # ---- table: val-loss per (E, steps) ----
    print("=" * 78)
    print("VAL-LOSS  (rows = arm, cols = step budget)")
    hdr = "  " + f"{'arm':<18s}" + "".join(f"{s:>10d}" for s in STEP_BUDGETS) + f"{'MACs/tok':>12s}"
    print(hdr)
    print("  " + "dense FFN".ljust(18) +
          "".join(f"{results[s]['dense']:>10.4f}" for s in STEP_BUDGETS) +
          f"{macs['dense']:>12d}")
    for E in E_SWEEP:
        print("  " + f"memory E={E}".ljust(18) +
              "".join(f"{results[s][E]:>10.4f}" for s in STEP_BUDGETS) +
              f"{macs[E]:>12d}")
    print("=" * 78)

    # ---- per-budget monotonicity verdict (the core question) ----
    print("\nMONOTONICITY of the memory-FFN E-sweep (val should DECREASE as E grows):")
    any_mono = False
    first_mono = None
    for steps in STEP_BUDGETS:
        seq = [results[steps][E] for E in E_SWEEP]
        mono = is_monotonic_decreasing(seq)
        any_mono = any_mono or mono
        if mono and first_mono is None:
            first_mono = steps
        seq_str = "  ".join(f"E={E}:{v:.4f}" for E, v in zip(E_SWEEP, seq))
        # locate the first non-monotonic step for diagnosis
        viol = next((f"E={E_SWEEP[i+1]} >= E={E_SWEEP[i]}"
                     for i in range(len(seq) - 1) if not seq[i + 1] < seq[i]), None)
        tag = "MONOTONIC" if mono else f"NON-monotonic (first break: {viol})"
        print(f"  steps={steps:5d}: {seq_str}   -> {tag}")

    # ---- verdict ----
    print("\n" + "-" * 78)
    if any_mono:
        print(f"VERDICT: M1 capacity-scaling becomes MONOTONIC at >= {first_mono} steps.")
        print("         The E-sweep was undertrained at 1500 steps; with enough training the larger")
        print("         table addresses correctly and bigger E -> lower loss, as the M1 claim needs.")
    else:
        print(f"VERDICT: M1 stays NON-MONOTONIC even at {max(STEP_BUDGETS)} steps.")
        print("         More training on this toy char corpus does NOT recover monotonic capacity")
        print("         scaling -> the corpus saturates and/or the product-key router undertrains")
        print("         regardless. M1's scaling claim needs real data and/or a better router; it")
        print("         is not supported by simply training the toy witness longer.")
    print("-" * 78)


if __name__ == "__main__":
    main()
