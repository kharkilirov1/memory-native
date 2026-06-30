"""End-to-end witness: does the WIRED ffn="moe" (M4) beat ffn="dense" in the full GPT?

The earlier M4 win (REALDATA_SCALING.md) was measured at the ISOLATED FFN level. This runs the
integrated model: same GPT, attention dense-fp in every arm (isolates the FFN variable), only the
FFN sub-block swapped via GPTConfig.ffn. Counter-MoE uses h=4d/top_k so top_k experts ~ the dense
FFN's active MACs/token -- an EQUAL-ACTIVE-COMPUTE comparison (capacity in experts is "free" MACs).

The val-loss comparison is HARDWARE-INDEPENDENT: CPU yields the same loss as a GPU (only wall-clock
differs). So this is a valid quality witness on CPU; tok/s is NOT measured here (that needs a GPU).

  python scripts/moe_gpt_witness.py
"""
from __future__ import annotations

import time

import torch

from memory_native.data import get_batch, load_corpus
from memory_native.models import GPT, GPTConfig

D, NL, NH, BLOCK = 128, 3, 4, 64
BATCH, STEPS, EVAL_BATCHES = 32, 500, 20
ADAM_LR = 1e-3
SEED = 0


def _dense_ffn_active_macs(d: int) -> int:
    return 2 * d * 4 * d                                  # fc (d->4d) + fc2 (4d->d)


@torch.no_grad()
def _val_loss(model, val, gen) -> float:
    model.eval()
    tot = 0.0
    for _ in range(EVAL_BATCHES):
        x, y = get_batch(val, BLOCK, BATCH, x_dev, gen)
        _, loss = model(x, y)
        tot += float(loss)
    model.train()
    return tot / EVAL_BATCHES


def run_arm(name, cfg, train, val, *, macs):
    torch.manual_seed(SEED)
    model = GPT("dense", cfg).train()                    # attention dense-fp in EVERY arm
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=ADAM_LR)
    data_gen = torch.Generator().manual_seed(123)        # identical data stream across arms
    val_gen = torch.Generator().manual_seed(999)
    t0 = time.time()
    for step in range(STEPS):
        x, y = get_batch(train, BLOCK, BATCH, x_dev, data_gen)
        opt.zero_grad()
        _, loss = model(x, y)                            # counter layers self-update in backward
        loss.backward()
        opt.step()
    vloss = _val_loss(model, val, val_gen)
    dt = time.time() - t0
    print(f"{name:28s} | val {vloss:.4f} | FFN active MACs/tok {macs:>8d} | {dt:5.1f}s")
    return vloss


if __name__ == "__main__":
    x_dev = torch.device("cpu")
    print(f"=== end-to-end MoE-GPT vs dense-GPT (CPU; loss is HW-independent) "
          f"d={D} L={NL} block={BLOCK} steps={STEPS} ===")
    train, val, vocab = load_corpus(x_dev)
    base = dict(vocab_size=vocab, block_size=BLOCK, n_layer=NL, n_head=NH, n_embd=D)

    dense_macs = _dense_ffn_active_macs(D)
    results = {}
    results["dense"] = run_arm("dense FFN (fp, AdamW)",
                               GPTConfig(**base, ffn="dense"), train, val, macs=dense_macs)
    for E in (8, 16):
        cfg = GPTConfig(**base, ffn="moe", ffn_experts=E, ffn_top_k=2)
        # h = 4d/top_k -> top_k experts ~ dense active MACs; capacity E is extra params, not MACs.
        moe_macs = 2 * 2 * D * (4 * D // 2)               # top_k * (d->h + h->d), h=4d/top_k
        results[f"moe E={E}"] = run_arm(f"counter-MoE FFN E={E} k=2",
                                        cfg, train, val, macs=moe_macs)

    best_moe = min(v for k, v in results.items() if k.startswith("moe"))
    print(f"\nVERDICT: dense {results['dense']:.4f}  vs  best MoE {best_moe:.4f}  -> "
          f"{'MoE WINS' if best_moe < results['dense'] else 'dense wins'} "
          f"(Δ {results['dense']-best_moe:+.4f}) at equal active compute")
