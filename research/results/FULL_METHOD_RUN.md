# Full-method run — d=1024, 12L, 2000 steps (T4)

The complete method as one model (`ReversibleGPT`): every linear is a finite-state counter
layer (counter_packed + 4-bit saved activations) and the whole transformer is wrapped in the
O(1) `ReversibleSequence`. Real tinyshakespeare, block 256, batch 16, 2000 steps.
Log: [`gpu_full_method_d1024_T4.log`](gpu_full_method_d1024_T4.log).

## Memory @ d=1024, 12 layers (one training step)

| config | training peak | persistent state |
|---|---|---|
| **full method (counter + reversible)** | **982 MiB** | **110 MiB** |
| dense + AdamW (same reversible arch) | 2.85 GiB | 577 MiB |
| dense + AdamW (plain GPT) | 3.69 GiB | 577 MiB |

The full method uses **3.76× less peak** than dense+AdamW (and 5.2× less persistent). Both
levers are visible and stack: reversible alone takes dense 3.69 → 2.85 GiB (activations), the
counter takes 2.85 → 0.98 GiB (weights/optimizer).

## Training (2000 steps, ~90 min, 1,510 tok/s)

train loss 4.46 → **1.45**; **val 1.72**; training peak **983 MiB**; counter zero-fraction 0.333
(healthy ternary). The full method trains a real d=1024 model end-to-end under 1 GiB.

## Overfitting

tinyshakespeare is tiny (~1M chars); 2000 steps × 4096 tokens ≈ 8 epochs over a 150M-coeff
model — overfitting was expected. Observed: **train 1.45 vs val 1.72 (gap 0.27)** — overfitting
is present but **mild, not catastrophic** (val did not diverge). This is consistent with the
ternary weights + stochastic counter updates acting as a strong regularizer: effective capacity
is far below the fp-parameter count. For a clean generalization number at this scale, use a
larger corpus (e.g. enwik8) where 2000 steps is < 1 epoch.

## Cost

Throughput 1,510 tok/s (reversible recompute + per-element counter update). The memory win
comes at a real compute cost; a fused update kernel and/or fewer reversible recomputes (anchors)
would narrow it.
