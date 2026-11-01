# mx6 latent grid: per-weight momentum closes the gap to AdamW (G4, 2026-11-01)

Raw logs: `mx6_g4_momentum_2026_11_01/` (`mom/` = momentum runs, `diag_no_momentum/` = the sweep before it,
`run_1b/` = the 1B TinyStories logs: lg6 full 5000 steps, dense stopped at step 1050).

## 1B TinyStories (byte-level, batch 128x512, cosine) -- why we looked

| step | dense AdamW val bpb | lg6_fp4_h2 (no momentum, lr 3e-3) |
|---|---|---|
| 500  | 1.376 | 3.275 |
| 1000 | 0.933 | 2.942 |
| 5000 | (stopped at 1050) | 0.947 |

lg sat on a ~3.3 bpb plateau for ~750 steps; dense at 1000 steps already beat lg at 5000. Dense was stopped to save
compute (gap obvious).

## Diagnosis on d512 (enwik8 20M chars, 4 layers, 3000 steps, batch 128, bf16+compile, cosine, NVFP4 scales)

No momentum (held-out val, bits/char; dense_nobias AdamW 1e-3 = 1.760):

| lr | 0.003 | 0.005 | 0.01 | 0.03 | 0.1 |
|---|---|---|---|---|---|
| lg6 val | 2.087 | 2.078 | 2.033 | 2.921 | 3.237 |

lg8 (lr 0.01) 2.354, buffer accumulation x4 2.174: not lr, not bits, not batch. Telemetry: 60-80 % of weights reverse
their step direction between consecutive updates early (33-45 % at the end) -- the update follows per-step gradient
noise. Adam's first moment averages it; the lg update had none.

## Fix: `set_momentum` (mx6/lg_accum.py, "ema" mode)

`buf <- mu*buf + (1-mu)*g` per weight, the RMS-normalised lg update consumes `buf`. Buffer bf16 (+2 B/param) or fp8 e4m3
+ per-row scale + stochastic rounding (+~1 B/param). Flags: `--momentum MU --momentum-dtype bf16|fp8 [--nesterov]`.

| arm (lg6_fp4_h2 unless noted) | seed | val |
|---|---|---|
| no momentum, lr 0.01 | 0 | 2.080 |
| mu 0.9, lr 0.003 | 0 | 1.981 |
| mu 0.9, lr 0.01 | 0 | 1.832 |
| mu 0.9, lr 0.01, fp8 buffer | 0 | 1.839 |
| mu 0.9, lr 0.01, nesterov | 0 | 1.848 |
| mu 0.95, lr 0.01 / 0.03 / 0.05 | 0 | 1.862 / 1.817 / 1.812 |
| **mu 0.9, lr 0.03** | 0 / 1 | **1.770 / 1.767** |
| **mu 0.9, lr 0.03, fp8 buffer** | 0 | **1.769** |
| mu 0.9, lr 0.05 | 0 | 1.776 |
| mu 0.9, lr 0.1 | 0 | 3.399 (diverges) |
| **dense_nobias AdamW 1e-3** | 0 / 1 | **1.761 / 1.752** |

Gap to AdamW: +0.18..0.27 bits/char -> **+0.012** (0.7 %), 2 seeds each. Early curve now matches dense
(step 250: 2.758 vs 2.766). fp8 momentum = bf16 momentum. Training state: ~6.5 bits idx + 8 bits momentum + row RMS
~ 1.9 B/param vs 12-16 B/param for AdamW (fp32 master + m + v).

## Next

1. 1B TinyStories with mu 0.9 fp8 momentum, lr tuned upward (~3x the no-momentum lr), compare to dense at steps 500/1000.
2. Fuse the EMA into the Triton update kernel (now separate torch ops: speed cost not yet measured at 1B).
3. Larger d / more steps to check the gap stays closed.
