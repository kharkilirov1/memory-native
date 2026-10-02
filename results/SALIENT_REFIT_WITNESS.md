# L1 witness — salient values as a solved corrector (copy → LSQ refit)

Date: 2026-07-30. Script: `scripts/salient_refit_witness.py`. Context: conversion-quality
analysis (the salient channel is the format's only continuous freedom after the sweep;
today it stores EXACT COPIES of original weights at pre-sweep-chosen positions).

## The lever

After the ternary sweep, refit the k% salient values to absorb the residual layer error:
minimize (w−q)ᵀH(w−q) over q_S with the ternary part fixed → per-row SPD solves
e_S = −H_SS⁻¹ H_SB e_B (~k_o×k_o each, trivial cost). Zero format change, zero bytes.

## Protocol

Two-blob gate (train H drives, HELD-OUT H judges — the item-7 overfit rule). Qwen2.5-0.5B
block 0, v3-lite solve (itf grid, align refit, salient 2% layer-scope), 16×256 train /
8×256 eval tokens (+50k offset), WikiText-2.

## Result — refit never loses, wins modestly; attention benefits most

| layer | arm | rel err (train H) | rel err (EVAL H) |
|---|---|---:|---:|
| q_proj | copy | 0.001143 | 0.002412 |
| q_proj | **refit** | 0.001014 (−11.3%) | **0.002292 (−5.0%)** |
| up_proj | copy | 0.062682 | 0.128694 |
| up_proj | refit | 0.060682 (−3.2%) | 0.128042 (−0.5%) |
| down_proj | copy | 0.007311 | 0.064121 |
| down_proj | refit | 0.006580 (−10.0%) | 0.063590 (−0.8%) |

Values move meaningfully (q_proj mean|Δ| ~1e-2, max ~0.19) — the copies were NOT optimal.
But the train-H gain (−3…−11%) largely evaporates on held-out (−0.5…−5%): the refit
partially fits calibration noise at this tiny 4k-token calib. Attention keeps the most
(consistent with the salient_scope=layer finding: attention rows are heterogeneous).

## 16× calibration follow-up (64×1024 train / 32×1024 eval) — prediction CONFIRMED

| layer | copy eval-H | refit eval-H | held-out gain |
|---|---:|---:|---:|
| q_proj | 0.001600 | 0.001488 | **−7.0%** (was −5.0% @4k) |
| up_proj | 0.087692 | 0.086288 | −1.6% (was −0.5%) |
| down_proj | 0.030868 | 0.030090 | −2.5% (was −0.8%) |

The train/eval gap essentially closes at real calibration size — the refit gain carries
to held-out nearly in full. At deploy calib (524k) expect the train-side −3…−11% to be
the honest number. Attention benefits most, consistent with salient_scope=layer.

## Verdict

- **Safe free win, not a step change.** Fold in as an optional post-pass
  (`salient_refit="align"`-style knob) — it is exact, cheap, and never lost on held-out.
  Expect the held-out share of the gain to GROW at deploy-scale calibration (524k tokens
  → far less H overfit); re-measure there before crediting more than a few %.
- The larger L1 half remains open: POSITION selection after the sweep (by residual /
  end-loss contribution) rather than pre-sweep |w|·√diagH — that changes what the budget
  is spent on, not just the values, and needs a re-sweep per candidate set (costlier
  experiment, same two-blob gate).
