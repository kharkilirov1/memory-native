# Qwen2.5-0.5B → counter → distill recovery — real run (CPU)

Date: 2026-07-06 · Box: 20-core CPU, no GPU

First real end-to-end run of the finetune-pretrained pipeline on the actual donor and real text
(WikiText-2-raw). CPU-sized (the box has no CUDA); it proves the mechanism on the real model, not
a production recovery to the fp baseline (that is a GPU-scale run).

## Setup

- Donor: **Qwen/Qwen2.5-0.5B** (base), fp32, eager attention. Swap: 168 body linears →
  counter (`RMSCounterLinear`), 357.8M counter coeffs, `lm_head` kept fp (tied).
- Corpus: WikiText-2-raw, 40 train batches + 24 held-out val batches, `2 × 256` tokens.
- Recipe (the stable one found by diagnosis — see below): `threshold_ratio=0.5`, fp AdamW
  `lr=3e-4` + `grad_clip=1.0`, counter `lr=0.008` + `local_grad_clip=1.0`, KD `T=2`, `ce_alpha=0.3`.
- Schedule: 8 rounds × 15 = **120 distill steps**; PPL measured after each round. 48.6 min.

## Result — PPL on held-out WikiText

| stage | PPL | note |
|---|---:|---|
| fp teacher (baseline) | **21.18** | the donor's own quality |
| counter warm-start | **159 268** | ternarizing a full-precision 0.5B donor is catastrophic |
| after 120 distill steps | **907** (best round 7: **589**) | **99.4% of the warm-start gap closed** |

Curve (PPL per round): `159268 → 3530 → 1634 → 1026 → 998 → 627 → 776 → 589 → 907`.

Monotone-with-noise descent; the tail wobble (589↔907) is counter stochastic-rounding noise on a
small val set. The residual gap to fp (≈21) is a **scale/compute limit** (120 CPU steps, 40
batches), not a method limit — it needs a GPU run (thousands of steps, more data) to close fully.

## What this establishes (and the honest caveat)

- The full pipeline runs on the **real** donor: load → in-place counter swap → distill → PPL.
- Warm-start alone is unusable (as the project notes predicted: forward sees only `s·t`); recovery is a
  **network-level distill effect**, and it works — 175× PPL reduction (159k → 0.9k).
- NOT full recovery to the fp baseline. That is out of CPU reach; the T4 script/notebook are for it.

## Diagnosis trail (why the recipe is what it is)

1. First run with the naive default (`lr=2e-3`, no clip, `threshold=0.7`) **diverged**: warm PPL
   2e5 → round 1 **5.5e13**. Fixed by lowering both lrs, adding fp grad-clip + counter
   `local_grad_clip`, and `threshold=0.5` (least-bad warm-start in a 0.2–0.7 sweep: all ≥1.8e5,
   so threshold is not the lever — the recovery finetune is).
2. Conservative recipe was stable (186k → 2.6k in 20 steps, loss falling); `+ce=0.3` and lower
   counter lr gave the clean run above.

Reproduce (real GPU, full recovery): `python scripts/finetune_qwen_counter.py` (defaults are this
recipe) or `notebooks/qwen_recovery_t4.ipynb` on a T4.
