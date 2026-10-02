# Decimation witness — 1/4-group round-robin update (CPU, toy teacher-recovery)

Date: 2026-07-30. Script: `scripts/decimation_witness.py`; contract test:
`tests/test_grouplocal_update.py::test_grouplocal_decimation_is_exact_restriction`.

## What decimation is here

`active_groups` on `group_counter_update_grouplocal_hashsr`: update only the scheduled
groups; inactive groups' state/scale/v stay BIT-untouched, and active groups get
BIT-identically what a full update would give them (pinned by the contract test). This is
only well-defined because group-local statistics cross no group boundary — under row scope
skipping groups would change every denominator. On the fused kernel it maps to launching
the grid over the scheduled groups only → S× fewer update FLOPs, no math change.

## Protocol

Ternary-teacher recovery (the project's standard first gate), out=64 in=128 group=16
(8 groups), C=11, clip=1.0, base lr=0.02, 600 steps (equal-FLOPs arms: 2400), schedule
`g % 4 == step % 4`. Full-update arms at lr×1/×2/×4 keep the comparison honest — without
them "dec4 lr×4 wins" is indistinguishable from "hotter lr wins".

## Results (seed 0; final mse, y_var=5.34, thr=0.02)

| arm | steps | final mse | steps<thr | update FLOPs |
|---|---:|---:|---:|---:|
| full lr×1 | 600 | 0.1176 | — | 1.00× |
| full lr×2 | 600 | 0.0814 | — | 1.00× |
| full lr×4 | 600 | 0.0341 | 27 | 1.00× |
| dec4 lr×1 | 600 | 0.1064 | — | 0.25× |
| dec4 lr×2 | 600 | 0.0574 | — | 0.25× |
| **dec4 lr×4** | 600 | **0.0063** | 81 | **0.25×** |
| dec4 lr×1 equal-FLOPs | 2400 | 0.0751 | — | 1.00× |
| dec4 lr×2 equal-FLOPs | 2400 | 0.0344 | — | 1.00× |

Seed robustness (full-lr×4 vs dec4-lr×4 final): seed 1: 0.0209 vs 0.0124; seed 2: 0.0341
vs 0.0071; seed 3: 0.0193 vs 0.0120. **dec4-lr×4 wins on every seed, 1.6–5.4×.**

## Reading

- The naive fear ("counters integrate 4× rarer signal → worse") is NOT what happens with
  lr×S compensation: the integrated signal per counter matches the full arm.
- The interesting part: full-lr×4 crosses the threshold FAST (step 27) then bounces back
  into a noise ball (0.034) — the constant-LR plateau the project already knows. dec4-lr×4
  crosses later (step 81) and settles ~5× LOWER. Mechanism hypothesis: only 1/4 of the
  groups take the hot SR tick per step, so per-step noise injection is staggered/averaged —
  decimation behaves like implicit noise reduction at the same effective signal.
- Equal-FLOPs cold arms (lr×1/×2 at 4× steps) do NOT catch up — the lever is
  "decimate + compensate lr", not "just run longer at 1/S".

## Status / next

Toy-scale, constant lr, single layer. Before any production use: repeat inside the KD
recovery loop (2-block CPU pre-gate protocol) and with the cosine schedule; then the fused
kernel grows a `groups_offset/groups_stride` grid restriction (trivial). Combined outlook
if it carries: fused one-launch update × 1/4 FLOPs ≈ update cost vanishes from the step.
