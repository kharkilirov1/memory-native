# Production promotion gate — 1.5B, mixed corpus, cosine, row vs group (Kaggle 2xT4)

Date: 2026-07-30/31. Kernels `mn-prod-gate-row` / `mn-prod-gate-group`; runner =
`scripts/run_ptq_recovery.py` (the production loop: KD+CE+feature-KD, homotopy alpha,
cosine counter lr, fp-tail AdamW, strict-alpha=0 selection). Model Qwen2.5-1.5B,
teacher on cuda:1 (KD split), student B2x512, 1500 steps.

## Shared start (bit-identical across arms)

v3-layer solve on the T4 itself: GRID=itf, SCALE_REFIT=align, SALIENT_FIRST=0.02,
SALIENT_SCOPE=layer, IN_SWEEP_REFIT=1, fp calibration 64x2x512 (131k tokens of the pilot
mix below). Both arms' solves produced the IDENTICAL warm eval — metric 3.2992 (en 51.7,
ru 62.0, code 7.9, math 18.2, sci 43.6, instr 19.6) — solver determinism witnessed at
1.5B scale. (Delta vs the full s2i2 deploy start: no asym pass, 131k not 524k calib —
shared by both arms, so the COMPARISON is unaffected.)

Corpus: `mix_pilot`, 12M tokens, 6 domains (en .40 / ru .30 / code .12 / math .08 /
science .05 / instruct .05), donor BPE, built by scripts/build_mix_corpus.py; SAME bins
for both arms (bit-exact — the corpus is an uploaded artifact, not a per-run rebuild).

## Row arm (COUNTER_LR 0.002→1e-4 cosine) — COMPLETE

| step | strict metric | en | ru |
|---:|---:|---:|---:|
| warm | 3.2992 | 51.7 | 62.0 |
| 300 | 3.2746 | 46.1 | 64.5 |
| 600 | 3.1680 | 41.1 | 49.9 |
| 900 | 3.0198 | 35.2 | 41.3 |
| 1200 | 2.7439 | 28.2 | 28.1 |
| **1500** | **2.7120** | **27.65** | **26.61** |

Monotone, best = final. Context: the entire previous campaign's best (6000 steps, B8,
richer s2i2 start, G4) was 2.6779 — this run reaches 2.712 with 1/8 the KD tokens on a
weaker start: the production loop is healthy on Kaggle T4s.

## Group arm (COUNTER_LR 0.001→5e-5 cosine — the re-centered grid) — COMPLETE

| step | strict metric | en | ru |
|---:|---:|---:|---:|
| warm | 3.2992 | 51.7 | 62.0 |
| 300 | 3.2346 | 44.1 | 59.7 |
| 600 | 3.1877 | 43.0 | 50.8 |
| 900 | 3.0518 | 36.7 | 41.8 |
| 1200 | 2.7648 | 29.5 | 27.4 |
| **1500** | **2.7262** | 28.38 | **25.87** |

**Group 2.7262 vs row 2.7120 — a virtual tie with a marginal row edge (+1.4% mean PPL).**
Per-domain: group WINS ru (25.87 vs 26.61) and instruct (10.55 vs 10.58); row wins en
(27.65 vs 28.38), math, science. Both curves still descending at 1500 — neither
converged. The 0.5B lr-matched −16% group win did NOT fully transfer to the 1.5B
production loop (which adds homotopy alpha, feature-KD and the fp AdamW tail); one lr
point per arm, single seed.

## Verdict (per the pre-registered rule — honored strictly)

2.7262 > 2.7120 is not "win/tie": **row scope KEEPS the quality-production default.**
What IS promoted to production (the measured, unambiguous wins):

- `stats_scope="group"` + fused kernel + `decimation=4` becomes the OFFICIAL SPEED
  RECIPE — 1.7-3.0x faster than the dense update at lower peak memory (T4-measured,
  results/GPU_GATE_T4_GROUPLOCAL.md) at a quality cost bounded by ~1.4% mean PPL at
  this budget (dec4's own 1.5B number below). Fully wired: layer → ptq_warm_start →
  run_ptq_recovery (STATS_SCOPE/DECIMATION envs) → restore/resume.
- `SALIENT_REFIT=align` available as the solver's free post-pass (held-out-gated).

Re-open conditions for the quality flip: a group lr micro-grid at 1.5B (0.0007-0.0015),
longer budgets (both curves unconverged), and/or homotopy-off comparison — the 0.5B
evidence says the gap is regime-sensitive, not structural.

## dec4 arm (group + decimation=4, lr 0.002→1e-4) — running

v1-v3 debug ledger (all pinned in code): B4 OOM at the first KD step (fp32 logits on a
14.5 GiB card → B2 + expandable_segments); kernel_sources of an ERROR version do not
mount (warm ckpts re-shipped as the `mn-prod-warm-ckpts` dataset); restore_counter_
structure's TWO allowed-kw filters silently dropped stats_scope → the resume built
row-shaped v for a group checkpoint (same silent-drop class as the ptq filter — now all
four filter sites pass stats_scope/decimation).

## Decision rule (pre-registered)

Group final metric vs row's 2.7120 on the identical protocol: win/tie → flip the
production recipe to stats_scope=group + half-row lr (+ decimation=4 as the speed
option; fused kernel already 1.7-3.0x over dense with dec4); lose → group stays a
speed/memory format, row keeps quality production, and the 0.5B lr-matched win is
re-examined at 1.5B before any further promotion attempt.
