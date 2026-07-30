# Full-model KD gate — row vs group vs dec4 on Qwen2.5-0.5B (T4, measured)

Date: 2026-07-30. Kernel `lirovkharki/mn-grouplocal-kd-gate`, Tesla T4. Script:
`scripts/grouplocal_kd_fullmodel_gate.py`. All 24 blocks converted (classic PTQ start,
3.7 min on T4, shared by every arm), KD vs the fp teacher on WikiText-2, 300 steps,
batch 2×512. Headline metric: strict WikiText-2 PPL (16×1024). fp teacher ppl 10.53;
warm (solver-only, classic config, EN-only calib) ppl 4173.6.

## Run 1 — single lr point (0.002; dec4 at the witness recipe lr×4)

| arm | final ppl | hidden mse (warm 64.12) | s/step |
|---|---:|---:|---:|
| **row** | **134.1** | **22.33** | 1.76 |
| group | 189.4 | 24.57 | 1.84 |
| dec4 (lr×4=0.008) | 531.8 | 33.62 (noisy curve) | 1.59 |

**The 2-block pre-gate result REVERSES at full-model depth.** The pre-gate (blocks 0-1,
frozen slice) had group at −29.7% vs row −15.9%; at 24 blocks with the identical recipe
row wins. dec4 at lr×4 is a clean instance of the project's own rule — "from a good PTQ
start, a hot lr damages it" — the ×4 compensation that dominated the toy
teacher-recovery is too hot for a deep model at step 0.

Honest reading of the reversal: the group denominator is FINER than row's, so at equal
lr the effective per-weight step is different (hotter where a group's g² is small).
Depth compounds what a shallow prefix forgives. A single-lr comparison is therefore
apples-to-oranges — the same trap the decimation witness closed with full-lr×2/×4
control arms.

## Run 2 — lr-matched arms (pending completion)

Arms: row:0.001 (control), group:0.001, group:0.0005, dec4:0.002 (=×1 of base),
dec4:0.004 (×2). Verdict rules: if group at its best lr matches/beats row at its best —
the speed/memory win (fused+dec4, 1.7-3.0× over dense) comes with quality intact and the
combination goes to the deploy recipe with a cosine schedule; if row stays ahead at
every matched lr — group scope remains a SPEED format (still strictly better memory,
and decimation still applies) whose quality gap must be priced, and the pre-gate's
shallow-regime optimism is pinned as a documented-negative for 2-block gates.

## Step-time note

s/step differences here are NOT a kernel benchmark (teacher forward dominates); see
results/GPU_GATE_T4_GROUPLOCAL.md for the update-path timings (dec4 1.7-3.0× over dense).
