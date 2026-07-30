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

## Run 2 — lr-matched arms: the confound was real, GROUP WINS the full gate

| arm | final ppl | hidden mse |
|---|---:|---:|
| row @0.002 (run 1 best) | 134.1 | 22.33 |
| row @0.001 | 136.5 | 23.20 |
| **group @0.001** | **112.4 (−16% vs row's best)** | **21.78** |
| group @0.0005 | 129.6 | 22.97 |
| **dec4 @0.002** | **131.8 (ties row's best, 1/4 update FLOPs)** | 22.39 |
| dec4 @0.004 | 264.8 (too hot) | 26.84 |

**Verdict:**
- The reversal in run 1 was an lr artifact: the group denominator is finer, so group's
  optimum sits at HALF of row's lr. At matched effective step, **group scope wins the
  full-model gate outright** — the pre-gate's direction was right after all.
- **dec4 recipe at depth: lr = 2× of group's base** (0.002 vs group's 0.001), NOT the
  toy-scale ×4 — ×4 is the documented "hot lr damages a good start" failure. At that
  recipe dec4 ties row's best quality at a quarter of the update FLOPs.
- Deploy candidate: `stats_scope="group"` + fused kernel + lr at half the row recipe
  (cosine at real recovery scale), `decimation=4` as the speed option at row-class
  quality. Combined with the kernel numbers (results/GPU_GATE_T4_GROUPLOCAL.md):
  **faster than the production dense path, lowest memory, and better quality.**
- Caveats that stay open for the production campaign: 300 steps, constant lr, EN-only
  stream, 0.5B; the s2i2-start mixed-corpus cosine run at 1.5B remains the final
  promotion gate (its lr grid should be re-centered at half of the row recipe).

## Step-time note

s/step differences here are NOT a kernel benchmark (teacher forward dominates); see
results/GPU_GATE_T4_GROUPLOCAL.md for the update-path timings (dec4 1.7-3.0× over dense).
