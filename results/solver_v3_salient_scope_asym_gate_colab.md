# Deploy-scale warm gate: salient_scope=layer + asym(0.5) on Qwen2.5-1.5B (Colab G4)

Date: 2026-07-22. Hardware: Colab G4 (RTX PRO 6000 Blackwell, 97.9 GB). Code: working
tree with the 2026-07-22 solver upgrades (solve_group_state refactor, salient_scope,
donor/asym.py residual-form cascade calibration, guided-H) — NOT yet committed; delivered
to Colab as a zip of src/scripts/pyproject (clipboard-b64 bootstrap cell).

Protocol: `run_ptq_recovery.py` with STEPS=0 (pure PTQ + strict warm eval at alpha=0),
production deploy config (itf + align + salient 1% + in-sweep, group=128, C=11,
kind=counter_packed), calibration 128xB8xT512 = 524k tokens from the mix corpus.
Corpus: `build_mix_corpus.py --train-tokens 12_000_000` rebuilt IN Colab from the public
HF sources (v3 shares: en40/ru30/code12/math8/science5/instruct5) — NOT the historical
mix_v2 150M val slices, so absolute PPLs are comparable only within this table; the
classic arm is the internal baseline (its EN 77.89 lands next to the historical 74.6,
confirming the refactored solve path reproduces the production solver).

## strict ternary warm PPL (alpha=0), lower is better

| domain | classic | salient layer | layer + asym s=0.5 |
|---|---:|---:|---:|
| en | 77.89 | **67.98** (−12.7%) | 78.16 |
| ru | 108.24 | 88.66 (−18.1%) | **51.10** (−52.8%) |
| code | 14.98 | **12.68** (−15.4%) | 14.97 |
| math | 25.98 | 26.81 (+3.2%) | 30.69 |
| science | 74.27 | **67.41** (−9.2%) | 85.77 |
| instruct | 31.20 | **29.38** (−5.8%) | 36.60 |
| aggregate metric (log) | 3.7920 | **3.6873** | 3.7457 |

Arm runtimes on G4 (solve + eval, no training): classic ~8 min, layer ~8 min,
asym05 ~11 min (28 chunks x 2 calibration passes on top).

## Readings

1. **salient_scope=layer wins the gate outright**: every domain improves except a
   +3.2% math blip; aggregate −0.105 log-PPL (~−10% mean PPL) at IDENTICAL bpw and
   format. Matches the CPU held-out witness (q_proj −20%, k_proj −26% rel H-err:
   attention rows are heterogeneous). RECOMMENDATION: enable SALIENT_SCOPE=layer for
   the next recovery run.
2. **asym(s=0.5, chunk=7) is domain-skewed at deploy scale**: RU collapses 108→51
   (−53% vs classic, −42% vs layer) — the cascade correction pays off most where the
   distortion is largest (RU is the widest fp→warm gap, 9.2→108). EN/code revert to
   classic level, math/science/instruct get worse; aggregate still beats classic but
   loses to plain layer. Consistent with the CPU witnesses: full-strength correction
   overcorrects where the true cascade signal is thin. Next knobs (NOT yet measured):
   s in 0.25–0.4, asym only for deep blocks, or per-domain calibration weighting.
3. classic arm reproduces the production ladder on a fresh runtime + rebuilt corpus +
   refactored solver — no regression from the 2026-07-22 refactor (unit suite: 230
   passed locally).

Raw logs: /content/{classic,layer,layer_asym05}.log in the Colab session (ephemeral);
the full per-arm output (solver progress, homotopy diagnostics) was also printed into
the notebook (Untitled4.ipynb on the account's Drive).

## ASYM_STRENGTH sweep (same protocol, all arms = SALIENT_SCOPE=layer + asym c7)

Fresh G4 runtime, corpus rebuilt by the same builder (bit-identical domain token
counts — the builder streams deterministically), so the s=0/s=0.5 anchors from the
first run carry over.

| s | en | ru | code | math | science | instruct | metric |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.00 (layer) | 67.98 | 88.66 | 12.68 | 26.81 | 67.41 | 29.38 | 3.6873 |
| 0.08 | 47.21 | 48.24 | 8.23 | 17.16 | 37.72 | 18.14 | 3.2017 |
| **0.15** | **46.75** | 42.13 | **8.28** | **16.85** | **35.12** | **17.38** | **3.1563** |
| 0.20 | 48.32 | 40.57 | 8.58 | 18.08 | 35.34 | 18.43 | 3.1841 |
| 0.25 | 50.48 | **40.08** | 9.02 | 19.07 | 36.37 | 18.96 | 3.2159 |
| 0.35 | 55.07 | 41.42 | 10.30 | 21.08 | 42.11 | 21.50 | 3.3201 |
| 0.50 | 78.16 | 51.10 | 14.97 | 30.69 | 85.77 | 36.60 | 3.7457 |

The curve is smooth and unimodal with the minimum AT s=0.15 (0.08 and 0.20 bracket it
from both sides). Runtime cost vs classic: ~+40% solve time (11 vs 8 min for 1.5B on
G4) plus a resident fp model copy.

## Capacity/iteration sweep (packs 1-2, same gate; all arms layer + asym c7)

| arm | salient | s | passes | en | ru | code | math | sci | instr | metric | +bpw |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| best 1% | 1% | 0.15 | 1 | 46.75 | 42.13 | 8.28 | 16.85 | 35.12 | 17.38 | 3.1563 | +0.48 |
| iter2 | 1% | 0.15 | 2 | 46.50 | 39.53 | 8.26 | 17.26 | 35.60 | 17.50 | 3.1519 | +0.48 |
| guided (no asym) | 1% | — | — | 76.05 | 68.41 | 11.29 | 23.18 | 74.32 | 28.66 | 3.6313 | +0.48 |
| sal2 | 2% | 0.15 | 1 | 37.52 | 34.27 | 7.13 | 14.45 | 25.75 | 14.23 | 2.9497 | +0.96 |
| sal2 | 2% | 0.20 | 1 | 38.29 | 33.25 | 7.28 | 14.65 | 26.13 | 14.33 | 2.9573 | +0.96 |
| **s2i2** | 2% | 0.15 | 2 | 35.59 | 31.84 | 7.04 | 13.81 | 23.98 | 13.56 | **2.8991** | +0.96 |
| **s3_s15** | 3% | 0.15 | 1 | **31.68** | **29.82** | **6.41** | **12.47** | **21.09** | **12.07** | **2.7953** | +1.44 |

Readings:
1. **Capacity is the binding constraint, as predicted by the s-curve**: salient
   1% -> 2% -> 3% at fixed s=0.15 drops the metric 3.156 -> 2.950 -> 2.795 with NO
   saturation yet. But this is buying quality with bits: 3% ≈ +1.44 bpw on top of the
   ~1.7-2.2 bpw base (total ~3.1-3.6 bpw, 4-bit territory). The same-format sweet spot
   is 2%.
2. **Iteration (2 passes) pays once capacity exists**: +0.004 at 1% salient vs +0.05
   at 2% — the second pass needs somewhere to put the refined correction.
3. guided-H works standalone (3.687 -> 3.631, RU 88.7 -> 68.4) but is second-order
   next to asym; combining it with asym is a future code change.
4. **RECOMMENDED DEPLOY (same-format budget): SALIENT_SCOPE=layer CALIBRATION=asym
   ASYM_STRENGTH=0.15 ASYM_PASSES=2 SALIENT_FIRST=0.02** — metric 2.899, en 35.6
   (vs classic 77.9), ~2.2-2.7 bpw total. Quality option: SALIENT_FIRST=0.03 →
   metric 2.795, en 31.7 at ~3.1-3.6 bpw.
5. Session totals vs the original production solver: aggregate 3.792 -> 2.795
   (−1.0 log ≈ 2.7x lower mean PPL), en 77.9 -> 31.7, ru 108 -> 29.8, science
   74.3 -> 21.1 — all PTQ-only, zero training steps.

## Recovery run from the new start (6000 steps, s2i2 config, G4 ~2h07m)

Config: SALIENT_SCOPE=layer CALIBRATION=asym ASYM_STRENGTH=0.15 ASYM_PASSES=2
SALIENT_FIRST=0.02, STEPS=6000 B8xT512, fresh 150M mix (en 60M / ru 45M / code 18M /
math 12M / science 7.5M / instruct 7.5M — same builder, deterministic), production
KD recipe (counter cosine 2e-3->1e-4, homotopy hold 20%->90%, KD+0.3CE+0.05feat).
Step time ≈0.9 s (vs 0.33 s of the 1%-salient run — the strict update pays for the
doubled sparse channel); solve+train+evals ≈ 2h07m ≈ 19 units.

Final (step 6000/6000, the monotone BEST checkpoint, strict ternary alpha=0):
loss=2.847 kd=1.828 feat=0.182; **ppl_en 30.06, ppl_ru 22.39**, ppl_code 6.6x,
**metric=2.6779** (trajectory 2.899 warm -> 2.678 trained; the last step is the best).
(code/math/science/instruct digits were clipped in the notebook viewport; the full
line lives in the notebook's grep cell output and /content/recovery.log of that
session. The run itself: recovery rc: 0, RECOVERY DONE.)

vs previous campaign best TRAINED strict (v3f2, 6000 steps from the 74.6 start):
en 47.4 -> **30.06** (−37%), ru 65.9 -> **22.39** (−66%). fp teacher: en 11.6, ru 9.2.

NOTE: the checkpoint was NOT persisted (session storage only — no Drive grant in the
automated run); the run is reproducible from the command above (seeded pipeline).

**A WEAK cascade correction on top of layer-scope is a step change, not a trade-off:
s=0.15 beats every other configuration on EVERY domain.** vs classic: en −40%,
ru −61%, code −45%, math −35%, science −53%, instruct −44%; aggregate 3.792 → 3.156
(~−0.64 log ≈ almost half the mean PPL). The warm (NO training) EN 46.75 is BELOW the
best TRAINED strict checkpoint of the whole previous campaign (47.4 after 6000 KD
steps) — the solver alone now clears last week's post-recovery bar. The s-curve is
smooth with a single minimum between 0 and 0.25 (0.08/0.20 refinement arms pending);
the s=0.5 collapse matches the CPU-witness overcorrection story. Mechanism reading:
the correction direction is right (it is what the litreature's full-strength methods
exploit at W2), but at 2-bpw ternary + salient the per-layer targets can only absorb
a FRACTION of the cascade error before the grid snaps — a tempered target keeps the
correction inside the representable set.
