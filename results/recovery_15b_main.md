# Qwen2.5-1.5B counter-recovery — main run (150M corpus) + plateau diagnosis

Full 150M-token mix (EN 82.5M / RU 37.5M / code 22.5M / math 7.5M), G4-class GPU (95.6 GB),
bf16, B8xT512, counter_packed + fused clipped kernel + int8 T-cache, KD(T=2)+0.3*CE,
CONSTANT counter lr 0.008, fp lr 3e-4. 0.40 s/step (~10.2k tok/s), epoch = 36622 steps.

## Per-domain PPL curve (fp base: EN 10.8 / RU 8.5 / code 2.6 / math 6.0)

| step | EN | RU | code | math |
|---:|---:|---:|---:|---:|
| warm-start | 575347 | 20672417 | 454477 | 369272 |
| 400  | 179.3 | 207.3 | 144.6 | 206.8 |
| 2400 | 108.3 |  60.8 |  31.8 |  95.3 |
| 6400 |  90.2 |  50.7 |  20.7 |  70.5 |
| 11600|  87.0 |  45.9 |  18.0 |  65.6 |
| 13200|  85.2 |  46.0 |  18.3 |  64.5 |
| 16800|  86.8 |  51.2 |  19.3 |  62.9 |
| 17200|  93.4 |  45.7 |  19.2 |  66.4 |

## PLATEAU — real, not false (confirmed by the two-eval rule)

7 consecutive evals over ~10M tokens with NO trend in any domain: EN 86.5-93.4, RU 43.6-51.2,
code 18.4-21.1, math 62.9-68.7. False plateaus in this project lived 1-2 evals; this lives
2400+ steps. From step 6400->17200 tokens grew 2.7x for a single-digit-percent gain. Progress
is above zero mathematically but below the noise floor.

## FORECAST MISS (recorded as-is, per protocol; no retroactive rewrite)

Claude predicted, twice, "final PPL to low-tens/single-digit, near the fp baseline." **Did NOT
happen.** The run plateaued at EN ~87 / RU ~46 / code ~19 / math ~65. Both misses came from
extrapolating a power-law tail; both times the system LEFT the power-law regime earlier. Early
phase was data-bound (why 150M easily beat the 12M pilot); the run then hit a different wall.

## DIAGNOSIS — two suspects, one cheap discriminator

1. **Constant-LR noise-ball floor (Prop 1, "boring"):** finite-state updates at a constant step
   don't converge to a point; they orbit the optimum in a ball of radius ~ lr (+ SR noise). The
   45-90 PPL shelf is that ball; the per-batch loss 3.4-5.9 with no trend is its signature.
2. **Accumulator ceiling (OPEN 1, "deep", never observed at this scale):** the bounded counter
   saturates; no lr helps.

**Confound (why naive "decay lr, watch PPL" lies on finite-state weights):** too small an lr
makes the tick (-lr*grad*C/s) fall below one quantum -> stochastic rounding stops firing ->
flips freeze -> looks identical to a ceiling but is a sub-quantum step. `state_statistics()`
separates all three:
- PPL falls + flips continue                  -> LR noise-ball (suspect 1).
- PPL flat + flips FROZEN + saturation low     -> sub-quantum lr (schedule artifact, NOT a wall).
- PPL flat + flips continue + saturation HIGH  -> accumulator ceiling (OPEN 1 in the wild).

`scripts/run_tail.py` logs {ppl, flip_rate, counter_edge (saturation), scale_mean, d_scale}.
Smoke already reproduced the confound: at lr=0.002 flip_rate=0.0052; at lr=0.0005 flip_rate=0.0000.

## FROZEN FORECASTS (before the tail run) — settle who is better calibrated

Tail experiment: resume final checkpoint, decay counter lr 0.008 -> 0.002 -> 0.0005 (stepped),
~3000 steps, telemetry on.

- **User's bet:** LR-floor dominates; decay removes 30-50% off the shelf; won't reach the teacher.
- **Claude's bet:** LR-floor dominant too, but (a) magnitude more conservative, ~15-30% off the
  shelf, mostly delivered by the 0.002 rung (EN ~68, RU ~37, code ~16, math ~52), NOT reaching
  fp; (b) DISTINCTIVE: the 0.0005 rung will FREEZE flips (flip_rate -> ~0) while counter_edge
  stays moderate -> its flatline is the sub-quantum artifact, not OPEN 1 -- so a naive reader
  would falsely cry "ceiling" at 0.0005 and the telemetry will refute it; (c) OPEN 1 is NOT the
  dominant wall at 87/46/19/65 -- the accumulator ceiling would bite closer to fp, not here.

Resolution in ~2h after the epoch finishes and the tail runs.

## TAIL RESULT (resolved) — LR-floor confirmed, accumulator ceiling refuted

Ran run_tail from the step-36622 checkpoint, counter lr 0.008 -> 0.002 -> 0.0005, 3000 steps.

| rung | steps | PPL EN | RU | code | math | behavior |
|---|---|---|---|---|---|---|
| 0.008 (control) | 0-1000 | 84-94 osc | 44-47 | 17-20 | 62-73 | NO trend = the plateau's noise ball reproduced |
| 0.002 | 1000-2000 | 88.5->78.5 | 44->38 | 20->15 | 68->57 | clean monotone descent |
| 0.0005 | 2000-3000 | 78.5->75.4 | 38->37 | 15->14.6 | 57->54 | still descending, slower |

`counter_edge` (saturation) = **0.081 constant** across the whole tail.

**Verdict:**
- **LR-floor CONFIRMED** — the 0.008 control rung reproduced the plateau (oscillation, no trend);
  0.002 broke it into a monotone descent. The plateau was the constant-LR noise ball (Prop 1).
- **Accumulator ceiling (OPEN 1) REFUTED as the wall** — saturation held flat at 0.081, no growth;
  recovery is LR-schedule-limited, not accumulator-limited. No ceiling in sight (0.0005 was still
  descending at step 3000).
- **Magnitude:** ~6-17% off the shelf in this short 3k-step ladder (EN 84->75, RU 45->37,
  code 17->14.6, math 58->54). A proper long cosine (0.008 -> ~1e-4, tens of k steps) would push
  further -- the user's 30-50% is plausible for a real schedule, not this crude ladder.

## OWNED: telemetry bug + wrong distinctive prediction (Claude)

1. **flip_rate telemetry was DEAD on the CUDA fused path.** weight_flips is never incremented by
   the Triton kernel (documented in CLAUDE.md; missed when building run_tail), so flip_rate read
   0.0000 at ALL rungs including 0.008 where weights obviously moved. The distinctive-prediction
   instrument was broken. FIXED: telemetry now measures the true flip fraction by diffing decoded
   ternary state t across the interval (validated: catches a forced 0.65 state change, 0 false
   positives, on both counter_rms and counter_packed).
2. **Claude's "0.0005 freezes flips (sub-quantum)" prediction is REFUTED** by the working channel:
   PPL kept dropping at 0.0005 (78.5->75.4), so weights kept updating -- no hard freeze. The user
   read the tail better ("decay keeps helping"); Claude was closer on the magnitude range but wrong
   on the mechanism. Net on the differentiator: draw leaning user.

## NEXT (clean experiment, now instrumented correctly)

Long cosine schedule (0.008 -> ~1e-4 over ~20k steps) from the checkpoint, with the fixed flip
telemetry, to (a) push PPL toward the teacher and (b) close the sub-quantum question with real
flip_frac numbers at low lr.

## COSINE TAIL (20k steps, 0.008 -> 1e-4) — clean resolution with fixed flip telemetry

Resumed the plateau checkpoint (EN 84.3), single cosine decay over 20000 steps.

| lr | step | EN | RU | code | math | flip_rate | edge |
|---|---|---:|---:|---:|---:|---:|---:|
| ~.008 | 1000 | 88.8 | 44.5 | 19.7 | 68.1 | 0.192 | 0.080 |
| ~.005 | 8000 | 81.2 | 40.2 | 16.1 | 58.8 | 0.147 | 0.081 |
| ~.002 | 13000| 75.9 | 37.8 | 13.9 | 53.7 | 0.077 | 0.082 |
| 1e-4  | 20000| **71.9** | **35.3** | **12.8** | **48.8** | 0.004 | 0.082 |

**Best recovered model: EN 71.9 / RU 35.3 / code 12.8 / math 48.8** (fp teacher 11.6/9.2/3.0/6.7)
-> the 1.5B in 6-bit counter format recovered to ~4-7x the fp baseline, all domains alive.

**Three regimes cleanly separated:**
- LR noise ball: confirmed (PPL descends as lr decays).
- Sub-quantum freeze: **confirmed + measured** -- flip_rate tracks lr DOWN 0.192 -> 0.004 (50x)
  as lr 0.008 -> 1e-4. Finite-state flips progressively freeze; LR decay has diminishing returns
  bounded by the quantum.
- Accumulator ceiling (OPEN 1): **refuted** -- counter_edge flat at 0.081-0.082 across all 20k.

**Forecast scoring (owned):**
- Claude predicted EN 58-65 / RU 28-33 / code 11-13 / math 42-48 (~30-40% off plateau). Actual:
  72/35/12.8/49 (~19%). Only code hit; EN/RU/math all worse than predicted -> OVERSHOT. This is
  Claude's SECOND consecutive optimism overshoot on LR-decay magnitude (stepped tail: predicted
  15-30%, got 12%). Stable bias: the counter-format floor is HIGHER than Claude keeps predicting.
- Sub-quantum: the fixed telemetry VINDICATES Claude's ORIGINAL (stepped-tail) prediction that
  low lr freezes flips -- which Claude wrongly RETRACTED after the dead telemetry read 0. flip_rate
  0.192->0.004 proves the freeze is real and gradual. The retraction was the error.

**Real next lever: MORE DATA, not more schedule.** At step 20000 PPL had nearly flattened
(72.4->71.9) and flips were frozen (lr at floor) -- more schedule steps won't help. The floor
here (~EN 72 / code 13) is set by the 150M-token data budget + the counter format, not by LR.
The pilot named the data wall; the main run named the schedule wall; the cosine tail removed the
schedule wall and returned to the data wall. To push lower: a B-token pretraining-scale corpus.

## PTQ WARM-START WITNESS (Bonsai-inspired; NO recovery training anywhere)

Three quantizers into the SAME counter format (per-row scale + ternary, counter_packed),
warm-start PPL on the held-out domains, Qwen2.5-1.5B, calib = 64x8x512 from the mix corpus.
Whole witness ran in 150s on the G4-class GPU.

| variant | EN | RU | code | math |
|---|---:|---:|---:|---:|
| fp teacher | 11.6 | 9.2 | 3.0 | 6.7 |
| naive TWN thr=0.5 (old default) | 575,347 | 20,672,417 | 454,477 | 369,272 |
| optimal ternary (exact per-row L2) | 187,431 | 2,211,696 | 164,113 | 178,622 |
| GPTQ (Hessian error feedback, act-order) | **17,553** | **11,971** | **35,064** | **22,812** |

- optimal vs naive: 2.1-9.3x better -- matches the frozen "3-10x, same order" call exactly.
- GPTQ vs naive: 33x (EN), 1727x (RU), 13x (code), 16x (math). Calibration is worth orders
  of magnitude, with the biggest wins where naive was worst.
- Forecast scoring (owned): predicted GPTQ warm PPL "~500-5000"; actual 12k-35k. Direction
  right, magnitude OVERSHOT for the THIRD consecutive time. RU crossed the naive-vs-optimal
  gap by 3 orders, but absolute quality is still far from usable.

**Verdict on the "no-retrain PTQ is enough" hypothesis (Bonsai claim, our scale/format):**
REFUTED at 1.5B with per-row scales -- 12k-35k PPL is still a broken model. Whatever carries
Bonsai's ~90%-retention-without-retraining at 27B (group-128 scale granularity = 12-70x finer
than our per-row, 27B redundancy, or undisclosed extras), it does NOT transfer down to this
regime. Recovery training remains essential here. The actionable win: the recovery now starts
from 17k instead of 575k -- PTQ + recovery is the combined recipe going forward.

Next measurement: short recovery (e.g. 2000 steps) from the GPTQ start vs the recorded naive
curve (naive reached EN 109 @ step 2000) -- does the calibrated start reach a lower floor or
the same floor faster?
