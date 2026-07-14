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
