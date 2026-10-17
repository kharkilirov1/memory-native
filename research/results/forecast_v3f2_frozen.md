# FROZEN pre-run forecast — run v3f2 (first run with a LIVE counter channel)

Written and committed BEFORE launch. Same config as v3f (solver, corpus, schedule,
6000 steps, G4) + the train-mode fix: counters and group scales update for the first
time. Calibration note: my absolute-PPL forecasts have a documented optimism bias
(4 consecutive overshoots earlier in the project); ranges below are deliberately wide
and the MECHANICS predictions are the ones I stake more on.

## Gate 0 — five-minute sanity (must-pass, not a forecast)

Warm strict alpha=0 must reproduce v3f: metric 3.770±0.005, ppl_en 74.5±0.5
(same seed, deterministic solver). Mismatch = the fix moved something it should not
have; stop and diagnose before burning the run.

## Mechanics (primary predictions — falsifiable)

- sr in the step log equals the step count from step 100 onward (guard enforces > 0).
- edge STARTS BREATHING: |edge − 0.0210| ≥ 0.001 by step ~1000 (in v3f it was frozen
  to 4 decimals for 6000 steps — that was the smoking gun).
- flip_alt in the soft phase (alpha≈1, steps ≤1200): 1e-4…3e-3 per 100-step window
  (sub-quantum drift + SR at the |c|≈C−1 fringe, ~2% of weights).
- flip_alt PEAKS during the anneal tail (steps ~3600–5400, alpha→0): 1e-3…3e-2 —
  the legalization pressure window; then decays with the cosine LR.
- Scales move: scale_mean drifts from the solver value (0.0558 on layer-0 q_proj)
  by ≥1% by mid-run.

## Outcomes (three worlds, per the reviewer's framing)

- World 1 — counters help: final strict ppl_en 32…44, arc_challenge retention pulls
  above v3f's 53%. My weight: moderate; the clip keeps per-element ticks small, so I
  do NOT expect a collapse of the gap to the teacher in 6000 steps.
- World 2 — small gain: final ppl_en 44…50 (within noise of 47.4). Then
  "solver skeleton + scales/tail" becomes a MEASURED characterization of the method
  at this LR, partially rehabilitating the retracted reading — as fact, not
  rationalization.
- World 3 — degradation/instability: final ppl_en > 50, or oscillating evals, or
  divergence in the anneal tail (ternary boundary oscillation is a real literature
  failure mode; the schedule was tuned — involuntarily — for the fp-tail-only system).
  Response if seen: not "method broken" but "re-tune lr/clip for the full system";
  best checkpoint remains protected by strict-metric selection.

I refuse to put sharp probabilities on 1 vs 2 vs 3; the honest statement is that
worlds 2 and 3 are jointly at least as likely as world 1 at THIS schedule, because
the clip math keeps counter ticks at ~0.01–0.02 quanta/step — the channel is live
but slow at lr 0.002.
