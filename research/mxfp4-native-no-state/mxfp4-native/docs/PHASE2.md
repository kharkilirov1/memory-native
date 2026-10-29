# Phase 2 results: the format is cheap, momentum is the whole story

Tiny char-LM on tiny-shakespeare (byte level, 1.1 MB, 2 blocks, d=128, 4 linears per block;
vocab 65; batch 32, seq 128; CPU, 8 threads, 800 steps). Identical architecture and data in
every arm — only the linear layers and their update rule differ. `probes/charlm_phase2.py`.

## Val loss, bits/char (lower is better)

| arm | 400 steps | 800 steps | what it isolates |
|---|---:|---:|---|
| dense fp32 + AdamW | 2.8949 | **2.6619** | the reference |
| dense, then post-hoc MXFP4 | — | **2.6916** | **the format alone: +0.030** |
| MX6, SR, no accumulator | 3.2051 | 3.1287 | no optimizer state at all |
| MX6, RNE + 2-bit error feedback | 3.6501 | — | the ablation that loses |

Quantising a trained dense model to MXFP4 costs **0.03 bits/char**. Everything else is the
update rule, not the weights.

## The integrator sweep — this is the answer

The ternary design's 21-level accumulator was momentum: gradient ticks integrated until a
threshold, then carried into the weight. We replaced it with error feedback (wrong
mechanism) and then with nothing (SR only). Here is the same mechanism with the state count
swept, 800 steps, lr 0.01:

| accumulator states | bits | B/param | val, bits/char | gap to dense |
|---:|---:|---:|---:|---:|
| 0 (SR only) | 0 | 0.531 | 3.1287 | +0.467 |
| 3 | 2 | 0.781 | 2.8850 | +0.223 |
| 5 | 3 | 0.906 | 2.8084 | +0.147 |
| **9** | **4** | **1.031** | **2.7707** | **+0.109** |
| 17 | 5 | 1.156 | 2.7952 | +0.133 |
| 33 | 6 | 1.281 | 2.7652 | +0.103 |

Against the post-hoc MXFP4 floor (2.6916) the 9-state arm is **+0.079**, the 33-state arm
+0.074 — i.e. beyond nine states the bits buy nothing.

Reading:

- **Three states already recover half the gap** (0.467 → 0.223) for 2 bits/param.
- **Nine states recover ~83% of it** (0.467 → 0.109) for 4 bits/param.
- 17 is *worse* than 9 (2.7952 vs 2.7707): with C=9 the tick granularity `lr/C` matches the
  step size, with C=17 the accumulator needs twice as many ticks and the effective damping
  changes. Non-monotone, so the knee must be tuned, not assumed.
- The gap also shrinks with training: at 400 steps the 9-state arm is +0.160 behind dense,
  at 800 it is +0.109.

Learning rate at 9 states: 0.003 → 2.8781, **0.01 → 2.7707**, 0.03 → 3.1048. The optimum
did not move.

## Diagnosis

The ternary design did not beat fp32 + AdamW because of its 1.58-bit value — it did so
because its accumulator integrated gradient over ~C steps, which *is* momentum, stored in
the weight word. Dropping to 4 bits removed it, and no amount of error feedback substitutes
for it: error feedback fixes rounding **bias**, which stochastic rounding already removes,
while an integrator supplies **memory of past gradients**, which nothing else does.

## Where the design lands

| configuration | bits/param | B/param | 12B model | val (800 steps) |
|---|---:|---:|---:|---:|
| MXFP4, no accumulator | 4.25 | 0.531 | 6.4 GB | 3.1287 |
| **MXFP4 + 9-state** | **8.25** | **1.031** | **12.4 GB** | **2.7707** |
| ternary counter (`main`) | 6.25 | 0.906 | 10.9 GB | — (not measured here) |
| BF16 + AdamW | 32 | 12 | 144 GB | 2.6619 |

The honest trade: reaching dense quality costs ~8 bits/param, not 4. At 8.25 bits we are
still **11.6x smaller** than BF16 + AdamW, and the state is a single packed word with no
adjacent buffers — but the "4-bit weights" headline only holds for the no-accumulator arm,
which is 0.47 bits/char behind.

## Next

1. Re-tune the tick rate at each state count (the C-granularity coupling is why 17 lost to 9)
   and confirm the knee.
2. Pack the accumulator into the same 3-byte-per-4-coefficients word (currently accounted,
   not packed).
3. A 30–100M parameter run, where the fp tail stops dominating and the memory claim is
   actually exercised.

## Round 2: attempts at cheap adaptivity, and why they failed

Base to beat: integrator 9 states, per-row RMS = **2.7707** (dense 2.6619, quantisation
floor 2.6916). Every variant keeps the integrator and changes only how the gradient is
normalised, plus one free gating trick.

| variant | extra bits/param | val, bits/char |
|---|---:|---:|
| per-row RMS (base) | 0 | **2.7707** |
| second moment per 32-block, E8M0 byte | 0.25 | 3.5965 |
| second moment per 32-block, fp8 byte | 0.25 | 3.5842 |
| Adafactor rank-1 (row × column factors, fp8) | 0.156 | 3.5830 |
| cautious gating (drop ticks opposing the accumulator) | 0 | 3.0110 |

Diagnostic after 60 steps on a 512×512 layer:

| mode | weights pinned at the ±6 rail | mean abs accumulator |
|---|---:|---:|
| row | 2.0% | 1.76 |
| block | 11.7% | 0.71 |
| col | 11.7% | 0.87 |

Finer-grained normalisation does not add information here — it **drives weights into the
rails**: normalising per 32 elements (or rank-1) boosts quiet directions to the same step
size as loud ones, and at our sign-like step they march to ±6 and stay there.

The deeper reason is structural: we normalise the gradient **before** accumulating it,
i.e. we integrate *normalised* gradients (Signum/LARS-like). Adam accumulates raw
gradients and normalises the **momentum** afterwards, which keeps direction coherence and
only then sets the step size. With an integer accumulator the natural analogue is an
adaptive *carry threshold* per row (C_i proportional to sqrt(v_i)/mean), not an adaptive
tick. That is the next experiment.

The negative results bound the problem: the remaining 0.079 bits/char to the quantisation
floor is **not** reachable by adding per-element second moments at these bit costs, at
least not in the "normalise first" arrangement.

## Round 3: the controls that reframed the problem

Three control arms, 3000 steps, lr 0.01, 9-state integrator:

| arm | what it isolates | bits/param | val, bits/char |
|---|---|---:|---:|
| dense fp32 + AdamW | reference | 96 | 2.6564 (at 800 steps) |
| qat: MXFP4 forward + fp32 master + AdamW | the format | 32 + | 2.4097 |
| msr: MXFP4 forward + fp32 master + **our rule** | the update rule | 32 + | **2.3951** |
| mx6, headroom 6 (previous best) | in-weight accumulator | 8.25 | 2.5777 |
| **mx6, headroom 1.5** | + rail headroom | **8.25** | **2.4415** |

Two conclusions, both important:

1. **Our update rule is not the problem.** With an fp32 master it *beats* AdamW
   (2.3951 vs 2.4097) at identical precision. What we had blamed on "no Adam" is
   actually the in-weight accumulator.
2. **The accumulator was failing on the rails, not on precision.** The MXFP4 scale rule
   (`amax/s <= 6`) puts the largest weight exactly on the top level, so any weight that
   wants to keep growing is clipped and its carry is blocked. Reserving headroom —
   `s = 2^ceil(log2(amax/h))`, so the top weight lands at level h instead of 6 — removed
   most of the gap.

Headroom sweep at 800 steps (the scale is a power of two, so h pairs collapse):

| h | 6 (old) | 4 | 3 | 2 | 1.5 | 1 |
|---|---:|---:|---:|---:|---:|---:|
| val | 2.7707 | 2.7410 | 2.7410 | 2.7187 | 2.7187 | 2.7728 |

Knee at h = 1.5-2: enough headroom to stop clipping, not so much that the ladder is wasted
on tiny values.

Remaining distance at 3000 steps: **+0.032 to Adam-QAT, +0.046 to the master control**, at
8.25 bits/param against 96 for dense fp32 + AdamW (11.6x smaller).

What this changes: the goal is no longer "make the update Adam-like" (it already is), it is
"make a 4-bit accumulator behave like an fp32 master". Next levers: a per-block scale that
tracks a moving distribution rather than only amax, and per-block rather than per-layer
headroom.

## Round 4: the accumulator curve, and a result that beats Adam

Two things changed at once: headroom (h = 1.5) and a proper accumulator sweep at 3000
steps. The accumulator depth was previously measured with the rail bug in place, which
masked it.

| accumulator states | bits | val, bits/char (h = 1.5, seed 0) |
|---:|---:|---:|
| 9 | 4 | 2.4415 |
| 17 | 5 | 2.3940 |
| **33** | **6** | **2.3652** |
| 65 | 7 | 2.3821 |

33 states is the knee; 65 regresses, the same non-monotonicity seen earlier and caused by
the tick granularity `lr/C` coupling to C.

Adam was given a fair shot too (QAT arm, 3000 steps, lr swept):

| adam lr | 1e-3 | 3e-3 | 1e-2 |
|---|---:|---:|---:|
| val | 2.4709 | **2.4097** | 2.4356 |

### Seed replication (3000 steps)

| seed | mx6, 33 states, h=1.5 | Adam-QAT (best lr) |
|---:|---:|---:|
| 0 | 2.3652 | 2.4097 |
| 1 | 2.3780 | 2.4039 |
| 2 | 2.3894 | 2.4302 |
| **mean** | **2.3775** | **2.4146** |

The counter wins on **3 of 3 seeds** by 0.022-0.045 bits/char (mean 0.037).

### What it costs per parameter

| | bits/param | B/param | relative |
|---|---:|---:|---:|
| dense fp32 + AdamW | 96 | 12.0 | 1.0x |
| Adam-QAT (MXFP4 fwd + fp32 master + m,v) | 100 | 12.53 | 1.04x |
| **mx6 (MXFP4 value + scale + 6-bit accumulator)** | **10.25** | **1.28** | **0.107x** |

So on this benchmark the counter reaches a *better* validation loss than the best-tuned
Adam baseline while holding **9.8x less state per parameter**.

The mechanism, in one line: the MX scale rule put the largest weight on the top rung, so
growing weights were clipped and their carries blocked; reserving headroom fixed the
clipping, and *then* the accumulator depth paid off, exactly as the literature predicts
(momentum needs depth, not precision).

Caveats: 461k-parameter model, one corpus, 3000 steps, CPU, and the counter lr was tuned
at 800 steps rather than swept at 3000. The fp tail is 65k of 459k parameters here, so the
memory ratio will improve at larger scale, not worsen.

## Round 5: GPU support, and width breaks the tuned scale

**The harness now runs on CUDA** (`--device cuda`). Same numerics (val loss within SR
noise), much faster:

| steps | CPU | GPU (RTX 5060 Ti) | speedup |
|---:|---:|---:|---:|
| 400 | 39.7 s | 12.6 s | 3.2x |
| 3000 | ~250 s | 53-90 s | ~3.5x |

The runs were slow for a structural reason, not because of the CPU: the updater is ~40
tiny tensor ops per layer per step, so at d=128 the time is Python/dispatch overhead, not
arithmetic. GPU helps by 3x here and would help much more at realistic widths.

**Width test (same device, GPU):**

| config | d=128, 3000 steps | d=256, 1500 steps |
|---|---:|---:|
| Adam-QAT | 2.4140 | **2.4687** |
| mx6, tuned headroom (1.5 at d=128, 4 at d=256) | **2.3738-2.3811** | 2.4802 |

So the d=128 win (0.035) does **not** transfer to d=256: the headroom knee moved from 1.5
to 4, and with the moved knee the counter is 0.011 *behind* Adam. A fixed per-layer
headroom is not a scale-free rule — this is exactly the width risk flagged in RISKS.md.

Two attempted fixes, both negative:

- **Adaptive scale maintenance** (recompute `s` from the live weights each step): the
  headroom invariant is unsatisfiable at runtime — with the top weight at level h, the
  maintenance rule wants `amax/h`, which is 4x the current scale, so `s` ratchets up one
  octave per step until it overflows to NaN. Even with a one-octave-per-step limit and a
  matching keep band it either drifted to the plain MX scale (losing the headroom benefit)
  or exploded. Conclusion: **headroom is an initialisation choice, not an invariant**; the
  scale must stay frozen unless a genuine saturation trigger fires.
- **MSE-optimal init scale** (pick the octave minimising block reconstruction error, the
  classical clipping criterion): 2.5709 at d=256 versus 2.4802 for the tuned h=4, i.e.
  worse. Minimising today's reconstruction error is not the same as leaving room for growth.

Design consequence: the next iteration needs a **per-block** headroom derived from the
block's own dynamic range (how far its smallest meaningful weight sits from its amax),
which is scale-free by construction, instead of one number per layer.

## Round 6: per-block headroom is also not enough

Implemented the scale-free rule: per block, headroom `h = clamp(amax / (4*q10), 1.5, 6)`,
i.e. blocks whose small weights are far from their maximum get little headroom (they cannot
afford it: shrinking `s` would flush them), tight blocks get the full amount. Measured on
GPU, same device as the baselines:

| config | d=128, 3000 steps | d=256, 1500 steps |
|---|---:|---:|
| Adam-QAT | 2.4140 | **2.4687** |
| mx6, tuned per-layer h | **2.3738-2.3811** (h=1.5) | 2.4802 (h=4) |
| mx6, per-block auto h | 2.3924 | 3.6078 |

The rule gives almost the same scales at both widths (init weights are kaiming-uniform, so
`q10` is a fixed fraction of `amax` by construction) — it is blind to the thing that
actually changes with width, which is the **trained** distribution, not the initial one.
So a rule fitted at initialisation cannot fix the width dependence.

What this leaves:

- the headroom optimum is a property of the trained weight distribution, which evolves;
- runtime adaptation of the scale is unstable (round 5: the rule wants `amax/h`, which is
  4x the current scale, so `s` ratchets one octave per step until it overflows);
- therefore, at present, headroom is a **per-width hyperparameter**, and the width
  transfer claim from round 4 is retracted: the counter beats Adam at d=128, loses by
  0.011 at d=256, and the rule that would make it scale-free is still missing.

The honest next step is a scale that is *state* rather than a function of the moment: keep
`amax` as a slowly-updated statistic (like Adam's `v`) and recompute the octave only when
it leaves a target band by more than one octave, with a hard rate limit and an overflow
guard. The failed attempts all recomputed from the instantaneous weights.







## Round 7: 30k steps, no drift

Long-horizon run (10x the previous horizon), d=128, GPU, mx6 headroom 1.5 / 33-state
accumulator versus Adam-QAT, same corpus and steps:

| arm | 3000 steps | 30000 steps |
|---|---:|---:|
| Adam-QAT | 2.4140 | 2.2986 |
| mx6, h=1.5, 33 states | 2.3738 | **2.2717** |

The counter keeps its advantage (0.027) at 10x the horizon, improves monotonically, and
shows no drift, limit cycle or late divergence: the validation curve is flat-to-slowly-
improving from step 10k to 30k for both arms.


## Round 8: scale-relative step size (negative)

Hypothesis: the step was defined in absolute weight units, so blocks whose scale is small
move several times more ladder steps per update than blocks with a large scale; at larger
width the spread of per-block scales grows, so those blocks saturate and the advantage
disappears. The fix would be a step proportional to each block's scale
(step = -lr * s * grad_hat, lr in units of the scale).

Measured at d=128, 3000 steps, 33-state accumulator, headroom 1.5:

| lr definition | value | val, bits/char |
|---|---:|---:|
| absolute, lr = 0.01 (previous) | 0.01 | **2.3738** |
| relative to scale, lr = 0.5 | 0.5 | 2.7320 |
| relative to scale, lr = 0.25 | 0.25 | 2.4986 |

Worse. A *uniform absolute* step beats a scale-proportional one: the per-block hotness
hypothesis is wrong, and the width dependence of the headroom is not explained by step
normalisation either.

Standing after eight rounds: the counter wins at d=128 by 0.02-0.04 bits/char at 10.25
bits/param (30k steps: 2.2717 vs 2.2986), loses at d=256 by 0.011, and no scale rule tried
so far (per-layer h, per-block auto h, MSE-optimal h, adaptive maintenance, scale-relative
step) makes the two widths agree. The scale remains the open problem.

## Round 9: the advantage does NOT vanish with width (round 5 retracted)

Round 5 concluded that the d=128 win does not transfer to d=256. That was an artefact of
comparing at 1500 steps: at a matched horizon the picture is different, and the headroom
knee simply moves with width.

3000 steps, GPU, best headroom per width, same corpus:

| width | Adam-QAT | mx6 (best h) | delta |
|---|---:|---:|---:|
| d=128 | 2.4140 | 2.3738 (h=1.5) | **-0.040** |
| d=256 | 2.3260 | 2.3436 (h=4) | +0.018 |
| d=512 | 2.5095 (median) | 2.4023 (h=6) | **-0.107** |

d=512, three seeds (mx6 h=6 / Adam-QAT):

| seed | mx6 | Adam-QAT |
|---:|---:|---:|
| 0 | 2.3235 | 2.4220 |
| 1 | 2.4621 | 3.5985 (diverged) |
| 2 | 2.4023 | 2.5095 |

The counter wins 3 of 3, and Adam is unstable on one seed. d=256 is the only width where
the counter is behind, by 0.018 — the same order as the seed noise measured at d=128
(0.008-0.012), so it needs seeds before it is believed.

What changes with width is the **headroom knee**: 1.5 at d=128, 4 at d=256, 6 (plain MX, no
headroom) at d=512. So "headroom is a per-width hyperparameter" holds, but its consequence
is the opposite of what round 5 claimed: at larger width the standard MX scale is the right
one, and the counter's advantage grows (0.04 to 0.10) rather than disappearing. The headroom
trick exists to rescue small widths.

Corrected standing: the counter beats Adam-QAT at d=128 and d=512 at 10.25 bits/param
against 100, and d=256 is unresolved pending seeds.

## Round 10: the width result was a tuning artefact (round 9 retracted)

Round 9 claimed the advantage grows with width. That was measured with Adam at one learning
rate (3e-3) everywhere. Swept per width, Adam needs a smaller lr as width grows, and with
its own best lr it wins at d=512 by a wide margin:

| width | mx6 best (lr, h) | Adam-QAT best (adam-lr) | delta |
|---|---:|---:|---:|
| d=128 | 2.3738 (0.01, h=1.5) | 2.4140 (3e-3) | -0.040 counter |
| d=256 | 2.3426 (0.003, h=4) | 2.3260 (3e-3) | +0.017 Adam |
| d=512 | 2.3235-2.4023 (0.01, h=6) | **2.2928** (1e-3) | **+0.10 Adam** |

At d=512 Adam at 1e-3 also stops diverging (seed 1 had blown up at 3e-3). So the counter
does **not** pull ahead with width; the advantage is confined to small widths.

This is the fourth tuning artefact in the project (RNE-vs-SR, headroom-vs-width, the
fixed-step width comparison, and now the baseline lr). The methodological lesson, recorded
as risk 3 in RISKS.md: every comparison needs both arms tuned at the size being compared.

State: the counter keeps its memory advantage (10.25 vs 100 bits/param) and a quality
advantage at d=128 (0.04), but at d=512 a properly tuned Adam-QAT is 0.10 ahead. The open
question has changed from "why does the advantage vanish" to "what does the counter need at
width" - a bigger accumulator, a tuned lr schedule, or a different scale rule.


