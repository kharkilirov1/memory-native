# Update: stochastic rounding on the E2M1 ladder — and why there is no error buffer

The rule that survived measurement. It needs no master weights, no optimizer state and —
this is the part that changed — **no residual buffer**.

## Notation

```
v      E2M1 level of the visible weight (4 bits)      w = s * v
s      E8M0 block scale (power of two)
g      grad_w for this tile, fp32, produced by backward
v_rms  per-row second moment of g (4 B per output row, the only extra state)
```

## The step

```python
# 1. normalise the gradient (RMS per row)
v_rms = beta * v_rms + (1 - beta) * mean_row(g**2)
g_hat = g / sqrt(v_rms).clamp_min(eps)

# 2. propose, in weight units
u = w - lr * g_hat

# 3. quantise the proposal to the grid, unbiased
v_new = SR_e2m1(u / s)          # stochastic rounding on the 8 magnitudes
w_new = s * v_new
```

`SR_e2m1` picks the upper neighbour with probability equal to the relative distance on the
non-uniform ladder, so `E[w_new | u] = u`. The scale is recomputed from `u` with the sticky
rule in FORMAT.md (kept while `3s <= amax <= 6s`), so it cannot ratchet on small updates.

## The measured result: error feedback is unnecessary once rounding is unbiased

Four cells, same teacher, 600 steps, medians over 5 seeds. Relative MSE, lower is better;
the quantisation floor for this teacher is ~1.4%.

| rounding | error feedback | lr = 0.003 | lr = 0.01 |
|---|---|---:|---:|
| stochastic (SR) | no | **0.0279** | **0.0283** |
| stochastic (SR) | yes | 0.0295 | 0.0337 |
| deterministic (RNE) | no | **3.0350** | 3.0039 |
| deterministic (RNE) | yes | 0.0319 | 0.0332 |

Read the two rows that matter:

- **Deterministic rounding without feedback cannot train at all.** Updates smaller than
  half a grid step are rounded away and the loss never moves (3.0 = no better than
  predicting the mean). This is the regime error feedback exists for, and with feedback it
  recovers to 0.032.
- **Stochastic rounding already does that job.** It is unbiased, so there is no bias left
  for a residual to compensate, and adding one makes things *worse* (0.0279 → 0.0295 at
  small lr, 0.0283 → 0.0337 at the larger one) because the buffer turns unbiased noise into
  a persistent random walk.

This is also why ECO needs a momentum buffer: it applies a *deterministic* optimizer step to
quantised weights, so it must carry the residual. We round stochastically, so we do not.

Consequence: the 2-bit residual plane is **off by default** and kept only as an ablation
(`MX6Linear(feedback=True)`). The training format is the deployed format.

## What this costs

Dropping the residual removes 2 bits/param and the whole carry logic, but it also gives up
the one thing that could close the gap to the 1.4% quantisation floor: measured training
lands at 2.8%, i.e. the grid is not the limiter, the dynamics are. The remaining levers are
learning-rate schedule and the gradient estimator (Agrusa et al. report an unbiased
estimator with lower variance than SR, arXiv:2610.24343), both Phase 2.

## Training contract

- Exactly one forward per backward. The update runs inside the autograd Function's
  backward and mutates the packed state in place; there is no `optimizer.step()`.
- No `param.grad` is retained for counter layers: the gradient tile dies with the step,
  which removes the 4 B/param full-model gradient buffer.
- Bias-free layers; the ordinary fp tail keeps its own optimizer and must be counted
  separately in every report.
- The scale is sticky (see FORMAT.md), otherwise re-quantising an already-quantised block
  rewrites codes on a zero-gradient step — found by a test, not by inspection.

## What Phase 2 still has to show

| metric | why | target |
|---|---|---|
| loss vs BF16 at equal tokens | is the format free | within the NVFP4 literature's gap |
| loss vs RNE+feedback | is SR really the better mechanism end-to-end | should match or beat 0.032 |
| error vs `lr` as lr decays | ECO's `1/lr` failure mode, now without a buffer | stationary error must not blow up |
| long run (10k+ steps) | SR noise may need a schedule | no drift, no monotone weight walk |
