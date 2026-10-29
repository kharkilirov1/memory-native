# No-extra-state quality experiments (2026-10-29)

Goal: improve the MXFP4-only (`integrator=0`) update without adding per-parameter
persistent optimizer state. All results here are exploratory tiny-char-LM measurements,
not evidence for 12B pretraining.

## Candidate A: stateless Muon-style polar direction

The probe applies five Newton–Schulz iterations to the current gradient matrix, with no
momentum buffer, then feeds that direction to the existing stochastic MXFP4 weight update.
This is deliberately called *stateless Muon-style*, not Muon: standard Muon uses a momentum
matrix, which is precisely the persistent state excluded by this experiment.

| mode | steps | seed | d / blocks | LR | validation bits/char |
|---|---:|---:|---:|---:|---:|
| independent SR baseline | 400 | 0 | 64 / 1 | 0.01 | 3.6270 |
| stateless Muon-style | 400 | 0 | 64 / 1 | 0.01 | 3.7169 |
| stateless Muon-style | 400 | 0 | 64 / 1 | 0.05 | 4.0327 |
| stateless Muon-style | 200 | 0 | 64 / 1 | 0.10 | 4.3154 |

The raw-gradient polar direction is worse at the tested settings. This does not rule out
standard Muon or a better-tuned stateless variant, but standard Muon requires momentum
state, and this first zero-state candidate should not replace the baseline.

## Candidate B: balanced dependent stochastic rounding (BSR)

Within each 32-value MX block, values are grouped by sign and adjacent E2M1 gap. For each
group, one uniform offset is applied to cumulative rounding probabilities. Every value
keeps the same stochastic-rounding marginal, while the group sum of rounding errors is
bounded by one local quantization gap. No persistent tensor or optimizer state is added.

For a value between adjacent levels l and h, let p=(x-l)/(h-l). The upper-level indicator
b_i is chosen by randomized cumulative rounding. It satisfies:

- E[b_i] = p_i, so E[q_i] = x_i (within the representable range);
- |sum_i (q_i - x_i)| < h-l for each sign/gap group.

This reduces blockwise aggregate rounding noise, not per-coordinate variance in general.

| seed | independent SR | balanced SR | delta (BSR − SR) |
|---:|---:|---:|---:|
| 0 | 3.6270 | 3.6267 | -0.0003 |
| 1 | 3.6199 | 3.6180 | -0.0019 |

Setup: `integrator=0`, `second_moment=layer`, `lr=0.01`, `headroom=1.5`,
400 steps, `d=64`, one block, CPU, 2 threads. Lower validation bits/char is better.
The observed mean improvement is only 0.0011 bits/char across two seeds and is too small
to claim a real quality gain; more seeds and longer runs are required. BSR adds no
persistent bytes, though temporary cumsum/grouping work may be slower than independent SR.

## Decision

- Do not adopt stateless Muon-style as the default: current evidence is negative.
- Keep BSR as an optional experiment, not as a claimed win.
- Do not assume published MXFP4 GEMM recipes transfer directly. This reference implementation
  forms `grad_w` in FP32 and quantizes the *weight update*, so benefits must be tested at
  that exact quantization site.

## Memory accounting caveat

The reference PyTorch path stores E2M1 codes in one byte each, so measured persistent bytes
are not the ideal packed MXFP4 footprint. The target format is about 4.25 bits/parameter
(4-bit E2M1 codes plus one E8M0 scale byte per 32 values), before metadata. BSR adds no
persistent bytes, but it does not fix runtime packing or full-size FP32 temporary tensors.
