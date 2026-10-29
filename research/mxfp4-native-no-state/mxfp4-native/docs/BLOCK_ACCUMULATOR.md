# Block-Shared Accumulator: 11-bit momentum at 4.50 bits/param

## Problem

MXFP4 values (E2M1, 4 bits) + E8M0 scale (0.25 bits) = 4.25 bits/param with **no
momentum**. Adding a per-parameter accumulator costs ≥2 bits (3 states → 6.25 total).
The PHASE2 sweep showed 9 accumulator states is the knee (83% of the quality gap
closed), but 9 states = 4 bits per-param → 8.25 bits/param total.

## Idea

The E8M0 scale byte (8 bits per 32-param block) uses only ~5 bits of exponent in
practice (weight scales span ~10 octaves, not 128). The other 3 bits are wasted.

**Split the scale byte** into 5-bit exponent + 3-bit accumulator-high. Add one
more byte per block for accumulator-low. The combined 11-bit block accumulator
(2048 levels) is shared across all 32 weights in the block.

The accumulator is **unsigned** and tracks the average gradient *magnitude* of the
block. When it crosses a threshold, every weight in the block steps one rung on
the E2M1 ladder — each in the direction of its own gradient sign. The block
decides **when** to step; each weight decides **where**.

## Bit budget

```
E2M1 values:        4.00 bits/param   (16 values, nibble-packed)
Scale+acc-high:     0.25 bits/param   (1 byte / 32, split: 5-bit exp | 3-bit acc)
Acc-low:            0.25 bits/param   (1 byte / 32, new)
─────────────────────────────────────────────────
Total:              4.50 bits/param
```

For comparison:

| configuration | bits/param | accumulator | levels |
|---|---:|---|---:|
| MXFP4, SR only | 4.25 | none | 0 |
| **MXFP4 + block-acc** | **4.50** | **11-bit block** | **2048** |
| MXFP4 + 3-state per-param | 6.25 | 2-bit per-param | 3 |
| MXFP4 + 9-state per-param | 8.25 | 4-bit per-param | 9 |
| ternary C=11 (main) | 6.25 | 6-bit per-param | 63 |

2048 block levels is far past the 9-state knee. The 3 stolen bits cost nothing;
the extra byte costs 0.25 bits/param.

## Byte layout

### Scale+acc-high byte (1 byte per 32-param block)

```
bit  7  6  5  4  3 | 2  1  0
     [  exponent  ] | [acc-hi]
      5 bits (0-31)   3 bits (0-7)

scale = 2 ** (exponent - 15)     # bias 15 → range 2^-15 .. 2^16
```

### Acc-low byte (1 byte per 32-param block)

```
bit  7..0
[  acc-low ]
  8 bits (0-255)
```

### Combined block accumulator (11 bits, unsigned)

```
acc = (acc_high << 8) | acc_low     # 0 .. 2047
```

The accumulator is unsigned (magnitude only). Direction comes from per-weight
gradient signs, not from the block accumulator.

## Update rule

```python
# 1. Per-weight gradient tick (same as per-param integrator)
tick_i = -lr * grad_hat_i * C_block / gap_i     # per-weight, signed

# 2. Block accumulator integrates average |tick|
avg_abs_tick = mean(|tick_i| for i in 32)
acc += stochastic_round(avg_abs_tick)

# 3. Carry: when pressure crosses threshold, all weights step
while acc >= C_block:
    acc -= C_block
    for i in range(32):
        direction = sign(tick_i)
        if direction != 0:
            code_i = ladder_step(code_i, direction)

# 4. Write back
acc_high = (acc >> 8) & 0x07
acc_low  = acc & 0xFF
```

`C_block` is the carry threshold (default 32, so ~32 average-magnitude ticks
before the block steps). `gap_i` is the local E2M1 ladder gap for weight `i`,
same as the per-param integrator's `delta()`.

### Why unsigned + per-weight direction

A signed block accumulator would force all weights to step in the *average*
direction, pushing weights with opposite gradients the wrong way. An unsigned
accumulator only decides **timing** ("enough pressure has accumulated, step
now"); each weight's direction comes from its own gradient sign. This preserves
per-weight direction coherence while sharing the momentum state.

### What is lost vs per-param

Weights with small gradients step just as often as weights with large gradients
in the same block. Per-row RMS normalization (already in the code as
`rms_update`) reduces the within-block spread, so the loss is bounded. The
PHASE2 data show that even 3 states (very coarse) recover 50% of the gap; 2048
block levels should be well past the knee.

## Scale range check

5-bit exponent with bias 15 gives range 2^(-15) to 2^16 (3e-5 to 65536).
Measured weight scales in the project span exponent −14 to +3 (amax 0.0001 to
10, headroom 1.5–6). The 5-bit field (exponent −15 to +16) covers this with
margin. If a block's scale would exceed the 5-bit range, it clamps and the
headroom invariant degrades gracefully (same as E8M0 clamping at 0/254).

## Forward pass

The forward matmul reads only the 5-bit exponent (bits 3–7 of the scale byte).
The 3 acc-high bits (bits 0–2) are masked off and invisible to the forward:

```python
exp = scale_byte >> 3           # 5 bits
s = 2.0 ** (exp - 15)           # float scale
weight = s * value_floats(codes)  # 16 E2M1 values
```

## Persistent state per layer

For a 4096 × 4096 layer (16.8M params, 131072 blocks):

| plane | bytes | bits/param |
|---|---:|---:|
| E2M1 codes (nibble-packed) | 8388608 | 4.00 |
| Scale+acc-high (1 byte/block) | 131072 | 0.25 |
| Acc-low (1 byte/block) | 131072 | 0.25 |
| **Total** | **8650752** | **4.50** |

Per-row RMS (4 bytes per output row) adds 16384 bytes (0.008 bits/param),
negligible.

## Deployed (inference) state

The accumulator is training-only state. At deployment, the acc-low byte is
dropped and the scale byte's acc-high bits are zeroed:

| plane | bytes | bits/param |
|---|---:|---:|
| E2M1 codes | 8388608 | 4.00 |
| Scale (5-bit exp, zeroed acc) | 131072 | 0.25 |
| **Total** | **8519680** | **4.25** |

The deployed format is standard MXFP4 (4.25 bits/param), identical to the
no-accumulator arm.

## Parameters

| parameter | default | meaning |
|---|---|---|
| `C_block` | 32 | carry threshold: ~32 avg-magnitude ticks per step |
| `scale_bias` | 15 | exponent bias for 5-bit scale field |
| `headroom` | 1.5 | same as per-param integrator (PHASE2 knee) |
| `rms_beta` | 0.9 | per-row second moment EMA decay |
| `rms_eps` | 1e-3 | per-row RMS denominator floor |

## What this doc does not claim

- The 2048 block levels are not 2048 per-weight states. The effective per-weight
  momentum depth depends on the within-block gradient spread, which per-row RMS
  bounds but does not eliminate. The smoke test measures the actual val loss.
- The 5-bit scale loses dynamic range vs 8-bit E8M0. The range check above shows
  it covers the measured distribution, but a width sweep (d=256, d=512) is needed
  before claiming scale-free behaviour.
- This is a design doc + reference implementation. The fused kernel and packed
  training layout are future work; the smoke test runs the eager PyTorch path.
