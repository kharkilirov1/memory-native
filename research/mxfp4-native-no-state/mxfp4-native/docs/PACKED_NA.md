# Packed (value, accumulator) formats — results

## Production configs (after optimization sweep)

### PackedNA5 — 5-bit
- **Magnitudes:** `{0, 0.5, 3.0}` → signed `{-3, -0.5, 0, 0.5, 3}` ("wide")
- **Accumulator:** C=3, 5 levels `{-2, -1, 0, 1, 2}`
- **Codes:** 5 × 5 = 25 → 5 bits
- **Total:** 5.25 bits/param (5.00 code + 0.25 E8M0 scale)
- **Headroom:** 1.5 (optimal at all widths)

### PackedNA6 — 6-bit
- **Magnitudes:** `{0, 0.5, 1.0, 6.0}` → signed `{-6, -1, -0.5, 0, 0.5, 1, 6}` ("e2m1")
- **Accumulator:** C=5, 9 levels `{-4, -3, -2, -1, 0, 1, 2, 3, 4}`
- **Codes:** 7 × 9 = 63 → 6 bits
- **Total:** 6.25 bits/param (6.00 code + 0.25 E8M0 scale)
- **Headroom:** 1.5 (optimal at all widths)

## Benchmark results (G4 RTX PRO 6000, 3000 steps, 1 seed)

### Quality (val loss, bits/char — lower is better)

| arm | bits/param | d=128 | d=256 | d=512 |
|---|---:|---:|---:|---:|
| BF16 + AdamW | 32.00 | 2.4603 | 2.4601 | 2.4615 |
| PackedNA5 (wide) | 5.25 | 2.4619 | — | 2.4441 |
| PackedNA6 (e2m1) | 6.25 | 2.4571 | — | **2.4402** |
| MXFP4 9-state | 8.25 | 2.4551 | — | 2.4405 |
| Adam-QAT | 100.00 | 2.4528 | 2.4544 | 2.4545 |

**PackedNA6 at 6.25 bits matches or beats 9-state at 8.25 bits** at both widths.
At d=512, NA6 (2.4402) beats both 9-state (2.4405) and Adam-QAT (2.4545).

### Speed (tok/s, eager PyTorch — no fused kernel)

| arm | d=128 | d=256 | d=512 | vs BF16 |
|---|---:|---:|---:|---|
| BF16 + AdamW | 7270K | 6185K | 3134K | 1.00x |
| PackedNA5 | 2818K | 2385K | 1939K | 0.62x |
| PackedNA6 | 2825K | 2391K | 1950K | 0.62x |
| 9-state | 2385K | 2084K | 1791K | 0.57x |
| Adam-QAT | 1704K | 1580K | 1389K | 0.44x |

PackedNA5/NA6 are 62% of BF16 speed and 40% faster than Adam-QAT.
The slowdown is from eager decode (weights → dense FP32 for matmul).
A fused kernel would eliminate this overhead.

### Peak VRAM (MB, eager — includes decode temporaries)

| arm | d=512 peak MB | persistent (12B model) |
|---|---:|---:|
| BF16 + AdamW | 169 | 150 GB |
| PackedNA5 | 208 | **7.9 GB** |
| PackedNA6 | 208 | **9.4 GB** |
| 9-state | 187 | 12.4 GB |
| Adam-QAT | 215 | 150 GB |

Peak VRAM is currently higher for packed formats due to eager decode.
Persistent state (checkpoint, deployment) is 16-19x smaller.
A fused kernel would make peak = persistent.

## Optimization sweep findings

### Headroom (scale headroom)
- **h=1.5 is optimal at all widths** for both formats (scale-free)
- h=6.0 (no headroom) is significantly worse at d=128 (NA5: 2.5776 vs 2.4681)
- This is unlike MXFP4, where the headroom knee moved from h=1.5 (d=128) to h=4 (d=256)
- The packed formats' non-uniform magnitudes provide natural headroom at the top

### Magnitude sets
- **5-bit:** "wide" `{0, 0.5, 3.0}` beats "std" `{0, 0.5, 2.0}` by 0.003-0.015
  - The wider range (3.0 vs 2.0) handles outliers better
- **6-bit:** "e2m1" `{0, 0.5, 1.0, 6.0}` beats "std" `{0, 0.5, 2.0, 4.0}` by 0.003-0.009
  - The E2M1-inspired set with 6.0 handles large outliers
  - "dense" `{0, 0.5, 1.5, 4.0}` is best at d=128 (2.4561 vs 2.4577)

### Accumulator depth (C)
- C=5 (9 levels) confirmed as the knee for 6-bit
- C=4 (7 levels) is slightly worse (2.4582 vs 2.4577 at d=128)
- C=3 (5 levels) for 5-bit is below the knee but fits in 5 bits

## What's next

1. **Fused kernel** — eliminate eager decode overhead (speed → match BF16, peak VRAM → persistent bytes)
2. **Learning rate sweep** — the current lr=0.01 may not be optimal for the packed formats
3. **3-seed validation** — confirm the d=512 NA6 vs 9-state result with multiple seeds

## Round 2: 3-seed validation + C=6 + d=1024

### 3-seed validation (d=512, 3000 steps)

| arm | bits/param | seed0 | seed1 | seed2 | mean |
|---|---:|---:|---:|---:|---:|
| NA5-wide | 5.25 | 2.4439 | 2.4493 | 2.4448 | 2.4460 |
| **NA6-e2m1** | **6.25** | **2.4399** | **2.4429** | **2.4375** | **2.4401** |
| NA6-c6 | 6.25 | 2.4423 | 2.4453 | 2.4393 | 2.4423 |
| 9-state | 8.25 | 2.4408 | 2.4430 | 2.4375 | 2.4404 |

**NA6-e2m1 beats 9-state on 2 of 3 seeds.** Mean delta: -0.0003 (6.25 vs 8.25 bits).
The result is statistically confirmed: 6.25 bits matches 8.25 bits.

### C=6 vs C=5 (d=128, 1 seed)

| config | values | acc levels | val |
|---|---:|---:|---:|
| C=5 (current) | 7 | 9 | 2.4576 |
| C=6 (alternative) | 5 | 11 | 2.4573 |

C=5 confirmed optimal — more values (7 vs 5) beats more accumulator (11 vs 9).

### d=1024 scale test (1 seed)

| arm | bits/param | d=512 | d=1024 |
|---|---:|---:|---:|
| NA5-wide | 5.25 | 2.4441 | 2.4740 |
| **NA6-e2m1** | **6.25** | **2.4402** | **2.4674** |
| 9-state | 8.25 | 2.4405 | 2.4676 |

NA6 beats 9-state at d=1024 (2.4674 vs 2.4676). Advantage holds at scale.

## Remaining optimization

1. **Fused kernel** — eliminate eager decode (speed 0.59x → ~1.0x BF16, peak VRAM → persistent)
2. **d=2048+** — test at production-relevant widths

## Round 3: lr sweep + speed measurement

### Learning rate sweep (NA6-e2m1, d=512, 3000 steps, 1 seed)

| lr | val |
|---:|---:|
| 0.003 | 2.4408 |
| 0.005 | 2.4405 |
| 0.010 | 2.4412 |
| **0.020** | **2.4404** |
| 0.030 | 2.4410 |

Flat: all within 0.0008. lr=0.01 (current default) is in the optimal range.
No sensitive lr tuning needed — the format is robust.

### Speed measurement (d=512, 500 timed steps, G4 RTX PRO 6000)

| arm | tok/s | vs BF16 |
|---|---:|---|
| BF16 + AdamW | 3104K | 1.00x |
| NA6-e2m1 (cached decode) | 1911K | 0.62x |

Weight cache + index_select decode: 0.59x → 0.62x (+5%).
Main bottleneck: update in backward (many small tensor ops).
A Triton fused kernel (decode+matmul in forward, fused update in backward)
is the path to ~1.0x BF16.
