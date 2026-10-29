# Format: MXFP4 values + 2-bit error, one E8M0 scale per 32

Working name `mx6`. Everything here is normative: the encoder, the decoder, the accounting
and the deployment artifact are all defined by this page.

## Element grid

MXFP4 = OCP microscaling FP4. An element is **E2M1**:

```
code 0..7  ->  +0, +0.5, +1, +1.5, +2, +3, +4, +6
code 8..15 ->  the same magnitudes negated
```

Eight magnitudes, non-uniform steps (0.5 between 0 and 2, then 1, then 2). Range per block
is set by one shared scale.

## Scale

One **E8M0** byte per block of 32 consecutive elements along the input dimension:

```
s = 2^(byte - 127)          byte 127 == 1.0
```

Choice rule (deterministic, no extra state):

```
byte = ceil(log2(amax / 6)) + 127        # largest power of two with amax/s <= 6
```

`amax` is the block maximum of `|w|` **before** quantization. Because `s` is a power of two,
recomputing it later is exact and lossless to renormalise (see UPDATE.md).

## The 2-bit residual

Two spare bits per element hold the quantization error from the previous step, in units of
the **local gap** to the neighbouring level (so it stays meaningful across the whole
non-uniform ladder and needs no second scale):

```
Delta(v)  = distance from level v to its neighbour toward zero-ward/far side (table)
kappa     in {-0.375, -0.125, +0.125, +0.375}      # 2-bit mid-rise, 4 levels
e         = kappa * Delta(v)
```

The latent value the optimiser acts on is `u = s * (v + kappa)`, but **the forward pass uses
`v` only** (strict readout). The residual is pressure, never a visible weight — exactly the
role the counter `c` played in the ternary design.

## Bit layout of one 32-element block

25 bytes per 32 parameters = **0.78125 B/param**:

| bytes | content |
|---:|---|
| 16 | 32 E2M1 nibbles, element `2i` in the low nibble of byte `i`, element `2i+1` in the high nibble |
| 8 | 32 residual codes, 4 per byte, element `i` in bits `2*(i%4)` of byte `i//4` |
| 1 | E8M0 block scale |

Encoding/decoding is pure bit manipulation; no mask, no sentinel, no padding. A layer whose
`in_features % 32 != 0` pads the last block (documented, not silent).

The layout deliberately keeps the **value nibbles contiguous** so that a matmul kernel can
consume them exactly as it consumes vanilla MXFP4 — the residual plane is a separate 8-byte
stride that no GEMM touches.

## Accounting

| artifact | bits/param | B/param | 12B model |
|---|---:|---:|---:|
| values only (deployed) | 4.25 | 0.531 | 6.4 GB |
| **values + residual (training)** | **6.25** | **0.781** | **9.4 GB** |
| per-row RMS `v` | 0.021 (4 B / 1536 rows) | 0.003 | 0.03 GB |
| **training total** | **6.27** | **0.784** | **9.4 GB** |

For comparison, the ternary project on `main` is 0.906 B/param persistent / 1.002 resident,
because it carries a salient channel (0.120), its runtime mirror (0.080), an act-order
permutation and flip telemetry. None of those exist here: MXFP4's block scale already
handles outliers, and there is no converter to compensate for.

Deployment drops the residual plane: the artifact is standard MXFP4 and can be served by
any MXFP4-capable runtime (`scaled_mm`, torchao, gpt-oss-style kernels).

## What the format costs in accuracy

Measured with `probes/` on a Gaussian weight matrix (std 0.02), best per-format scale:

| format | B/param | relative reconstruction error |
|---|---:|---:|
| ternary counter (main, 3x21) | 0.906 | 19.2% |
| MXFP4, values only | 0.531 | 1.41% |
| MXFP4 + 2-bit residual | 0.781 | ~1.41% (the residual does not change the grid) |

The residual does not improve *representation*; it improves *training* by carrying error
forward (measured separately: 2.63% without feedback vs 1.41% floor, see UPDATE.md).

## Invariants the implementation must assert

1. `decode(encode(w))` is exactly the MXFP4 rounding of `w` (bit-identical to the reference
   `e2m1_round` + `scale` path).
2. `persistent_bytes() == ceil(N/32) * 25 + 4*out_features` for a layer of N parameters.
3. The residual plane never influences the forward pass, including under `torch.no_grad()`.
4. Renormalising the block scale (power-of-two change) leaves `v` and `kappa` unchanged and
   the latent `u` within one representable step.
