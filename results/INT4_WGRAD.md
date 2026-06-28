# int4 weight-gradient — compute the correlation, not the value

The counter flip needs only the **sign** of `G_oi = (Delta^T X)_oi` (which way to tick) and the
**in-row rank** of its magnitude (is it big enough to push a flip). The fp32 value of G is thrown
away on a binary decision. So the update correlation should not be an fp32 GEMM — it should be an
**int4** GEMM (INT4 IMMA, present on Turing/T4, ~4x fp16 / ~2x int8), a *different hardware
operator*, not a faster fp one. `update_compute="int4"` / `int4_correlation` do this.

This is specific to the counter: threshold + stochastic rounding + error-feedback **tolerate a
quantized signal** — a coarse but directionally-correct correlation drives the same flip trajectory
and the accumulated error rescues the sub-threshold part. dense SGD can't (it needs the value).

## Witness (CPU, before any CUDA kernel) — `scripts/int4_wgrad_witness.py`

Fidelity vs the exact fp32 G on correlated (activation-like) data:

| operator | sign agree | Spearman rank | top-0.5% overlap |
|---|---|---|---|
| **int4** | 0.90 | **0.948** | 0.61 |
| int8 | 0.99 | 1.000 | 0.97 |
| 1-bit (XNOR, sign only) | 0.71 | 0.620 | 0.12 |

**Teacher recovery (the decisive test) — final MSE:** `fp 0.00000 · int8 0.00000 · int4 0.00000`.

int4 reproduces ~95% of the row rank and, crucially, **recovers the teacher to the same MSE as
fp32**. The static top-0.5% overlap here is 0.61 (data-distribution dependent; more strongly
correlated activations push it higher), yet the *dynamic* outcome is identical — because error-
feedback accumulates the missed sub-threshold signal over steps. So the static metric understates
int4; the training witness is what matters, and it's perfect. **1-bit is too coarse** (rank 0.62,
top-overlap 0.12) — fine only as a cheap candidate screen, int4 is the sweet spot.

## Step-speedup framework

The three GEMMs per counter layer can each drop off fp: forward `X T^T` int8 ~x2.05 (measured),
grad_x `Delta T` int8 ~x2, **wgrad `Delta^T X` int4 ~x3-4** (INT4 IMMA ~2x the int8 presaved
x1.45-2.16). With the GEMM block a fraction `f` of the step running `k`x faster, the step speedup is
`1/(1 - f + f/k)`; at d=2048 the GEMMs dominate the non-reversible cost, so int8-forward + int4-wgrad
+ int8-grad_x make the GEMM block ~2.5-3x — the full-step ceiling then set by the reversible-recompute
share (cut with anchors).

## Status / honesty

CPU-validated (`test_int4_wgrad`): int4 correlation is unbiased, recovers the teacher like fp, and
clearly beats 1-bit on rank. The **int4 speedup needs a real INT4-IMMA kernel** (CUTLASS — torch's
`_int_mm` is int8 only); on CPU the int4 GEMM is emulated with int32 matmul (correctness, not speed).
This is the wgrad twin of the int8 forward, and it composes with slow/fast (int4 only on the rare
base recompute), sparsity (int4 over live weights), and strided sketch (int4 over a token subsample).
