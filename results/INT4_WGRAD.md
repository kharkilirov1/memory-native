# int4 weight-gradient — compute the correlation, not the value

> **Strategic principle (T4, read this first).** On a T4 the GEMM is **no longer the bottleneck** —
> the bottleneck is *everything around the GEMM*: quantization, the epilogue, reversible-recompute,
> and the gather/scatter of the update. Once the forward and grad_x already run on int8 Tensor Cores,
> dropping the wgrad to int4 buys almost nothing, because the wgrad's cost is dominated by the
> quant + epilogue that int4 does **not** touch. So further speed engineering on T4 must aim at
> **data movement and not-GEMM ops**, not at lower GEMM precision. Every future "use fewer bits"
> idea must be Amdahl-checked against the measured step breakdown *before* it is built — the GEMM
> share is the only thing lower bits can shrink, and on T4 that share is already small.

This doc keeps **three things separate**, because they have different verdicts:

1. **int4-math** — the correlation `G = Delta^T X` carries enough signal at int4 to drive the same
   counter flips. **Verdict: VALIDATED.** An asset, banked for hardware that makes 4-bit cheap.
2. **int4-on-T4** — running that math faster than int8 on a T4. **Verdict: REJECTED by Amdahl** (the
   wgrad is quant-bound, not compute-bound; int4 shrinks the part that is already small).
3. **quant-as-the-new-bottleneck** — the open lever is **killing the quantize cost** by reusing a
   presaved int8 activation, *not* lowering GEMM bits.

---

## 1. int4-math — VALIDATED (the asset)

The counter flip needs only the **sign** of `G_oi = (Delta^T X)_oi` (which way to tick) and the
**in-row rank** of its magnitude (is it big enough to push a flip). The fp32 value of G is thrown
away on a binary decision. So the update correlation does not need an fp32 GEMM — an **int4** GEMM
(INT4 IMMA) carries enough of it. `update_compute="int4"` / `int4_correlation` do this.

This is specific to the counter: threshold + stochastic rounding + error-feedback **tolerate a
quantized signal** — a coarse but directionally-correct correlation drives the same flip trajectory,
and the accumulated error rescues the sub-threshold part. Dense SGD can't (it needs the value).

### Witness (CPU) — `scripts/int4_wgrad_witness.py`

Fidelity vs the exact fp32 G on correlated (activation-like) data:

| operator | sign agree | Spearman rank | top-0.5% overlap |
|---|---|---|---|
| **int4** | 0.90 | **0.948** | 0.61 |
| int8 | 0.99 | 1.000 | 0.97 |
| 1-bit (XNOR, sign only) | 0.71 | 0.620 | 0.12 |

**Teacher recovery (the decisive test) — final MSE:** `fp 0.00000 · int8 0.00000 · int4 0.00000`.

int4 reproduces ~95% of the row rank and **recovers the teacher to the same MSE as fp32**. The
static top-0.5% overlap (0.61) understates it — error-feedback accumulates the missed sub-threshold
signal over steps, so the *dynamic* training outcome is identical. **1-bit is too coarse** (rank
0.62, top-overlap 0.12) — fine only as a cheap candidate screen; int4 is the sweet spot. CPU-tested
in `test_int4_wgrad` (unbiased, recovers the teacher like fp, clearly beats 1-bit on rank).

**This math is banked.** It is the wgrad twin of the int8 forward and it composes with slow/fast
(int4 only on the rare base recompute), sparsity (int4 over live weights), and strided sketch (int4
over a token subsample). It is the right operator on hardware that makes 4-bit cheap — see §3.

---

## 2. int4-on-T4 — NOT HARDWARE-JUSTIFIED (rejected by Amdahl)

We probed the actual operators on a T4 (`mn-int4-bench`). The decisive measurement is *where the
update's time goes*, not the GEMM's peak rate.

```
Status: VALIDATED (math) / NOT HARDWARE-JUSTIFIED on T4.

T4 update breakdown (int8 path, K=2048):
  int8 _int_mm GEMM .................. ~1.8 ms   <- the only part int4 could shrink
  quantize + epilogue (per-col scale,
    error-feedback, pack/unpack) ...... ~3.7 ms   <- int4 does NOT touch this
  ------------------------------------------------
  total update ........................ ~5.6 ms

The bottleneck has MOVED: the update is quantize-bound, not compute-bound. Halving the
~1.8 ms GEMM with a hand-written INT4-IMMA kernel saves <1 ms — a ~5% step gain — while the
~3.7 ms of quant/epilogue is untouched. That gain does not justify a bespoke CUTLASS kernel,
and `import cutlass` is unavailable on Kaggle anyway (torch `_int_mm` is int8-only).

int4 becomes worth it only if EITHER:
  (a) the quant cost is first removed (so the GEMM is again the dominant term — see §3), OR
  (b) the hardware is NVFP4-class (Blackwell): fast 4-bit IMMA + hardware micro-block scale +
      stochastic round = cheap quant AND no fidelity tax. On T4, int4 also carries a fidelity
      tax (the per-block scale is software); NVFP4 removes that too.

Recommendation: stop at int8 on T4.
```

**Base caveat — state the honest denominator.** int8 vs **fp32** is ×4; int8 vs **fp16** is ×2. T4
training already runs in fp16, so the real, banked speedup of the int8 path over the training base
is **×2**, not ×4. Quote both so nobody double-counts the ×4.

---

## 3. quant-as-the-new-bottleneck — the open lever

The Amdahl breakdown points at the move that *does* pay: the ~3.7 ms is quantizing the activation
every step. But the int8 forward **already quantized that same activation** this step. Reusing it
(`int8_correlation_presaved`, `act_save_bits=8`) means the wgrad quantizes only `Delta` — the
activation's per-token scale is folded into `Delta` and the saved int8 `X` is matmul'd directly.
That removes most of the 3.7 ms *without changing GEMM precision*.

So the order of operations is:
1. **Now / T4:** cut the quant cost via presaved int8 activation. This is the open lever.
2. **Only after (1):** if the GEMM is again the dominant term, int4 returns to the table — but at
   that point you are likely on NVFP4-class hardware where it is free of both the quant cost and the
   fidelity tax.

## Step-speedup framework (kept, with the Amdahl caveat)

Each of the three GEMMs per counter layer can drop off fp: forward `X T^T` int8 ~×2.05 (measured),
grad_x `Delta T` int8 ~×2, wgrad `Delta^T X` int8-presaved ~×1.45–2.16. With the GEMM block a
fraction `f` of the step running `k`× faster, the step speedup is `1/(1 - f + f/k)`. **The lesson of
§2 is that `f` is what governs the ceiling, and on T4 `f` is small** once quant/epilogue/reversible
dominate — which is exactly why lower GEMM bits (int4) move the step so little, and why the next win
is shrinking the *non-GEMM* terms, not `k`.
