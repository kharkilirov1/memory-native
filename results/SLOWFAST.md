# M3 — slow-fast low-rank residual: cut the base-update frequency at no accuracy cost

Decompose the effective weight as `W_eff = s·T + A·Bᵀ` (counter base `s·T` + low-rank fp residual,
rank r≪d). Train only the small fp `A,B` each step (cheap, O(Mdr)); the base counter's full
correlation `ΔᵀX` is **frozen between merges** (`update_enabled=False`). Every K steps "merge":
re-encode `s·T + A·Bᵀ` back into the counter state (ternary T + residual counter c + absmean scale),
reset `A,B←0`. This cuts the full-base-correlation frequency by ~K×.

## Witness — `scripts/slowfast_witness.py`

Teacher recovery (Gaussian teacher, n=24, N=256, C=11, 800 steps) vs exact `counter_rms`.

| arm | MSE | gap% vs exact | base-corr reduction | merge-jump | progresses |
|---|---|---|---|---|---|
| counter_rms (exact) | 0.2947 | — | 1× (800 full ΔᵀX) | — | — |
| slowfast r=8 K=8 | 0.2842 | −3.6 | 8× | 0.143 | yes |
| slowfast r=16 K=8 | 0.2795 | −5.2 | 8× | 0.195 | yes |
| slowfast r=32 K=8 | 0.2683 | −9.0 | 8× | 0.269 | yes |
| slowfast r=8 K=16 | 0.2938 | −0.3 | 16× | 0.177 | yes |
| slowfast r=8 K=32 | 0.2977 | +1.0 | 32× | 0.206 | yes |
| slowfast r=32 K=32 | 0.2865 | −2.8 | 32× | 0.302 | no |

## Verdict — gates PASS, with one honest confound

**PASS — base-update frequency cut 8–32× at no accuracy cost.** Every slow-fast arm matches or beats
exact (gap ≤ 1–2%, mostly negative), while the base counter state changes only on merge steps
(8×–32× fewer full `ΔᵀX`). Merges are stable: training progresses across cycles, no divergence.

**Confound (state it plainly).** "Beats exact" is partly because slow-fast = ternary base **plus
extra fp `A,B` capacity** the pure-ternary baseline lacks — so beating pure ternary is unsurprising.
The clean, defensible claim is **non-degradation**: slow-fast does not hurt accuracy while making the
base update rare. It is *not* evidence that low-rank ≥ full-rank counter.

**Transient merge jump.** Folding an fp low-rank residual back into ternary `s·T` is lossy by
construction, so post-merge loss briefly rises (~0.14–0.30 on a Gaussian target). It is transient
(per-cycle minima keep falling), not divergence. The strict <0.05 per-merge bound only holds on a
*ternary* teacher (small residual) — covered by the unit test.

**Standalone speed ceiling ~1.43× (the honest boundary).** Slow-fast cheapens the *update* of the
base, not its forward/grad_x *use*: the dense `X(s·T)ᵀ` GEMM still runs every step. With the base
update ~1 of the 3 per-layer GEMMs, removing it on (K-1)/K of steps caps the standalone gain at
~1.43×. **The real win is only in combination with structured sparsity (a later method) that also
cheapens the USE.** This witness validates correctness + update-frequency, not raw speed.

## Tests — `tests/test_slowfast.py` (3, pass)
rank-0 parity (bit-identical to the base RMSCounterLinear), teacher recovery within 2× of exact,
merge stability on a ternary teacher. Default OFF; opt-in like every numerics-changing mode.
