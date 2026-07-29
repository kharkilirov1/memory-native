# Group-local update + one-launch fused kernel — design & CPU witnesses

Date: 2026-07-29. Goal: **strict-class memory AND GEMM-class speed for the weight update**,
by changing a computation principle rather than optimizing the existing form.

## The principle change

The row-scope update (production today) keys the RMS denominator and the grad-norm clip to
the whole output row. That makes the tick of weight [o, p] depend on a full-row reduction
of the CURRENT gradient — the structural blocker for a one-pass tiled-GEMM epilogue
(FUSION_PLAN lever #1: "exact blocks it"; `lagged` unblocked only the denominator, and the
clip=1.0 recovery recipe stayed row-coupled).

**Change: put every update statistic on the storage geometry.** `stats_scope="group"` makes
`v` `[out, n_groups]`; the denominator and the clip are computed over the weight's own
128-wide scale group. Then the tick of [o, p] depends only on its group's slice of grad_w —
exactly what one group-aligned GEMM tile holds. Notes:

- This is a **different optimizer** (parity-gated, checkpoint-incompatible with row scope
  on purpose), not a kernel trick. It is *finer*-grained adaptivity than row scope
  (denominator over 128 weights vs the whole row), i.e. a step toward per-element Adam-like
  normalization, not away from it; the EMA (`rms_beta=0.9`) smooths the noisier estimate.
- SR keys (original-column hash), carry, saturation, 6-bit encoding, scale-gradient math,
  salient freezing: all unchanged.
- With a single group (`group == in_features`) the group-local math **is** the row math —
  pinned bit-exactly (below), so the change is statistic granularity only.

## What landed

- `group_counter_update_grouplocal_hashsr` / `..._from_io_hashsr`
  (`src/memory_native/group_scale_kernels.py`) — deterministic torch reference, the oracle
  the kernel mirrors. Mutates scale/v in place, returns act-ordered codes, like the row
  reference.
- `_group_fused_update_kernel` + `triton_group_counter_update_fused` — **one launch per
  layer**: grid (⌈N/32⌉, G); each program owns a [32 rows × one group] tile, accumulates
  its slice of `grad_w = go^T x` with `tl.dot` over the whole M in registers (tensor-core
  path, fp16/bf16 inputs, fp32 accumulate), then runs the entire automaton in the epilogue:
  group g²/scale-grad → v-EMA → denom → clip → scale step → SR tick → 6-bit repack.
  `grad_w` never exists in HBM; there are **no fp32 input casts, no [out,in] temporaries,
  no scratch launches**. Requires power-of-two group ≥ 16 (`tl.dot` tile constraint).
- `PackedGroupScaleCounterLinear(stats_scope="group")` — CUDA routes to the fused kernel,
  CPU to the reference; salient codes re-zeroed after the tick (identical to the dense
  flow); row scope untouched and stays the default.
- Benchmark arm `[L3]` in `scripts/benchmark_group_kernels.py`: fused timing + peak memory
  + single-step quanta-level parity vs the group-local oracle.

## Why this is the right shape for "memory of strict + speed of GEMM"

| update path | correlation form | temporaries | launches |
|---|---|---|---|
| strict 3-launch from-IO | scalar dot-loops (no MMA), computed **twice** | O(out·groups) scratch | 3 |
| dense (L2, production) | fp32 cuBLAS + `[:, perm]` gather | fp32 x/go casts + 2×[out,in] fp32 (570 MiB peak at Qwen M=4096) | 1 GEMM + 3 slim |
| **fused group-local (L3)** | `tl.dot` tiles in registers, computed **once** | none (state/scale/v only) | **1** |

The dense path's remaining cost is dominated by the fp32 cuBLAS GEMM itself
(group_kernel_opt_stage01.md); the fused kernel replaces it with a 16-bit tensor-core
correlation (T4: ~65 TFLOPS TC vs ~8 fp32; A100: bf16 native), *and* deletes the fp32
casts, the [N,K] temporary and the gather copy. Expected T4 at M=4096 FFN shapes:
~1.5–3× vs dense with peak transient memory dropping from ~570 MiB to ~0 beyond inputs.
To be measured, not asserted — see gates below.

## CPU witnesses (this box, `tests/test_grouplocal_update.py`, 8 passed; full suite 222 passed / 31 skipped)

- **Locality** (`test_grouplocal_is_group_local`): perturbing `grad_w[0,5]` changes state
  only inside row 0 / group 1 (cols 4–7) for every seed tried — the tile-independence
  invariant the fused kernel rests on. Row scope with clip=1.0 provably couples other
  groups of the row (`test_row_scope_couples_whole_row`) — the blocker is real, not
  hypothetical.
- **Degeneracy** (`test_grouplocal_degenerates_to_row_math`): `group == in_features`, 3
  steps, random perm, clip=1.0, residual_alpha=0.35 → codes, scale AND v **bit-identical**
  to the row reference. (Required aligning the g² reduction to the same op order:
  `view(out, G, group).square().mean(-1)`; scatter_add drifted in the last ULP — exactly
  the class of drift that tips SR boundaries.)
- **Learning** (`test_grouplocal_layer_recovers_teacher`): `stats_scope="group"` layer,
  clip=1.0, 500 steps → recovers a ternary teacher (mse < 0.05 gate passes).
- Salient stays frozen through group-local updates; state_dict round-trips; row↔group
  checkpoints refuse to cross-load (v shapes differ — intentional).

## GPU gates — pending (next CUDA session)

1. `scripts/benchmark_group_kernels.py` on T4/A100, Qwen FFN shapes (1536↔8960),
   M ∈ {512, 4096}: `[L3]` time + peak vs `[L2]` dense; **quanta contract**: code mismatch
   vs oracle at the ~1e-4–1e-3 level (same class as the dense-kernel gate), scale max|Δ|
   ~1e-6. The kernel's `tl.split`/`tl.reshape` lane unpacking needs triton ≥ 3.0 (gate ran
   3.6.0).
2. Occupancy check (Nsight): the register-resident [32×group] fp32 accumulator + epilogue
   is the known risk (FUSION_PLAN #1's open question). If occupancy tanks, halve BLOCK_N
   before anything else.
3. **KD-parity before production** (the discipline that caught rotations/naive-asym):
   short recovery from the s2i2 start, row vs group scope, strict alpha=0 curves. Only a
   parity/win there promotes `stats_scope="group"` into the deploy recipe.

## Honest caveats

- Numerics change (denominator/clip granularity K→group). All quality claims wait for
  gate 3; the CPU witnesses pin correctness and locality, not recovery quality.
- Single-rank only, like every packed-group update path; a fused epilogue additionally
  removes the natural all-reduce point (grad_w never materializes) — DDP would need
  activation-side reduction instead.
- tl.dot with fp32 inputs may engage TF32 on Ampere (same as the existing matmul kernels);
  the real training inputs are bf16/fp16, where products are exact in the fp32 accumulator.
