# Native Memory: strict-readout failure analysis and 2-bit compute reference
Status: **research/ablation**, not production acceleration. 2026-10-08.

## Reproduced constraints and failure
- Training can read `W_alpha = s*(t + alpha*c/C)`, but deployment reads `W_0=s*t`.
- The 12B cached-KD report records warm strict aggregate log-PPL 3.2585 and step-750 11.5862; training loss alone failed to predict deployable quality. See `results/GEMMA12B_CACHED_KD.md`. The original 12B artifacts are external; this change DOES NOT reproduce that run.
- `_PackedGroupScaleFn.backward` changes state during backward. The forward-reuse guard prevents a naïve two-forward/two-backward dual-loss implementation. Such an implementation must first introduce a collection/single-commit update scheduler.
- A formal **loss mismatch witness**: with one scalar input x=1, scale s=1, t=0, c=C-1 and target y=(C-1)/C, alpha=1 gives zero squared error and therefore no loss gradient, but the deployable alpha=0 output is zero and has squared error ((C-1)/C)^2. This is an existence counterexample to the implication L_alpha small => L_0 small, **not evidence that this exact trap caused the 12B failure**.

## A safe ablation, NOT a claimed quality fix
`src/memory_native/recovery/strict_exposure.py` selects alpha=0 every S-th train step when `STRICT_EXPOSURE_EVERY=S>0`. **Only one forward and one backward run per step.** Zero disables it, preserving the original schedule and optimizer. The selected alpha is set before forward, so the state update uses one coherent alpha per step.

Both the live-teacher `scripts/run_ptq_recovery.py` and the cached-teacher `scripts/kd_cached_recovery.py` can opt in. The cached-KD runner now also accepts `HOMOTOPY_ALPHA_START=0` as a strict-only control. Controls/arms:

| arm | HOMOTOPY_ALPHA_START | STRICT_EXPOSURE_EVERY | interpretation |
|---|---:|---:|---|
| Baseline | 1 | 0 | existing homotopy, identical default |
| Strict-only | 0 | 0 | train/eval alpha=0 always |
| Strict exposure | 1 | 4 | every fourth step trains alpha=0 |
| Early anneal | 1 | 0 | hold=0.1, end=0.6 (existing env) |

These are hypotheses, not promoted recipes. Compare identical warm checkpoints, teacher cache, token ordering, total training tokens, learning rates, evaluation settings, and seeds. Include warm checkpoint as candidate for strict-alpha selection (existing protection); report best and final strict alpha=0 held-out metrics, grad/counter engagement, time and peak VRAM. Run cheap 0.5B smoke, then a 1.5B matched gate before a full 12B re-run. Separate effects of schedule, learning rate, and dataset.

## CPU toy falsification result (not a language-model benchmark)
On a three-layer residual-tanh student, 24 paired seeds, 150 counter-update steps, same teacher/training/eval inputs per seed, average strict-test MSE:
- ordinary homotopy: **0.1710608**
- strict pulse period 4: 0.1715862
- early anneal: 0.1720556
- always strict: 0.1746444

Pulse won 10/24 paired seeds relative to homotopy. The toy does not reproduce the catastrophic 12B collapse and does **not** establish a quality improvement. This is a *negative* gate against making strict pulses a default. The standalone CPU reproducer is a research artifact and not part of model-scale verification.

## A separate hypothesis: compressed visible compute state
At alpha=0 the forward only needs ternary `t`, not the residual counter `c`. Keep 6-bit packed (t,c) canonical for learning (0.75 byte/weight), and derive a 2-bit t cache (0.25 byte/weight), so the two representations total **1.00 byte/weight**, excluding scales, permutations, salient channels, other parameters and temporary storage. This compares against 1.75 byte/weight with a 1-byte int8 t-cache; it is a **memory layout** result, not an end-to-end model memory measurement.

The `visible2_reference.py` module supplies:
- 4-ternary-per-byte encode/decode (reserved code 3 rejected)
- collision-safe CPU reference flip patching (several flipped weights can share one byte)
- group-scaled, permuted strict forward `sum_g s[n,g] * sum_{p in g} t[n,p] * x[perm[p]]`

Tests check all 81 four-value ternary combinations, random nontrivial permutations/group scales, byte collisions, and malformed codes. This is **CPU/PyTorch reference only**: it materializes unpacked int8 for checking and has no CUDA/Triton speed claim. Salient FP overrides must be added separately to match the complete donor layer; alpha>0 needs the hidden counter and is not represented by t-cache alone.

## GPU work gates, not yet passed
1. First benchmark a packed-2bit, group-scale direct GEMV for M=1/2/4/8/16/32, with pre-gather cost included. Compare against fp16 cuBLAS, current packed6 decoding, and resident int8 T-cache, reporting absolute latency.
2. Use physical HBM bytes, executed instructions, regs/thread, spills, occupancy, and output error (not just theoretical byte count). A repeated byte load for four lanes can erase a theoretical 4x traffic advantage.
3. For larger M, a 2-bit tile decode feeding FP16 Tensor Core GEMM must be compared with dense cuBLAS on equal numerical precision and the same semantics. Do not infer speed from GEMV.
4. 2-bit cache coherence under *GPU* counter updates needs a dedicated collision-safe update protocol: single-owner packed-byte tiles or a two-pass flip/patch kernel. CPU patching is NOT production-ready.

## Decision
No new training recipe has passed a 12B quality gate; no GPU speedup has been measured. The safe delivered work is an explicit, zero-default-change strict-exposure ablation plus a tested 2-bit storage/correctness oracle.
