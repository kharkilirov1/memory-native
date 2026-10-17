# Benchmarks and status

[Documentation](README.md) / Evidence

Memory Native is an experimental library. Tests establish behavior on covered cases;
measured performance depends on the workload, hardware and implementation path.

| Area | Evidence | Scope |
|---|---|---|
| Packed counter state | [Packing and dynamics tests](../tests/test_packed.py) | Six-bit coefficient codes; other state is additional |
| Reversible backward | [CPU comparison](../research/results/REVERSIBLE_SINGLE_PASS.md) | Fewer branch evaluations; one measured CPU workload |
| Counter stability controls | [CPU ablation](../research/results/COUNTER_STABILITY.md) | Equation and resume checks; mixed quality results, opt-in controls |
| Small-model CUDA memory | [T4 comparison](../research/results/SHOOTOUT.md) | Short character-model runs; counter steps were slower in the recorded comparison |
| Donor conversion and recovery | [Artifact manifest](../research/results/ARTIFACT_MANIFEST.md) | Experimental pipelines; some historical artifacts are missing |
| MLX | [Port guide](MLX_PORT.md) | Covered layer/state behavior; Apple-silicon performance remains an open gate |
| Per-parameter bytes | [State accounting](../research/results/STATE_ACCOUNTING.md) | 0.750 B/param is the code body; 0.906 persistent / 1.002 resident measured; ~23 B/param for one pure-PyTorch step |
| Step speed (CPU) | [Step speedup A/B](../research/results/STEP_SPEED_2026_10_17.md) | −15.5% on the pure-PyTorch step, bit-identical; the automaton chain still dominates |
| PTQ recipe (GPU, 0.5B) | [PTQ recipe + rotation](../research/results/PTQ_RECIPE_2026_10_17.md) | align+salient+iters≥4 = 12.3× PPL vs defaults; rotation +1.40× (does not replace salient); sketching rejected |
| LUT-GEMM (A100, M=1) | [LUT-GEMM measurement](../research/results/LUT_GEMM_2026_10_17.md) | 2.7× faster than fused decode-in-GEMM for single-token decode, exact fp32; M=1 only, no salient path, not wired into a layer |
| Counter step vs dense AdamW (A100) | [Speed vs AdamW](../research/results/SPEED_VS_ADAM_2026_10_17.md) | gap 3.55× → 2.33× after hybrid mode + weight cache; matmuls now equal dense; remaining = one decode/step + updater |

Six bits are the coefficient code size, not total training memory. Count scales, RMS
statistics, floating-point model components, activations and temporary buffers. Kernel
speedups are not whole-model speedups. Large-model quality parity remains unproven.
Measured on CPU (`scripts/state_accounting.py`): a deployed packed group-scale layer is
0.906 B/param persistent and 1.082 resident, and one pure-PyTorch forward+backward step
peaks near 23 B/param — the code body is the smallest part of a working set.

The documented 60-step character-model gate is a plumbing/regression check, not a parity
claim: on CPU it reproduced the counter-optimizer cost over ternary QAT (+2.1…+5.7%) but
not "within ~0.1% of dense" (+3.9…+6.4% above dense across the default run and seeds 0-2,
and no win over vanilla counter in 3 of 4 runs).

## LUT-GEMM multi-token scaling and a verification-first counter/AdamW re-measurement (2026-10-17)

[`research/results/lut_scaling_2026_10_17/REPORT.md`](../research/results/lut_scaling_2026_10_17/REPORT.md)
adds a real multi-token LUT-GEMM pipeline (M=1..512), a CUDA-graph replay harness and a
counter-vs-AdamW harness that verifies real gradients, real AdamW state/parameter changes
and exactly one counter update per step before timing anything. On an A100 at
`ab39b3f`: multi-token LUT is exact (2.2e-6 relative) and 3-35x faster than the in-repo
fused-decode kernel, but loses to bf16 tensor-core GEMM at every M and to fp32 GEMM from
M~16-32; the counter step is 0.97-1.06x of an AdamW step at 4096^2/B=512 (decimation 4/8
help) and slower on 1024^2. Limits: strict ternary only (no alpha>0, no salient), single
A100, and **no controlled time-to-quality training experiment** - per-step cost is not
shown to translate into equal-quality wall-clock savings yet.

## Compute bottleneck: step decomposition, the no-fp32-cache path and a real training gate (2026-10-17/18)

[`research/results/compute_bottleneck_2026_10_17/REPORT.md`](../research/results/compute_bottleneck_2026_10_17/REPORT.md)
measures, on an A100 at `ab39b3f`: a 4096^2/B=512 packed-counter step is 0.87x the wall
time of FP32 AdamW with 3.8x less peak allocated VRAM (2.67 vs 3.07 ms, 17 vs 64 MB) using
the persistent fp32 cache + decimation; dropping the cache is a memory regression (144 MB
peak with two live decodes) and the truly fused decode-in-GEMM step is 98 ms even after a
3.2x tile retune. Small layers stay 1.1-1.3x slower. In the repo's real char-LM gate
(2 seeds) counter_rms reaches a **better** val loss than dense AdamW at the same budget
(1.9605 vs 2.1753) with 3.7x less persistent state but ~4x lower tokens/s, so time to equal
quality is ~2.1x worse - the open problem is the model-level counter layer overhead, not the
matmuls. A new verification harness confirms the decimation contract and one-update-per-step
and documents an open defect: on 4096^2 the fused updater moves 2 of 16.8M codes by one
quantum at lr=0. Peak memory at model scale is activation-dominated (1.05-1.12x total win),
so storage savings must not be quoted as training-memory savings.

For the full capability matrix, read [project status](status.md). For raw measurements,
negative results and historical protocols, use the [results index](../research/results/README.md).
To reproduce or challenge a claim, follow [reproducibility](reproducibility.md).
