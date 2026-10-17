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

For the full capability matrix, read [project status](status.md). For raw measurements,
negative results and historical protocols, use the [results index](../research/results/README.md).
To reproduce or challenge a claim, follow [reproducibility](reproducibility.md).
