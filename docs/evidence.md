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

Six bits are the coefficient code size, not total training memory. Count scales, RMS
statistics, floating-point model components, activations and temporary buffers. Kernel
speedups are not whole-model speedups. Large-model quality parity remains unproven.

For the full capability matrix, read [project status](status.md). For raw measurements,
negative results and historical protocols, use the [results index](../research/results/README.md).
To reproduce or challenge a claim, follow [reproducibility](reproducibility.md).
