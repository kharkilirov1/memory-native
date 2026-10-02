# Project status

`memory-native` is an active research implementation, not a production training framework.
Its purpose is to make finite-state optimizer-in-weight training inspectable, reproducible,
and falsifiable.

## Status vocabulary

- **Verified** — covered by an executable test or a committed hardware witness.
- **Experimental** — implemented, but validated only on limited shapes, hardware, or data.
- **Open** — a stated research or engineering question; no positive claim is made.

## Current capability matrix

| Area | Status | Evidence |
|---|---|---|
| Six-bit coefficient/update code and deterministic update | Verified on CPU for supported `C=1..11` | Unit tests under `tests/test_counter.py`, `tests/test_packed.py`, and related suites |
| Packed coefficient codes at 0.75 byte/weight | Verified on CPU; additional state excluded | Packed round-trip and dynamics tests; scales, RMS statistics, metadata and FP tail add memory |
| PyTorch training integration | Verified on CPU/CUDA | CLI gates and `results/VERIFICATION_RESULTS.md` |
| Triton packed forward and fused update | Verified on Tesla T4 | `results/KERNEL.md`, `results/GPU_KERNEL_VERIFICATION.md` |
| Reversible activation memory | Verified on Tesla T4 for tested depths | `results/POOLS.md` |
| 1.21B-coefficient allocation/2000-step training | Historical T4 report; primary raw log missing | `results/SCALE_1B.md`; cannot independently audit the original run from this checkout |
| CUDA data-parallel state synchronization | Verified on 2x T4 for the recorded run | `scripts/fineweb_1b_2xt4.py` and linked results |
| MLX layer/state compatibility, group scales and Bonsai helpers | Verified for covered operations on Linux CPU | `docs/MLX_PORT.md` and MLX tests; full donor model import/training is not covered |
| Custom Metal kernel and Mac peak-memory/throughput | Open hardware gate | No committed Apple-silicon execution witness |
| Dense donor recovery / solver-v3 | Experimental | `docs/solver_v3_consolidated.md` and `results/solver_v3_*` |
| Group-local fused updates and decimation | T4 kernel witness and limited recovery evidence | `results/GPU_GATE_T4_GROUPLOCAL.md`, `results/PROD_GATE_15B.md`; kernel speed is not end-to-end speed |
| 12B cached-KD recovery | Infrastructure executed; recipe failed quality gate | `results/GEMMA12B_CACHED_KD.md`; preserve warm state when recovery degrades it |
| Convergence parity at 7B+ on a real corpus | Open | Primary scientific milestone |
| Strict update-from-IO without materialized `grad_w` | Verified on Tesla T4, impractical as the default | `results/ACCELERATION.md`, `results/group_kernel_opt_stage01.md` |

## Claims boundary

The historical 1.21B report describes fit and execution under one configuration; its named
primary raw log is absent from the public repository. Even with that log restored, a
2000-step run would not establish convergence parity for a language model trained to completion.

Six bits describe each packed counter coefficient and accumulator, not all model/training
memory. Scales, RMS statistics, permutations, salient FP channels, optional visible-weight
caches, FP embeddings/head/norms and their optimizer, activations, and temporary correlation
buffers must be counted separately. Some strict/group-local update paths avoid dense `grad_w`;
the default cuBLAS path materializes it transiently.

The corrected Qwen 1.5B v3f2 report records EN PPL 34.41 versus donor 11.6, RU 30.34,
and 70.4% mean task-accuracy retention. A later s2i2 report records EN 30.06/RU 22.39,
but its checkpoint and full metrics were not preserved publicly; corpus slices differ.
The 12B recipe degraded warm EN 46.4 to 46,608. Quality preservation remains unproven.
Training without master copies has prior work (for example ECO); novelty claims should concern
the specific finite-state representation, not absence of FP master copies alone.

Likewise, synthetic, character-level, calibration, and short-run recovery experiments are
reported as such. They are useful regression witnesses, not substitutes for long-horizon
pretraining evidence.

## Supported use

The most reliable uses today are:

1. reproducing finite-state counter dynamics;
2. measuring persistent and activation-memory trade-offs;
3. testing new update rules or packed kernels;
4. running controlled small-model comparisons;
5. evaluating donor recovery experiments with explicit baselines;
6. comparing strict zero-`grad_w`, tiled-correlation, and fast cuBLAS update paths.

For production training, treat the package as experimental and pin the exact commit, Python,
PyTorch, CUDA, GPU, seed, and command used.

[`notebooks/cached_kd_public.ipynb`](notebooks/cached_kd_public.ipynb) uses the committed
conversion/cache/recovery scripts. The older 27B notebooks describe an archived workflow
that requires an externally supplied runner ZIP; that runner is not included in a cold clone.

## Maintenance

The primary maintainer reviews changes for:

- correctness and regression coverage;
- traceable empirical claims;
- reproducible commands and environment disclosure;
- explicit separation of measured results from forecasts;
- compatibility with the MIT license.

See `CONTRIBUTING.md`, `REPRODUCIBILITY.md`, and `SECURITY.md`.
