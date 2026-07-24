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
| Six-bit counter state and deterministic update | Verified on CPU | Unit tests under `tests/test_counter.py`, `tests/test_packed.py`, and related suites |
| Packed persistent state at 0.75 byte/weight | Verified on CPU | Packed round-trip and dynamics tests; `results/SUMMARY.md` |
| PyTorch training integration | Verified on CPU/CUDA | CLI gates and `results/VERIFICATION_RESULTS.md` |
| Triton packed forward and fused update | Verified on Tesla T4 | `results/KERNEL.md`, `results/GPU_KERNEL_VERIFICATION.md` |
| Reversible activation memory | Verified on Tesla T4 for tested depths | `results/POOLS.md` |
| 1.21B-parameter allocation/training witness | Verified on one Tesla T4 for the recorded run | `results/SCALE_1B.md` |
| CUDA data-parallel state synchronization | Verified on 2x T4 for the recorded run | `scripts/fineweb_1b_2xt4.py` and linked results |
| MLX state compatibility | Verified for covered operations | `docs/MLX_PORT.md` and MLX tests |
| Dense donor recovery / solver-v3 | Experimental | `docs/solver_v3_consolidated.md` and `results/solver_v3_*` |
| Convergence parity at 7B+ on a real corpus | Open | Primary scientific milestone |
| Strict update-from-IO without materialized `grad_w` | Verified on Tesla T4, impractical as the default | `results/ACCELERATION.md`, `results/group_kernel_opt_stage01.md` |

## Claims boundary

The committed 1.21B witness demonstrates that the represented model and training path fit and
execute under the recorded configuration. It does **not** establish convergence parity for a
1.21B language model trained to completion.

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

## Maintenance

The primary maintainer reviews changes for:

- correctness and regression coverage;
- traceable empirical claims;
- reproducible commands and environment disclosure;
- explicit separation of measured results from forecasts;
- compatibility with the MIT license.

See `CONTRIBUTING.md`, `REPRODUCIBILITY.md`, and `SECURITY.md`.
