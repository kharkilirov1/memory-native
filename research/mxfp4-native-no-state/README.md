# MXFP4 no-extra-state experiments

This is an isolated research subproject added as a new branch of the original Memory Native repository. It does not modify `main` or the existing ternary-counter implementation.

## Goal

Test whether MXFP4 weights can recover quality without introducing a persistent per-parameter optimizer state. The target is to retain the original low-memory premise rather than silently adding Adam moments or FP32 master weights.

## Current evidence (exploratory only)

- Balanced dependent stochastic rounding (BSR) is an optional candidate. In a tiny char-LM run at 400 steps, validation loss was 3.6270 vs 3.6267 for seed 0 and 3.6199 vs 3.6180 for seed 1 (lower bits/char is better). Two seeds are insufficient to claim a real gain.
- A stateless Muon-style polar direction was worse than the independent stochastic-rounding baseline at tested settings: 3.7169 vs 3.6270 at 400 steps, seed 0, LR 0.01. Do not enable it by default.
- The 2-bit residual/error-feedback plane did not improve the tested stochastic-rounding baseline; unbiased stochastic rounding already removes much of the motivation for a persistent error buffer.
- The current eager PyTorch path is a reference implementation, not a 12B-ready kernel. It can expand MXFP4 codes to FP32 during the forward/backward path, so ideal packed-state accounting is not peak VRAM accounting.

## Research rules

1. Keep `main` untouched; all work stays on this branch.
2. Never describe the tiny char-LM results as evidence for 12B pretraining.
3. Keep BSR optional until multi-seed, longer-run results support it.
4. Do not adopt stateless Muon-style as the default without a controlled sweep.
5. Report persistent bytes, peak allocated memory, throughput, and validation loss separately.

## Documents

- [No-extra-state experiments](docs/NO_STATE_EXPERIMENTS.md)
- [Original MXFP4 project notes](docs/PROJECT_NOTES.md)

## Reproduction

The experiments expect PyTorch and a local Tiny Shakespeare text file at `data/tinyshakespeare.txt`. Run scripts from this directory after installing the dependencies listed in `pyproject.toml`. The full original ZIP is not embedded as a binary Git object in this branch; this branch contains the research notes and tracked source files added here.
