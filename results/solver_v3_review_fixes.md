# Solver v3 — review blockers closed

Date: 2026-07-15

Branch: `agent/solver-v3-group-recovery`

This patch addresses the executable review findings discovered after the packed group layer and
Triton kernels landed.

## Correctness fixes

### Non-power-of-two group boundary

The original strict stats kernel used a next-power-of-two block but masked only the matrix boundary.
For groups such as 12, 24 or 96, inactive lanes could therefore enter the following scale group.
The production solver uses group 128, so the safe fix in this patch is explicit: strict Triton
updates now reject non-power-of-two groups before launch. Non-power-of-two groups remain valid on
the torch/reference path. A CPU gate proves rejection occurs before touching CUDA, and a CUDA gate
keeps the same contract on GPU.

### One carry implementation

The active group-reference update now resolves `_carry_resolve` to the canonical implementation in
`counter.py`, preventing future carry/saturation fixes from diverging across formats.

### Legacy PTQ kwargs

Group-only controls (`residual_alpha`, `kernel_mode`, `strict_update`, `flip_sample_size`) are
filtered before legacy row-scale constructors are called. `PTQ_MODE=gptq` and `optimal` ablations
therefore remain executable.

## Honest evaluation and checkpoint selection

Training may use residual homotopy, but every unprefixed PPL, selection metric and best-checkpoint
decision is now measured at strict inference `alpha=0`.

At each evaluation the runner can log two separate records:

- `strict_*`: the deployable ternary model at `alpha=0`;
- `homotopy_*`: a diagnostic at the current training alpha.

The model's training mode and alpha are restored after evaluation. Early alpha≈1 checkpoints can no
longer win merely because the counter residual is visible.

## Reproducible resume

`sr_step` is now a persistent scalar buffer in every packed group layer, mirrored by the Python
hot-path seed counter. Hash stochastic rounding therefore continues from the exact next seed after
`state_dict` restore without adding a per-step GPU-to-host synchronization.

The recovery runner now resumes by reconstructing counter module structure from saved tensors,
loading model/optimizer states, restoring Python/Torch/CUDA/NumPy RNG state, and continuing from the
saved global step. It does not collect Hessians or rerun PTQ. Legacy checkpoints without `sr_step`
initialize it from the completed global step.

## Fused-path telemetry

The Triton path no longer relies on the dead `weight_flips` counter. Each packed layer keeps a
deterministic decoded sample and reports interval net-change `flip_rate_alt`, plus a sampled
counter-edge fraction. The runner aggregates observations by actual sample size.

## Validation

Targeted CPU/reference checks in the reconstructed worktree:

```text
review-fix suite                         9 passed
new CUDA contract gate                  1 skipped (CUDA + Triton required)
existing packed/PTQ affected subset    10 passed, 2 deselected
Python bytecode compilation             passed
```

The tests pin exact SR continuation across resume, decoded flip sampling, strict-alpha evaluation,
legacy kwargs filtering, resume without solver invocation, trained-bias restoration, reference
support for non-power-of-two groups, and strict-Triton rejection of unsafe group sizes.

## Remaining hardware gate

The earlier forward/grad-x/update parity suite and the new CUDA contract gate still require T4/G4
execution. No new CUDA performance or parity result is claimed by this patch.
