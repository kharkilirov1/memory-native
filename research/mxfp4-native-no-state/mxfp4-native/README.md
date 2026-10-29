# mxfp4-native

Training with **no full-precision master weights or Adam moments**. The visible weights live in MXFP4 (E2M1 values, one E8M0 scale per 32 elements); a low-bit in-weight accumulator can carry optimizer pressure during training. Updates run in the backward pass. The compact 3-state path stores weights and accumulator at 6.25 bits/parameter asymptotically, while shipping only the MXFP4 values + scales (0.531 B/param). A fused path that avoids expanded full-size temporaries is still future work.

This branch is a **clean-room rewrite**. The idea (finite-state weights, updates applied in backward, no fp32 master, no Adam moments) is carried over from the ternary `memory-native` project on `main`; nothing else is. No salient channel, no act-order permutation, no PTQ solver, no converter — the model is trained in the format it ships in.

Status: **design phase**. There is no implementation yet, on purpose. The documents below are the specification to build against.

## Read in this order

| doc | what it settles |
|---|---|
| [docs/PRIOR_ART.md](docs/PRIOR_ART.md) | who already trains in 4-bit, and what is actually new here |
| [docs/FORMAT.md](docs/FORMAT.md) | bit layout, scale rule, encode/decode, byte accounting |
| [docs/UPDATE.md](docs/UPDATE.md) | the update: stochastic rounding on the E2M1 ladder + 2-bit error feedback |
| [docs/KERNELS.md](docs/KERNELS.md) | which community kernels are reusable, measured on this GPU |
| [docs/PLAN.md](docs/PLAN.md) | phases, acceptance criteria, what to measure before believing anything |
| [docs/PHASE2.md](docs/PHASE2.md) | char-LM results: the format costs 0.03 bits/char, the update rule costs 0.44 |

## Why rewrite instead of porting

The ternary project accumulated converter machinery that only exists because ternary cannot hold outliers: a sparse fp16 salient channel (0.12 B/param), its int32 runtime mirror (0.08), act-order permutation, and a ~2k-line PTQ solver. MXFP4 holds outliers with its block scale, so all of that disappears. Porting would keep the cost and lose the point.

Measured on the same synthetic dense teacher (`probes/`, reported in FORMAT.md):

| format | bytes/param | representation error |
|---|---:|---:|
| ternary counter, 6-bit (main) | 0.906 | 19.2% |
| **MXFP4 (this project)** | **0.531** | **1.41%** |

41% fewer bytes and ~14x less quantization error, at the same 6-bit-class budget.

A 2-bit residual plane was specified first and **lost its own A/B**: with stochastic rounding it makes training worse, because SR is already unbiased and the buffer only adds a random walk. Deterministic rounding without such a buffer cannot train at all (relative MSE 3.0 vs 0.028), which is exactly why ECO needs momentum. The measurement is in [docs/UPDATE.md](docs/UPDATE.md); the plane remains as MX6Linear(feedback=True).

## Hard constraint found before writing code

`tl.dot_scaled` on this box (RTX 5060 Ti, sm_120) compiles and is bit-exact, but the generated PTX contains **no `tcgen05`** — Triton enables native block-scaled MMA only for compute capability 10/11. So on consumer Blackwell MXFP4 buys **memory, not compute**. See [docs/KERNELS.md](docs/KERNELS.md) for the evidence and the paths that might still be native (cuBLASLt, CUTLASS).

## Current standing (char-LM, 3000 steps, 3 seeds)

| arm | bits/param | val, bits/char |
|---|---:|---:|
| dense fp32 + AdamW | 96 | — (2.6564 at 800 steps) |
| Adam-QAT: MXFP4 forward + fp32 master + AdamW | 100 | 2.4146 |
| **mx6: MXFP4 value + E8M0 scale + 6-bit accumulator** | **10.25** | **2.3775** |

The 33-state counter beats the best-tuned Adam baseline on the documented 3-seed tiny char-LM experiment at **9.8x less ideal packed state** per parameter. Two mechanisms produced that: a 6-bit gradient accumulator and headroom in the block scale. However, the original eager implementation did not physically pack that accumulator; actual persistent buffers were larger than the table implied. The new opt-in 3-state path closes much of the storage gap (6.25 bits/parameter before row-state overhead), with a small quality regression in one exploratory 800-step run. The result is not yet statistically validated, and peak memory still needs a fused/tiled kernel. Full evidence and caveats: [docs/PHASE2.md](docs/PHASE2.md).
