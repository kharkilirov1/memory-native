# mxfp4-native

Training with **no full-precision master weights or Adam moments**. The visible weights live in MXFP4 (E2M1 values, one E8M0 scale per 32 elements); a low-bit in-weight accumulator can carry optimizer pressure during training. Updates run in the backward pass. The compact 3-state path stores weights and accumulator at 6.25 bits/parameter asymptotically, while shipping only the MXFP4 values + scales (0.531 B/param). A fused path that avoids expanded full-size temporaries is still future work.

This branch is a **clean-room rewrite**. The idea (finite-state weights, updates applied in backward, no fp32 master, no Adam moments) is carried over from the ternary `memory-native` project on `main`; nothing else is. No salient channel, no act-order permutation, no PTQ solver, no converter — the model is trained in the format it ships in.

## Status (2026-10-31)

**Implemented** (package `mx6/`): `layer.py`, `pack.py` (real bit packing), `update.py`, `packed_na.py` (PackedNA5/PackedNA6 layers, opt-in `enable_compile()`), `triton_kernel.py` (fused decode-in-GEMM, optional). The documents below remain the specification.

**Tested on CPU** (run 2026-10-31, from this directory: `..\..\..\.venv\Scripts\python.exe -m pytest tests -q`): `33 passed, 15 skipped`. The 15 skips are `tests/test_triton_kernel.py`, which needs Triton + CUDA.

**Not verified on this machine:** the fused Triton path (skipped here; its 15/15 parity result is from the A100 report). All GPU speed/memory numbers below come from reports, not from a re-run here.

**Known open defect:** the block scale of `MXNLinear` is frozen at init and the learned weights drift onto the outermost ladder rung; the "follow the scale" fixes tried so far collapse to the bigram floor. See [research/results/mx6_e1_quality_2026_10_31.md](../../results/mx6_e1_quality_2026_10_31.md). Update 2026-10-31: on the full E2M1 ladder, block-scale headroom 2 removes most of the cost of the frozen scale (see Current results).

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

## Historical standing (char-LM, 3000 steps, 3 seeds) — superseded, see "Current results" below

The 10.25 bits/param figure below is the *ideal* bit budget of the original eager layout, before commit a315447 ("fix(mx6): put the accumulator in the code word and make the bit budget real"). It is not what the current code holds and is kept only as history. Its quality column also predates the frozen-scale finding in `research/results/mx6_e1_quality_2026_10_31.md`, which says such benchmarks are devalued.

| arm | bits/param | val, bits/char |
|---|---:|---:|
| dense fp32 + AdamW | 96 | — (2.6564 at 800 steps) |
| Adam-QAT: MXFP4 forward + fp32 master + AdamW | 100 | 2.4146 |
| mx6: MXFP4 value + E8M0 scale + 6-bit accumulator (historical, superseded layout) | 10.25 | 2.3775 |

The 33-state counter beats the best-tuned Adam baseline on the documented 3-seed tiny char-LM experiment at **9.8x less ideal packed state** per parameter. Two mechanisms produced that: a 6-bit gradient accumulator and headroom in the block scale. However, the original eager implementation did not physically pack that accumulator; actual persistent buffers were larger than the table implied. The new opt-in 3-state path closes much of the storage gap (6.25 bits/parameter before row-state overhead), with a small quality regression in one exploratory 800-step run. The result is not yet statistically validated, and peak memory still needs a fused/tiled kernel. Full evidence and caveats: [docs/PHASE2.md](docs/PHASE2.md).

## Current results (claimed in reports, not re-run here)

**Newest (2026-11-01, G4, commit `e9830bc`): scaling, 4 seeds, step speed and memory: [research/results/mx6_g4_scaling_speed_2026_11_01.md](../../results/mx6_g4_scaling_speed_2026_11_01.md)**, raw logs and bench files in `research/results/mx6_g4_scaling_speed_2026_11_01/`. Held-out enwik8, 4 layers, 2 seeds per point; gap to the best-tuned `dense_nobias` at d=128 / 256 / 512:

- `lg6_fp4_h2` (6.55-6.69 bits) +0.283 / +0.231 / +0.179, `tern_counter` +0.380 / +0.299 / +0.200, `qat_fp4` +0.029 / +0.016 / +0.000: the gap shrinks with width; `na_fp4_c8` (8.55-8.69 bits) +0.143 / +0.134 / +0.106 is flat within its seed noise.
- AdamW lr 3e-3 diverges at d=512 for dense and `qat_fp4` (val 3.17 vs 1.80 at 1e-3); the counter / grid arms did not diverge. Dense d=128 / 256 were not re-tuned.
- 4 seeds, d=128 tinyshakespeare: `na_fp4_c8` h2 2.317, `lg7_fp4_h2` 2.353, `lg6_fp4_h2` 2.414 (dense_bf16 2.286).
- Speed (fp32 GEMMs, no autocast; step vs dense eager at d=128 / 256 / 512): `lg6` 3.28x / 2.94x / 1.38x, `na_fp4_c8` 8.19x / 7.23x / 3.44x; with `torch.compile` `lg6` 1.11x at d=512 (1.22x vs compiled dense), `na` 2.64x. Eager steps are launch-bound (~10 us per launch); no fused update kernel exists yet.
- Memory: block-linear state 15x smaller (lg6 9.82 vs 144.00 MB at d=512) and resident 3x smaller (72 vs 211 MB), but peak is unchanged at these sizes (794 vs 827 MB); at equal state memory lg6 d=512 beats dense d=128 by 0.043, at equal peak dense wins.

**Newest frontier (2026-10-31, held-out, G4, d=128, 6000 steps, 2 seeds per arm): [research/results/mx6_g4_headroom_lg_2026_10_31.md](../../results/mx6_g4_headroom_lg_2026_10_31.md)**, raw logs in `research/results/mx6_g4_headroom_lg_2026_10_31/` (uncommitted code; arm flags are in the Colab job scripts, not in the logs). Val bits/char; references: dense bf16 2.286, `qat_fp4` (fp32 latent) 2.298:

- 8.69 bits/param: **2.315** (`na_fp4_c8`, full E2M1 ladder, nvfp4 scale, block-scale headroom 2 via `--headroom-mode ladder --headroom 2`; 2.372 at the default headroom in the same batch, 2.361 in the earlier run below)
- 7.69 bits: **2.351** (latent grid k=7, headroom 2; counter C=4 at headroom 2: 2.384)
- 6.69 bits: **2.416** (latent grid k=6, headroom 2; counter C=2 at headroom 2: 2.502; the 6-bit mx6 `na_e2m1_c5` below: 2.547, and it does not improve with headroom 2)
- Reading: static headroom 2 is the main lever for the counter (-0.057 at C=8); pressure growth and a learnable block scale at the default headroom help less; the latent grid (one k-bit index into a fine E2M1-aligned grid, no counter, `mx6/latent_grid.py`) beats the counter at 6.69 and 7.69 bits and ties at 8.69. The latent grid is eager torch only: no fused kernel, no speed or memory measurement. Two seeds, d=128, 1 MB corpus; see the report's Limits.

**Held-out decomposition (2026-10-31, same split; the frontier above supersedes its best-arm numbers): [research/results/mx6_g4_decomp_2026_10_31.md](../../results/mx6_g4_decomp_2026_10_31.md)**, raw logs in `research/results/mx6_g4_decomp_2026_10_31/`. Quality numbers produced before this (the E1 report, the quality / rescale tables of `mx6_g4_2026_10_31.md`, the earlier 6000-step G4 runs and the "Quality" paragraph below) were measured with validation == training data (`probes/e1_quality.py` had no held-out split) and are **memorisation, superseded**. Held-out, G4, d=128, 6000 steps, cosine, val bits/char (dense / qat: 4 seeds; mx6 / fp4 arms: 2-3 seeds):

- dense bf16 (96 bits/param): **2.286**; full-E2M1 QAT with an fp32 latent (`qat_fp4`): 2.298
- mx6 on the full E2M1 ladder, C=8 (`na_fp4_c8`): 2.361 at 8.69 bits/param
- main project's ternary counter (`tern_counter`, best lr 3e-2): 2.481 at 6.56 bits/param
- mx6 `na_e2m1_c5` (6-bit word, ladder {0,.5,1,6}, nvfp4 scale): 2.547 at 6.69 bits/param (3 seeds; seeds 0-1 alone: 2.545)
- Reading: the RMS-normalised no-momentum update costs nothing on dense; most of the mx6 gap is the reduced ladder forced by the 6-bit word (+0.14 with an fp32 latent), not the counter (+0.05 to +0.12). See the report's decomposition.

Sources: `research/results/mx6_gpu_2026_10_30.md` (A100-SXM4-40GB, two linears 1024->4096->1024, batch 64) and `research/results/mx6_e1_quality_2026_10_31.md` (Tesla T4, 2-block transformer d=128, tinyshakespeare, 1500 steps, 4 seeds).

**Bits per trained parameter, measured** (gpu report): PackedNA5 5.26, PackedNA6 6.26, MXFP4 9-state with dense codes 8.26, dense bf16 + fp32 master + AdamW 96.00. Storage-state win: ~15x smaller than bf16 + AdamW.

**Speed vs dense bf16 + AdamW (1.00x, 1.30 ms/step)** (gpu report): PackedNA6 eager 0.23x, PackedNA6 + `enable_compile()` 0.33x (batch >= 32 only), PackedNA6 fused decode-in-GEMM 0.06x. The earlier 0.62x / 0.73x BF16 figures (commit messages, older docs) predate a315447 and must not be quoted.

**Peak training memory** (gpu report): dense bf16 177.7 MB, PackedNA6 eager 388.0 MB, PackedNA6 + compile 168.0 MB, fused 356.0 MB.

**Quality, bits/char (4 seeds, mean) - SUPERSEDED, memorisation (validation == training data); see the newest evidence above** (quality report): na_wide_c3 (5.44 bits) 2.4203, na_wide_c5 (6.44 bits) 2.4085, na_e2m1_c3 (6.44 bits) 2.6402 (one seed at 3.13, near the bigram floor 3.469); dense bf16 2.0394 (2 seeds). The counter arms are about 0.37 bits/char behind dense; the report attributes this to a degenerate, effectively 3-level quantizer (80-84% of weights on the outermost rung) caused by the frozen block scale.

**Bottom line:** a storage-state win only. There is no speed win and no peak-memory win over dense bf16 except with `torch.compile` at batch >= 32 (peak 168 vs 177.7 MB, speed still 0.33x). Quality is behind dense (held-out, d=128, 2 seeds: +0.03 at 8.69 bits, +0.13 at 6.69 bits with the latent grid; the older 6-bit counter was +0.26) and the scale rule is only partly solved (static headroom 2 helps; no dynamic rule beat it). 2026-11-01 update: the gap to dense shrinks with width for the latent grid and the ternary counter (not detectable for `na_fp4_c8`), eager steps are 1.4x-8x slower than fp32-GEMM dense (compiled lg6 1.11x at d=512), and resident memory is 3x lower while peak memory is unchanged at d<=512 (see the newest report above).