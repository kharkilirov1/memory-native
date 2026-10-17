# State accounting: 6 bits is 0.75 B/param, the layer is 0.9-1.1, a step is ~23

Measured 2026-10-17 on the 20-core CPU box (Windows, 31.9 GiB RAM, no CUDA),
PyTorch 2.12.1+cpu. Everything below is printed by
[`scripts/state_accounting.py`](../../scripts/state_accounting.py); the process-peak
numbers come from the platform working-set probe in the same script.

```bash
python scripts/state_accounting.py --step
```

## Why this exists

The method is described as a **6-bit** code (0.75 B/param). That figure is the
packed code body only. Published big-run numbers are consistently ~20% higher,
which reads like a discrepancy until the components are counted. This file is
the counting.

## One deployed layer: 1536x1536, group 128, C=11, salient 2%

`PackedGroupScaleCounterLinear`, state loaded the way the PTQ solver loads it
(2,359,296 params; 47,185 salient entries).

| component | shape | dtype | bytes | note |
|---|---|---|---:|---|
| `state` | (1536, 1152) | uint8 | 1.69 MiB | checkpoint — 0.750 B/param |
| `scale` | (1536, 12) | float32 | 72.0 KiB | checkpoint — one group scale per 128 |
| `v` | (1536, 1) | float32 | 6.0 KiB | checkpoint — RMS second moment, per row |
| `perm` | (1536,) | int32 | 6.0 KiB | checkpoint — act-order permutation |
| `salient_idx` | (47185,) | int32 | 184.3 KiB | checkpoint — 0.080 B/param |
| `salient_val` | (47185,) | float16 | 92.2 KiB | checkpoint — 0.040 B/param |
| `_salient_perm_flat` | (47185,) | **int64** | 368.6 KiB | **runtime only** — 0.160 B/param |
| `_flip_sample_indices` | (4096,) | int64 | 32.0 KiB | runtime only, fixed per layer |
| `_flip_sample_prev` | (4096,) | int8 | 4.0 KiB | runtime only, fixed per layer |

| accounting level | bytes | B/param | vs 6-bit ideal |
|---|---:|---:|---:|
| 6-bit ideal | 1.69 MiB | 0.750 | 1.00x |
| `persistent_bytes()` (checkpoint) | 2.04 MiB | **0.906** | 1.21x |
| resident (checkpoint + runtime tensors) | 2.43 MiB | **1.082** | 1.44x |
| strict-update scratch (transient) | 222.0 KiB | 0.096 | — |

So the honest per-coefficient numbers are **0.906 B persistent / 1.082 B resident**,
not 0.750. The largest single non-body item is the salient channel
(0.12 B/param persistent at 2%), and the largest *runtime* item is its int64
flat-index mirror, which is twice the int32 index it mirrors.

`--salient 0` gives 0.797 B/param: state + group scales + RMS + perm only.

## Reference vs packed layer (1024x1024, no salient)

| class | bytes | B/param |
|---|---:|---:|
| `RMSCounterLinear` (reference, one code per uint8) | 1036.0 KiB | 1.012 |
| `PackedRMSCounterLinear` (4 codes / 3 bytes) | 780.0 KiB | 0.762 |

The 0.75 B/param story belongs to the packed classes. The reference layer that
the quickstart and most unit tests use stores one code per byte: 1.0 B/param,
1.35x the ideal. Do not mix the two when quoting per-parameter bytes.

## One training step: 23 B/param, ~29x the state

`--step` measures one forward+backward through 4 layers of 1536x1536
(9,437,184 params) in a **fresh interpreter** (the working-set peak is a
process high-water mark, so prior allocations would hide part of the delta):

```
one fwd+bwd through 4 layers (9,437,184 params): persistent 7.08 MiB,
peak working-set growth 206.95 MiB = 22.99 B/param (x29.2 the state)
```

A separate ad-hoc probe with the same method split the phases: forward under
`no_grad` grew the peak by 7.9 B/param, forward+backward by 22.9 B/param. The
backward half is the decode-in-GEMM fp32 visible weight (4 B/param), the fp32
`grad_w` (4 B/param) and the elementwise update chain in
[`counter._update_tile`](../../src/memory_native/counter.py) (several full
`[out, in]` tensors for rebase/ticks/rounding/carry). This is the pure-PyTorch
path; the Triton kernels are what avoid the dense `grad_w`, which is why the
README credits 0.75 B/param to the kernel path and not to the default.

## Reconciliation with the recorded big runs

Measured per-coefficient rate above (0.906 B) predicts the recorded state sizes:

| run | recorded state | converted coefficients | implied B/param |
|---|---:|---:|---:|
| gemma-4-12B streaming conversion (`GEMMA12B_TERNARY_INFERENCE.md`) | 9.16 GiB | 10,899,947,520 | **0.903** |
| qwen3.8-27B streaming conversion | 20.5 GB | 24.35B (per-layer recount) | **0.84** |

The 12B number matches the measured 0.906 to 0.3%. Note the 27B report quotes
*15.59B* coefficients in the same sentence as 20.5 GB; 20.5 GB / 15.59B would be
1.31 B/param, which no measured layer reaches, so the 15.59B figure is the
converter's counter and undercounts (the report itself gives the 24.35B
per-layer recount). Quote 0.84-0.90 B/param for that state, not the counter.
Resident (`CLAUDE.md`, 12B on 2xT4) was 11.7 GiB = 1.15 B/param — consistent with
1.08 B/param resident + fp-tail buffers.

## Solver-side transients (the other "more than expected")

Conversion peak is set by the Hessian workspace, not by the stored state:

| item | size at a 27B-shaped target |
|---|---|
| one fp32 Hessian `[in, in]` for `in = 17408` | 1.21 GB (`in = 5120`: 105 MB) |
| `_prep_hinv` chain (clone + eye + cholesky + inverse) | process peak rose to **4.81 GB** with one live 17408^2 H |
| `align_scales_output` `A[out, K, K]`, `K = 2*(in/group)` | 5120 x 272 x 272 x 4 = **1.51 GB** per target |
| `_refit_salient_values` `.double()` copy of H | 17408^2 x 8 = 2.42 GB |
| chunking knob `hessian_gpu_budget_gib` (default 24.0) | caps live Hessians per chunk |

That is why the 27B streaming run reports 2.65 GiB resident while the solve phase
plateaus around 18.6 GiB commit: "memory bounded by block width" is true for the
stored weights, and the Hessian budget is a separate default.

## What this accounting does not cover

Activations and saved low-bit codes, the CUDA allocator's reserved-vs-allocated
gap, the fp tail (embeddings/head/norms) with its own AdamW moments, fp32 logits
(one vocabulary-sized tensor), and load-time duplication when a state dict is
copied into fresh modules. Add those per model before quoting a training peak.

## Open levers (measured, not applied)

1. `_salient_perm_flat` is int64 for every model. At 2% salient it costs 0.160
   B/param (17% of the body). A flat index needs 64 bits only when
   `out*in >= 2**31` (true at 27B, false at 1.5B): a dtype chosen from that
   condition halves it below the threshold.
2. `_flip_sample_indices` is 4096 int64 per layer; int32 would halve it (0.03
   B/param on a 1536^2 layer, more on narrow ones).
3. The pure-PyTorch update chain's temporaries (below) are what the Triton fused
   kernel exists to remove; on CPU they are the step peak, on GPU they are
   latency.
