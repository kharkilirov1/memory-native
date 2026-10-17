# Counter step vs dense AdamW — measured 2026-10-17 (A100)

The project's speed target is not "fast for a counter" but "no slower than the dense
AdamW step it replaces". This note records the first A/B that measures both sides the
same way and attributes the gap, plus the two changes that landed from it.

Harness: `scripts/bench_vs_adam.py` (phase medians, `blocked_autorange`), A100-SXM4-40GB,
one 1536x1536 or 4096x4096 group-128 C=11 layer, batch 64 / 512. Both full steps include
their state update (dense `opt.step()`, counter `_update_from_io`).

## Baseline A/B, batch 64, 1536x1536 (row scope)

| phase | dense | gemm | triton | hybrid |
|---|---:|---:|---:|---:|
| fwd | 0.059 ms | 0.537 (9.1x) | 0.270 (4.6x) | 0.270 (4.6x) |
| grad_x | 0.029 | 0.483 (16.9x) | 0.242 (8.5x) | 0.242 (8.5x) |
| state update | 0.031 | 0.445 (14.4x) | 1.530 (49.5x) | 0.414 (13.4x) |
| **full step** | **0.654** | **2.322 (3.55x)** | **3.605 (5.51x)** | **1.547 (2.36x)** |

The phases disagree about which kernel path wins, which is why a single per-layer flag was
leaving time on the table: decode-in-GEMM is ~2x faster for the matmuls at batch <= 64,
but the from-I/O updater is ~3.7x slower than the dense (cuBLAS correlation + dense
kernel) one. `kernel_mode="hybrid"` takes the best phase of each and lands at 2.36x.

Group scope (`stats_scope="group"`, the decimation-capable path) is better still for the
update: 0.371 ms instead of 0.414, full step **1.390 ms = 2.09x** dense.

## Decoded-weight cache removes the entire matmul gap

The decode is paid twice per step (forward and grad_x) for the SAME weights. With
`weight_cache_bytes` enabled the matmuls match dense exactly:

| shape / batch | phase | cache off | cache on | dense |
|---|---|---:|---:|---:|
| 1536 / 64 | fwd | 0.548 | **0.062** | 0.059 |
| 1536 / 64 | grad_x | 0.496 | **0.028** | 0.028 |
| 1536 / 64 | full step | 2.350 | **1.873** | 0.662 |
| 4096 / 512 | fwd | 2.918 | **1.107** | 1.106 |
| 4096 / 512 | grad_x | 2.741 | **0.932** | 0.932 |
| 4096 / 512 | full step | 9.184 | **7.375** | 3.173 |

So the whole step-time gap is now: one decode per step (cache is invalidated by the
update) plus the updater itself:
- 4096/512: +4.2 ms = ~1.8 ms decode + 1.8 ms update.
- 1536/64: +1.2 ms = ~0.4 ms decode + 0.4 ms update.

## Mode crossover (decode-once+cuBLAS vs decode-in-GEMM), fwd

| shape | B=1 | B=16 | B=64 | B=128 | B=256 | B=512 |
|---|---:|---:|---:|---:|---:|---:|
| 1536x1536 | fused 1.60x | fused 1.96x | fused 1.90x | tie | decode-once | decode-once |
| 4096x4096 | fused 1.41x | fused 1.49x | tie | decode-once | decode-once | decode-once |

Fused wins only for small M; at real training batch sizes the dense path is correct, so
"hybrid" is an inference/small-batch mode, not a training default. Consistent with the
earlier T4 result that decode-in-GEMM is 10-20x slower at 4096^2/B=512 (12.6 ms vs 2.9 ms
here).

## What is left before parity

1. **Fused decode kernel** (~1.8 ms at 4096^2): the torch decode is ~10 elementwise passes
   over [out, in]; one Triton launch reading packed codes and writing the dense weight
   should be ~0.1-0.2 ms. This is the largest single remaining term.
2. **Updater** (~1.8 ms at 4096^2/B=512, ~0.4 ms at 1536^2/B=64): decimation is the
   measured lever (1.7-3.0x isolated); it needs per-class `lr` before a quality test.
3. Batch/scope heuristics: small-M steps should pick the fused matmul; group scope should
   be preferred when its update is cheaper.

Tests: `tests/test_group_scale_packed.py` green; full CPU suite 639 passed / 24 skipped
after the hybrid mode and the cache.

## Round 2: fused decode + group scope + decimation, and a measurement bug

`triton_decode_dense` landed (one launch; 0.145 ms at 1536^2, 0.557 ms at 4096^2 vs ~1.8 ms
for the torch chain; bit-identical at the deployable alpha=0, fp32-close at alpha>0).
`--decimation` and `--stats-scope` were added to the harness. Also found that the earlier
"AdamW state update = 0.031 ms" was wrong: `opt.step()` on a parameter whose `.grad` is None
is a no-op, so that row measured launch overhead only. With a gradient loaded, AdamW costs
**1.21 ms** at 4096^2 (and 0.318 ms at 1536^2), which changes the attribution.

A100, 4096x4096, B=512, cache on, fused decode:

| config | fwd | grad_x | update | full step | vs dense |
|---|---:|---:|---:|---:|---:|
| dense + AdamW | 1.105 | 0.932 | 1.211 | **3.047** | 1.00x |
| group, dec1 | 1.108 | 0.932 | 1.075 | 4.532 | 1.49x |
| **group, dec4** | 1.108 | 0.932 | **0.721** | **3.862** | **1.27x** |
| group, dec8 | — | — | 0.620 | 3.856 | 1.27x |
| group, dec16 | — | — | 0.568 | 3.864 | 1.27x |
| row, dec1 | 1.106 | 0.931 | 1.844 | 6.061 | 1.99x |

At 1536x1536/B=64: group dec1 1.233 vs dense 0.580 = 2.12x (decimation is *counter*
productive there: the update is already only 0.345 ms and the mask path costs more).

Two things stand out:
- **The updater already beats AdamW** at 4096^2 (0.721 vs 1.211) — dec4 is not the
  bottleneck any more. The gap is now elsewhere: fwd/grad_x are exactly 1.00x, so the
  remaining +0.8 ms is the **per-step decode** (~0.56 ms, because the update invalidates
  the cache) plus ~0.25 ms of autograd/Function overhead.
- Decimation saturates: dec4/8/16 all land at 1.27x, so the update has a fixed ~0.57 ms
  floor at this shape.

Next lever, aimed at that: make the updater **emit the dense weight itself** (it already
computes each element's new code and scale) so the per-step decode disappears. Expected
~3.1 ms ≈ 1.0-1.05x at 4096^2/B=512. The alternative — a faster decode — cannot go below
its ~0.5 ms read/write floor at this size.


## Round 3: the updater emits the dense weight, and the counter step is FASTER than AdamW

Two fixes, one discovery:

- `triton_group_counter_update_fused(..., dense_out=)`: the fused updater writes the
  post-update dense weight as it goes (it already has every new code and scale in
  registers). The per-step decode disappears: update 0.721 -> 0.856 at 4096^2/B=512 (the
  store costs, the decode it replaces cost 0.56) and the step drops 3.862 -> 3.520.
- **The counter updater runs INSIDE the autograd Function's backward**, so the earlier
  harness counted it TWICE (forward -> backward(update) -> explicit `_update_from_io`).
  The fair step is dense `fwd+bwd+opt.step()` vs counter `fwd+bwd`. Re-measured:

A100, cache on, fused decode, group scope (row scope is strictly worse now):

| shape | batch | dec | dense full | counter full | ratio |
|---|---:|---:|---:|---:|---:|
| 4096x4096 | 512 | 8 | 3.047 ms | **2.715 ms** | **0.89x** |
| 4096x4096 | 512 | 4 | 3.047 | 2.750 | **0.90x** |
| 4096x4096 | 512 | 1 | 3.047 | 3.425 | 1.12x |
| 4096x4096 | 1024 | 8 | 4.823 | 4.516 | **0.94x** |
| 2048x2048 | 256 | 4 | 0.586 | 1.075 | 1.83x |
| 1024x1024 | 512 | 4 | 0.572 | 1.011 | 1.77x |
| 1536x1536 | 64 | 1 | 0.585 | 0.760 | 1.30x |

At the 4096^2 shapes that a 12B-class model actually uses, a counter step with decimation
is now **faster than the dense AdamW step it replaces (0.89-0.94x)**, with fwd and grad_x
already equal to dense (1.00x) and the updater cheaper than AdamW (0.695 vs 1.209).

What is left (medium/small shapes only): the updater has a ~0.6 ms floor that does not
scale with N*K (0.607 at 2048^2, 0.593 at 1024^2, 0.695 at 4096^2/dec8), and at those
sizes AdamW itself is ~0.3-0.4 ms, so the ratio stays above 1. The prime suspect is
per-call host work in the update path (a `torch.tensor([seed])` H2D copy each step plus the
decimation index tensors rebuilt every call), not the kernel: the same kernel is 0.4 ms at
1536^2/dec1. Next: reuse the seed buffer and per-phase decimation indices, then re-measure.

Semantics: `tests/test_group_scale_kernels_cuda.py` 5 passed (bit-exact vs the torch
oracle), `tests/test_group_scale_packed.py` 18 passed, full CPU suite 639 passed / 24
skipped. The dense emission only writes a derived buffer; the state math (SR, carry, group
scales) is untouched.


## Round 4: re-measured after the seed experiment; the per-shape picture

Tried to remove the last host-side cost by passing the SR seed as a scalar kernel argument
(no per-step H2D tensor). Triton renders a python int as a constant, not a tensor, so
`seed.to(tl.uint32)` stops compiling and the tick's RNG stream would change; reverted. The
decimation column-index helper (`_decim_index`) stayed. Semantics re-verified: 5 CUDA
bit-exactness tests + 18 packed tests pass.

Confirmed table (A100, cache on, fused decode, group scope):

| shape | batch | dec | dense full | counter full | ratio |
|---|---:|---:|---:|---:|---:|
| 4096x4096 | 512 | 8 | 3.047 ms | **2.742 ms** | **0.90x** |
| 4096x4096 | 512 | 1 | 3.047 | 3.430 | 1.13x |
| 4096x4096 | 1024 | 8 | 4.823 | 4.516 | **0.94x** |
| 2048x2048 | 256 | 4 | 0.590 | 1.065 | 1.81x |
| 1024x1024 | 512 | 4 | 0.591 | 1.035 | 1.75x |
| 1536x1536 | 64 | 1 | 0.596 | 0.768 | 1.29x |

Reading: for the layer shapes and batches a 12B-class model uses, one counter step is now
**faster than the dense AdamW step it replaces (0.90-0.94x)**. Smaller layers stay above
1.0 because the updater carries a ~0.6 ms floor (0.597 at 1024^2, 0.621 at 2048^2, 0.710 at
4096^2/dec8) while AdamW is 0.27-0.41 ms there: the floor is flat in N*K, so it is per-call
overhead in the update path (host work, launch, the decimation index build), not the
kernel's arithmetic. That is the next target; until it is fixed the objective is met for
large shapes only.
