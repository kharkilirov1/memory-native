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
