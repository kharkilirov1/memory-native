# Group-kernel optimization — Stage 0+1 results (T4-measured)

Date: 2026-07-16. Code @8944e20 (on top of solver-v3 @ad44753).
Plan: docs/group_kernel_optimization_plan.md. Witness kernel:
`lirovkharki/mn-v3-group-kernel-gate` v2 (baseline, @4f762a6) and v3 (final, @8944e20),
Tesla T4, torch 2.10.0+cu128, triton 3.6.0, bf16, group=128. Parity gate v3: **18 passed**
(13 solver-v3 + 3 gemm-mode CPU + 2 dense-update CUDA), verdict PASS on both runs.

## Stage 0 — the plan's run-blocker math, confirmed harder than forecast

Per-layer times on Qwen2.5-1.5B FFN shapes ([K->N], M = tokens per step):

| path | up 1536->8960, M=512 | down, M=512 | up, M=4096 | down, M=4096 |
|---|---:|---:|---:|---:|
| triton decode-in-GEMM fwd | 33.4 ms | 56.6 ms | 435.9 ms | 823.4 ms |
| triton grad_x | 39.6 ms | 34.8 ms | 315.8 ms | 314.5 ms |
| **strict 3-launch from-IO update** | **520 ms** | **1319 ms** | **30 458 ms** | **46 203 ms** |
| cuBLAS fwd (same math) | 5.1 ms | 5.3 ms | 50.0 ms | 51.3 ms |
| decode(visible_weight), M-independent | 10.4 ms | 10.6 ms | 10.5 ms | 10.9 ms |

The strict update at the real step size (M=4096) is 30-46 s per layer -- hours per training
step over 196 layers. Hypothesis H5 (hand-rolled BLOCK_M=16 correlation loops) confirmed as
the dominant pathology; H2 (M-times repeated decode) and H3 (uncoalesced gather) confirmed by
the M-scaling of the matmul kernels vs the M-independent decode cost.

## Stage 1 — landed (L1 gemm mode + L2 slim dense update), same gate PASS

| path | up, M=512 | down, M=512 | up, M=4096 | down, M=4096 |
|---|---:|---:|---:|---:|
| gemm-mode fwd (decode+cuBLAS) | 14.5 ms | 15.4 ms | 59.3 ms | 56.5 ms |
| gemm-mode grad_x | 15.3 ms | 15.6 ms | 67.7 ms | 72.1 ms |
| semi update (cuBLAS grad_w + reference chain) | 64.0 ms | 64.3 ms | 86.5 ms | 84.7 ms |
| **dense update (cuBLAS grad_w + slim kernels)** | **5.9 ms** | **7.1 ms** | **32.5 ms** | **27.9 ms** |

- gemm-mode parity: max_abs=0 vs the dense reference (identical code path).
- dense update parity: quanta-level vs the bit-exact reference (same contract as the
  existing strict-kernel gate); SR keys unchanged (original-column hash).
- dense update speedup at M=4096: **681-1665x vs strict**, 2.6-3.0x vs semi; peak temp
  memory 570 MiB (fp32 x/go casts + one [out,in] fp32 grad -- transient, not a pool).
- Remaining dense-update cost is dominated by the fp32 cuBLAS correlation GEMM; on A100
  (TF32/fp32 throughput + 3x bandwidth) expect a few ms/layer -- the plan's <=2 ms/layer
  gate is within reach there; T4 was never the run target.

## What changed in code

- `kernel_mode="gemm"` (and `"auto"` -> gemm): dense decode + cuBLAS matmuls; `"triton"`
  decode-in-GEMM stays as the explicit low-memory path.
- `triton_group_counter_update_dense` + `_group_stats_dense_kernel` /
  `_group_state_dense_kernel` (finalize kernel reused; group-boundary mask included, so the
  non-pow2 stats hazard does not exist on this path — pow2 still enforced for consistency).
- gemm/auto layers on CUDA route updates through the dense kernels automatically;
  CPU keeps the bit-exact reference.
- Benchmark now prints cuBLAS/decode/update references and the gemm-layer witness rows.

## Step-cost outlook for the 6000-step run (extrapolation — verify with the 5-min A100 baseline)

T4 per-block (M=4096): ~510 ms -> ~14-16 s/step full model. A100 estimate: ~2.5-4 s/step.
6000 steps ~ 35-50 units (over the ~23-28 remaining budget); **3000 steps ~ 17-25 units, fits**
— and the Stage-B compounding evidence says most of the PTQ-start gain lands in the first
hundreds of steps. Decision knobs: STEPS, EVAL_HOMOTOPY=0, B4 vs B8.

## Not done / next

- Stage-0 A100 5-min baseline (first thing next Colab session; decides STEPS/batch).
- Decode-once-per-step cache (saves one 10 ms decode per layer per step; rejected keeping
  the full-pool variant — a per-step cache held across fwd->bwd adds a transient ~3 GB at
  1.5B, acceptable on A100 but only worth it after the A100 baseline says decode matters).
- Stage 2 (autotune, x-perm materialization, fp16-dot on SM75) only if a low-memory
  deployment target appears; the triton decode-in-GEMM path is otherwise dormant.
- review_fixes monkey-patch fold into sources — unchanged prerequisite before merging
  `agent/solver-v3-group-recovery`.
