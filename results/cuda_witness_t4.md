# CUDA witness on Kaggle T4 (2026-07-11)

Kernel: `lirovkharki/mn-cuda-perf-witness` v2 (private), code = branch
`claude/finetune-pretrained-model-fwyuun` @ `9cc4e1a` embedded as tarball.
Raw numbers: `results/cuda_witness_t4.json`. One T4, torch 2.10.0+cu128, triton 3.6.0.

## A. Components (Qwen2.5-0.5B MLP shape 896->4864, M=4096)

| lever | result |
|---|---:|
| forward: decode -> fp16 T-cache | 11.9 -> 9.0 ms (**1.33x**) |
| forward: decode -> int8 T-cache | 11.9 -> 9.0 ms (1.32x) |
| update: torch path -> fused Triton | 7.75 -> 0.49 ms (**15.8x**) |
| layer fwd+bwd: `compile_update=True` (inductor->triton) | 11.2 -> 8.4 ms (**1.32x**), ~11 s compile |

## B. Full distill step, 0.5B geometry (random init, bf16, B4xT512)

| config | steady step | teacher part |
|---|---:|---:|
| OLD defaults (eager attn, no cache, resident teacher) | 4.03 s | 0.826 s |
| NEW defaults (sdpa, fp16 cache, top-k teacher cache) | **3.28 s (1.23x)** | **0.004 s** |
| NEW + emulated pre-9cc4e1a `.item()` syncs | 3.295 s | — |

- The top-k teacher cache erases the teacher forward on hit steps (0.83 s -> ~0).
- **Sync-cost hypothesis refuted at this scale**: the emulated ~336 per-layer host syncs cost
  only **+0.014 s/step (0.4%)** — a 3.3 s step hides them. The fix stays (it is free and
  matters for small/fast step configurations), but it is NOT a T4/0.5B lever.

## C. Fused-kernel recovery stability (REAL Qwen2.5-0.5B, WikiText-2, 150 steps, B4x512)

| run | PPL warm -> end | loss first -> last | s/step | fused engaged |
|---|---|---|---:|---:|
| control: `counter_rms`, clip=1.0, lr .008 | 117k -> **782** | 13.24 -> 13.55 | 4.25 | 0 |
| fused: `counter_packed`, clip=0, lr .008 | 117k -> 16.7k | 13.24 -> **21.2 (rising)** | 4.04 | 25 200 |
| fused: `counter_packed`, clip=0, lr .004 | 117k -> 16.2k | 13.24 -> **21.2 (rising)** | 4.04 | 25 200 |

Engagement proof: 25 200 fused tile-updates = 168 layers x 150 steps — the kernel really ran.

**Verdict — the "wake the x16 kernel by dropping the clip" idea is refuted as-is:**
1. Quality: without the row clip the recovery stalls and the loss RISES (21x worse end-PPL
   than control). `local_grad_clip` is essential for recovering a heavily degraded warm-start.
2. Speed: the x15.8 isolated update shrinks to **~5% of a full step** (4.25 -> 4.04 s), because
   the T4 step is dominated by GEMMs/attention/KD, not by the update transition (unlike CPU).

**Right next move**: put the row clip INSIDE the Triton kernel — it needs no extra pass:
`row_norm(grad_eff) = sqrt(g_sq * in_features) / denom`, and both `g_sq` and `denom` are already
computed in the kernel's pass 1. Then the recovery recipe keeps its stabilizer AND the fused
transition. (Expected end-to-end gain stays bounded by Amdahl: ~5% of a distill step at 0.5B —
worth taking mainly as part of a larger kernel-side consolidation.)

## v3 follow-up (same day): row clip landed INSIDE the kernel — verdict reversed to GO

`local_grad_clip` now folds into the kernel/reference RMS denominator (zero extra passes) and
the `clip==0` gate in `PackedRMSCounterLinear._fused_update` is gone. T4 witness
(kernel v3, `results/cuda_witness_t4_v3.json`):

- **Parity at clip=1** (hot grads, clip engaged on 88% of codes): kernel vs
  `counter_update_hashsr` = **0 mismatched codes, 0 quanta drift**, scale/v allclose.
- **Recovery with fused AT clip=1** (real Qwen2.5-0.5B, 150 steps, B4x512, same seed/corpus):

| run | PPL warm -> end | loss | s/step | fused engaged |
|---|---|---|---:|---:|
| fused `counter_packed` clip=1 lr .008 | 117k -> **727** | 13.24 -> 13.54 | **4.31** | 25 200 |
| control `counter_rms` clip=1 lr .008 | 117k -> 782 | 13.24 -> 13.55 | 4.54 | 0 |

The stable recipe now runs on the fused kernel: recovery is healthy (slightly better end-PPL,
within run-to-run noise), step is ~1.05x — the Amdahl bound predicted for a 0.5B T4 step.
Local suite after the change: 170 passed / 12 skipped (4 new clip tests in
`tests/test_kernel_clip.py`; clip=0 and huge-clip paths are bit-identical to unclipped).
`finetune_qwen_counter.py` now defaults to `counter_packed` on CUDA.

## Bottom line for the recovery workflow on T4

- NEW script defaults are **1.23x per step** end-to-end; with rounds >= 2 the teacher cache is
  the main contributor.
- `compile_update=True` is worth testing in the full run (+1.32x on the layer step; interacts
  with the same Amdahl budget).
- Control run after the saturation bugfix reached PPL 782 in 150 steps (prior CPU run: 907
  after 120 steps; different batch/steps — not a controlled comparison, but consistent with
  the improved toy recovery).
