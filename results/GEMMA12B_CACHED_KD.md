# gemma-4-12B cached-KD recovery on 2xT4 — infrastructure PROVEN, recipe FAILED the gate

Date: 2026-08-03. Kernel `lirovkharki/mn-g12-kd-recovery` v5 (1500-step schedule, cut at
750 by the 11h cap) and v6 (750-step schedule, completed, WITH warm baseline). Student:
the R1 quality conversion of gemma-4-12b (131k mixed gemma-BPE calib, itf+align+salient
2% layer + salient_refit). Teacher: top-64 logit cache of the fp donor over the exact
deterministic KD stream (R2). Recipe: production group+dec4 + fused kernel, counter lr
0.002 -> 1e-4 cosine, homotopy alpha hold 0.2 / end 0.9, KD_T=2, CE_ALPHA=0.3, B1x512.

## The infrastructure result (all new, all now proven at 12B)

- **2-GPU model-parallel split** (`split_across_gpus`): layers 24..47 -> cuda:1, embed +
  norm + lm_head + layers 0..23 on cuda:0 (embed/head tie untouched); device movers as
  forward pre-hooks; Triton launch wrappers now pin `torch.cuda.device(<tensor>.device)`
  — the multi-GPU correctness fix. Why it is REQUIRED, not optional: the 12B counter
  buffers are ~11.7 GiB RESIDENT (packed state 8.2 + salient idx/val 1.3 + salient perm
  int64 1.7 + scales/v 0.5) + frozen embedding 2.0 = ~13.8 GiB on a 14.56 GiB T4 before
  a single activation. v3 (14.31 GiB) and v4 (14.15 GiB) both OOMed in the first forward;
  no activation trick fixes resident state.
- **Reentrant grad checkpointing** compatible with the eager-only counter guard (no-grad
  first pass takes the plain path; the recompute builds the Function graph exactly once).
  Used in v5; v6 ran WITHOUT it (2-GPU headroom) — 46 s/step -> 31-34 s/step.
- **FREEZE_EMBED**: AdamW moments for the tied 1.0B embedding would be ~8.5 GiB fp32 at
  the first step; frozen tail = norms + (never-exercised) multimodal linears, 53M.
- **Slim best-only checkpoint** (drop frozen fp + immutable salient/perm + v stat, stage
  through /kaggle/tmp): a full 12B state_dict is ~13-15 GiB and best+latest+tmp would
  blow Kaggle's ~20 GiB output cap at the FIRST eval.
- Cost points (2xT4, B1x512): 31-34 s/step without ckpt, 46-53 s with; strict eval ~19
  min per 12k tokens x 6 domains (~8 s per 512-token forward — T4 has no bf16 tensor
  cores). Restore + placement ~7 min.

## The quality result — the run FAILS its gate, and the warm baseline is the story

v6 curve (strict alpha=0, aggregate metric = mean log-ppl over 6 domains):

| point | metric | en | ru | code | math | science | instruct |
|---|---:|---:|---:|---:|---:|---:|---:|
| **warm (step 0)** | **3.2585** | 46.4 | 53.7 | 6.79 | 28.5 | 53.3 | 12.0 |
| step 250 | 5.6023 | 213 | 2707 | 184 | 158 | 304 | 77.9 |
| step 500 | 5.5877 | 243 | 1252 | 256 | 182 | 370 | 69.1 |
| step 750 (final, alpha=0) | 11.5862 | 46608 | 1.24M | 135k | 70.8k | 160k | 17.6k |

v5 (same recipe at 1500-step schedule, cut at 750): warm not measured, step 300 metric
9.46, step 600 11.57 — consistent with v6's shape once the baseline is known.

- **The R1 solver-only 12B warm start is strong: metric 3.2585** — the same class as the
  1.5B production-gate warm start (3.2992) that recovery then improved to 2.71. The
  12B conversion chain (itf + align + salient 2% layer + refit at 131k calib) stands.
- **Cached-KD training made it strictly worse at every eval**, ending 3.6x above warm.
  Diagnostic signature: kd-loss falls to 2.7-7.9 while alpha anneals 0.32 -> 0.02 (the
  residual c absorbs the objective), then jumps back to 22-25 at alpha=0 — c compensated
  while t drifted destructively. The 1.5B precedent (dec4 turbulence 3.30 -> 3.49 by
  step 600, then collapse to 2.71 BEST) does NOT transfer at this depth/compression.

## Candidate causes, ranked (next-experiment matrix, needs quota)

1. **lr too hot for 12B group+dec4** (first lever, halving is the documented group-scope
   fix): kd/ce are flat-noisy through the whole alpha>0.4 phase — no net learning while
   strict degrades monotonically even by step 250 (en 46 -> 213 during the alpha=1 HOLD).
   Arm: COUNTER_LR_START=0.001 (and 0.0005), same 750-step envelope.
2. **Homotopy compressed too hard**: alpha 1 -> 0 inside 500 steps at 12B leaves the
   post-anneal phase (steps 650+) with lr <= 2e-4 — too cold to repair t. Arm:
   HOMOTOPY_HOLD=0.1, HOMOTOPY_END=0.6 (anneal early, leave 300 hot-ish strict steps).
3. **No feature-KD** (uncacheable on this path) — the 1.5B production loop had it; logit
   KD alone may under-constrain 48 blocks. Arm: live-teacher feature-KD on 8 blocks via a
   second pass (needs >T4 memory or CPU teacher; deferred).
4. dec4 at depth (staggered noise at 48 blocks) — arm: DECIMATION=1 control.

Cheap CPU-side sanity first: rerun the 0.5B cached-KD smoke at 24 blocks and verify the
cached path RECOVERS there (it improved held-out MSE in the 2-block pre-gate); if even
0.5B degrades under the exact v6 config, the bug is in the cached-KD loop itself, not
scale.

## Artifacts

- Warm state: dataset `mn-gemma12b-counter-state` v2 (unchanged, still the best 12B).
- v6 best.pt (step 500, 5.5877 — WORSE than warm, kept only as evidence) + metrics.json
  in the kernel output; both curves pinned here.
- Quota: exhausted for the week (v5 11h + v6 7.6h on the 2xT4 shape); next arms wait for
  the Saturday reset.
