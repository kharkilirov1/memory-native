# Perf audit + speedups of the distill step (CPU witness, 2026-07-11)

Box: Windows, torch 2.12.1+cpu, 10 threads, no CUDA/Triton/MSVC. Model: Qwen2.5-0.5B
*geometry* (h896, L24, GQA 14/2, inter 4864, vocab 151936), random init — speed does not
depend on weight values. Step = resident-teacher KD + CE, AdamW on fp slice, counter body
self-update in backward, i.e. exactly `scripts/finetune_qwen_counter.py` mechanics.

## Where a step's time goes (torch.profiler, B2xT128, old defaults)

- `backward(+counter update)` = **75–80% of the step**; the update is O(params), batch-independent.
- Single most expensive op: `aten::uniform_` (the `torch.rand_like` inside `stochastic_round`)
  — **20.4% of the whole step**, more than ALL matmuls combined (`aten::mm` 16.6%).
- The rest of the update is ~15 separate elementwise passes per weight tile
  (mul/div/add/sub/remainder/ne/where/trunc/sign/clamp) + 8267 `copy_` dtype conversions.
- Refuted: uint8 `randint` as a cheaper RNG — it is ~2x SLOWER than fp32 `rand` on CPU
  (64.4ms vs 33.7ms per 4.4M elems). CPU RNG cost is only fixable by fusing/decimation/GPU.

## What was changed (all verified by tests + bit-for-bit state hashes)

1. **`attn_implementation="sdpa"`** everywhere (donor loader default, script). SDPA fuses
   only the attention math — each counter projection still runs once per forward; the
   eager-only reuse guard does NOT fire (test: `test_sdpa_full_steps_do_not_trip_reuse_guard`,
   logits match eager at atol 1e-4). Teacher forward measured 2.1–2.8x faster on CPU.
2. **`cache_mode="fp16"` default in `qwen_to_counter`** — forward reads the derived T-cache
   instead of re-decoding the full state every call (student fwd ~2x on CPU). Costs +2 B/weight
   resident; pass `cache_mode="none"` to trade back.
3. **`TopKLogitCache`** (`memory_native.recovery`) — TeacherSource wrapper caching teacher
   top-k logits per unique batch (CPU fp16/int32, ~3 MiB per 8x512 batch at k=128). Batches
   repeat every epoch, so epoch 2+ skips the teacher forward entirely. NOTE: replay is
   top-k-renormalized KD — numerically NOT full-vocab KD. On a *random-init* toy teacher a
   small k visibly distorts the loss series (near-uniform tails carry mass); on a trained
   0.5B teacher top-128 at T=2 should hold ~all mass — **verify on the T4 curve** or set
   `TEACHER_TOPK = 0` for exact KD.
4. **No host syncs in the update hot path**: `weight_flips`/`update_events`/flip-rate now
   accumulate as on-device tensors (readers `int()`/`float()` lazily). Before: 2 `.item()`
   per layer per step = ~336 forced CUDA pipeline stalls/step on the 0.5B (CPU-neutral;
   the win is GPU-only and still needs a T4 measurement). Same for `distill_finetune` loss
   history and `perplexity` NLL (one sync at the end instead of per step/batch).
5. **`compile_update=True` opt-in** (any counter layer): the deterministic update chain
   (`_rms_eager_pre_sr` + `_carry_resolve`) goes through `torch.compile`; `stochastic_round`
   stays eager so the global RNG stream — and the DDP bit-identity contract — is untouched.
   Falls back to eager permanently if no backend (verified: without vcvars the fallback is
   bit-identical to the plain layer). MEASURED with MSVC reachable (vcvars64 + Build Tools
   14.44, follow-up pass): `dynamic=False` is required — static kernels run the chain
   **1.46x** (RNG-preserving split) and a real layer fwd+bwd step **1.30x** (321ms -> 247ms,
   ~15s one-time compile per layer shape); the initial `dynamic=True` choice generated code
   0.61x SLOWER than eager and was fixed. Fusing SR inside too reaches 2.01x on the chain but
   changes the RNG stream — kept out of the default design. Refuted alongside: hash-SR as a
   cheaper CPU RNG (0.46x vs rand) and `counter_update_hashsr` as a faster one-call torch
   update (0.54x) — both slower on CPU.

## Step benchmark (B4xT256, same run, quiet box)

| config | steady step | teacher part |
|---|---:|---:|
| OLD: eager + cache none + resident teacher | 43.2 s | 4.9 s |
| NEW: sdpa + fp16 cache + top-k cache (hit steps) | **32.0 s** | **~0 s** |

**1.35x per steady step** from config alone (no math change: state hashes across 6 layer
configs are bit-identical before/after the refactor; suite 163 passed / 12 skipped).
An earlier "1.79x" reading was taken while the test suite ran in the background — discard it.
CPU steps stay dominated by the elementwise update chain (batch-independent), so the CPU
ceiling is modest; the same levers on T4 (fused Triton update, no `.item()` stalls, sdpa
memory) are where the big multiple should come from.

## Mini end-to-end (tiny random Qwen, real pipeline calls)

swap -> ppl -> 3 rounds x 8 steps of `distill_finetune` sharing ONE `TopKLogitCache(k=vocab)`
-> ppl: loss 1.2431 -> 1.2108, ppl 64.33 -> 54.12, teacher ran once per unique batch
(4 misses / 20 hits).

## Not closed here (needs the T4 box)

- `KIND = "counter_packed"` enables the fused Triton update on CUDA, but the kernel requires
  `local_grad_clip=0` while the stable recovery recipe uses 1.0 — a stability-vs-speed
  experiment, or extend the kernel with row clip.
- Actual GPU effect of items 4 and 5 (sync stalls, inductor fusion) — measure on T4.
- ~~Latent finding~~ **FIXED same day (follow-up pass)**: the dead `blocked` branch in the
  carry step (`clamp_` in-place aliasing made `proposed_t != new_t` all-False) was a real
  torch-vs-kernel DIVERGENCE, not a design choice: the fused Triton kernel and its CPU
  reference (`fused_update.counter_update_hashsr`) pin a blocked flip's residual to the
  counter edge +-(C-1), while the torch path silently reset it. Fixed in `_carry_resolve`
  (counter.py) + the two remaining copies (group_counter.py, memory_ffn.py now call the same
  function). Witness: `tests/test_carry_saturation.py` (red before / green after, exhaustive
  torch==kernel-reference equality); full suite 166 passed / 12 skipped; toy distill recovery
  IMPROVED from 87% to **98.7%** (KL after distill 0.0012 -> 0.0001, same test/seed).
  NOTE: training dynamics legitimately changed (all 6 fixed-seed state hashes moved), so
  pre-fix run numbers (qwen_recovery_cpu / kaggle_mixed) are not bit-comparable with post-fix
  runs; the stable recipe itself re-validates on the next T4 run.
