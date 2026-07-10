# CLAUDE.md — project + active initiative context

Auto-loaded by Claude Code. Read this first to resume work cold.

## What this repo is

`memory-native` — finite-state **counter synapses** + **reversible activations** for
memory-efficient training, pure PyTorch (CPU/CUDA, no custom engine).

**What the method actually saves (read this — it is the whole point):** it is NOT a weight
compressor. It *eliminates the training-state pools*. For a Gemma-4-E2B-shaped model
(35 layers, d=2048, vocab 262K, seq 2048, batch 8) the pool breakdown is:

| pool | BF16 AdamW | counter |
|---|---:|---:|
| persistent (weights) | 4.28 | 2.24 |
| grad | 4.28 | **0** |
| optim (Adam m/v/master) | **25.68** | **0** |
| activations | 13.12 | **0.47** |
| **total** | **47.4 GiB** | **2.7 GiB** |

The dominant wins are `optim → 0` (no Adam moments, no fp master), `grad → 0`, and reversible
activations (~28×). Embeddings stay bf16 (~1 GiB) — a trivial slice, NOT a bottleneck. This
matters for finetuning because finetune memory is dominated by exactly the optim + activation
pools the method zeroes.

Key files: `src/memory_native/counter.py` (counter layers), `packed.py` (6-bit storage),
`glm.py` (`MNGLM` — GLM/Llama-class decoder: RMSNorm+GQA+RoPE+QK-norm+SwiGLU+Counter-MoE, the
"receiver" architecture for modern donors), `moe_ffn.py` (stacked-expert Counter-MoE),
`reversible.py`, `budget.py` (4-pool memory model), `convert.py` (NEW, see below).

## Active initiative: finetune a pretrained model in counter format

Branch: **`claude/finetune-pretrained-model-fwyuun`**. Goal: take a modern open-weights model,
warm-start it into the counter format, and recover quality with a finetune.

Donor decision (DECIDED 2026-07-06): **Qwen2.5-0.5B base** — dense, open weights (not gated),
near-mechanical map — chosen to close the first end-to-end recovery witness. **Gemma 4 E2B**
(2.3B; needs a Per-Layer-Embeddings + sliding-window/global-attention + proportional-RoPE adapter;
weights HF **gated**) is deferred to a second pass. Phases 1–2 were kept donor-independent on purpose.

### Plan (phases)
1. **Conversion primitive** — `weight → (scale, ternary, counter)`.  ✅ DONE
2. **Model-level swapper** — replace every `nn.Linear` with a warm-started counter layer.  ✅ DONE
   - Qwen HF loader + in-place swap: `src/memory_native/donor/qwen.py` (`qwen_to_counter`,
     `load_qwen_donor`). Dense Qwen needs no arch adapter — HF owns GQA/RoPE/RMSNorm/SwiGLU; we
     only swap the body linears (skip tied `lm_head`, preserve q/k/v bias). Gemma adapters: 2nd pass.
3. **MoE import** — fill the stacked `[E,out,in]` expert buffers (`moe_ffn.py`) for a MoE donor.
   (Not needed for dense Qwen; deferred with Gemma.)
4. **Recovery finetune** — distillation from the fp teacher.  ✅ toy + REAL-CPU witness (T4 for full recovery)
   - `src/memory_native/recovery/distill.py` (`distill_finetune` with `grad_clip`, `ResidentTeacher`,
     a `TeacherSource` seam for a later offline logit cache) + `eval/ppl.py`. 0.5B teacher resident
     (fits beside the student on a T4), online KL; offline cache deferred to big donors.
   - REAL run (results/qwen_recovery_cpu.md): Qwen2.5-0.5B on WikiText-2, PPL fp=21.2, warm-start
     =159k (ternarization is catastrophic), 120 CPU distill steps → **907 (99.4% of the gap closed)**.
   - STABILITY (critical): the naive default (lr 2e-3, no clip, thr 0.7) DIVERGES (warm 2e5 → 5e13).
     The stable recipe (now the script default): thr=0.5, fp lr=3e-4 + grad_clip=1.0, counter
     lr=0.008 + local_grad_clip=1.0, ce_alpha=0.3, T=2. A threshold sweep (0.2–0.7) is all ≥1.8e5,
     so threshold is not the lever — the recovery finetune is.
5. **Validation** — round-trip loss, PPL-recovery curve vs original, memory-gate/shootout.

### Done so far (committed + pushed)
- `d048bd5` Phase 1: `weight_to_counter_state()`, `from_dense`/`from_linear`/`load_dense_weight`
  on the counter layers (packed inherits). Tests: `tests/test_from_dense.py`.
- `4e57949` Phase 2 core: `swap_linears_to_counter(model)` + `CounterLinearWithBias` +
  `SwapReport` in `convert.py`. Tests: `tests/test_convert.py`. Exported from package `__init__`.
- (2026-07-06, UNCOMMITTED on this branch) Qwen donor end-to-end: `donor/qwen.py`,
  `recovery/distill.py` (with `grad_clip`), `eval/ppl.py`; tests `test_qwen_donor.py` (4) +
  `test_distill.py` (2; KL 0.0092→0.0012 on a toy, 87% recovered); runnable
  `scripts/finetune_qwen_counter.py` + `notebooks/qwen_recovery_t4.ipynb` (T4). Design:
  `docs/superpowers/specs/2026-07-06-qwen-counter-recovery-design.md`. `[donor]` extra in pyproject.
  Full suite: 156 passed / 12 skipped. REAL CPU run: 99.4% PPL gap closed (results/qwen_recovery_cpu.md)
  — BUT inference (results/qwen_inference_samples.md) shows the metric LIED: English-only distill on
  tiny WikiText-2 overfit to its hockey topic, and Russian + code were forgotten (Russian/code prompts
  return English hockey text). Lesson: judge recovery by broad generation, not PPL on the train set;
  the distill corpus MUST be diverse + multilingual + close to the pretraining mix (a data problem).
- (2026-07-06) REAL GPU run on Kaggle T4, MIXED corpus EN+RU+code (results/qwen_recovery_kaggle_mixed.md):
  per-lang PPL fp {EN 21, RU 8.5, CODE 9} -> warm {170k, 6.6M, 251k} -> recovered {307, 91, 224}.
  Russian prompts now return RUSSIAN (not English) — the mixed corpus brings back ALL abilities, no
  topic collapse. Still ~10-25x above baseline (300k tok/lang, 1500 steps is tiny). Confirms: corpus
  decides which abilities return; full recovery is a data+compute scale-up, not a method fix.
- Kaggle infra (results file + scratch build_and_push.py): embed src as base64 tarball in a T4 script
  kernel; MUST set machine_shape=NvidiaTeslaT4 (else preinstalled torch has no kernel image), keep the
  preinstalled torch (no pip), unpack to /tmp (clean output). donor/qwen.py now re-.to(device) after
  swap (counter buffers build on CPU) — fixes a cpu/cuda mismatch on GPU donors.
  NEXT: scale the mixed corpus (B-token, pretraining-like) for FULL ability-preserving recovery.

### Honest finding to respect (encoded in tests)
Recovering a *full-precision* donor's own outputs is NOT a single-layer win: the TWN warm-start
already sits near the ternary weight optimum, so per-layer self-update only adds
stochastic-rounding noise. **Real recovery is a network-level effect** (composed layers + task
loss + distillation = Phase 4). Do not write a test asserting a single full-precision layer
self-improves — it does not.

### Gotchas
- Counter layers are **eager-only**: exactly one forward per backward. For eval/measurement
  forwards, wrap in `torch.no_grad()` or the "reused before backward" guard fires.
- Packed kind (`counter_packed`) needs `in_features % 4 == 0`.
- Counter layers are bias-free; `swap_linears_to_counter` preserves a donor bias via
  `CounterLinearWithBias` (modern donors are mostly bias-free anyway).

## Local setup (to continue this work on your own machine)

```bash
git fetch origin claude/finetune-pretrained-model-fwyuun
git checkout claude/finetune-pretrained-model-fwyuun
pip install -e .                     # torch>=2.1, numpy
pip install pytest                   # dev
python -m pytest tests/test_from_dense.py tests/test_convert.py -q
```

Everything worth keeping is pushed to the branch — the cloud container is ephemeral. Do the
gated Gemma download and any GPU finetune locally (HF credentials + disk + CUDA live there).
