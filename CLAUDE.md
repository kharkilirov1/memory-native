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

Donor decision (OPEN — needs the human): **Gemma 4 E2B** (2.3B; needs an adapter for Per-Layer
Embeddings + interleaved sliding-window/global attention + proportional-RoPE; weights are HF
**gated** — accept the license under your account) **vs a small dense Qwen** (near-mechanical
mapping, open weights). Phases 1–2 were kept donor-independent on purpose.

### Plan (phases)
1. **Conversion primitive** — `weight → (scale, ternary, counter)`.  ✅ DONE
2. **Model-level swapper** — replace every `nn.Linear` with a warm-started counter layer.  ✅ DONE (core)
   - remaining: donor-specific HF loader + arch adapter (Gemma PLE/sliding-window/p-RoPE, or Qwen).
3. **MoE import** — fill the stacked `[E,out,in]` expert buffers (`moe_ffn.py`) for a MoE donor.
4. **Recovery finetune** — distillation from the fp teacher (cache teacher logits offline to
   avoid holding it resident) + slowfast fp residual on start + output-calibrated scales.
5. **Validation** — round-trip loss, PPL-recovery curve vs original, memory-gate/shootout.

### Done so far (committed + pushed)
- `d048bd5` Phase 1: `weight_to_counter_state()`, `from_dense`/`from_linear`/`load_dense_weight`
  on the counter layers (packed inherits). Tests: `tests/test_from_dense.py`.
- `4e57949` Phase 2 core: `swap_linears_to_counter(model)` + `CounterLinearWithBias` +
  `SwapReport` in `convert.py`. Tests: `tests/test_convert.py`. Exported from package `__init__`.

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
