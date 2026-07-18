# memory-native — project notes

Auto-loaded context. Read this first to resume work cold.

## What this repo is

`memory-native` — finite-state **counter synapses** + **reversible activations** for
memory-efficient training, pure PyTorch (CPU/CUDA, no custom engine), plus an MLX/Metal
port for Apple silicon (`src/memory_native_mlx`).

**What the method saves (the whole point):** it is NOT a weight compressor — it *eliminates
the training-state pools*. For a Gemma-4-E2B-shaped model (35 layers, d=2048, vocab 262K,
seq 2048, batch 8):

| pool | BF16 AdamW | counter |
|---|---:|---:|
| persistent (weights) | 4.28 | 2.24 |
| grad | 4.28 | **0** |
| optim (Adam m/v/master) | **25.68** | **0** |
| activations | 13.12 | **0.47** |
| **total** | **47.4 GiB** | **2.7 GiB** |

The dominant wins: `optim → 0` (no Adam moments, no fp master), `grad → 0`, reversible
activations (~28×). This matters most for finetuning, which is dominated by exactly the
optim + activation pools the method zeroes.

## Key source files

- `src/memory_native/counter.py` — counter layers; carry/saturation via `_carry_resolve`
  (single source of truth — group layers import it).
- `packed.py` — 6-bit storage; `fused_update.py` — row-scale Triton update (hash-SR,
  clip folded into the RMS denominator).
- `group_scale_packed.py` / `group_scale_counter.py` / `group_scale_kernels.py` — the
  group-128 act-ordered packed format: trainable group scales, salient channel (exact fp16
  overrides, frozen base), kernel modes `gemm` (decode+cuBLAS, default via `auto`),
  `triton` (decode-in-GEMM + strict O(out·groups)-scratch update), `torch` (reference).
- `donor/qwen.py` — HF donor loader/swap; `donor/ptq.py` — the calibrated PTQ solver
  (GPTQ-style group ternary v3: act-order, post-sweep s↔t alternation with a monotone
  Hessian gate, `scale_refit="align"` exact joint solve, `grid="itf"` asymmetric grid,
  `salient_first` BiLLM-style split).
- `recovery/distill.py` + `recovery/runtime.py` — KD recovery finetune, resumable runner
  helpers (strict α=0 evaluation, RNG capture, counter-structure restore).
- `glm.py`, `moe_ffn.py`, `reversible.py`, `budget.py` — receiver architecture, MoE,
  reversible blocks, 4-pool memory model.

## Where the work stands

1. **Conversion + swap + recovery pipeline: DONE.** Qwen2.5-0.5B/1.5B donors convert,
   warm-start, and recover. Mixed multilingual corpus (EN/RU/code/math) is REQUIRED —
   a narrow corpus recovers only its own domain and forgets the rest (measured).
2. **Best trained 1.5B floor so far:** EN 71.9 / RU 35.3 / code 12.8 / math 48.8
   (fp baseline 11.6 / 9.2 / 3.0 / 6.7). The wall ordering is data → LR schedule → data.
   Constant-LR plateau = noise ball; cosine decay to ~1e-4 is the recipe. From a good PTQ
   start, begin the counter schedule LOW (cosine 0.002 → 1e-4) — a hot lr damages it.
3. **PTQ solver chain (no training), 1.5B EN warm PPL:** naive 575k → optimal 187k →
   GPTQ row 17.5k → group-128 5.6k → v3 solver ~0.9k-class. Full-chain layerwise gate:
   −22.3% rel. H-error over v3 base. Rotations (QuaRot-style) are measurably HARMFUL for
   the ternary grid (pinned by a documented-negative test); salient/outlier ISOLATION is
   the correct direction.
4. **PTQ-start + recovery compounds:** EN ~75 after 400 steps (~100× less compute than
   the naive path to the same neighborhood).
5. **Kernels (T4-gated):** gemm-mode forward/grad_x 4–14× over decode-in-GEMM with
   bit-exact parity; slim dense update ~950× over the strict from-IO update at M=4096
   (strict is 30–46 s/layer there — unrunnable); step outlook A100 ~2.5–4 s at B8×T512.
6. **Next planned step:** a short A100 baseline (~5 min) to pick STEPS/batch, then the
   full recovery run from the v3-solver start (~3000 steps fits the remaining budget).

## Gotchas (hard-won, keep)

- Counter layers are **eager-only**: exactly one forward per backward; wrap measurement
  forwards in `torch.no_grad()` or the reuse guard fires.
- Packed kinds need `in_features % 4 == 0`; strict Triton group update needs a
  power-of-two group size (guard rejects others before launch).
- Counter layers are bias-free; the swap preserves donor bias via `CounterLinearWithBias`.
- `weight_flips` does NOT increment on fused/Triton paths — use the decoded-sample
  telemetry (`flip_rate_alt`, `observe_flip_sample`) instead.
- Recovering a full-precision donor's outputs is a NETWORK-level effect (composed layers +
  distillation). A single layer does not self-improve from its own optimum — do not write
  that test.
- Evaluate and select checkpoints at strict ternary `alpha=0`; homotopy (`alpha>0`) PPL is
  a diagnostic, not a deployable number.
- Salient channel: base (t, c) is zero and FROZEN at salient entries; checkpoints with
  salient load into fresh layers (buffers resize on load).
- torch.compile of the update chain: `dynamic=False` only (dynamic emits slower code);
  SR stays eager to preserve the RNG stream.

## Setup

```bash
pip install -e .            # torch>=2.1, numpy
pip install -e .[donor]     # + transformers/safetensors/accelerate for donor work
pip install pytest
python -m pytest tests/ -q  # CUDA/Triton tests skip on CPU-only boxes
```

Recovery runs live in `scripts/` (`run_ptq_recovery.py` is the resumable v3 runner;
env-driven: MODEL, DATA_DIR, CKPT_DIR, STEPS, GRID, SALIENT_FIRST, …). Results and
measurement protocols are under `results/` — treat them as the evidence record.
