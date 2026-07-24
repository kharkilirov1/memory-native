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
  `salient_first` BiLLM-style split, `salient_scope="layer"` global top-K budget,
  `solve_group_state` = the shared one-layer deploy solve,
  `collect_hessians_guided` = GuidedQuant loss-weighted H via `hessian_weighting`).
- `donor/asym.py` — GPTAQ-style cascade calibration (`calibration="asym"`): sequential
  two-tower collection of H_q = X_qᵀX_q and G = X_qᵀX_fp, residual-form target
  w̃ = w + (H_q+λI)⁻¹(G−H_q)w solved by the unchanged v3 solver under H_q.
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
7. **Solver upgrades (2026-07-22, implemented + unit-gated, awaiting A100 warm-PPL
   gates):** `calibration=asym` (GPTAQ-class cascade objective; residual-form target —
   the naive Tikhonov-to-zero form measurably LOST 4.3× network-KL, pinned by a test),
   `salient_scope=layer` (global fp16 budget, same bpw), `hessian_weighting=end_loss`
   (GuidedQuant g=1; collect with [1, seq] batches on CPU — the LM backward graph is
   the memory hog). Gate protocol fix: the 2048-token layerwise gate overfits (base
   train/eval rel-err gap 4–18×; an ICD flip post-pass "won" −20…−49% on train-H and
   LOST +2…+12% on held-out H) — collect a second probe blob with OFFSET and pass
   EVAL_CALIB to the layerwise witness; judge arms on BOTH numbers.
8. **CPU network witnesses (1.5B, blocks 0-1 quantized, KL vs fp on held-out):**
   `salient_scope=layer` carries to held-out (q_proj −20%, k_proj −26% rel H-err;
   attention rows are heterogeneous, MLP neutral) — safe to enable. `asym` at FULL
   strength loses on a SHALLOW cascade (KL 0.392–0.441 vs classic 0.360 @2048tok;
   the true cascade signal there is tiny), `asym_strength=0.5` already beats classic
   on top-1 (77.39 vs 76.90) and CE (3.0985 vs 3.1094) at 8192 tok. Deploy gate on
   A100 = FULL quantization + 524k calibration, arms classic vs asym(c7, s=0.5/1.0);
   chunk=1 is the cleanest objective but costs 2×n_layers calibration passes.
9. **Deploy-scale warm gate DONE (Colab G4, STEPS=0, 524k calib, rebuilt 12M mix —
   results/solver_v3_salient_scope_asym_gate_colab.md):** `salient_scope=layer` WINS
   (aggregate 3.792→3.687; en 77.9→68.0, ru 108→88.7, code 15.0→12.7, math +3%) —
   ENABLE for the next recovery run. asym(s=0.5, c7): ru 108→51.1 (−53%!) but the
   other domains revert/worsen (aggregate 3.746) — cascade correction is real and
   largest where distortion is worst; tune s∈0.25–0.4 / deep-blocks-only next.
   classic arm reproduced the ladder (en 77.9 ≈ historical 74.6 on a fresh val slice)
   — the solve_group_state refactor is non-regressive at deploy scale.
10. **ASYM_STRENGTH sweep DONE — s=0.15 is a STEP CHANGE (same gate):** curve
   0/0.08/0.15/0.20/0.25/0.35/0.5 → metric 3.687/3.202/**3.156**/3.184/3.216/3.320/
   3.746, smooth, unimodal, minimum at 0.15. At s=0.15 EVERY domain beats every other
   config: en 46.75 (classic 77.9, −40%), ru 42.1 (−61%), code 8.28, math 16.85,
   science 35.12, instruct 17.38. Warm EN 46.75 is BELOW the best TRAINED strict
   checkpoint of the previous campaign (47.4 after 6000 KD steps) — solver-only now
   clears the post-recovery bar. Mechanism: tempered target keeps the cascade
   correction inside what the ternary+salient grid can absorb before snapping.
11. **Capacity+iteration sweep DONE (packs 1-2): the s-curve bottleneck IS capacity.**
   `ASYM_PASSES` implemented (multi-pass asym: re-collect on the quantized net,
   re-solve from ORIGINAL weights w0; fp tower + w0 captured once — pinned by test;
   NOTE pass 2 is a bit-exact no-op at chunk=1 on a pure chain, signal comes from
   intra-chunk staleness). Results (layer+asym s=0.15): salient 1/2/3% → metric
   3.156/2.950/2.795 (no saturation; +0.48 bpw per 1%); 2 passes at 2% → **2.899**;
   guided-H standalone 3.687→3.631 (RU-heavy gain, second-order vs asym).
   **NEW DEPLOY DEFAULT (same-format budget): SALIENT_SCOPE=layer CALIBRATION=asym
   ASYM_STRENGTH=0.15 ASYM_PASSES=2 SALIENT_FIRST=0.02** (en 35.6, metric 2.899);
   quality option SALIENT_FIRST=0.03 (en 31.7, metric 2.795, ~3.1-3.6 bpw total).
   Session total vs production solver: 3.792 → 2.795 (~2.7× lower mean PPL), zero
   training. Full sweep tables: results/solver_v3_salient_scope_asym_gate_colab.md.
12. **Recovery run from the s2i2 start DONE (6000 steps, G4):** strict alpha=0 final
   (= best, monotone) **en 30.06, ru 22.39, metric 2.6779** (warm 2.899 → trained
   2.678); vs previous campaign best en 47.4 / ru 65.9. Step ≈ 0.9 s (2× salient
   strict channel; was 0.33 s at 1%) — budget ≈19 units for solve+6k steps.
   Checkpoint NOT persisted (no Drive grant) — rerun is seeded/reproducible.

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
