# Design — Qwen2.5-0.5B → counter warm-start + recovery finetune

Date: 2026-07-06
Status: approved (chat), implementation via TDD

## Goal / witness

Take a real pretrained open-weights donor (**Qwen2.5-0.5B base**), warm-start it into the
counter format (Phase 1–2 already done), and **recover quality with a distillation finetune**
(Phase 4). The end-to-end witness is a **PPL-recovery curve** on a held-out slice:

- `PPL(fp Qwen)` — baseline (donor's own quality),
- `PPL(counter, warm-start, pre-distill)` — degradation from ternarization (expected large),
- `PPL(counter, post-distill)` — recovered, approaching the fp baseline.

Recovery is a **network-level effect** (composed layers + task/distill loss), per the honest
finding recorded in the project notes: a single full-precision layer does NOT self-improve.

## Decisions (locked)

| dimension | choice | why |
|---|---|---|
| donor | Qwen2.5-0.5B **base** | dense, open weights (not gated), near-mechanical map; base = meaningful PPL |
| scope | full end-to-end recovery | Phase 2-remaining + Phase 4 + Phase 5 |
| compute | Colab/Kaggle **T4 (bf16)** | no local CUDA; matches project history (T4 sweeps, fineweb script) |
| distill | resident fp teacher, **online KL** (B), with a `TeacherSource` seam for offline cache (C) | 0.5B teacher (~1 GiB fp16) fits beside student on 16 GiB; offline cache is a later scale knob |
| receiver | **in-place swap of the HF model** (no MNGLM) | `swap_linears_to_counter` walks any `nn.Module`; HF already implements GQA/RoPE/RMSNorm/SwiGLU/KV-cache |

## Architecture

Counter layers are a drop-in replacement for `nn.Linear` (same `x → y`). So the "arch adapter"
for a dense donor is almost empty: we replace the transformer-body linears
(`q/k/v/o_proj`, `gate/up/down_proj`) **inside the loaded HF `Qwen2ForCausalLM`** and leave the
donor's own embeddings, norms, RoPE, attention and LM head intact and fp.

All architecture numbers (`n_kv_heads`, `head_dim`, `rope_theta`, `intermediate_size`, tied
embeddings, per-layer bias) come from the donor's `config.json` — **nothing is hardcoded**.

## Components (new)

- `src/memory_native/donor/qwen.py`
  - `load_qwen_donor(name="Qwen/Qwen2.5-0.5B", dtype=...) -> (model, tokenizer)`
  - `qwen_to_counter(model, *, kind="counter_rms", **ckw) -> SwapReport`
    - `skip=["lm_head"]` (tied with `embed_tokens`; keep it fp + trained by AdamW),
    - `keep_bias=True` (Qwen2 q/k/v carry a bias → preserved via `CounterLinearWithBias`),
    - disable gradient checkpointing (counter is eager-only: one forward per backward).
- `src/memory_native/recovery/distill.py`
  - `TeacherSource` (protocol) + `ResidentTeacher` (holds the fp donor, returns logits).
  - `distill_finetune(student, teacher, batches, *, steps, kd_alpha, temperature, lr, ...)`:
    forward student (counter self-updates in backward), loss = `KL(student‖teacher)/T² + α·CE`;
    fp params (embed/final norm/lm_head) trained by AdamW; optional slowfast fp-residual on start.
- `src/memory_native/eval/ppl.py`
  - `perplexity(model, batches) -> float` (no_grad, counter-safe).

## Data flow

corpus text → tokenizer → batches → {teacher logits (resident), student logits} →
KD+CE loss → AdamW on fp params + in-backward counter self-update → periodic PPL on held-out.

## Tests (TDD, CPU, no network)

- `tests/test_qwen_donor.py` — build a **tiny** `Qwen2Config` (transformers instantiates a random
  model locally, no download); `qwen_to_counter` swaps every body linear, `lm_head`/`embed`
  stay fp, q/k/v bias preserved, forward runs, and pre-distill logits differ from fp (degradation
  is real). Assert the eager-guard does not fire in a standard forward.
- `tests/test_distill.py` — on the tiny model, one `distill_finetune` step **decreases** the
  student↔teacher KL (convergence on a toy).

## Artifacts

- `scripts/finetune_qwen_counter.py` — self-contained runnable (real download + T4 finetune).
- `notebooks/qwen_recovery_t4.ipynb` — Kaggle/Colab, bf16, in the style of `fineweb_1b5_glm.py`.

## Dependencies / risks

- Add `transformers`, `safetensors`, `accelerate` as an optional extra `[donor]` in
  `pyproject.toml` (core stays torch+numpy only).
- Risk: HF `Qwen2Attention` under SDPA/checkpointing might call a projection other than exactly
  once and trip the counter eager-guard. Mitigation: covered by `test_qwen_donor.py`; if it
  fires, add a thin per-projection wrapper. Eager attention path is the safe default.
- Real weights download + finetune run on T4 only, not locally.

## Out of scope (explicit)

Gemma 4 E2B donor and its adapters (PLE / sliding-window / p-RoPE), offline logit cache, and
MoE import (Phase 3) — deferred; the `TeacherSource` seam keeps the offline cache cheap to add.
