# Donor selection for the next conversion run — measured, 2026-07-27

Decision: **google/gemma-4-12B**. The MoE candidates were priced first and dropped on
cost; this file records the numbers so the choice does not have to be re-derived.

All solve timings are the real `solve_group_state` at the model's real `[out, in]`,
on synthetic weights and a synthetic Hessian, deploy config

    group=128 C=11 grid=itf salient_first=0.02 salient_scope=layer
    in_sweep_refit=True scale_refit=align

on this box (10 threads, torch 2.12.1+cpu). They are NOT extrapolated from the
DeepSeek-class figure in CLAUDE.md — that extrapolation was tried first and was wrong
by ~8x, because the sweep is not a clean power law in `in^2 * out`.

## The pick: gemma-4-12B

`Gemma4UnifiedForConditionalGeneration`, model_type `gemma4_unified`. Text tower
48 layers, hidden 3840, intermediate 15360, vocab 262144, `gelu_pytorch_tanh`,
16 heads / 8 KV, head_dim 256, sliding window 1024. On disk: a SINGLE
`model.safetensors`, 22.3 GiB, not gated.

| linear | shape [out, in] | solve | H size |
|---|---|---:|---:|
| q_proj | 4096 x 3840 | 48.96 s | 0.05 GiB |
| k_proj | 2048 x 3840 | 31.14 s | 0.05 GiB |
| v_proj | 2048 x 3840 | 40.48 s | 0.05 GiB |
| o_proj | 3840 x 4096 | 34.71 s | 0.06 GiB |
| gate_proj | 15360 x 3840 | 86.64 s | 0.05 GiB |
| up_proj | 15360 x 3840 | 115.25 s | 0.05 GiB |
| down_proj | 3840 x 15360 | 232.71 s | **0.88 GiB** |
| **per layer** | | **589.89 s** | |
| **x48 layers** | | **7.87 CPU-h** | |

Plumbing checks, all run rather than assumed:

- `_target_paths` finds 7 linears per layer (336 for the full model) plus 3 multimodal
  projectors — `embed_vision.patch_dense`,
  `embed_vision.multimodal_embedder.embedding_projection`,
  `embed_audio.embedding_projection`. The runner's existing `VISION_SKIP` drops exactly
  those three and leaves 0 non-text targets. Converting them would be wrong: a text-only
  corpus gives them no calibration signal.
- `_assert_no_unhandled_moe` passes (dense model, nothing to miss).
- `_WeightSource` in `donor/streaming.py` already handles a single-file checkpoint with
  no `model.safetensors.index.json`, which is how this donor ships.
- `run_ptq_recovery.py` already loads it: `_load_donor` falls back to
  `AutoModelForMultimodalLM` when `AutoModelForCausalLM` refuses.

The 0.88 GiB `down_proj` Hessian x 48 layers is why `_hessian_chunks` exists; that path
is on the critical route for this donor and has not been exercised at this size.

## The MoE candidates, priced and dropped

Same protocol; one "expert" = gate_up + down.

| donor | disk bf16 | experts (E x L) | solve | note |
|---|---:|---:|---:|---|
| OLMoE-1B-7B | 12.9 GiB | 64 x 16 = 1024 | 7.7 CPU-h | cheapest real MoE |
| Qwen1.5-MoE-A2.7B | 26.7 GiB | 60 x 24 = 1440 | 15.3 CPU-h | + shared expert |
| DeepSeek-V2-Lite | 29.3 GiB | 64 x 27 = 1728 | 15.9 CPU-h | see routing note |
| Mixtral-8x7B | 87.0 GiB | 8 x 32 = 256 | 46.9 CPU-h | disk-infeasible |
| Qwen3-30B-A3B | 56.9 GiB | 128 x 48 = 6144 | 46.7 CPU-h | |
| **gemma-4-26B-A4B** | 48.1 GiB | 128 x 30 = 3840 | **72.7 CPU-h** | not discovered, see below |
| **Qwen3.6-35B-A3B** | 67.0 GiB | 256 x 40 = **10240** | **75.5 CPU-h** | discovered, ready |

The two current-generation MoEs cost ~9.5x the dense 12B for the same solver. That is
the whole reason for the pick.

Architecture facts established while pricing them, worth keeping:

- In transformers 5.13 **every** one of these keeps experts as STACKED parameters
  (`mlp.experts.gate_up_proj`), including OLMoE — there is no live legacy-ModuleList
  donor among them.
- `router_indices` is present in every MoE modeling file EXCEPT `deepseek_v2`, so
  teacher-forced routing (what `calibration=asym` needs on MoE) does not apply to
  DeepSeek-V2-Lite as implemented.
- **gemma-4-26B-A4B is not discovered by `_stacked_moe_targets`**: its experts use
  `GELUTanh` and the gate at `ptq.py:661` hard-requires `silu`. This is not a silent
  hole — `_assert_no_unhandled_moe` raises and refuses the conversion, exactly as
  `efb9879` intended. Taking that donor later needs the act gate widened to GEGLU and
  the `gate_up` layout of gelu experts verified.

## Open item: a crash that would not reproduce

The first cost probe died in `align_scales_output` at
`torch.linalg.solve` — "Pivots given to lu_solve must all be greater or equal to 1",
i.e. LAPACK `getrf` failed and left invalid pivots. It has not recurred since:

- the full sweep re-run with per-dim seeding: 20 solves (5 models x 2 matrices x
  full-rank and rank-deficient H), zero failures;
- a byte-faithful replay of the original RNG sequence — same single global seed, same
  model order, same draw order inside each call: 10 solves, zero failures;
- hand-built degenerate inputs, none of which reproduce it: H with inf, H with nan,
  H all zeros (dead expert), H of rank 1, T all zeros, T with no negative entries,
  H scaled by 1e20 and by 1e-20.

A batched `linalg.solve` does raise on an exactly-singular batch element, but with a
different and clearer message ("the input matrix is singular"), so that is not what
happened either.

Conclusion: not diagnosed, not reproducible, seen once in ~30 solves. It is still worth
a seatbelt before any multi-hour unattended run — a guard that changes nothing when the
solve succeeds and degrades to a warned fallback when LU fails — but it should be
labelled as exactly that, not as a fix for a understood defect.
