# Qwen3.8-27B full streaming counter conversion — results

Date: 2026-08-18/19. Box: 20-core CPU, 32 GiB RAM, Windows, C: spinning disk with
pagefile. All numbers measured, not projected.

## Donor

`Qwen/Qwen3.8-27B` = `Qwen3_5ForConditionalGeneration` (VLM): text decoder 64 layers
(48 linear-attention GatedDeltaNet + 16 full-attention at interval 4) under
`model.language_model`, dense SwiGLU MLP 17408, hidden 5120, vocab 248320, untied
lm_head; vision tower + MTP stay fp. Download 51.7 GiB (18 shards, 78 min).

## VLM enablement (code)

- `_skeleton_for_checkpoint(src, config)`: skeleton class taken from
  `config.architectures` first, multimodal auto-class retry second, embed-key probe
  against the real checkpoint decides; loud error on mismatch. Fixes the
  qwen3_5 case where `AutoModelForCausalLM` maps to the text-only class and the
  decoder prefix (`model` vs `model.language_model`) would not match the checkpoint.
- `_build_probe(config, device, cls=None)`: probe built with the SAME top-level class
  as the skeleton so computed-buffer paths align (`model.rotary_emb.inv_freq` vs
  `model.language_model.rotary_emb.inv_freq`); `_shallow_config` also cuts vision
  tower depth to 2 (a 27x1152 tower is ~1.6 GiB of probe nobody reads).
- `restore_witness.py` phases 1-2 both build through `_skeleton_for_checkpoint`.

Witness: `.codex-tmp/tiny_qwen35_smoke.py` — synthetic 2-layer qwen3_5 VLM through
the real `convert_streaming` twice (resume), `load_streamed_state`, full-VLM
phase2-restore with vision uncovered via `extra_skip`, forward finite. ALL GREEN.

## Conversion run

Config (deploy defaults): GROUP=128, C=11, GRID=itf, SALIENT_FIRST=0.02,
SALIENT_SCOPE=layer, IN_SWEEP_REFIT=1, CASCADE=1, CALIB 16x512 tokens from the
4.01M-token mix (en .40 / ru .30 / code .12 / math .08 / science .05 / instruct .05,
Qwen3.8 tokenizer), DEVICE=cpu, DTYPE=fp32, MICRO_BATCH=1.

- 64/64 blocks, 317 targets, 496 counter linears on restore, **15,591,342,080 coeffs**
  (24.35B counted with per-layer linear recount), solver-only, zero training.
- Wall 671 min (11.2 h) + 23-block replay after a restart; steady ~14.5 min/block.
- Peak resident 2.65 GiB during streaming; solve-phase plateau ~18.6 GiB commit
  (fp64 Cholesky workspaces) — flat across block landings after the leak fix below.
- Output: 64 `block_*.pt`, **20.5 GiB** total (~2.68 bpw over 61B total params;
  ternary body 17.4 GiB + scales fp32 + salient fp16 2% + perm int32).

### Memory bugs found at 27B scale (each pinned, fixed, regression-tested)

1. **Commit growth ~0.6-0.7 GiB/block** (would die ~block 50): `block.to('meta')`
   moves registered params/buffers only; the packed counter layer keeps
   `_salient_perm_flat` / `_salient_sparse_cache` as PLAIN attributes.
   Fix: `_release_block()` clears private tensor attributes; resume replay rebuilds
   them from disk via the load post-hook. Commit stayed flat across all block
   landings after the fix (`watch.csv`: 18.56-18.63 GiB through block
   boundaries); tiny witness and all 21 streaming/group-update tests green.
2. **Access violation building fresh counter layers over a resident 20.5 GiB state**:
   the constructor zero-filled a dense [out, in] int16 + int32 pack intermediates
   (~0.9 GiB transients/layer). Fix a: zero state packs to a constant 3-byte
   pattern, `repeat`ed — bit-exact vs the dense path on widths 48, 5120 and
   17408, including both extremes of the model's matrices.
3. **Double residency of the state dict**: construct-zero-then-`load_state_dict`
   copies. Fix b: `PackedGroupScaleCounterLinear(..., state=tensor)` registers the
   solver's packed tensor BY REFERENCE; `restore_witness.py` reference-loads all
   3472 tensors and re-derives salient runtimes. Restore now peaks ~24.5 GiB RSS
   (vs >41 naive) and survives on the 32 GiB box.

## Restore witness + PPL/KL gate (WikiText-2 test, GPTQ protocol)

- Phase 1 (true fp32, block-streamed, 61 min): **fp ppl = 7.407** over 4092 tokens.
- Phase 2 (bf16 fp-parts): 496 counter linears, 0 unexpected keys, 688 fp params
  materialized from shards (vision + embeddings + head + MTP), 3 computed buffers
  from the probe.
- Phase 3: forward finite; top-5 after "The capital of France is":
  `[' Paris', ' the', ' ______', ' known', '\n']`.
- **RESULT: counter ppl = 15.856 vs fp 7.407 (x2.14);
  KL(counter||fp) = 0.8764 nats over 256 positions.**

Context for the number: solver-only, no recovery finetune, first shot at this
donor. The 1.5B classic chain measured x6.7 EN on its (harder, mixed-domain)
protocol; asym s=0.15 cut that to x4.0 and KD recovery to ~x2.6. This run used
classic cascade; the measured next levers, in order: `CALIBRATION=asym
ASYM_STRENGTH=0.15 ASYM_PASSES=2` re-run (~15 h CPU), then KD recovery. CPU
counter-forward of the 27B is slow (~45 min/1024-token window under pagefile
pressure) — full-set WikiText eval needs the MotifCL Vulkan path or a GPU.

## Artifacts

- State: `C:\Users\Kharki\Desktop\donors\out\qwen38_27b\block_0000..0063.pt`
  (+ `manifest.json`, `convert.log`, `watch.csv`, `fp_eval_cache.pt`).
- Donor kept at `C:\Users\Kharki\Desktop\donors\Qwen3.8-27B` (needed for future
  MNCC export / re-gates).
- MotifCL (RX 580) inference wiring deliberately deferred per user decision;
  port ladder remains `motifcl_production/docs/MN_SOLVER_PORT_PLAN.md`.
