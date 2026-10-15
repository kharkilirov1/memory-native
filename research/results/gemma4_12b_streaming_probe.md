# gemma-4-12B streaming conversion — first real run, 2026-07-27

Two blocks of `google/gemma-4-12B` converted end to end on CPU. The point was to
find what breaks at real size before spending GPU quota. Four defects surfaced,
all of them on the critical path; the driver now runs the donor.

## Result

| | |
|---|---|
| blocks | 2 of 48, `cascade=True`, `micro_batch=1` |
| calibration | 8 x 512 tokens from the new gemma-BPE mix |
| targets | 7 per block (q/k/v/o + gate/up/down), 0 non-text |
| coefficients | 448,266,240 |
| wall | 23.7 min = **710 s/block** -> 48 blocks ≈ **9.5 h** CPU |
| tensor peak | 1.66 GiB |
| process RSS | ≈ **15.9 GiB** (sampled externally; the in-script sampler read 0 and is not to be trusted) |
| state written | 385.9 MiB for 2 blocks -> ≈ 9.3 GiB for 48 |

710 s/block against the 590 s/layer the synthetic solve probe predicted: the
extra 2 min is Hessian collection plus the cascade re-run, not solver cost.

Format, measured per linear rather than assumed — `q_proj` [4096, 3840]:

| tensor | shape | dtype |
|---|---|---|
| `.state` | [4096, 2880] | uint8 — 6 bits/weight packed |
| `.scale` | [4096, 30] | fp32 — one per group of 128 |
| `.salient_idx` / `.salient_val` | [314573] | int32 / fp16 — exactly 2% |

≈ 7.2 bits/weight all in. This is the TRAINING format; the counter state is what
makes the method trainable, and shrinking it to a deploy-only ternary body is a
separate question this run does not answer.

## What was broken

1. **Hardcoded decoder path.** The driver assumed `skeleton.model.layers` and a
   `model.` checkpoint prefix. gemma-4 keeps the text tower at
   `model.language_model`, so it raised AttributeError before reading a block —
   and the prefix is also the safetensors name prefix and the state-dict prefix,
   so a wrong value would have asked the file for keys that do not exist. Fixed
   by `_resolve_decoder`, which finds the module carrying both a non-empty
   `.layers` and `.embed_tokens` and returns its dotted path; it refuses rather
   than guesses when there is none or two at the same depth.

2. **The probe built the whole model.** `_build_rotary` instantiated the donor at
   full depth on a real device just to read `rotary_emb`. For gemma-4-12B that is
   ~24 GiB — more than this 32 GiB box has free, and the exact cost streaming
   exists to avoid. Now built from a one-layer config: no computed buffer depends
   on depth (rotary comes from head_dim/rope_theta, gemma's `embed_scale` from
   hidden_size), so depth is the safe dimension to cut.

3. **Computed buffers had no source.** `_materialize` is strict — it refuses to
   run on uninitialized memory — and it correctly stopped on
   `model.language_model.embed_tokens.embed_scale`, which is `sqrt(hidden_size)`
   and lives in no checkpoint. The same class as the rotary `inv_freq` gotcha.
   The criterion is now PyTorch's own: a buffer registered `persistent=False` is
   by construction absent from every state dict, so those are filled from the
   one-layer probe and never looked up on disk. gemma-4 has five of them.

4. **One rope per layer type.** gemma-4 interleaves sliding and full attention and
   keeps one `inv_freq` set per kind, selected by a `layer_type` argument;
   calling the rotary without it dies on `None_inv_freq`. The value lives on the
   attention submodule, not the decoder layer, so the lookup walks the block.

## Guard added, not silently worked around

Blocks are run bare. With `attention_mask=None` the HF attention path is already
causal — checked directly: editing the last token leaves the first token's output
bit-identical (max|diff| = 0). But it is FULL causal, not sliding. For sequences
inside the window the two masks coincide exactly; beyond it, sliding layers would
attend to the whole prefix and their Hessians would describe a model the donor is
not. `_assert_window_covers` refuses that case instead of computing it wrong.
gemma-4-12B's window is 1024, so calibration sequences must be <= 1024.

## Correction to a previously recorded number

CLAUDE.md item 14 said the streamed Qwen2.5-1.5B ternary body was "63.0 MiB".
That was measured with a filter matching `counter.state`. Only the 84
bias-carrying attention linears nest inside `CounterLinearWithBias` and pick up
that `.counter.` infix; the 112 bias-free ones end in plain `.state` and were
skipped. Recomputed on the same output directory: **937.1 MiB** of `.state`,
1130.6 MiB with scales and salient. The old figure was 15x too low.

## Still open

- **`scripts/convert_streaming.py` imports `MixCorpus` from
  `memory_native.recovery.runtime`, which does not exist.** The CLI cannot run
  with `DATA_DIR` set. The probe read the corpus bins directly to get around it.
- Process RSS ≈ 15.9 GiB is dominated by the one-layer probe (262144 x 3840
  embeddings plus the vision and audio towers `from_config` builds anyway). That
  is the number to beat before this runs on a 30 GiB Kaggle box.
- GPU speed is still unmeasured; 9.5 h is the CPU figure.
