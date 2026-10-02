# gemma-4-12B ternary restore + inference — the item-15 gate CLOSED (Kaggle T4)

Date: 2026-08-01. Kernel `lirovkharki/mn-gemma12b-ternary-infer` v2 (conversion) + v3
(restore+inference). Donor: `google/gemma-4/transformers/gemma-4-12b` (Kaggle Models,
23.9 GB single shard — the same donor as the CPU-box conversion of item 15).

## Conversion (v2, T4): 48/48 blocks in 66.5 minutes

Fresh streaming conversion ON GPU: real EN calibration (16x512 WikiText via gemma BPE),
deploy-class solver (itf + align + salient 2% layer-scope), `device="cuda"`. All 48
blocks, 10,899,947,520 coeffs, peak resident 1.99 GB, state 9.16 GiB (2296 tensors).
Context: the same conversion took ~10.5 h on the 285-GFLOPS CPU box — **~10x on a free
T4**, memory-boundedness intact. The full-attention no-v_proj quirk handled (layer 47
shows 6 targets — correct, not a skip).

## Restore + inference (v3): the first witnessed 12B restore

The hardened witness path (meta skeleton + shard materialization + probe-path computed
buffers + re-tie), plus the new multimodal auto-skip (gemma-4 unified carries 3
vision/audio linears the decoder-only conversion never touches — they stay fp now
instead of KeyError):

- 328 counter linears rebuilt, 301 params + 53 buffers (15 computed) from shards,
  0 unexpected keys, 0 meta left; peak RSS **28.1 GiB** (fits Kaggle's 29 GB; the naive
  from_pretrained path needs 44+ GiB and died silently on the 32 GiB box — item 15).
- Logits finite. **Top-5 after "The capital of France is": [' Paris', ' the',
  ' located', ' a', ' called'] — the solver-only ternary 12B ranks the fact first.**
- Greedy 30-token continuations: "The capital of France is Paris." then whitespace
  loops; "def fibonacci(n): return n" loops; "the class class class". This is the
  expected WARM class: 8k-token EN-only calib (deploy gates use 524k), zero recovery
  steps, greedy decoding (loop-degenerate even on fp models). The 1.5B witness of item
  14 gave top-5 the/not/a/, — this one is qualitatively ahead (fact first).

## What it means / next

- Item 15's pending gate (restore + liveness at 12B) is CLOSED; the remaining quality
  ladder is the standard recipe, both steps measured elsewhere: 524k mixed calibration
  (+ asym s=0.15 + salient_refit) for the warm floor, then KD recovery (group+dec4
  production recipe) for coherent text.
- Speed ladder for real 12B inference: GPU + T-cache + the inference export pack
  (state -> t, drop c, base-3 repack: ~35 GiB for 12B, ~81 GiB for a 284B V4-Flash) —
  the export utility remains the open utility task.
