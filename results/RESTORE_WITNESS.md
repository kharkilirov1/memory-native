# Hardened restore witness — machinery proven end-to-end at 0.5B, ready for 12B

Date: 2026-07-30. Script: `scripts/restore_witness.py`. Target gate: CLAUDE.md item 15
(gemma-4-12B restore + PPL/KL vs fp — pending; the first attempt died silently).

## Why the naive restore dies on the 12B box

`from_pretrained(dtype=fp32)` holds the ENTIRE fp donor (44.6 GiB for gemma-4-12B) before
the swap starts. The witness never does that:

- **phase 1**: fp reference (per-token NLL + logits at sampled positions for KL) computed
  BLOCK-STREAMED from the safetensors shards with the conversion's own machinery
  (`_WeightSource` — plain file I/O, the Windows-mmap-safe reader; `_run_block`; config
  probe for rotary). Cached to disk. Peak = one block + activations.
- **phase 2**: meta skeleton → `restore_counter_structure` from the streamed state →
  every remaining meta tensor materialized straight from the shards; computed tensors
  (rotary `inv_freq` class) resolve from the config probe BY FULL PATH (depth remapped to
  `layers.0`), never from the checkpoint — `to_empty()` garbage is a pinned trap; tied
  lm_head re-tied after materialization (named_parameters dedup skips shared tensors).
  Refuses to proceed on ANY unexplained meta tensor or unexpected state key.
- **phase 3**: finiteness probe, top-5 continuation, PPL + KL vs the phase-1 cache with
  logits computed in chunks ([tokens, vocab] never materializes).
- `faulthandler` + per-phase RSS logging: a silent death cannot stay silent.

## End-to-end witness on Qwen2.5-0.5B (this box, real streamed state)

Chain exercised for real: `convert_streaming` (24/24 blocks, RANDOM-id calibration —
plumbing, not quality; peak resident **0.14 GiB**, 311 MB state on disk, 100.5 min on 4
shared cores) → `restore_witness.py`:

- phase 1: 24/24 fp blocks streamed, peak RSS 1.32 GiB; fp ppl 17.41 (seq 512, 2048
  test tokens, bf16 shard dtype);
- phase 2: 168 counter linears rebuilt (357.8M coeffs), `missing` fully accounted (51 =
  shard-side tensors), 122 params + 2 computed buffers materialized, **0 meta left,
  0 unexpected keys**;
- phase 3: logits finite; counter ppl 856 vs fp 17.4, KL 3.77 nats — the expected
  outcome of a random-calibration sym-grid solve (the point was the machinery; a real
  conversion uses the deploy solver config and real calibration data).
- Peak RSS across restore+eval: **2.89 GiB** on a model whose naive restore path costs
  ~4.5 GiB — the ratio is what matters: for 12B this path projects ~12-13 GiB peak vs
  ~44 GiB naive.

Debug ledger (all pinned in code now): 1-D windows fed to blocks reshaped heads into the
rotary dim; shard-dtype vs head-dtype mismatch at the logits matmul; leaf-name computed-
buffer lookup missed `original_inv_freq`; tied lm_head left meta by named_parameters dedup.

## Running the real 12B gate (user's box, 32 GiB RAM)

```
MODEL=<gemma-4-12b snapshot dir> STATE_DIR=<streamed out_dir> EVAL_TOKENS=8192 \
  SEQ=1024 KL_POSITIONS=256 PYTHONPATH=src python scripts/restore_witness.py
```

Phase 1 ≈ donor-shard read + 48 block forwards over 8k tokens (hours on CPU, one-off,
cached); phases 2-3 need the restored model resident (~11-12 GiB). If phase 2 still dies,
the RSS log now shows exactly where.
