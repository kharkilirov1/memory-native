# gemma-4-12B — full 48-block streaming conversion, 2026-07-29

First complete conversion of a 12B multimodal donor, entirely on this 32 GiB
CPU box. Solver-only; no recovery finetune. Quality numbers (PPL/KL vs the fp
donor) are NOT in this record — only the conversion itself and its witnesses.

## Result

| | |
|---|---|
| blocks | **48 / 48**, manifest complete, none skipped |
| errors across both runs | **0** (no MKL, no access violation, no solver RuntimeError) |
| peak resident | **1.86 GiB** — the donor's fp weights are 22.3 GiB |
| output | 9.2 GB counter state (48 `block_NNNN.pt` + manifest) |
| wall | ~11 h (blocks 0–45) + ~3 h resume (replay 46 + solve 2) |
| per block | ~13 min solve+swap; replay-only ~45 s |
| calibration | 16 x 512 tokens, gemma-BPE mix (RU 30%), sequences <= the 1024 sliding window |
| config | group=128, C=11, itf, salient 2%, salient_scope=layer, in_sweep_refit, cascade |

The peak-resident number is the point of the exercise: memory stayed set by the
block width, not the model size, for the entire depth — 12x under the donor's
own weight footprint.

## The resume was exercised for real, not just designed

The first run died at block 46 (the `shared_kv_states` defect below). The
resume replayed blocks 0–45 through their already-converted counter layers at
~45 s/block — the cascade requires it, block i+1 calibrates on converted
block i's outputs — then solved 46–47. Interrupted long conversions losing
nothing is now a measured property, not a claim.

## Defects found by this run (all fixed, all with the real cause named)

1. **Batched-LU MKL failure in the scale-alignment solve.** `torch.linalg.solve`
   over a batch of small systems floods `Intel oneMKL ERROR: Parameter 6 was
   incorrect on entry to SLASWP` (the pivot array), which surfaced three ways:
   the never-reproduced "Pivots must be >= 1" RuntimeError, a process-killing
   access violation mid-run, and the possibility of silently wrong scales.
   Intermittent, so hand-built degenerate cases all passed. The matrix is a
   Gram matrix in the ternary-code basis — symmetric PSD by construction — so
   Cholesky (`cholesky_ex` + `cholesky_solve`) is the right factorization on
   the merits and avoids the pivot path entirely. The exact solve that produced
   the flood (down_proj: 3840 systems of 240x240) runs clean in 242 s.

2. **Intermittent access violation reading the checkpoint through mmap.**
   safetensors returns views over its memory map; on this box (safetensors
   0.8.0, torch 2.12.1, Windows) pulling a large tensor out of the 22 GiB file
   dies inside `UntypedStorage.__getitem__` — no exception, the process just
   ends — and the identical call can succeed minutes earlier. A 256 MiB
   threshold was not enough (block weights at 118 MiB also faulted). The reader
   now uses plain file I/O + `torch.frombuffer` for every tensor described by
   the header; the file itself was verified intact (size matches the header
   exactly, all 677 tensors).

3. **`shared_kv_states=None` at block 46.** The last layer of each attention
   type in gemma-4 has `store_full_length_kv=True` and WRITES its K/V into a
   caller-supplied dict; running blocks bare passed None and died 46 blocks in,
   because only two layers in the model do it. `_run_block` now passes a fresh
   dict per block — correct precisely because `num_kv_shared_layers=0`, nothing
   ever READS across blocks. Donors that declare real KV sharing are refused
   outright: block-at-a-time conversion is not sound there.
   Correction to the earlier probe note: the missing `v_proj` on layers
   5, 11, ..., 47 is NOT KV sharing — those full-attention layers use
   `value_states = key_states` (K reused as V). Predicted block 47 would
   convert with 6 targets; it did.

4. **Two avoidable full-size copies.** `_materialize` asked the source to cast
   (a second fp32 embedding on top of the destination), and the computed-buffer
   probe was built at full depth and full vocab (~24 GiB for this donor —
   more than the box has). Probe is now 1 layer x 2048 vocab (0.85 GiB, every
   computed buffer bit-identical to the full-vocab probe, checked), and the
   embedding table is loaded as only the rows the calibration touches, with the
   module's own forward kept — gather+scale matches the module exactly (0.0);
   dropping `embed_scale` would err by 6.22, so no hand-rolled lookup.

## Open

- Restore witness (rebuild through `restore_counter_structure`, top-5
  continuations on CPU) not yet green: the first attempt died before its first
  output line; undiagnosed. Restored model needs ~11–12 GiB resident.
- PPL/KL warm gate vs the fp donor: not run (CPU ~2–4 h, or minutes on a GPU).
- GPU speed for the solve remains unmeasured.
