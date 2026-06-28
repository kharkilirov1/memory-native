# GPU kernel verification (T4) — the Triton/CUDA kernels run and pass on real hardware

The earlier code review flagged the project's biggest coverage gap: **every Triton/CUDA kernel test
is `skipif(not (CUDA and HAS_TRITON))`, so the kernels never ran in CI** — they were only ever
hand-checked on a T4. This run executes the full suite on a Kaggle GPU so those assertions actually
fire.

## Run — `mn-gpu-tests` (Kaggle, Tesla T4, torch 2.10+cu128, triton 3.6.0)

First attempt landed on a **Tesla P100 (sm_60)**, which the cu128 PyTorch build does not support
(needs sm_70+), so every CUDA kernel launch raised `no kernel image available for execution` — a
hardware/build mismatch, not a code fault (all CPU-runnable tests passed there). Re-pushed with
`--accelerator NvidiaTeslaT4`; on the T4:

```
======================== 87 passed in 76.94s ========================
PYTEST_RETURNCODE 0
```

**All previously-skipped kernel tests now execute and PASS on the T4:**

| test | what it proves on GPU |
|---|---|
| `test_fused_update::test_triton_update_matches_reference` [24-64-11, 512-2048-11, 256-4096-8] | the fused RMS+SR update kernel matches the CPU reference within one SR quantum, across sizes |
| `test_tiled_fused_update::..._refreshes_only_touched_cache_rows` | the row-slice fused update refreshes only the touched T-cache rows (the deep-iterations-v2 patch) |
| `test_triton::test_triton_forward_matches_reference` | decode-in-GEMM forward matches dense |
| `test_triton::test_triton_grad_x_matches_reference` | grad_x straight from packed state matches |
| `test_triton::test_triton_full_backward_trains` / `..._handles_3d_input` | end-to-end Triton backward trains; 3-D input path |
| `test_update_from_io::test_from_io_kernel_bitquantified` | the grad_w-in-kernel (no dense gradient) update is bit-quantified vs the reference |
| `test_review_fixes::test_cache_consistent_after_fused_update` | the cache-rebuild-after-CUDA-fused-update fix (from the review) holds on GPU |

So the kernels are correct on real hardware, and the "bit-quantified, not bit-exact" framing (chunked
fp reduction tips ~O(1) SR boundaries) is confirmed by the GPU assertions — they pass at the
documented within-one-SR-quantum tolerance, exactly as the corrected docstrings say.

## Witnesses re-run on the T4 (sanity that the CPU witnesses hold on GPU)

M1 memory-FFN (same conclusion as CPU): memory-FFN matches dense val-loss at a fraction of active MACs.
```
  dense FFN (fp, AdamW)            val 0.9591  active MACs/tok 131072  persist 1543.5 KiB
  counter-memory FFN E=16384 k=16  val 0.9601  active MACs/tok  26880  persist 6816.0 KiB   (~4.9x less compute)
```
M3 slow-fast: `OVERALL: ALL GATES PASS` on GPU.

## Caveats
- The packaged suite predated the M2/M4 test files, so this run covers 87 tests (kernels + M1/M3 +
  all prior); a follow-up GPU run can include `test_group_counter.py` / `test_moe_ffn.py` (both pass
  on CPU and use no GPU-only path).
- T4 int8 `_int_mm` is exercised indirectly by the int8 forward/update paths; the dedicated int8/int4
  speed numbers live in `results/ACCELERATION.md` / `INT4_WGRAD.md`.
