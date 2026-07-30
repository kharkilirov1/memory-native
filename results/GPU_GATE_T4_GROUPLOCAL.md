# T4 gate — fused one-launch group-local update (Kaggle, measured)

Date: 2026-07-30. Kernel: `lirovkharki/mn-grouplocal-gpu-gate` v4-v6, Tesla T4 x2,
torch 2.10.0+cu128, triton 3.6.0. Source = the working branch (dataset
`mn-grouplocal-src`). Gate driver: `scripts/benchmark_group_kernels.py` (`[L3]` arm),
tests: `test_grouplocal_update.py` (9) + `test_dense_update_cuda.py` +
`test_group_scale_kernels_cuda.py` + `test_gemm_mode.py` — **all green on GPU**.

## Correctness + memory: GATES PASS

- **Quanta parity is essentially exact**: code mismatch vs the group-local torch oracle
  7.3e-08…9.5e-07 (single-digit codes out of 13.7-52M), scale max|Δ| ≤ 6e-08 — far inside
  the dense-kernel contract class. The tl.split/reshape lane packing works on triton 3.6.
- **Peak memory is the lowest of any GPU update path**: 152-159 MiB (M=512) / 377-435 MiB
  (M=4096, dominated by the bf16 x_perm copy + inputs) vs dense 244 / 570 MiB and semi
  1204-1560 MiB. No [out,in] grad_w, no fp32 input casts.

## Speed: two iterations measured, honest status

| shape (M, N, K) | strict | dense L2 | fused v4 (gather-in-kernel) | fused v5/v6 (x_perm + BLOCK_M=64) |
|---|---:|---:|---:|---:|
| 512, 8960, 1536 | 1233 ms | 5.6-7.2 ms | 42.8 ms | **11.6 ms** (0.5x dense) |
| 512, 1536, 8960 | (n/m) | 6.2 ms | (n/m) | **8.0 ms** (0.8x dense) |
| 4096, 8960, 1536 | 22 724 ms | 27.6-32.4 ms | 287.8 ms | **53.7-59.0 ms** (0.5x dense) |
| 4096, 1536, 8960 | 46 500 ms | 25.3-27.7 ms | 819.3 ms | **56.7-67.4 ms** (0.4x dense) |

- v4→v5 lesson (pinned): the in-kernel gather through perm was the dominant pathology
  (H3 confirmed at kernel scale) — one coalesced `x_perm` copy + BLOCK_M 32→64 bought
  3.7-14x (worst case 819→57 ms).
- fp16-vs-bf16 hypothesis CLOSED NEGATIVE (v6): fp16 tiles are not faster on T4
  (70-75 ms vs 59-67) — the remaining ~2x gap to dense is NOT tensor-core dtype
  emulation. It is structural: each of the G group-programs re-reads its whole
  [M, BLOCK_N] go stripe (G-fold redundancy) and runs a serial M loop; dense's cuBLAS
  reads each operand ~once with L2 tiling. Next levers, in order: (1) decimation grid
  restriction (below), (2) multi-group programs (BLOCK_K spanning 2-4 groups amortizes
  the go stripe), (3) split-M cooperative scheme, (4) Ampere+ (native bf16 TC).

## Why 0.4-0.8x of dense is already deployable arithmetic

The fused kernel is the only path that supports **decimation** (row-scope statistics
forbid it): launching the grid over 1/4 of the groups cuts its cost ~4x at the same
integrated signal (dec4+lr×4), i.e. **~14-17 ms vs dense's 26-32 ms at M=4096 — faster
than dense, at lower peak memory, with the KD-pregate quality edge (−29.7% vs −15.9%)
and the decimation quality edge on top.** The grid restriction is a trivial kernel
change (offset/stride on program_id(1)); it is the next kernel task.

## Session quota spent

4 T4 runs ≈ 1-1.5 h of the weekly 30 h.
