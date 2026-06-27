# Acceleration milestones — status

Tracks the acceleration memo (memory-native → also GEMM/Tensor-Core-native). The thesis: the
6-bit state is *truth/optimizer*, but it must not also be the *compute layout* — forward/grad_x/
update correlations belong on Tensor-Core GEMM, with the visible ternary weight kept as a derived
cache and the counter transition as a fused epilogue. Strict sub-byte memory stays the default
(`strict6`); the speed modes are opt-in so the memory claim stays honest.

| # | Milestone | Status |
|---|---|---|
| M1 | Layer profiler truth table | **done** — `scripts/layer_profiler.py` (per-phase: forward, GEMM fwd/grad_x/grad_w, decode, pack, act-quant, update torch vs fused) |
| M2 | Fused QKV counter layer | **done** — `CounterQKVLinear` (d→3d); bit-identical to three separate layers (test), one saved activation + one update + one larger GEMM. Opt-in via `ReversibleGPT(fused_qkv=True)` |
| M3 | Shared activation handle | **partial** — the QKV case (the important one) is subsumed by M2 (one layer ⇒ one saved activation). A general cross-layer handle is still open |
| M4 | Lagged RMS one-pass + lazy scale rebase | **done** — `rms_mode={exact,lagged}`, `scale_rebase={eager,lazy}` on RMSCounterLinear. lagged uses last step's v, lazy rebases the counter at the next read via a per-row `s_base`, so the tick needs no row-stat of the current grad → one pass. Parity gate: all 4 combos recover the teacher to MSE 0.00000 (`test_lagged_rms`). Fused kernel stays exact/eager-only; other modes use the torch path. (`proxy` RMS = M7, needs grad_out/x plumbing) |
| M5 | Derived visible cache (`cache_mode={none,fp16,int8}`) | open — the central pivot; keep the cache outside truth state, refresh on visible flips |
| M6 | int8 Tensor-Core compute path | open — `Q(Δ)^T Q(X)` unbiased; needs GPU + a parity gate (numerics change) |
| M7 | Reversible anchors (`anchor_every`) | **done** — `ReversibleSequence(anchor_every=A)` / `ReversibleGPT(anchor_every=A)`. Stores the activation every A blocks and recomputes each chunk forward from its anchor instead of inverting: skips the inverse pass (~1 fwd/block faster) and is *exact* (no float-inverse error), at O(L/A + A) memory. Gradient parity vs plain autograd verified for A∈{1,2,3,5,8} (`test_anchors`); model trains, counters fire. (the peak-memory/speed frontier number is a GPU benchmark, queued with M5/M6.) `proxy` RMS still open (needs grad_out/x plumbing) |
| M8 | Adaptive update decimation by flip-rate | open — late-training speed mode |

## Profiler reading (illustrative, CPU d=256 M=512)

```
forward (decode+GEMM)   0.998 ms     pure GEMM fwd  x@W^T    0.140 ms
update torch tile       3.119 ms     pure GEMM grad_w go^T@x 0.151 ms
decode unpack+state     0.456 ms     act quant int4         0.792 ms
```

The decode is ~7× the matmul here and the torch update dominates — i.e. the wall is the decode
tax (→ M5 derived cache) and the update (→ the fused kernel, already ×45.9, and the strict
update-from-IO). Re-run on a T4 (`--device cuda`) for the numbers that drive the kernel work.

## Sequencing note

M1/M2/M3 are layout changes with **no numeric change** (M2 is bit-identical) — landed first.
M4–M6 change numerics (unbiased, higher variance) and trade a little memory for the cache, so each
ships behind a parity gate with `strict6` remaining the default. GPU-dependent milestones (M5/M6,
and the GPU benchmark of M2/the profiler) wait on free quota.
