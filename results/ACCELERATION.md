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
| M5 | Derived visible cache (`cache_mode={none,fp16,int8}`) | **done (mechanism)** — `CompactCounterLinear(cache_mode=...)` keeps the visible ternary T in fp16/int8 (a derived view, persistent=False, rebuilt from truth state, refreshed on every visible flip in `_write_rows`). Forward routes through `_visible_t` so the GEMM never unpacks 6-bit. Bit-exact vs the decode forward (T=−1/0/1 is exact in fp16/int8) and tracks the state through updates (`test_cache`). Profiler (CPU d=256): forward 1.28→0.29 ms with the cache (decode tax removed, now near the pure GEMM). The int8 *Tensor-Core GEMM itself* is M6 (GPU). |
| M6 | int8 Tensor-Core compute path | open (GPU) — the cache already stores int8 T; what's left is doing `X_int8 @ T_int8` / `Δ_int8 @ T_int8` / `Δ_int8^T X_int8` on the Tensor Cores (`torch._int_mm` / cuBLASLt). `Q(Δ)^T Q(X)` is an unbiased estimator of the update correlation; ships behind a parity gate (numerics change). Needs a T4. |
| M7 | Reversible anchors (`anchor_every`) | **done** — `ReversibleSequence(anchor_every=A)` / `ReversibleGPT(anchor_every=A)`. Stores the activation every A blocks and recomputes each chunk forward from its anchor instead of inverting: skips the inverse pass (~1 fwd/block faster) and is *exact* (no float-inverse error), at O(L/A + A) memory. Gradient parity vs plain autograd verified for A∈{1,2,3,5,8} (`test_anchors`); model trains, counters fire. (the peak-memory/speed frontier number is a GPU benchmark, queued with M5/M6.) `proxy` RMS still open (needs grad_out/x plumbing) |
| M8 | Adaptive update decimation by flip-rate | **done** — `decimate_updates=True`. A near-stable layer (tiny flip-rate) fires the update only every `_dec_period` steps (1→2→4→8 as flip-rate crosses 1e-3/1e-4/1e-5), lr scaled by the period to compensate; grad_x always runs, only the update is skipped. Parity gate (`test_decimation`): recovers the teacher to MSE 0.0 and engages (period→8 once stable); default off leaves the path untouched. Uses the torch update path (kernel doesn't report flip-rate) |

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

All CPU-validatable milestones are landed: **M1, M2, M4, M7, M8 done; M3 partial** (QKV case
covered). Each numerics-changing mode (M4 lagged/lazy, M8 decimation) ships behind a teacher
parity gate and defaults off, so `strict6` / exact-eager / no-decimation remain the default and
the memory + dynamics claims stay intact. Anchored reversible (M7) is exact (no inverse error).

Still open, **GPU-blocked** (wait on the 1B run freeing the T4 quota): M5 derived visible cache
(fp16→int8) and M6 the int8 Tensor-Core compute path — the memo's main long-term speed path —
plus the GPU benchmark numbers for M2 (fused QKV), M4, M7's memory/speed frontier, and the
profiler truth table on a real T4. CPU-side remainders: the general cross-layer shared-activation
handle (M3) and proxy RMS (needs grad_out/x plumbed into the update).
