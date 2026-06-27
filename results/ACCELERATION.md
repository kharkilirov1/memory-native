# Acceleration milestones — status

Tracks the acceleration memo (memory-native → also GEMM/Tensor-Core-native). The thesis: the
6-bit state is *truth/optimizer*, but it must not also be the *compute layout* — forward/grad_x/
update correlations belong on Tensor-Core GEMM, with the visible ternary weight kept as a derived
cache and the counter transition as a fused epilogue. Strict sub-byte memory stays the default
(`strict6`); the speed modes are opt-in so the memory claim stays honest.

## GPU-measured (Tesla T4, d=2048, M=4096) — what actually moved

```
forward  decode + cuBLAS GEMM        10.60 ms
forward  int8 cache + cuBLAS GEMM     7.71 ms   (cache removes the 6-bit decode tax, -27%)
forward  int8 _int_mm, GEMM only      3.24 ms   (raw Tensor-Core GEMM, NOT a usable forward)
forward  int8 correct (row-scale)     5.18 ms   *** x2.05 vs decode -- the honest int8 forward ***

update   torch tile                   7.57 ms
update   fused Triton kernel          0.42 ms   x17.6 (the update transition)

reversible O(1)        peak 427 MiB   818 ms/step
reversible anchor=8    peak 769 MiB   715 ms/step   (-13% time for +340 MiB -- the M7 knob)
```

The validated headline is **the int8 Tensor-Core forward at ×2.05 over the decode path** (the memo's
pivot: keep T as a derived int8 cache and run the GEMM on the integer Tensor Cores instead of
unpacking 6-bit). The raw `_int_mm` is ×3.3, but a *correct* forward needs the per-token (row) scale
quantize epilogue (a per-column scale cannot be pulled out of X T^T), which costs ~2 ms — so ×2.05
is the honest number, not ×3.3. Honest negatives the same run found:
- **int8 update correlation lost to fp32 cuBLAS** (8.21 vs 7.34 ms): the per-call stochastic
  quantize overhead outweighs the int GEMM at this shape. It only pays off by *reusing the
  already-int8-saved activation* (act_save_bits=8) instead of re-quantizing — the open refinement.
- **fused-QKV forward showed no speedup** (27.2 vs 27.7 ms): the forward is decode-bound, so 3×
  d→d and 1× d→3d decode the same weights. M2's win is fewer saved activations + fewer update
  launches (backward), and it *composes* with the int8 cache (which removes the decode floor).

Bench: `scripts/gpu_acceleration_bench.py` (log `gpu_acceleration_bench_T4.log`). One T4; directional.

| # | Milestone | Status |
|---|---|---|
| M1 | Layer profiler truth table | **done** — `scripts/layer_profiler.py` (per-phase: forward, GEMM fwd/grad_x/grad_w, decode, pack, act-quant, update torch vs fused) |
| M2 | Fused QKV counter layer | **done** — `CounterQKVLinear` (d→3d); bit-identical to three separate layers (test), one saved activation + one update + one larger GEMM. Opt-in via `ReversibleGPT(fused_qkv=True)` |
| M3 | Shared activation handle | **done for QKV** (the high-value case) via M2 — one fused layer ⇒ one saved/quantized activation + one update. A general cross-layer handle (sharing Q(h) across arbitrary layers with independent dither streams) is **deferred**: it needs autograd coordination across layers for marginal gain beyond QKV — documented, not built. |
| M4 | Lagged RMS one-pass + lazy rebase + proxy | **done** — `rms_mode={exact,lagged,proxy}`, `scale_rebase={eager,lazy}` on RMSCounterLinear. lagged uses last step's v; lazy rebases the counter at the next read via a per-row `s_base`; **proxy** takes the row second-moment from grad_out norms × activation energy (post-LN E[r_o²]~‖Δ_o‖²) instead of the full ‖G_o‖² reduction. Parity gate: every mode recovers the teacher to MSE 0.0 (`test_lagged_rms`). Fused kernel stays exact/eager-only; other modes use the torch path. |
| M5 | Derived visible cache (`cache_mode={none,fp16,int8}`) | **done (mechanism)** — keeps the visible ternary T in fp16/int8 (a derived view, persistent=False, rebuilt from truth state, refreshed on every visible flip in `_write_rows`, **and after the CUDA fused update**, which mutates packed state directly). Forward routes through `_visible_t` so the GEMM never unpacks 6-bit. Bit-exact vs the decode forward and tracks the state through updates (`test_cache`, `test_review_fixes`). **Not free: a speed mode that adds live memory** — `int8` is 0.75 B truth + 1.0 B cache ≈ **1.75 B/weight** (`fp16` ≈ 2.75); still far below dense+Adam's ~12–16, but count it honestly. |
| M6 | int8 Tensor-Core compute path | **done (forward path) / partial (update)** — `forward_compute="int8"` wires the int8 forward into training (`int8_forward_ternary`, per-token scale, deterministic round-to-nearest so reversible/eager stay valid; grad_x + update stay fp, straight-through). T4 end-to-end: it has a **size crossover** — net-negative at d=512 (17.6k→16.8k tok/s, quantize overhead > GEMM saving on small layers) but net-positive at d=768 (5.9k→6.1k, 0.91→0.95× dense), and ×2.05 on an isolated d=2048 forward → wins more as models grow. Opt-in (fp default), recommended at d≥768. The int8 *update* correlation (`update_compute="int8"`) still re-quantizes x/Δ each call (can lose to fp32 cuBLAS) — the open refinement is reusing the already-int8-saved activation. |
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

## What is and isn't closed (honest)

**Done:** persistent 6-bit packed state; fused update-from-grad_w (×17.6 on a T4); derived visible
cache (decode-tax removed); QKV fusion; lagged/lazy/proxy RMS; reversible anchors; late update
decimation; int8 compute helpers (forward row-scale, update column-scale, both unbiased); 1B
single-T4 evidence. Each numerics-changing mode is behind a teacher parity gate and defaults off,
so `strict6` / exact-eager / fp / no-decimation stay the default and the memory + dynamics claims
hold.

**Strict update-from-IO — built and validated, and it settles the question.**
`update_from_io.triton_counter_update_from_io` forms grad_w in-kernel (one program/output-row
streams M, accumulates the four packed-lane grad_w vectors) and applies the exact RMS+SR update in
one launch — the dense gradient is never materialized. T4 (`gpu_update_from_io_T4.log`):

```
correctness   bit-EXACT vs the reference (0 mismatches, all sizes)
memory        from-IO kernel        +0.00 MiB   (no dense grad_w)
              cuBLAS grad_w + fused +16.00 MiB  (the [2048x2048] grad_w)
speed         from-IO one launch    7386 ms     <-- ~860x SLOWER
              cuBLAS grad_w + fused    8.6 ms
```

Verdict: zero grad_w materialization is *achievable and correct*, but a hand-written GEMM-in-kernel
is ~860× slower than cuBLAS — so the 16 MiB transient grad_w per layer is **not worth eliminating**
at that cost. The practical strict-enough path is cuBLAS grad_w (transient, or per-tile) + the fused
update kernel. The from-IO kernel stays as the strict-memory bound for memory-constrained regimes.

### Tiled cuBLAS grad + fused tile update — the practical bridge (T4-confirmed)

`PackedRMSCounterLinear._fused_update` is now row-slice aware, so `tile_rows=R` materializes only an
`[R, in]` grad tile (cuBLAS) and fuses the transition on that slice. Frontier on a T4
(`gpu_tiled_update_frontier_T4.log`, d=2048 M=4096):

```
full-gw+fused   R=2048   10.86 ms   +17 MiB grad peak
tile-gw+fused   R=128     9.45 ms    +4 MiB   = 0.87x full
tile-gw+fused   R=256     8.96 ms    +8 MiB   = 0.83x full
tile-gw+fused   R=512     8.58 ms   +17 MiB   = 0.79x full
strict-from-IO   0      7381 ms     +512 B
```

The witness criterion (some R≤512 within 1.5× of full at ≤4 MiB grad tile) is met *for this large
isolated layer*: tiled is even faster than full here, at a fraction of the transient gradient.

**But the full-training-step measurement refutes making it the default** (`gpu_training_throughput_T4.log`,
s512 d=512, B·T=4096):

```
dense + AdamW                        19190 tok/s   (1.00x)
counter_packed untiled               17089 tok/s   (0.89x dense)   <-- fastest counter config
counter_packed tile_rows=256         14235 tok/s   (0.74x dense)
counter_packed tile_rows=128         10817 tok/s   (0.56x dense)
```

For a *real* model the layers are small (512², 2048×512) and tiling turns each into several tiny
fused-kernel launches whose overhead dominates — so untiled is faster end-to-end. The frontier win
was specific to one big 2048² layer. **Verdict: `tile_rows=0` (untiled) stays the default**; tiling
is a *memory* knob (bounds the transient grad to R·in·4 bytes) worth it only for very large layers
or a tight VRAM budget, not a speed default. Three-way: untiled fast for normal training, tiled for
large-layer/low-VRAM, strict from-IO only when *zero* grad materialization is mandatory.

## Training speed (the bottom line)

At s512 the counter method now trains at **17.1k–17.6k tok/s — 0.89× of dense+AdamW (19.2–19.8k)**,
i.e. ~11% slower than dense while using a fraction of the memory (no FP master, no Adam moments,
0.75 B/weight state). That is up from ~0.48× before the fused update kernel (the old torch-tile
update was the wall). This lands inside the memo's "1.1–1.4× dense" target.

The int8-forward path (`forward_compute="int8"`) closes more of the gap as models grow: it's
net-negative at d=512 (overhead) but at d=768 reaches **0.95× dense** (5.9k→6.1k), and an isolated
d=2048 forward is ×2.05 — so on large models it is the path to *match or beat* dense speed. Net: the
counter method is ~0.89–0.95× dense speed today (fp / int8 forward, scale-dependent), trending to
parity at scale, at multiples-less memory.

**Still genuinely open (and now lower priority, given the above):**
- int8 Tensor-Core forward is correct (`int8_forward_ternary`) but **not yet a built-in training
  forward path** on the layer;
- `cache_mode` accelerates but **adds live memory** (≈1.75 B/weight at int8) — a speed mode;
- the int8 **update** correlation re-quantizes x/Δ each call, so it can lose to fp32 cuBLAS until
  it reuses the already-int8-saved activation.
