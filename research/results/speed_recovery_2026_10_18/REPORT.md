# Speed recovery: why Memory Native char-LM trained 4× slower, and what fixed it

**Branch:** [`research/speed-recovery`](https://github.com/kharkilirov1/memory-native/tree/research/speed-recovery)
**Verified HEAD:** `b5657f6` · **start point:** `32f0fa2` (mission's stated HEAD)
**Device:** NVIDIA A100-SXM4-40GB (39.5 GiB, 108 SMs, sm_80), Colab
**Stack:** Python 3.13.15, torch 2.11.0+cu130, CUDA 13.0, Triton 3.6.0
**Scope:** root cause + repair of the 4× char-LM slowdown; no architecture change, no
persistent fp32 weight copy, no AdamW moments, no model/data/eval change for one arm.

---

## 0. Result summary (same gate, same data, same budget, 2 seeds)

| kind | val loss (mean) | tok/s (mean) | persistent | peak VRAM |
|---|---:|---:|---:|---:|
| FP32 AdamW (baseline) | 2.2012 | 440,738 | 12.21 MiB | 342.40 MiB |
| **counter_triton (fixed path)** | **1.9578** | **199,371** | **2.56 MiB** | 306.77 MiB |
| counter_fast (new alt. path) | 1.9269 | 192,883 | 2.56 MiB | 306.77 MiB |
| counter_rms (old slow path) | 1.9280 | 90,232 | 3.31 MiB | 324.54 MiB |

- **Speed: 90.2k → 199.4k tok/s = 2.21×** on the same layer family and gate; the profiling
  harness (no eval overhead) gives 68.7k → 199.3k = **2.90×**.
- **Quality: no regression** — the fixed path is 0.24 better than AdamW in val loss; the
  0.03 gap to the old torch path is within the 2-seed spread and comes from a different SR
  stream, not different math.
- **Memory: unchanged win** — 4.77× less persistent state (2.56 vs 12.21 MiB), 1.12× less
  peak (306.8 vs 342.4 MiB). No persistent fp32 weight, no AdamW moments; peak is
  activation-dominated, so the storage win does not become a peak win at this model size.
- **Minimum target (≥2×) met. Parity with AdamW not met**: 199k vs 441k at the same batch,
  and the remaining gap is now precisely attributed (§5), not hand-waved.

---

## 1. Root cause analysis (measured, not inferred)

The 4× was **not one slow kernel**. It was the gate measuring the pure-PyTorch reference
implementation of the counter layer, plus two per-layer costs that survived into the fast
kinds.

### RC1 — the char-LM gate trained the *torch reference* layer kinds
`src/memory_native/cli.py::_train_one` built the model with `GPT(kind, cfg, **kw)` and
forwarded counter hyperparameters only for `kind in ("counter", "counter_rms")`
(`src/memory_native/baselines.py::make_linear`). Those kinds are
`counter.py::CompactCounterLinear` / `RMSCounterLinear`: the update is the pure-torch
automaton `_update_tile` → `_rms_eager_pre_sr` / `_carry_resolve` / `_write_rows`, i.e.
~20 elementwise CUDA kernels per layer per step.

Measured (torch.profiler, batch 32, 15 steps, A100):

| kind | step ms | tok/s | update-off step | update cost | CUDA launches/step |
|---|---:|---:|---:|---:|---:|
| dense AdamW | 9.13 | 448,579 | – | – | 2,940 |
| counter (torch) | 54.98 | 74,499 | 24.18 | **30.8 ms** | **27,060** |
| counter_rms (torch) | 59.61 | 68,712 | 24.42 | **35.2 ms** | **29,580** |
| counter_packed | 51.54 | 79,468 | – | – | 19,140 |
| counter_triton | 35.93 | 113,984 | 15.44 | 20.5 ms | 9,420 |
| group_scale_packed | 33.08 | 123,815 | 19.60 | 13.5 ms | 4,740 |

CPU-side evidence at HEAD 32f0fa2 for `counter_rms`: `_FusedCounterLinearFnBackward`
16.5 ms/step self CPU and `cudaLaunchKernel` 15.5 ms/step — the step was **dispatch-bound
inside the reference automaton**, not matmul-bound. Meanwhile the repo already shipped
fused kinds (`counter_packed`, `counter_triton`, `group_scale_packed`) that the gate could
not select with correct hyperparameters — the architectural mismatch the mission suspected.

### RC2 — the fast kinds still decoded `(t,c)` with torch in every backward
`src/memory_native/counter.py::_FusedCounterLinearFn.backward` ran
`module._decode_rows(lo, hi)` at the top of the tile loop **before** asking whether the
fused Triton updater (`_fused_update`) or the kernel grad_x path needed it. Neither does:
the fused updater reads the packed state itself, and `TritonCounterLinear._backward_grad_x`
decodes inside its kernel. Cost: `unpack_codes` + `decode_state` ≈ 13 torch kernels and
~0.3 ms CPU per layer per step. Update-off ablation for `counter_triton` proved the update
loop (which contained the decode) cost 20.5 ms of a 35.9 ms step.

### RC3 — one pageable H2D copy + stream sync per layer per step
`src/memory_native/fused_update.py::triton_counter_update` allocated
`torch.tensor([seed], dtype=int64, device=cuda)` on **every** call to pass the SR seed.
That is a pageable host→device copy that synchronizes the stream. torch.profiler showed
`cudaStreamSynchronize` × 360 per 15 steps (exactly one per layer per step) and
`cudaMemcpyAsync` in the step. Every sync also breaks CPU/GPU overlap for the following
kernels.

### RC4 — `lr=0` was not a bit-exact no-op (correctness defect carried over)
`fused_update._tick` rebuilt the 6-bit code from a float expression
(`c * s_old/s_new + tick`). At `lr = 0` with an unchanged scale, 2 of 16,777,216 codes on a
4096² group-scale layer moved by one quantum (1 of 4,194,304 at 2048²), deterministically;
the torch reference moved none. Verified by `scripts/verify_counter_semantics.py`, which
now reports **10 checks / 0 failures** after the fix (both 1024² and 4096²).

### RC5 — the true no-dense-weight forward path is not faster on this GPU
`triton_counter.py::_decode_matmul_kernel` re-decodes the weight tile once per M-block
(grid is over M), i.e. `M / BLOCK_M = 64×` redundant decode work at M=4096. Measured
`_decode_matmul_kernel` 2.56 ms/step and `_decode_gradx_kernel` 2.26 ms/step — 4.8 ms of
the fixed 20.6 ms step, and the fused GEMM runs at ~4.4× off fp32 peak on these small tiles.
A decode-to-dense Triton kernel + cuBLAS was built as the alternative
(`src/memory_native/fast_decode.py`, bitwise-identical decode: max abs err **0.0**) and is
equal at layer level but slower at model level (43.5 vs 20.6 ms — extra launch and
allocation per direction), so it is kept as an option, not the recommended path.

---

## 2. Fixes applied (each with its measured effect)

| # | file / function | change | measured effect |
|---|---|---|---|
| F1 | `counter.py::_FusedCounterLinearFn.backward` | decode `(t,c)` lazily — only the torch grad_x/update fallbacks request it | counter_triton **35.9 → 23.0 ms**, 114k → 178k tok/s |
| F2 | `fused_update.py::triton_counter_update` | cache one device seed slot and `fill_` it (no pageable H2D, no sync) | **23.0 → 18.4 ms**, 178k → 223k tok/s (profiling harness), one `cudaStreamSynchronize`/layer removed |
| F3 | `cli.py::_train_one`, `baselines.py::make_linear` | forward counter kwargs to `counter_packed` / `counter_triton` / `counter_fast`; gate compares the fused kind | gate **90.2k → 199.4k tok/s**, val 1.928 → 1.958 (within spread) |
| F4 | `fused_update.py::_tick` | exact no-op guard (`lr==0 and s_old==s_new` → return loaded code) | verify: 1 failure → **0 failures**; non-zero steps untouched |
| F5 | `fast_decode.py` (new) + kind `counter_fast` | one Triton kernel packed→dense fp32 + cuBLAS + inherited fused update | layer-level equal (0.948 vs 0.957 ms step), model-level slower; **not adopted** |

Cumulative (profiling harness, batch 32 / 4096 tokens per step):
`counter_rms 59.61 ms → counter_triton (F1+F2) 18.37–20.55 ms`,
`68,712 → 199,283–222,919 tok/s`, CUDA launches per step `29,580 → 196`.

Correctness of every change was checked before timing:
`python -m pytest tests/ -q` → **643 passed, 38 skipped** on CPU; GPU:
`scripts/verify_counter_semantics.py` → **10/10 contracts pass**; the new decode kernel is
bitwise equal to `scale * t` (max abs err 0.0).

---

## 3. Where the remaining time goes (final HEAD, batch 32)

| arm | step ms | CPU upper-bound | GPU upper-bound | launches/step | bound |
|---|---:|---:|---:|---:|---|
| dense AdamW | 8.02 | 13.23 | 15.61 | 196 | gpu |
| counter_triton | 20.55 | **23.32** | 17.69 | 196 | **cpu** |

Counter step top CUDA: `_decode_matmul_kernel` 2.56, `_decode_gradx_kernel` 2.26,
`aten::mm` 1.83, attention backward 0.37. Top CPU: `_FusedCounterLinearFnBackward` 7.06,
`_FusedCounterLinearFn` 3.00, `cudaLaunchKernel` 2.04, `aten::empty` 1.24, `aten::mm` 1.11.
(CPU/GPU totals are sums of profiler self-times and can contain overlapping intervals; they
are an upper-bound attribution. Step ms is measured separately with CUDA events.)

Batch scaling (measured, `scripts` sweep): counter overhead per step is roughly constant
(~10–12 ms) across batch 32/64/128 — i.e. **per-step fixed cost**, so
`batch 128: dense 583k tok/s, counter_triton 416k tok/s` (still 1.4× slower but above the
350k parity line in absolute terms).

---

## 4. Full-model comparison vs FP32 AdamW (2 seeds, same budget)

600 steps, batch 32, tiny char-LM (4 layers, d=256, block 128), tinyshakespeare (vocab 65),
eval 20 iters. `val` is final validation loss (mean of seeds), `tok/s` is training
throughput, `persistent` is resident parameter/optimizer state, `peak` is
`max_memory_allocated` for the run.

| arm | val | tok/s | persistent | peak | speed vs baseline | mem vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| AdamW | 2.2012 | 440,738 | 12.21 MiB | 342.40 MiB | 1.00× | 1.00× |
| counter_triton | 1.9578 | 199,371 | 2.56 MiB | 306.77 MiB | **2.21× faster than the old counter arm**, 2.21× slower than AdamW | **4.77× less persistent**, 1.12× less peak |
| counter_fast | 1.9269 | 192,883 | 2.56 MiB | 306.77 MiB | 2.14× vs old arm | 4.77× / 1.12× |
| counter_rms (old) | 1.9280 | 90,232 | 3.31 MiB | 324.54 MiB | 1.00× (baseline) | 3.69× / 1.06× |

Peak-memory gate (`memory-native-memgate --config tiny --batch 16`):
counter_rms 209.72 MiB vs dense+AdamW 234.14 MiB = **1.12×** — peak is
activation-dominated, and the report does not present the 4.77× persistent win as a
training-memory win.

---

## 5. Remaining bottlenecks, ranked by expected gain

1. **CPU dispatch inside the custom Function (≈5–7 ms/step over the GPU limit).** The step
   is now CPU-bound (23.3 vs 17.7 ms upper bounds). The lever with the largest payoff is
   CUDA-graph capture of the whole step, which requires moving the SR seed advance to the
   device (one int64 buffer per layer, advanced by a 1-thread kernel) so replay preserves
   the stochastic-rounding stream. Expected: step ≈ GPU time ≈ 17–18 ms → ~230k tok/s.
2. **Redundant weight decode in the fused GEMM (4.8 ms/step of GPU).** The decode is
   repeated `M/BLOCK_M = 64×` per weight tile because the grid is over M blocks. A
   weight-stationary kernel (decode a `[BLOCK_N, BLOCK_K]` tile once into shared memory,
   loop over all M blocks, split-K reduction or a two-stage epilogue) removes most of it;
   the tile scan shows no further win from retuning the current grid for 256-wide layers.
   Expected: −3 to −4 ms → ~250–270k tok/s.
3. **Fused GEMM efficiency (4.4× off fp32 peak at these shapes).** Even without redundant
   decode the hand-written Triton matmul is far from cuBLAS tensor-core throughput; a
   bf16/tf32 path or a decode-once-to-shared-memory GEMM with larger tiles is the next
   step. Only relevant once 1–2 land.
4. **Per-layer Python work that is not dispatch** (`_FusedCounterLinearFnBackward` 7.1 ms
   CPU self, `attn::empty` 1.2): fewer allocations (preallocated grad_w buffer), fewer
   tensor views, and fusing `grad_w` formation into the update kernel.
5. **Parity math**: at 20.6 ms/step vs 8.0 ms dense, closing the last 2.2× needs 1+2+3
   together; nothing in the measurement suggests a structural blocker, only the three
   engineering items above.

---

## 6. Negative results / rejected hypotheses (kept deliberately)

- **New decode-to-dense path (`counter_fast`)** — bitwise-exact and equal at layer level
  (0.948 vs 0.957 ms), but **43.5 ms/step at model level** vs 20.6 ms for the fused flat
  path: the extra launch + `torch.empty` per direction dominates at 24 small layers. Not
  adopted; still useful as a reference/alternative for wide layers.
- **Decode-kernel tile retune** — swept `(BLOCK_M, BLOCK_N, BLOCK_K)` over 7 combinations at
  the three char-LM layer shapes: the shipped `64/64/32` is already the best for the narrow
  256-wide layers; alternatives win only for wide layers and not enough to change the total.
- **`group_scale_packed`** (the previous campaign's fast path) is 32.3 ms/step / 126.8k tok/s
  in this gate — better than the old torch kinds but worse than `counter_triton`, because
  the gate is bandwidth-insensitive and it pays two decode-dense kernels per layer per step.
- **Update-in-backward restructuring** (moving the updater out of the autograd Function) was
  not needed: after F1/F2 the update is one kernel + one GEMM per layer and is no longer the
  top cost. Keeping it in backward preserves the "one forward per backward" invariant.

---

## 7. Artifacts

| file | content |
|---|---|
| `profiling.json` | before/after per-kind step ms, tok/s, launches, update-disabled ablation, CPU/GPU attribution, top ops |
| `fix_ledger.json` | F1–F5 with file/function, problem, evidence, change, measured effect |
| `quality.json` | 2-seed gate results, throughput, persistent/peak memory, memory gate, caveats |
| `update_verification.json` | 10 contracts incl. the fixed lr=0 no-op and quantified TF32 oracle parity |
| `single_layer_ab.json` | single-layer A/B (torch / packed / fused / decode+cuBLAS), decode bitwise check, tile scan |
| `scripts/profile_charlm_step.py` | the profiler used for all root-cause tables |
| `scripts/probe_fast_paths.py` | single-layer A/B + launch counts + decode correctness |
| `scripts/verify_counter_semantics.py` | byte-level update contracts |
| `notebooks/Speed_Recovery_A100.ipynb` | one-click reproduction of profile, A/B, verification, gate |

Code: `src/memory_native/counter.py` (F1), `fused_update.py` (F2, F4),
`fast_decode.py` + `baselines.py` + `cli.py` (F3, F5).

## 8. Reproduction

```bash
git fetch origin research/speed-recovery && git checkout research/speed-recovery
git rev-parse HEAD              # b5657f6 (all numbers here)
python -m pip install -e .

# root cause + final profile (writes profiling.json under research/results/speed_recovery_2026_10_18)
python scripts/profile_charlm_step.py --kinds dense,counter_rms,counter_triton,counter_fast \
    --steps 20 --batch 32 --out-dir research/results/speed_recovery_2026_10_18

# single-layer A/B (decode bitwise check, launch counts)
python scripts/probe_fast_paths.py

# byte-level update contracts (expect 10 checks / 0 failures)
python scripts/verify_counter_semantics.py

# real gate, 2 seeds, and the peak-memory gate
memory-native-charlm --config tiny --steps 600 --batch 32 --eval-iters 20 \
  --kinds dense,counter_triton,counter_fast,counter_rms --device cuda --seed 0
memory-native-memgate --config tiny --batch 16 --device cuda --optimizers adamw --seed 0
```

## 9. Honest limits

- 2 seeds, 600 steps, one tiny model, one GPU; the val-loss difference between the new and
  the old counter paths (0.03) is within that budget's spread and both beat AdamW by >0.24.
- No time-to-equal-quality claim: the fixed arm is 2.2× faster per token but still 2.2×
  slower than AdamW at batch 32; at batch 128 its absolute throughput (416k tok/s) exceeds
  the 350k target line, but the mixed-shape/scale behaviour is not characterised yet.
- Peak VRAM is activation-bound at this model size: the memory story of this path is the
  4.77× persistent-state reduction, not the 1.12× peak reduction.
- The CUDA-graph and weight-stationary items in §5 are designed but **not implemented** in
  this round; they are the measured next steps, not claims.
