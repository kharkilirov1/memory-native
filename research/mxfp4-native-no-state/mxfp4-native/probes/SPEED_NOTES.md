# Speed notes: where a training step's bytes go (ESTIMATE from reading the code - not measured)

Status: hand counts from `mx6/layer.py` (`MXNLinear.apply_update_`), `mx6/packed_na.py` (`step`, `pack_state`,
`unpack_state`, `dequantize_fast`), `mx6/update.py` (`rms_update`), `mx6/latent_grid.py` (`LatentGridLinear.apply_update_`,
`LatentGrid.move`), `probes/e1_quality.py` (`fq_nvfp4`). No GPU was available. `probes/bench_train_step.py` measures the
real numbers (ms, kernels, launches); use this file to know what to expect and where to look.

Counting rules: every eager PyTorch elementwise op reads its full-size inputs and writes its full-size output
(no fusion); uint8 = 1 B, fp32 = 4 B, `.to(torch.long)` temporaries = 8 B; per-row / per-block / table operands are
ignored (< 1 B/param). "B/param" = bytes moved over HBM per weight per training step. Accuracy about +-20 %.

## 1. Passes over the weights per step

### na_fp4_c8 (MXNLinear, nvfp4, directional gap, track_clip=True, uint8 code, 8-bit state) - eager

| # | pass (code) | B/param | notes |
|---|---|---|---|
| 1 | grad_w = grad_out^T x (GEMM output) | 4 | common to every arm |
| 2 | `rms_update`: square, mean, normalise | 20 | read g twice, write g_hat |
| 3 | `_step_pre_sr_dir`: scale expand (E4M3 decode of the byte plane + `repeat_interleave` -> fp32 [out,in]), `mag_idx` (to long, //, clamp), `gu`/`gd` gathers, `acc` (to long, %, -, to f32), v, v*C/gap | 250 | ~38 launches; int64 temporaries dominate |
| 4 | SR draw #1 (`_sr_int`: floor, `torch.rand`, <, +) | 50 | one fp32 uniform per weight |
| 5 | `_step_mid_dir`: carry, rem, `mag` again, moved, clamp, blocked, new_mag, 4 gap gathers, rem rescale | 306 | ~25 launches |
| 6 | SR draw #2 (re-quantise the leftover) | 50 | |
| 7 | `_step_enc_dir`: re-encode code word | 77 | |
| 8 | `_clip_mask` (telemetry, track_clip=True) + block sums | 140 | recomputes carry / `mag` / `moved` a 3rd time |
| 9 | `pack_state` (8-bit **generic** path) | 414 | codes -> int32 -> **int64 [.., 8] bit matrix** (shift, &1, *byte_w, sum) |
| 10 | `unpack_state` (next forward, generic path) | 412 | same 8x int64 expansion in reverse |
| 11 | `dequantize_fast` (decode to fp32 weight, cache) | 77 | scale expand again + to long, //, clamp, index_select, * s |
| | **total update path** | **~1800** | ~140 kernel launches per layer |

Biggest single finding: for 8-bit codes `pack_state` / `unpack_state` take the generic bit-matrix path (only 6-bit has a
fast path) although a packed 8-bit group IS the byte. These two passes are ~46 % of the bytes. An identity fast path is
bit-exact; `bench_train_step.py --fast-pack` measures it (`na_fp4_c8+fastpack`, ~975 B/param). Other cheap eager wins:
int32/int16 instead of int64 temporaries, one `mag`/`gap` gather per step instead of three, and no `_clip_mask` when the
clip telemetry is off. na6 (6-bit) already has the fast pack path, so its update is ~900-1000 B/param.

### lg6 / lg8 (LatentGridLinear, uint8 index at run time, exact update) - eager

| pass | B/param |
|---|---|
| grad_w write | 4 |
| `rms_update` | 20 |
| `delta = -lr*g_hat / scale` (fp32 expanded scale read from `_scale_cache`) | 20 |
| `move`: idx->long, nan_to_num+clamp, `values[idx]`, two `_u` evaluations (bucketize + 6 gathers/arith each) and their difference, `i + d`, `over` mask, clamp, floor, ONE `torch.rand`, SR compare/add, to long, clamp | 368 |
| `over` mean (clip telemetry) + `idx.copy_` | 20 |
| next forward `weight()` = `deploy[idx.long()] * scale_cache` | 33 |
| **total** | **~465** (~60 launches per layer) |

lg is ~4x cheaper than na_fp4_c8 eager (~2x cheaper than na+fastpack): one index, one SR draw, no carry / directional gaps /
clamp-to-rung / re-encode / pack. Run-time memory is not the persistent size: lg keeps `_scale_cache` (fp32 [out,in], 4 B/param,
never freed) and `_weight_cache` (4 B/param until the next update) on top of the 1 B index, and k=6 is held in a dense uint8
(the k-bit packing exists only in `pack_state()`); na keeps `_codes_cache` (1 B) + the decoded fp32 weight (4 B) while a step runs.
So **use `peak_mem_mb` / `resident_mb` of the bench for equal-memory claims; `state_mb` is the checkpoint-format size.**

### Other arms

| arm | update-related B/param/step | note |
|---|---|---|
| dense_nobias | 4 (grad) + 84 (AdamW, default foreach: 8 multi-tensor passes) = ~88; **28 for a single fused AdamW kernel** (read p,g,m,v = 16, write p,m,v = 12) | the harness uses the default (foreach); persistent state 12 B/param (+4 grad) |
| qat_fp4 | ~142 (fake-quant EVERY forward: abs/amax twice, block scale decode+expand, `w/s`, 2x bucketize + where + clamp, gather, *s) + 88 (AdamW) = ~230 | STE backward is free |
| tern_counter | not analysed in detail: `PackedRMSCounterLinear._fused_update` -> `memory_native/fused_update.py:triton_counter_update` is ALREADY one fused Triton launch (RMS + hash-SR + carry), followed by `_refresh_t_cache` (decode of the touched rows) | the bench's tern row is the "fused" reference point |

(Correction to the working assumption "dense AdamW ~12-16 B/param/step": 12 B is the persistent state; a fused AdamW step moves
28 B, the default foreach implementation ~84 B.)

## 2. What the numbers predict (bandwidth floor vs launch floor), 4 layers, batch 32 x seq 128

Block-linear params: d=128 786,432 | d=256 3,145,728 | d=512 12,582,912.

| update path | B/param | d=128 | d=256 | d=512 |   (ideal HBM time at 1.5 TB/s) |
|---|---|---|---|---|---|
| na_fp4_c8 eager | 1800 | 0.94 ms | 3.8 ms | 15.1 ms | |
| na_fp4_c8 + fast pack | 975 | 0.51 | 2.0 | 8.2 | |
| lg eager | 465 | 0.24 | 0.98 | 3.9 | |
| qat_fp4 (fq + AdamW) | 230 | 0.12 | 0.48 | 1.9 | |
| AdamW foreach / fused | 88 / 32 | 0.05 / 0.02 | 0.19 / 0.07 | 0.74 / 0.27 | |
| fused update kernel (section 3) | ~18 | 0.01 | 0.04 | 0.15 | |
| GEMMs (fp32 SGEMM, 6 x params x 4096 tokens at ~60 TFLOPS) | | ~0.3 | ~1.3 | ~5 | TF32 would be several x less |

At d=128 the tensors are tiny (qkv = 49k weights = 200 KB), so the step is **launch / host bound, not bandwidth bound**:
~140 launches x 16 layers = ~2,200 launches for the na update (~60 x 16 = ~1,000 for lg); at ~10 us of eager-PyTorch host time
per op that is ~20-30 ms (na) and ~10 ms (lg) per step vs a few ms for dense. At d=512 the bandwidth floor takes over
(na ~15 ms + GEMMs ~5 ms vs dense ~6 ms). Expect the na/dense ratio to be large at both ends and the lg/dense ratio ~2x smaller;
the bench prints measured launches per step and the kernel table to check this. Whether the ratio shrinks or grows with d is
exactly what the d=128/256/512 grid shows.

## 3. What one fused Triton update kernel saves

Per element the information needed is: grad (fp32), state byte, the block scale byte (+ per-tensor scale), the row's v / sum(g^2).
Everything in rows 2-11 above is arithmetic on those; the intermediates (`mag`, `gu`, `gd`, `acc`, `cc`, `carry`, `rem`, int64
copies, bit matrix) never need to leave registers.
- Memory traffic: grad 4 (read once if the program owns a whole row - rows are 128-2048 fp32 = 0.5-8 KB - else 8 for a
  separate row-reduction kernel) + state 1 r + 1 w + scale 1/16 + optional fp32 weight write 4 (so the next forward needs no
  unpack/decode) = **~10-18 B/param instead of ~1800 (na) / ~465 (lg)**: ~100x / ~25-45x fewer bytes. That is below fused
  AdamW's 28 B.
- Launches: ~140 (na) / ~60 (lg) per layer -> 1-2 (RMS reduction + update), i.e. ~2,200 -> ~32 per step; at d=128 the host-bound
  ~20-30 ms becomes ~1-2 ms.
- Expected end state: the update drops from the dominant cost to a few percent of the step (d=512: ~15 ms -> ~0.2-0.3 ms),
  leaving the GEMMs; an na / lg step then lands within ~10-30 % of dense AdamW. (If the decoded fp32 weight is written by the
  kernel the runtime memory is 4 B/param again - a bf16 weight (2 B) or decode-in-GEMM (`mx6/triton_kernel.py`, which the
  layer notes "never wins" at fp32) trade speed for memory; pick it consciously for the equal-memory comparison.)
- Caveats: not bit-identical to the eager path (in-kernel Philox `tl.rand` or the hash-SR of `fused_update.py` replaces
  `torch.rand`; test statistically - E[step] unbiased as in tests/test_gap_arms.py - and for same-seed reproducibility); the row
  RMS needs the whole row before the elementwise part (two-pass or row-resident); the `track_clip` telemetry needs a per-block
  count (or a flag to drop it); E4M3 decode in-kernel by bit arithmetic already exists in `triton_kernel.py:_decode_w`.
- Templates in the repo: `src/memory_native/fused_update.py:triton_counter_update` (fused RMS + hash-SR + carry for the
  ternary counter, one launch) and `mx6/triton_kernel.py` (E4M3 decode, autotune).

### Is lg's update simpler to fuse than na's? Yes.
- one state (idx), one SR draw (na: two), no accumulator carry, no directional gap pair, no clamp-to-rung / blocked / rescale
  of the remainder, no re-encode, no clip-mask recomputation;
- `G[idx]` is a closed form (rung = idx // m, sub-step = idx % m: value = v[r] + (v[r+1]-v[r]) * (idx % m) / m) so no table is
  needed, and the piecewise-linear map `U` is a 14-threshold compare chain (or a 15-entry constant table);
- the deployed value `RTN_E2M1(G[idx])` is a pure function of idx (253-entry LUT, or computed), so the same kernel can emit the
  weight; the fp32 `_scale_cache` disappears (scale decoded from the byte in-kernel).
Roughly 40-50 lines of Triton against ~100+ (and a harder equivalence story) for na.

## 4. Things to read off the bench
- `launches` in each kernel-table header (kernels per step) and `update (in bwd)` / `decode (in fwd)` columns: na vs lg vs dense.
- `na_fp4_c8+fastpack` vs `na_fp4_c8`: the pure pack/unpack share (predicted ~45 % of the na update bytes).
- `+compile` rows: how much of the elementwise fusion `torch.compile` already recovers (it does not cover rms_update, pack /
  unpack, `_sr_int`, `_clip_mask`; lg's compiled `move` is experimental).
- `tern_counter` row: the cost of a fused update (+ `_refresh_t_cache`) in this harness.
- `peak MB` / `resident MB` vs `state MB`: the gap is the run-time caches + update temporaries (the int64 intermediates make the
  eager update peak at several x the persistent state for the biggest layer).
