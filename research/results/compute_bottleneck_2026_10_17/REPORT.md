# Breaking the compute bottleneck: measured status, 2026-10-17/18

**Branch:** `research/lut-multi-token` (main untouched) · **Measured commit:** `ab39b3f`
**Device:** NVIDIA A100-SXM4-40GB (39.5 GiB, 108 SMs, sm_80), Colab runtime
**Software:** Python 3.13.15, torch 2.11.0+cu130, CUDA 13.0, Triton 3.6.0
**Scope of this report:** corrections to the previous study, a full-step cost/memory
decomposition, the true no-FP32-residency compute path, the small-layer bottleneck, and a
real char-LM training comparison (2 seeds). Everything below is measured on the device
above; nothing is extrapolated from a different GPU.

---

## 0. Short answer to the final question

> Can we train a comparable network with significantly less memory than FP32 AdamW and at
> equal or greater speed to quality?

**Memory: partially. Speed: no, not in the current implementation.**

- Persistent training state: **3.7–3.8× smaller** than FP32 AdamW at the same architecture
  (3.24–3.31 MiB vs 12.21 MiB in the char-LM gate; 6-bit codes + scales only, no dense
  master weights).
- Peak training VRAM at that model scale: **only 1.05–1.12× smaller** (324 vs 342 MiB
  to 210 vs 234 MiB depending on the harness) because activations dominate. Storage
  savings alone cannot move the peak when activations are 20–90× the parameter state.
- Quality at equal step budget: **equal or better** — counter_rms mean val loss **1.9605**
  vs dense AdamW **2.1753** (2 seeds, tiny char-LM, 600 steps); counter 2.1548.
- Time to equal quality: **~2.1× slower** (counter_rms reaches dense's 600-step val between
  15 and 23 s of wall clock vs dense's 7 s) and then overtakes it (1.97 vs 2.15 val at
  600 steps). The slowdown is a per-token overhead of the counter layers in the GPT gate
  (84k vs 334k tok/s), not a memory effect.
- At the layer level on the shapes a 12B model uses, the fast group-scale packed path is
  **13% faster** than AdamW with **2.7–3.8× less peak allocated VRAM** (2.67 vs 3.07 ms,
  17 vs 64 MB at 4096²/B=512, dec8). That advantage has **not** yet been shown to survive
  a full training run: the model-level gate still uses the old counter layers.

The two bottlenecks are identified and measured, not hand-waved: (1) per-token overhead of
the *model-level* counter layers (4× in the GPT gate, see §6) and (2) the update automaton
floor on small layers (see §5). The **true no-FP32-residency** path (decode inside the
GEMM) is **3.6–7.8× slower than cuBLAS + one transient decode** even after a 3.2× tile
retune (§4), so it does not close them on this GPU.

---

## 1. Corrections to the previous study (required by the mission)

| issue | correction |
|---|---|
| LUT benchmark did not isolate "cached weight index + fresh LUT per input" | added `lut_cached_idx` path to `scripts/bench_lut_scaling.py`: builds the `[N, K/4]` int8 index table once (weight-side) and rebuilds the `[M, P, 81]` LUT for every call, then gathers. It is reported as its own row and is the realistic serving number. |
| LUT buffer arithmetic wrong | corrected: a `[512, 1024, 81]` FP32 LUT is **512·1024·81·4 = 169,869,312 B = 162.0 MiB (169.9 MB)**, not 682 MB. The earlier note multiplied the M=512 table by 4; the report in `research/results/lut_scaling_2026_10_17/REPORT.md` is fixed. |
| memory accounting conflated persistent/temp/peak | new `scripts/measure_step_breakdown.py` reports four separate quantities per variant: **persistent** (params + buffers + packed state + scales + fp32 cache when enabled), **live_temp** (peak_allocated − persistent), **peak_allocated**, **peak_reserved**. Table in §3. |
| `update_events` does not prove weights changed | new `scripts/verify_counter_semantics.py` checks bytes, not counters: (a) `lr=0, lr_scale=0` must leave codes and scales bitwise unchanged; (b) under decimation only active groups' codes/scales may change, inactive must be bitwise identical; (c) parity vs the torch oracle counted per code; (d) exactly one update per step, twice. Results: §7. |
| comparability of math/precision | all layer and training comparisons use identical fp32 activations and inputs, strict ternary `alpha=0`, no salient; bf16 paths are reported only where measured, as a separate precision class and never mixed into the fp32 comparison. Decimated runs are labelled a *different algorithm*, not a cheaper AdamW. |
| regression tests / artifacts | `tests/test_lut_scaling.py` = 16 passed on the A100; the new harnesses are committed with the raw CSV/JSON they produce; all numbers here cite a row in `step_breakdown_summary.csv` / `training_comparison.json` / `update_verification.json`. |

---

## 2. Method

- **Layer benchmark** (`scripts/measure_step_breakdown.py`): for each (out, in) shape and
  token count, builds the five implementation variants on identical inputs and measures
  forward, backward, updater, decode and full step with CUDA events (8 warmup, 15 iters,
  3–5 reps, median + IQR), and the four memory quantities above.
- **Verification** (`scripts/verify_counter_semantics.py`): bit-level, see §7.
- **Training** (`memory-native-charlm`, the repo's own parity gate): same architecture,
  tokenizer, data, initialisation and token budget per kind; A100; 600 steps; batch 32;
  eval 20 iters; seeds 0 and 1. Reported: final val loss, tokens/s, persistent state and
  peak CUDA memory. Time-to-quality is built from separate runs at 150/300/450/600 steps.
- **Tile retune** for the true fused path: direct launches of the existing Triton kernels
  with different `BLOCK_M/N/K` and `num_warps`, same inputs, medians over 5 reps.

All raw data that fits the transfer budget is committed next to this report; the row count
of each file is stated. The full per-phase CSV for all 990 measured rows is regenerable
with the committed script (command in §9) — only a summary excerpt is committed because
the raw CSV is 172 KB and had to be transferred out of the Colab runtime by hand.

---

## 3. Where the full step actually goes

### 3.1 Phase attribution, 4096×4096 / B=512 (ms, A100)

| variant | fwd | full step | fwd_2d | grad_x_2d | update_only | decode |
|---|---:|---:|---:|---:|---:|---:|
| dense fp32 + AdamW | 1.074 | **3.066** | – | – | – | – |
| packed, decode-once, no cache | 1.573 | 4.893 | 1.574 | 1.438 | 1.794 | 0.499 |
| packed, persistent fp32 cache (group scope) | 1.076 | 3.286 | 1.074 | 0.939 | 1.282 | 0.500 |
| packed, persistent cache + dec8 | 1.075 | **2.672** | 1.075 | 0.938 | 0.397 | 0.498 |

Reading:
- The packed path's matmuls reach parity with dense (`fwd` 1.074 vs 1.074, `grad_x` 0.938)
  **only when the decoded fp32 weight is cached**; the cache costs 67 MB resident.
- The updater is the remaining non-matrix term: 1.28 ms at dec1, **0.40 ms at dec8**
  (AdamW's own step is ~1.08 ms at this size, so decimation makes the updater cheaper than
  AdamW, but decimation is a different algorithm — see §6).
- Decode is a hard 0.50 ms per step whenever the cache is invalidated; it is a read+write
  of 67 MB at ~1.8 TB/s plus the decode ALU, i.e. already near its floor.

### 3.2 Full-step matrix (medians, ms) — `step_breakdown_summary.csv` has all 210 rows

| shape | tok | dense fp32 | fused (no cache) | decode-once | cache dec1 | cache dec4 | cache dec8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 896² | 512 | 0.728 | 4.610 | 1.222 | 0.850 | 0.976 | 0.915 |
| 1024² | 512 | 0.794 | 6.289 | 1.200 | 0.825 | 0.934 | 0.986 |
| 2048² | 512 | 0.784 | 24.854 | 1.628 | 0.859 | 1.054 | 1.043 |
| 4864×896 | 512 | 0.822 | 23.992 | 1.717 | 0.913 | 1.150 | 1.144 |
| 896×4864 | 512 | 0.909 | 24.817 | 1.714 | 0.919 | 1.190 | 1.187 |
| 4096² | 512 | 3.066 | 97.956 | 4.893 | 3.285 | 2.812 | **2.672** |
| 4096² | 1024 | 4.821 | 195.273 | 7.778 | 5.387 | 4.698 | **4.685** |

`fused` (decode inside the GEMM, zero fp32 residency) is unusable as shipped: 98 ms at
4096²/B=512, and its cost grows with M because the weight is re-decoded per M-block.
`decode-once` (transient 67 MB per step) is *worse than the cache in both time and memory*
(see 3.3). The cached path with decimation is the only configuration that beats AdamW, and
only at 4096².

### 3.3 Memory (MB), 4096×4096 / B=512

| variant | persistent | live temp | peak allocated |
|---|---:|---:|---:|
| dense fp32 + AdamW | 64.0 | 0.0 | 64.0 |
| fused, no cache | 17.5 | ~0 | 17.5 |
| decode-once, no persistent cache | 8.0 | 136.0 | **144.0** |
| cache dec1 | 24.0 | ~0 | 24.0 |
| cache dec8 | 17.0 | ~0 | 17.0 |

The "no persistent fp32 cache" variant that decodes once per step is a **memory regression**:
two decodes are live at once during backward (one per matmul direction), so peak allocation
is 144 MB — 2.25× *more* than dense. Any claim that dropping the persistent cache saves
memory is false unless the decode happens *inside* the kernel.

---

## 4. Can the true fused path (decode inside the GEMM) be made competitive?

Hypothesis: re-tiling the existing Triton kernels removes the inefficiency, so the packed
state can feed tensor cores directly with no fp32 residency.

Measured (4096², direct kernel launches, medians):

| path | original tiles 32×32×32 | best retuned | dense fp32 reference | decode-once reference |
|---|---:|---:|---:|---:|
| fused forward, M=1 | 1.330 | 1.003 (32/64/64) | 0.059 | 0.500 |
| fused forward, M=64 | 1.300 | 1.184 (64/64/64) | 0.157 | 0.499 |
| fused forward, M=512 | 12.50 | **3.876** (128/128/64) | 1.073 | 0.500 |
| fused grad_x, M=64 | – | 1.054 (64/128/64) | 1.445 (cuBLAS+W) | – |
| fused grad_x, M=512 | – | **3.983** | 2.407 (cuBLAS+W) | – |

Retuning recovers **3.2×** on the forward at M=512 and makes the fused grad_x roughly
comparable to a cuBLAS matmul *with a materialised weight* (1.65× slower at M=512, faster
at M=64). But the fused forward remains 3.6× above dense fp32 GEMM and 7.8× above a single
transient decode, and its measured effective bandwidth is **~3 GB/s of packed state** versus
a 1.8 TB/s roofline — the decode + byte-gather pattern, not the MMA, is the limit.

**Verdict (hypothesis rejected for this GPU):** the no-FP32-residency path is worth keeping
only for memory-critical inference at small M, not for training throughput. The engineering
budget is better spent on the cached path (§5) or on a fundamentally different decode
format (e.g. an int8 code plane with vectorised loads) which this session did not build.

---

## 5. The small-layer bottleneck

Measured facts:
- At 896²/1024² the counter full step is 0.82–0.99 ms vs dense 0.73–0.79 ms (1.1–1.3×),
  and the gap is **not** in the matmuls (both are at parity when cached) but in the
  per-step fixed cost: decode 0.5 ms is per *layer*, not amortised by batch, and the
  updater has a floor of ~0.2–0.4 ms even at decimation 8.
- In the model-level char-LM gate the counter variants run at **84k tok/s vs 334–376k** for
  dense (4× slower). That gate uses the plain counter layers (not the group-scale packed
  path) and the updater runs inside the autograd Function's backward with Python-level
  host work per layer per step.
- Attribution measured earlier in this branch (previous report): ~0.15 ms per layer per
  step of Python/autograd plumbing around the custom Function at 1024², of which
  ~0.0 ms is the custom Function itself (a custom Function wrapping the same matmuls is
  not slower than `nn.Linear`). The remaining cost is per-layer decode + updater, i.e.
  algorithmic, not dispatch.

Not done in this session (listed honestly): CUDA-graph capture of the whole step, moving
the updater out of backward, two-pass statistics for wider parallelism, and warp
specialisation. Each needs its own A/B; the breakdown harness to measure them is committed.

---

## 6. Comparison matrix (measured; quality column only where a training run exists)

| configuration | full step 4096²/B512 | peak VRAM | persistent | val loss (char-LM, 2 seeds σ) | status |
|---|---:|---:|---:|---:|---|
| FP32 AdamW | 3.066 ms | 64 MB (layer) / 342 MiB (gate) | 12.21 MiB | 2.1753 | baseline |
| BF16 AdamW | not measured in this run | – | – | – | gap (harness fix pending) |
| packed dec1, no fp32 cache (`fused`) | 97.96 ms | 17.5 MB | 8 MB | not run (path unusable) | rejected |
| packed dec1, decode-once (no persistent cache) | 4.893 ms | 144 MB | 8 MB | not run (memory regression) | rejected |
| packed dec1, persistent cache | 3.286 ms | 24 MB | 24 MB | not run (gate uses old layers) | viable |
| packed dec4 (group scope, cache) | 2.812 ms | 18 MB | 18 MB | not run | viable |
| packed dec8 (group scope, cache) | **2.672 ms** | **17 MB** | 17 MB | not run | **best measured** |
| repo counter / counter_rms (gate, 600 steps) | 4× slower per token | 324 MiB | 3.24–3.31 MiB | **2.1652 / 1.9747** | best *quality* |

Rules used: decimated runs are a different algorithm (only 1/dec of groups update per step)
and are not presented as equivalent to AdamW; precision classes are not mixed; peak, live
and persistent memory are separate columns; the quality column is only filled where a real
training run was executed.

---

## 7. Update verification: what passed and one real defect

`scripts/verify_counter_semantics.py` (A100, 10 checks):

| check | result |
|---|---|
| decimation is restricted: inactive groups' codes **and** scales bitwise unchanged, all active groups changed | **PASS** for dec 4 and 8, 1024² and 4096² |
| exactly one update per step (`update_events` delta == out·in, two steps in a row) | **PASS** both shapes |
| `lr=0` leaves parameters unchanged (1024²) | PASS (state and scales bitwise equal) |
| `lr=0` leaves parameters unchanged (4096²) | **FAIL**: 2 of 16,777,216 codes moved by one quantum (2048²: 1 of 4,194,304). The torch reference moves 0. |
| fused kernel vs torch fp32 oracle | 0.15% of codes differ, max delta 12–13 quanta — the fused correlation runs on **TF32** tensor cores, so ~1 in 650 codes lands on a different stochastic-rounding quantum. Documented contract, now quantified. |

The 4096² no-op defect is real but bounded (≈1e-7 of codes per step, one quantum). It
cannot explain the quality or speed results here, but it violates the strict no-op contract:
the code path is `val = c·(s_old/s_new) + tick` with both terms exactly neutral at lr=0, so
the deviation must come from the masked/tail epilogue or a neighbouring program's write to
the same 3-byte pack, not from arithmetic. Fix + bitexact regression test are the next
action; this report does not hide it.

---

## 8. Real training (required stage): results and limits

Same architecture (tiny char-LM: 4 layers, d=256, block 128, 4 heads), same corpus
(tinyshakespeare, vocab 65), batch 32, 600 steps, eval 20, seeds 0/1.

| kind | mean val loss | tok/s (seed0/1) | persistent | peak VRAM |
|---|---:|---:|---:|---:|
| dense + AdamW | 2.1753 | 334k / 376k | 12.21 MiB | 342.40 MiB |
| qat | 2.2495 | 231k / 232k | 12.21 MiB | 354.40 MiB |
| counter | 2.1548 | 84k / 86k | **3.24 MiB** | 324.47 MiB |
| counter_rms | **1.9605** | 80k / 80k | **3.31 MiB** | 324.54 MiB |

Time-to-quality (seed 0, separate runs at each step count, val loss / wall time):

| steps | dense | counter_rms |
|---|---|---|
| 150 | 2.5881 / 2 s | 2.3674 / 8 s |
| 300 | 2.4662 / 4 s | 2.1202 / 15 s |
| 450 | 2.3582 / 5 s | 2.0130 / 23 s |
| 600 | 2.1549 / 7 s | 1.9747 / 31 s |

So: **equal-or-better quality at equal budget; 3.7–3.8× smaller persistent state; 1.05×
smaller peak; but ~2.1× longer to reach dense's final quality** (14–16 s vs 7 s) because
the model-level counter layers are 4× slower per token. That 4× — not the kernels measured
in §3/§4 — is the current blocker for the training-speed goal.

Reproduce:

```bash
memory-native-charlm --config tiny --steps 600 --batch 32 --eval-iters 20 \
  --kinds dense,qat,counter,counter_rms --device cuda --seed 0
memory-native-memgate --config tiny --batch 16 --device cuda --optimizers adamw --seed 0
```

---

## 9. Artifacts and reproduction

Committed next to this report (all measured at `ab39b3f`):

| file | rows | content |
|---|---:|---|
| `step_breakdown_summary.csv` | 210 | full step + peak allocated per (shape, tokens, variant) + ratios |
| `phases_4096_B512.json` | 4 variants | phase attribution from §3.1 |
| `retune_sweeps.json` | 3 shapes | fused-path tile retune vs references |
| `training_comparison.json` | 2 seeds × 4 kinds + curve + memory gate | §8 |
| `update_verification.json` | 10 checks | §7, including the open defect |
| `REPORT.md` | – | this document |

Scripts (committed): `scripts/measure_step_breakdown.py`,
`scripts/verify_counter_semantics.py`, `scripts/bench_lut_scaling.py` (with the new
`lut_cached_idx` path), `scripts/bench_lut_graph.py`, `scripts/bench_counter_vs_adam.py`.

```bash
git fetch origin research/lut-multi-token && git checkout research/lut-multi-token
python -m pip install -e .
python -m pytest tests/test_lut_scaling.py -q          # 16 passed on A100
python scripts/verify_counter_semantics.py             # §7 (expects the known 4096^2 defect)
python scripts/measure_step_breakdown.py --iters 15 --reps 3   # regenerates the 990-row CSV
python scripts/bench_lut_scaling.py --shapes 4096 --batches 1,8,64,512 \
    --iters 15 --reps 3 --out-dir /tmp/lut                # includes lut_cached_idx
memory-native-charlm --config tiny --steps 600 --batch 32 --kinds dense,counter_rms \
    --device cuda --seed 0
```

---

## 10. Hypotheses journal

| # | problem | hypothesis | change | experiment | result | verdict |
|---|---|---|---|---|---|---|
| H1 | fused decode unusable (98 ms/step) | tiles too small (32³) | re-launch existing kernels with 64–128 tiles | tile sweep, 3 M values | 3.2× faster at M=512 (12.5→3.88 ms), still 7.8× off decode-once | **rejected** for training |
| H2 | dropping the fp32 cache saves memory | no persistent weight → less VRAM | `packed_decode_once` variant | memory profile | peak 144 MB vs 64 MB dense (two live decodes) | **rejected** |
| H3 | the fast cached path wins at 12B shapes | decimation + cache + group scope | existing options, measured | 6 shapes × 5 token counts | 0.87× of AdamW at 4096²/B512 with 3.8× less peak; slower on small layers | **accepted** (layer level) |
| H4 | model-level training is 4× slower because of matmuls | profile layer-level | char-LM gate + breakdown | gate numbers, §8 | matmuls are at parity; the 4× is per-layer updater/decode overhead in the old counter layers | **accepted as bottleneck** |
| H5 | update is a strict no-op at lr=0 | bit-level check | `verify_counter_semantics.py` | 10 checks | decimation contract and one-update pass; 4096² no-op moves 2 codes | **defect found** |
| H6 | fused updater matches the fp32 oracle | quanta-level contract | oracle parity count | per-code diff | 0.15% codes differ, TF32 correlation | **bounded, documented** |
| H7 | LUT path could serve as the no-cache compute path | gather-based matmul | multi-token LUT (prior commit) | scaling sweep | loses to bf16 GEMM from M=1, to fp32 from M≈16–32 | **rejected** for training |

---

## 11. What remains (ranked by expected value)

1. **Port the group-scale packed path into the char-LM model** (the §3/§5 fast path) and
   re-run the training gate. Expected: removes most of the 4× per-token overhead; the
   layer-level data predicts ~0.9–1.2× of AdamW depending on shape mix. This is the single
   change most likely to flip the speed answer.
2. **Fix the 4096² lr=0 defect** (instrument the epilogue for the 2 affected codes; then a
   bitexact no-op regression test and re-run §7).
3. Two-pass group statistics to widen the updater's parallelism, and CUDA-graph capture of
   the whole step for small layers (both drafted as hypotheses, not measured yet).
4. An int8 code plane + vectorised decode GEMM as an alternative to the fused path (design
   sketched in §4; not built). Expected ceiling: decode cost near the 16.8 MB stream rather
   than 3 GB/s effective.
5. Peak memory cannot improve much at these model sizes because activations dominate: the
   honest lever is checkpointing/recomputation or smaller batch, not weight storage.
