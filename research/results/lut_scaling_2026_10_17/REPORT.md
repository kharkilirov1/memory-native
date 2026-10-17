# LUT-GEMM scaling to multiple tokens, and a fair counter-vs-AdamW re-measurement

**Date:** 2026-10-17 · **Branch:** `research/lut-multi-token` (main untouched)
**Measured commit:** `ab39b3f3b13fc663f58730e25eaf214f6ec17570`
**Hardware:** NVIDIA A100-SXM4-40GB (39.5 GiB, 108 SMs, sm_80), 12 vCPU, 83 GiB host RAM
**Software:** Python 3.13.15, torch 2.11.0+cu130, CUDA 13.0, Triton 3.6.0
**Raw data:** `results.csv`/`results.json` (eager sweep), `graph_results.csv`/`.json`
(CUDA-graph replay), `counter_vs_adam.csv`/`.json` (training-step A/B + verification),
`latency_scaling.svg` (figure). Harnesses: `scripts/bench_lut_scaling.py`,
`scripts/bench_lut_graph.py`, `scripts/bench_counter_vs_adam.py`.
Kernel module: `src/memory_native/lut_scaling.py`; tests: `tests/test_lut_scaling.py`
(16 passed on the A100: 4 CPU + 12 CUDA).

This is an independent reproduction and extension of the earlier M=1 result
(`research/results/LUT_GEMM_2026_10_17.md`), not a re-statement of it.

---

## 1. What was built

The prior module supported **one token per launch** (`triton_lut_group_matmul` raises for
M > 1) and used one atomic add per (row, group). This branch adds a real multi-token
pipeline in `src/memory_native/lut_scaling.py`, split so every cost can be measured and
charged:

| stage | kernel | what it does | scales with |
|---|---|---|---|
| 1. weight-side index | `_ternary_index_kernel` | packed 6-bit state `[N, 3P]` → LUT index table `[N, P]` int8 (0..80) | weights only, built **once per weight version** |
| 2. LUT build | `_lut_build_kernel` | activations `[M, K]` (act order) → `[M, P, 81]` fp32 partial sums (4 broadcast FMAs over the 81 base-3 digits, no 81 dot products) | M × K/4 × 81 |
| 3. gather | `_lut_multi_kernel` | `out[M, N] = Σ_g scale[n,g] · Σ_{p∈g} LUT[m,p,idx[n,p]]`, **no atomics**, one store per output element, groups looped in-kernel | M × N × K/4 scalar loads |

Semantics: **strict ternary only** (`residual_alpha = 0`), **no salient channel**,
no training gradients. Those modes are explicitly unsupported and the code says so; the
homotopy (`alpha > 0`) and salient paths remain on the existing kernels.

Correctness is pinned by `tests/test_lut_scaling.py`: CPU algebra agreement with the
dense product and with the existing reference; CUDA exactness of each kernel against its
torch counterpart and of the composed matmul against `x @ visible_weight().T` for
M ∈ {1, 2, 8, 33, 64}; and `amortized == total`. Measured errors over the whole sweep are
**2.3e-7 … 2.2e-6 relative** (fp32 rounding; dense fp32 is the reference).

Two measurement harnesses were added because they answer different questions:

- **eager** (`bench_lut_scaling.py`): what a loop pays per call, Python dispatch and
  Triton launch marshalling included. In this Colab VM a single small Triton launch costs
  tens of µs, which dominates every path at small M.
- **CUDA-graph replay** (`bench_lut_graph.py`): capture once, replay as one launch. This is
  the GPU-side ceiling and the honest basis for comparing kernels.

Both use CUDA events, ≥10 warmup calls, `iters` per rep, median over `reps`, IQR reported,
first-call (JIT) cost measured separately (never inside steady state), and per-path peak
allocated memory via reset/reset-peak.

---

## 2. Headline numbers (4096×4096, CUDA-graph replay, ms)

| M | dense fp32 | dense bf16 | fused decode (repo) | LUT gather only | LUT full (idx+prep+gather) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0557 | **0.0315** | 1.3294 | 0.0376 | 0.1347 |
| 2 | 0.0658 | **0.0317** | 1.3246 | 0.0381 | 0.1343 |
| 8 | 0.0854 | **0.0333** | 1.3018 | 0.0597 | 0.1563 |
| 32 | **0.0948** | 0.0357 | 1.3456 | 0.1843 | 0.2900 |
| 128 | 0.2547 | **0.0520** | 3.3118 | 0.7852 | 0.9181 |
| 512 | 1.0673 | **0.1107** | 12.495 | 4.0220 | 4.2319 |

1024×1024 (graph, ms): at M=1 dense fp32 **0.0069**, dense bf16 0.0136, fused decode
0.2280, LUT gather 0.0134; at M=512 dense fp32 0.0767, bf16 **0.0188**, fused decode
0.8364, LUT gather 0.2017.

Eager (per-call, ms) at 4096²: M=1 dense fp32 0.0587, LUT amortized 0.0482, LUT total
0.2107, fused decode 1.3349; M=512 dense fp32 1.0724, dense bf16 0.1164, LUT amortized
3.9931, LUT total 4.2804, fused decode 12.5048.

Figure: `latency_scaling.svg` (latency vs M for 1024²/4096², ratio to dense bf16, and the
LUT-vs-fused-decode speedup).

---

## 3. The seven questions, answered from the measurements

### 3.1 Why does LUT-GEMM speed up one token?

Three distinct effects, and only two of them are about LUT itself:

1. **The in-repo fused-decode kernel wastes 31/32 of its work at M=1.** It tiles M in
   32-row blocks, so a single token runs a 32-row tensor-core GEMM. Measured 1.33 ms at
   4096²/M=1 = **9.4 GB/s of packed state**, against a 7 µs memory roofline (12.6 MB at
   1.8 TB/s) — i.e. ~190× off. LUT's gather has no M padding: 0.0376 ms, **35× faster**
   than that kernel.
2. **LUT streams 6-bit codes, not fp32 weights, and turns MACs into table adds.** At M=1
   it beats cuBLAS **fp32** GEMV: 0.0376 vs 0.0557 ms (1.48×), because dense streams
   67 MB of fp32 weight while LUT streams 12.6 MB of packed state plus a 324 KiB table
   per token (L2-resident) and pays one add per 4 weights instead of decode+multiply.
3. **Against cuBLAS bf16 the M=1 LUT does not win**: 0.0376 vs 0.0315 ms. bf16 GEMV
   streams 33.5 MB and uses tensor cores; the LUT gather is scalar ALU/L2 traffic.

The earlier reported "≈2.7×" is reproduced as **6.3× eager** (0.211 vs 1.335 ms) and
**35× under graph replay** at 4096²/M=1 — but that speedup is relative to the repo's own
fused-decode kernel, which is far from its roofline, not relative to cuBLAS. This report
therefore does not claim a 2.7× (or 35×) win over dense GEMM.

### 3.2 Can it scale to 512 tokens?

**Functionally yes, performance-wise no.** The multi-token kernels are exact for
M = 1…512 (max relative error 2.2e-6, and ≤4e-7 up to M=256), and the pipeline runs
end-to-end for all M in the sweep. But the gather cost grows with M: 0.0376 → 4.02 ms
(107× for 512× more tokens), because the inner loop is `M·N·K/4` scalar L2 loads — 2.1
billion gathers at 4096²/M=512, ≈2–3 TB/s effective L2 bandwidth, no tensor cores. Dense
bf16 does the same arithmetic at 0.111 ms.

### 3.3 At what batch size does LUT stop winning?

| baseline | LUT wins at | crossover | loses from |
|---|---|---|---|
| repo fused-decode kernel | **all M** (35× at M=1 → 3.1× at M=512) | — | never |
| cuBLAS fp32 GEMM | M ≤ 8 (graph: 0.0376 vs 0.0557 at M=1; 0.0597 vs 0.0854 at M=8) | **M ≈ 16–32** (eager: LUT 0.0906 vs dense 0.0915 at M=16, LUT loses at M=32) | M ≥ 32 |
| cuBLAS bf16 GEMM | never at 4096² (0.0376 vs 0.0315 at M=1) | below M=1 | M ≥ 1 |
| anything, 1024² layers | never | — | M ≥ 1 (dense fits in cache: 0.0069 ms) |

Explanation, measured rather than asserted: the LUT path's work is
`M·(K/4)·81` prep adds plus `M·N·(K/4)` scalar table loads. The gather is memory/ALU-bound
(~2–3 TB/s effective), while a tensor-core GEMM turns the same MACs at 100–300 TFLOP/s.
The LUT's advantage is not arithmetic — it is that it never materializes 4 B/param weights
and never pads M. Both advantages disappear as soon as M is large enough to amortize the
weight stream, which is exactly the M≈8–32 region measured above.

### 3.4 Can LUT be combined with Memory Native without losing correctness?

Yes, at the kernel level: the LUT kernels consume the *same* packed 6-bit state, the same
group scales and the same act-order permutation; nothing about the packed format changes,
and the path is bit-identical to decoding `W` and multiplying, up to fp32 summation order
(max rel err 2.3e-7 at M=1, 2.2e-6 at M=512; tests pin CPU and CUDA behavior).
It is an **inference-only, strict-α=0** path: `alpha>0` homotopy, the salient channel and
gradients are unsupported. A serving path would use the weight-side index table built once
per weight version (the amortized numbers above) and could be graph-captured.

### 3.5 Is the 6-bit counter-updater really faster than AdamW under a fair comparison?

Verification first (this was the part most likely to be wrong):

- Dense baseline: gradient present before the step, parameter **values changed** after it,
  AdamW state non-empty — verified for all three shapes.
- Counter layer: the updater runs inside the autograd Function's `backward`. The harness
  counts `update_events` and asserts **exactly one update per step** (delta == out·in),
  twice in a row, and that the packed state changed — verified for decimation 1/4/8 and
  with the weight cache on and off. No double update, no missed update.

Measured full steps (eager, batch 512, ms):

| shape | dense `nn.Linear`+AdamW | counter dec1 | dec4 | dec8 |
|---|---:|---:|---:|---:|
| 1024² | **0.643** | 0.790 (1.23×) | 0.875 | 0.857 |
| 2048² | 0.961 | **0.859 (0.89×)** | 1.293 | 1.177 |
| 4096² | 3.104 | 3.286 (1.06×) | **3.017 (0.97×)** | 3.054 (0.98×) |

Forward is at parity or better (4096²: 1.078 counter vs 1.125 dense). So the honest
statement is: **at 4096² with decimation 4/8 the counter step is ~3% faster than a real
AdamW step; at 2048² dec1 is 11% faster; at 1024² the counter path is 23% slower** in this
harness. Two caveats that matter:

- Decimated steps are a **different algorithm** (only 1/dec of groups move per step); they
  are not a cheaper equivalent of an AdamW step. Comparing them is only meaningful with the
  lr compensation and quality control described in `research/results/DECIMATION_WITNESS*`.
- Cross-harness variance is real. The earlier warm campaign
  (`research/results/SPEED_VS_ADAM_2026_10_17.md`) measured 0.83–0.91× at 4096²/B=512 with
  more iterations and cached decimation indices; this independent harness rebuilds host-side
  state every call and reports 0.97–1.06×. The direction is consistent (counter ≈ AdamW or
  better at 4096², worse on small layers), the exact percentage depends on harness
  overhead and should always be quoted with its configuration.

### 3.6 Does the final system reduce time-to-equal-quality?

**Not measured in this session, and this report does not claim it.** A controlled
time-to-loss experiment (same architecture, tokenizer, data, init, token budget and
eval procedure, counter vs dense) needs a multi-hour KD/CharLM training run; it was not
run here. What is established is the per-step cost side (3.5) and bit-exact kernel
semantics for strict α=0; the loss-vs-wall-clock comparison remains open.

### 3.7 Which change produced the largest measured win?

1. **Not padding M / not decoding per weight** — LUT gather vs the repo's fused-decode
   kernel: **35×** at M=1, 3.1× at M=512 (graph replay).
2. **Amortizing the weight-side index table** (build once per weight version instead of per
   call): 0.1347 → 0.0376 ms at 4096²/M=1, i.e. **3.6×** within the LUT path; this is the
   difference between "LUT total" and "LUT amortized" in every table.
3. In the separate training-speed campaign (main-branch, reported in
   `SPEED_VS_ADAM_2026_10_17.md`): the cached decoded weight (fwd/grad_x parity at 1.00×)
   and the fused updater emitting the dense weight (removing a per-step decode) are what
   brought the counter step to 0.83–0.91× of AdamW at 4096².

---

## 4. VRAM accounting (4096×4096 layer)

| object | bytes |
|---|---:|
| packed 6-bit state (0.75 B/param) | 12.58 MB |
| weight-side index table `[N, K/4]` int8 | 4.19 MB |
| LUT `[M, K/4, 81]` fp32 | M=1: 1.33 MB · M=32: 42.5 MB · M=128: 170 MB · M=512: 682 MB (not materialized in the sweep beyond what peak reports show) |
| measured peak, LUT total | M=1: 4.3 MB · M=32: 15.1 MB · M=128: 48.5 MB · M=512: 182 MB |
| measured peak, LUT gather only (amortized) | output only: 8 MB at M=512 |
| measured peak, fused decode | 8 MB at M=512 |
| measured peak, dense `nn.Linear`+AdamW full step | 64 MB at 4096²/B=512 |
| measured peak, counter full step (cache on) | 24 MB at 4096²/B=512 |
| measured peak, counter full step (cache off) | 80 MB (the fp32 weight cache is 67 MB — this is the cost of the cache) |

The LUT preprocessing buffer is the dominant transient cost at large M; a serving engine
that tiles tokens would cap it at the per-tile LUT size.

---

## 5. Improvements, before → after

| change | before | after | evidence |
|---|---|---|---|
| multi-token LUT | M=1 only (older kernel raised for M>1) | M=1…512 exact | `results.csv`, tests |
| LUT vs repo fused decode, 4096², M=1 (graph) | 1.3294 ms | 0.0376 ms gather / 0.1347 full | `graph_results.csv` |
| LUT vs repo fused decode, 4096², M=512 (graph) | 12.495 ms | 4.022 ms gather | same |
| LUT vs fused decode, eager M=1 | 1.3349 ms | 0.2107 ms | `results.csv` |
| index-table amortization (per call) | 0.1347 ms | 0.0376 ms | same |
| counter vs AdamW, 4096², B=512 | 1.06× (dec1) | 0.97× (dec4) / 0.98× (dec8) | `counter_vs_adam.csv` |
| update-count correctness | unverified before | exactly 1/step, dec1/4/8, twice | `counter_vs_adam.json` checks |

Negative results worth keeping:

- LUT **loses to bf16 tensor-core GEMM at every M** on this device and to fp32 GEMM from
  M≈16–32; it is not a general GEMM replacement.
- At 1024² even M=1 goes to dense fp32 (0.0069 vs 0.0134 ms): small layers fit in cache and
  the LUT machinery cannot pay for itself.
- The in-repo fused-decode kernel is ~20–190× above its memory roofline and degrades with M
  (3.3 ms at M=128). That is a separate, larger opportunity than LUT: fixing its M tiling
  would beat both current paths at M ≤ 32.
- Multi-token LUT does not need atomics (the M=1 kernel's atomic design is removed), but a
  smaller `block_n` sweep showed the gather is flat vs M up to ~16 only because it is
  launch-bound, not because it is efficient.

---

## 6. Reproduction

```bash
# exact state
git fetch origin research/lut-multi-token
git checkout research/lut-multi-token
git rev-parse HEAD          # measured runs: ab39b3f3b13fc663f58730e25eaf214f6ec17570
python -m pip install -e .

# correctness (16 tests; CPU parts run anywhere, CUDA parts need a GPU)
python -m pytest tests/test_lut_scaling.py -q

# 1) eager sweep: M=1..512 x 1024^2/2048^2/4096^2, 6 paths, JSON+CSV
python scripts/bench_lut_scaling.py --shapes 1024,2048,4096 \
    --batches 1,2,4,8,16,32,64,128,256,512 --iters 20 --reps 3 \
    --out-dir research/results/lut_scaling_2026_10_17

# 2) CUDA-graph replay sweep (GPU ceiling)
python scripts/bench_lut_graph.py --shapes 1024,4096 \
    --batches 1,2,8,32,128,512 --out-dir research/results/lut_scaling_2026_10_17

# 3) counter vs AdamW with verification checks
python scripts/bench_counter_vs_adam.py --shapes 1024,2048,4096 --batch 512 \
    --iters 20 --reps 3 --out-dir research/results/lut_scaling_2026_10_17
```

Each script prints an `artifacts:` line with its JSON/CSV paths. Plot: regenerate the SVG
with `python research/results/lut_scaling_2026_10_17/_make_plot.py` (stdlib only).

Colab: `notebooks/LUT_Scaling_A100.ipynb` performs all of the above in order with
"Run all" (it pins the same commit and writes the same artifacts).

## 6b. Notebook validation (executed end-to-end)

`notebooks/LUT_Scaling_A100.ipynb` was executed with
`jupyter nbconvert --to notebook --execute` on the same A100 runtime at commit
`fa14032` (clone + pin + install + tests + all three harnesses + plot + tables):
8/8 code cells executed, **0 error outputs**, 1 plot, and it regenerated the artifact
files under `/content/results/lut_scaling_2026_10_17/`. The second, independent run
reproduced the headline numbers within noise (4096^2, graph): dense fp32 0.0551 vs 0.0557
ms, dense bf16 0.0286 vs 0.0315, fused decode 1.3289 vs 1.3294, LUT gather 0.0375 vs
0.0376, LUT full 0.1349 vs 0.1347. Its crossover table (reproduced):
LUT/dense_fp32 = 0.68x at M=2, 0.70x at M=8, 1.95x at M=32, 3.09x at M=128, 3.73x at
M=512 — the same M~16-32 crossover.

## 7. Limits of this study

- Single device (A100-SXM4-40GB in a shared Colab runtime); medians over 3–7 reps with IQRs
  are reported, but absolute times on a shared VM carry noise — ratios are the stable part.
- Eager timings include this VM's high per-launch host overhead (tens of µs), which is why
  both eager and graph numbers are published rather than either alone.
- Strict ternary only; `alpha>0`, salient and training gradients are out of scope here.
- No controlled time-to-quality training experiment (see 3.6).
- The counter-vs-AdamW comparison uses one layer per measurement; whole-model composition
  and small-batch regimes are covered in `SPEED_VS_ADAM_2026_10_17.md`, not re-run here.
