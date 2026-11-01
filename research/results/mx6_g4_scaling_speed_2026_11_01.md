# mx6 on G4: scaling with width, 4-seed means, step speed and memory - 2026-11-01

Raw evidence (31 unedited files, copied byte-for-byte from the Colab G4 archive `q_logs.tar`, sha256 `5901E70F...12BA`; the copy was checked file by file against the extracted tree): `mx6_g4_scaling_speed_2026_11_01/` (`q/` training logs, `bench_out/` + `bench_*.log` speed/memory benchmark). `computed_tables.md` in the same folder is the unedited output of the two analysis scripts (`analysis_parse.py.txt`, `analysis_tables.py.txt`, stored as .txt; they read the absolute paths of the extraction folder) that produced every mean, sd, difference and ratio below.
Continues `mx6_g4_headroom_lg_2026_10_31.md` (same harness; arms, h=2 ladder headroom, latent grid). Code of these runs: commit `e9830bc` (stated in the task; the logs print no commit and no hashes, so it is not verifiable from the files).

Every number is copied from the logs/bench files or derived from them (marked "derived"). Statements are labelled **[verified]** (read off the logs or computed from them) or **[hypothesis]** (not tested by these runs).
sd = sample sd (n-1), the logs' convention. "M2" = rule M2 of `docs/LESSONS.md`: a difference smaller than 3x the larger sd of the two arms is "not detectable". **Every scaling arm here has n=2 seeds** (M2 asks for 4); only the d=128 tinyshakespeare block (section 2) has n=4.

## TL;DR

1. **[verified]** The gap to the best-tuned dense shrinks with width for the stateless arms (enwik8, 4 layers, 2 seeds): `lg6_fp4_h2` +0.283 -> +0.231 -> +0.179 (d=128/256/512), `tern_counter` +0.380 -> +0.299 -> +0.200, `qat_fp4` (fp32 latent) +0.029 -> +0.016 -> +0.000. `na_fp4_c8` is flat within its noise (+0.143 / +0.134 / +0.106; seed sd 0.025 / 0.035 / 0.002), so its trend is not detectable. Endpoint change vs the larger sd: lg6 4.6x, tern 11x (both pass M2), na 1.0x (does not).
2. **[verified]** AdamW lr 3e-3 DIVERGES at d=512 for `dense_nobias` and `qat_fp4` (val 3.17 vs 1.80 at 1e-3; both seeds); the counter / grid arms, whose emb / pos / LN / head still use adam-lr 3e-3, train normally at d=512. Dense best of {5e-4, 1e-3, 3e-3} is 1e-3 (1.7982). All d=512 gaps are taken against that tuned dense. d=128 / 256 dense were not re-tuned (still 3e-3).
3. **[verified]** 4-seed d=128 tinyshakespeare (seeds 0-3, mixed code snapshot): `na_fp4_c8_h2` **2.3172** +- 0.0087, `lg7_fp4_h2` **2.3530** +- 0.0123, `lg6_fp4_h2` **2.4141** +- 0.0059. Against `dense_bf16` 2.2864 (decomp, other snapshot): +0.031 / +0.067 / +0.128. The 2-seed numbers of the previous report moved by at most 0.0021.
4. **[verified]** Speed, eager, fp32 GEMMs (no autocast), vs dense eager at d=128/256/512: `lg6` 3.28x / 2.94x / 1.38x, `na_fp4_c8` 8.19x / 7.23x / 3.44x, `tern_counter` 3.61x / 3.41x / 1.72x, `qat_fp4` 3.09x / 2.66x / 1.49x. lg is 2.5x faster than na at every d. With `torch.compile`: lg 2.41x / 2.18x / **1.11x** (1.22x vs dense+compile at d=512), na 5.75x / 5.09x / 2.64x. `--fast-pack` takes 14-15 % off the na step.
5. **[verified]** Memory: block-linear training state is 15x smaller for lg6 (9.82 MB vs 144.00 MB at d=512), 11x for na and resident memory 3x lower (72 vs 211 MB), but **peak** memory is not: lg6 794 MB vs dense 827 MB (-4 %), na 857 MB (+4 %), tern 673 MB (-19 %). At equal state memory lg6 d=512 (9.82 MB) beats dense d=128 (9.00 MB) by 0.043 bits/char (derived); at equal PEAK memory dense wins at every size measured.

## Setup

| item | value |
|---|---|
| GPU | `NVIDIA RTX PRO 6000 Blackwell Server Edition` (log headers); bench meta: torch 2.11.0+cu130, triton 3.6.0, CUDA 13.0 (the training logs print no versions) |
| scaling data | enwik8, byte-level latin-1, `--chars 20000000`, vocab 202; contiguous split train = first 90 % (18,000,000), val = last 10 % (2,000,000); `epochs=1.37` (underfit regime, unlike tinyshakespeare's 24.5 epochs) |
| scaling model / schedule | causal transformer, 4 layers, heads = d/64, mlp 4d, seq 128, batch 32, 6000 steps, cosine (warmup 0.02, min 0.1), eval every 500; d = 128 / 256 / 512 with 786,432 / 3,145,728 / 12,582,912 block-linear params (total 905,984 / 3,384,832 / 13,061,120; other = emb/pos/LN/head 119,552 / 239,104 / 478,208) |
| arms | `dense_nobias` (AdamW), `qat_fp4` (fp32 latent + fake-quant E2M1, `qat_lr` = adam-lr), `tern_counter` (`tern_lr=0.03`, C=11, from the log), `na_fp4_c8` (counter C=8), `lg6_fp4_h2` (latent grid k=6, `probes/lg_ab.py`); counter / grid: `--lr 0.01 --scale-format nvfp4 --headroom-mode ladder --headroom 2` |
| adam-lr | default 3e-3 for the dense parts (dense/qat: all weights; counter, tern and grid arms: emb, pos, LN, head); extra d=512 runs at 1e-3 (dense, qat, lg6) and 5e-4 (dense) |
| seeds | 0, 1 for every scaling arm (n=2) |
| 4-seed block | tinyshakespeare, d=128, **2 layers, 1 head** (log header), 6000 steps, cosine, lr 0.01, h=2, held-out last 10 % (`val_tokens=111540`), the same setup as the headroom/lg report. Seeds 2, 3 from this batch, seeds 0, 1 from `mx6_g4_headroom_lg_2026_10_31/{fp4c8_h2,lg6_fp4_h2,lg7_fp4_h2}.log` (older uncommitted snapshot; equivalence with `e9830bc` is assumed and not checked, no hash comparison) |
| bench | `probes/bench_train_step.py`-style: batch 32 x seq 128, 4 layers, heads d/64, vocab 256, warmup 30 + 100 timed steps, phases from 20 steps, `matmul_precision highest` (**no autocast: every GEMM is fp32 simt sgemm, dense included**), two runs, `bench_std` (eager; arms dense, qat, tern, na, lg6, lg8) and `bench_x` (adds `--compile`, `--fast-pack`). The task states they ran alone on the idle GPU; the files cannot confirm that. `bench_smoke.*` / `smoke.*` are in the archive and were not used |

Arm flags (headroom mode, h, lr, `--chars`) are not all printed in the log headers; they are taken from the task description. The wall-times in the `q/` logs (e.g. 2136 s vs 351 s for two seeds of the same na arm) suggest the training jobs shared the GPU, so they are not speed data; speed comes only from `bench_out/`.

## 1. Scaling with width (enwik8, held-out, 2 seeds)

Val = bits/char at step 6000 on the 2 M-char tail. `train` = loss on a fixed train subset; `gap val-train` is the logs' own `gap`. `edge` = share of deployed weights on the outer rungs (`tern_counter`: the share on the +-1 states, so not comparable with the E2M1 arms); `flip` = share of deployed weights whose rung changed in the last eval interval; `clip` = clipped updates (counter) or requested moves beyond the grid ends (lg), not comparable across families. `kurt` as in the headroom report (Gaussian 3, uniform 1.8). Bits/param is the harness number; it falls with d because the per-row/tensor scale state is amortised.

| d | arm (adam-lr) | bits/param | val mean +- sd (n) | per-seed val | train | gap val-train | edge % | flip % (end) | kurt | clip % |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| 128 | dense_nobias | 96.00 | **2.0200** +- 0.0035 (2) | 2.0225, 2.0175 | 1.8820 | +0.1380 | - | - | 3.04 | - |
| 128 | qat_fp4 | 96.00 | **2.0491** +- 0.0023 (2) | 2.0474, 2.0507 | 1.9188 | +0.1302 | - | - | 3.13 | - |
| 128 | tern_counter | 6.56 | **2.3996** +- 0.0013 (2) | 2.3987, 2.4006 | 2.2959 | +0.1038 | 75.75 | 21.7 | 1.33 | - |
| 128 | na_fp4_c8 | 8.69 | **2.1631** +- 0.0250 (2) | 2.1454, 2.1808 | 2.0504 | +0.1127 | 21.11 | 51.5 | 2.17 | 0.01 |
| 128 | lg6_fp4_h2 | 6.69 | **2.3025** +- 0.0037 (2) | 2.2999, 2.3051 | 2.1905 | +0.1120 | 14.72 | 68.3 | 2.33 | 2.43 |
| 256 | dense_nobias | 96.00 | **1.8508** +- 0.0059 (2) | 1.8550, 1.8466 | 1.7109 | +0.1398 | - | - | 3.27 | - |
| 256 | qat_fp4 | 96.00 | **1.8663** +- 0.0086 (2) | 1.8602, 1.8724 | 1.7289 | +0.1373 | - | - | 3.33 | - |
| 256 | tern_counter | 6.28 | **2.1495** +- 0.0149 (2) | 2.1601, 2.1390 | 2.0530 | +0.0965 | 75.37 | 18.6 | 1.33 | - |
| 256 | na_fp4_c8 | 8.59 | **1.9847** +- 0.0352 (2) | 1.9598, 2.0096 | 1.8667 | +0.1179 | 17.62 | 47.3 | 2.42 | 0.01 |
| 256 | lg6_fp4_h2 | 6.59 | **2.0821** +- 0.0003 (2) | 2.0823, 2.0819 | 1.9712 | +0.1108 | 12.96 | 67.0 | 2.48 | 2.16 |
| 512 | dense_nobias (adam-lr 3e-3) DIVERGED | 96.00 | **3.1658** +- 0.1228 (2) | 3.2527, 3.0790 | 3.0831 | +0.0828 | - | - | 3.33 | - |
| 512 | dense_nobias (adam-lr 1e-3) | 96.00 | **1.7982** +- 0.0011 (2) | 1.7974, 1.7989 | 1.6517 | +0.1464 | - | - | 3.04 | - |
| 512 | dense_nobias (adam-lr 5e-4) | 96.00 | **1.8563** +- 0.0072 (2) | 1.8614, 1.8512 | 1.7170 | +0.1393 | - | - | 2.75 | - |
| 512 | qat_fp4 (adam-lr 3e-3) DIVERGED | 96.00 | **3.1691** +- 0.0033 (2) | 3.1668, 3.1714 | 3.0813 | +0.0878 | - | - | 3.45 | - |
| 512 | qat_fp4 (adam-lr 1e-3) | 96.00 | **1.7983** +- 0.0071 (2) | 1.8033, 1.7933 | 1.6565 | +0.1418 | - | - | 3.15 | - |
| 512 | tern_counter | 6.14 | **1.9982** +- 0.0083 (2) | 2.0041, 1.9923 | 1.8918 | +0.1064 | 75.16 | 15.6 | 1.34 | - |
| 512 | na_fp4_c8 | 8.55 | **1.9042** +- 0.0023 (2) | 1.9026, 1.9059 | 1.7723 | +0.1319 | 13.90 | 43.2 | 2.70 | 0.00 |
| 512 | lg6_fp4_h2 (adam-lr 3e-3) | 6.55 | **1.9863** +- 0.0204 (2) | 1.9719, 2.0008 | 1.8717 | +0.1146 | 11.74 | 64.0 | 2.62 | 1.96 |
| 512 | lg6_fp4_h2 (adam-lr 1e-3) | 6.55 | **1.9771** +- 0.0226 (2) | 1.9931, 1.9612 | 1.8595 | +0.1176 | 12.01 | 63.6 | 2.59 | 2.01 |

**Reading [verified].**
- Dense at d=512: 3e-3 diverged (3.2527 / 3.0790, per-seed), 1e-3 = **1.7982** +- 0.0011, 5e-4 = 1.8563 +- 0.0072 (+0.0581, 8.1x the larger sd). So 1e-3 is the best of the three lrs tried; the optimum may lie between 5e-4 and 3e-3 (not swept). `qat_fp4` shows the same split: 3e-3 diverged (3.1691), 1e-3 = 1.7983. Dense and qat at d=128 / 256 were run at 3e-3 only and trained normally (2.0200 / 1.8508 and 2.0491 / 1.8663), so their lr was not re-tuned at those widths either (M8).
- The arms whose block linears are not trained by AdamW (counter / grid / tern) did not diverge at d=512 although their emb / pos / LN / head use the same 3e-3: so the divergence is in the AdamW update of the block linears at that lr, not in the fp parts [inference from the contrast].
- `lg6` at d=512 with adam-lr 1e-3 vs 3e-3: 1.9771 vs 1.9863 (-0.0092, both seeds spread 0.02; not detectable). This is the only measurement of how much the fp-part lr matters to a counter/grid arm at d=512; it bounds the confound for lg6 to about 0.01 (**[hypothesis]** that the same holds for `na_fp4_c8` and `tern_counter`, which were run only at 3e-3).
- Val falls with width for every arm; dense -0.169 then -0.053 (d=512 at tuned lr), `lg6` -0.220 then -0.105, `tern_counter` -0.250 then -0.151, `na_fp4_c8` -0.178 then -0.080.

### Gap to dense and its trend

Delta = arm val - dense_nobias val at the same d (negative would mean better than dense). At d=512 the reference is the best-tuned dense, **adam-lr 1e-3 (1.7982)**; `qat_fp4` and `lg6` use their own adam-lr 1e-3 runs in the main rows so that they differ from dense only in the arm.

| arm | d=128 | d=256 | d=512 | change 128->512 | note |
|---|---:|---:|---:|---:|---|
| qat_fp4 | +0.0291 | +0.0155 | +0.0001 | -0.0289 | adam-lr 1e-3 at d=512 (same as the dense reference) |
| tern_counter | +0.3796 | +0.2987 | +0.2001 | -0.1796 | fp emb/LN/head at adam-lr 3e-3 at d=512 (confound) |
| na_fp4_c8 | +0.1431 | +0.1339 | +0.1061 | -0.0370 | fp emb/LN/head at adam-lr 3e-3 at d=512 (confound) |
| lg6_fp4_h2 | +0.2825 | +0.2313 | +0.1790 | -0.1035 | adam-lr 1e-3 at d=512 (same as dense) |
| lg6_fp4_h2 (adam-lr 3e-3 at d=512; dense ref still a1e-3) | +0.2825 | +0.2313 | +0.1882 | -0.0943 | unconfounded vs 3e-3 dense is impossible: dense diverged |

Sample sd of the Delta (sqrt of the two arms' variances, n=2): qat 0.0042 / 0.0105 / 0.0072; tern 0.0038 / 0.0161 / 0.0084; na 0.0253 / 0.0357 / 0.0026; lg6 0.0051 / 0.0059 / 0.0226. Relative to dense val: qat 1.4 % / 0.8 % / 0.0 %, tern 18.8 % / 16.1 % / 11.1 %, na 7.1 % / 7.2 % / 5.9 %, lg6 14.0 % / 12.5 % / 10.0 %. Against `qat_fp4` instead of dense: tern +0.351 / +0.283 / +0.200, na +0.114 / +0.118 / +0.106, lg6 +0.253 / +0.216 / +0.179.

Trend per arm:
- **`lg6_fp4_h2`: +0.283 -> +0.231 -> +0.179 [verified].** Monotone; step 128 -> 256 is -0.051 (Delta sd 0.006, clear), step 256 -> 512 is -0.052 (Delta sd 0.023, 2.3x: below M2), endpoint change -0.104 = 4.6x the larger sd (M2 passes). With the d=512 lg6 run at 3e-3 the last point is +0.188 and the endpoint change -0.094 (4.6x of the Delta sd 0.0204).
- **`tern_counter`: +0.380 -> +0.299 -> +0.200 [verified].** Both steps clear M2 (5.0x, 11.7x of the larger sd). Confound: fp parts at 3e-3 at d=512.
- **`na_fp4_c8`: +0.143 -> +0.134 -> +0.106, no detectable trend [verified numbers].** Seed spread is large at d=128 (2.1454 / 2.1808, sd 0.0250) and d=256 (1.9598 / 2.0096, sd 0.0352), then tiny at d=512 (1.9026 / 1.9059); seed 1 is the worse seed at all three widths. The endpoint change -0.037 is 1.0x the larger Delta sd (0.0357). The d=512 point is also confounded by adam-lr 3e-3 for the fp parts. The cause of the large small-d seed variance of `na_fp4_c8` is not investigated here; on tinyshakespeare its 4-seed sd is 0.0087 (section 2), the lg6 sd on enwik8 d=128 is 0.0037.
- **`qat_fp4`: +0.029 -> +0.016 -> +0.000 [verified].** The fake-quant cost against dense vanishes at d=512 (1.7983 vs 1.7982); the endpoint change is 4.0x the larger sd.
- **Mechanism [hypothesis]:** the counter / grid quantisation noise is roughly a fixed relative perturbation per weight, and a wider model averages it over more weights per output; nothing here tests that. Another candidate is that the quantised arms are further from convergence at d=128 (1.37 epochs) and gain more from the extra capacity. No ablation was run.
- **Against the O1 kill line (dense + 0.15 at d=128, gap shrinking at d=256-512)** on this corpus: at d=128 lg6 (+0.283) and tern (+0.380) are above the line, na (+0.143, sd 0.025) is at it; the second clause (gap not shrinking) is **not met** for lg6 and tern, so the kill criterion does not fire. The enwik8 d=128 gaps are not comparable with the tinyshakespeare ones (lg6 +0.128 vs dense_bf16 there): tinyshakespeare's dense overfits (val-train +0.36), enwik8's does not (val-train +0.14 at all d). A linear extrapolation of the lg6 trend (-0.052 per doubling) would cross +0.15 near d=750 (**[hypothesis]**, three points, two seeds).
- **Bits:** the arms sit at different state widths: na 8.55-8.69, lg6 6.55-6.69, tern 6.14-6.56 bits. lg6 vs tern at d=512: 1.9771 vs 1.9982, lg6 better by 0.0211 (larger sd 0.0226: not detectable), at 6.55 vs 6.14 bits.

## 2. Four-seed means, d=128 tinyshakespeare

| arm | bits | seeds 0,1 mean (old report) | seeds 2,3 mean | per-seed val (s0, s1, s2, s3) | 4-seed mean | sd (n-1) | train | val-train | edge % |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| na_fp4_c8 (h2) | 8.688 | 2.3151 | 2.3194 | 2.3191, 2.3111, 2.3100, 2.3287 | **2.3172** | 0.0087 | 1.9952 | +0.3220 | 24.09 |
| lg6_fp4_h2 | 6.688 | 2.4155 | 2.4127 | 2.4202, 2.4108, 2.4075, 2.4179 | **2.4141** | 0.0059 | 2.1395 | +0.2747 | 16.09 |
| lg7_fp4_h2 | 7.688 | 2.3513 | 2.3546 | 2.3586, 2.3440, 2.3417, 2.3676 | **2.3530** | 0.0123 | 2.0484 | +0.3046 | 18.99 |

- **[verified]** The 4-seed means differ from the 2-seed means of the previous report by +0.0021 (na), -0.0014 (lg6), +0.0017 (lg7); seeds 2, 3 alone differ from seeds 0, 1 by +0.0043 / -0.0028 / +0.0033. No snapshot drift of the size seen in decomp (+0.0117) is visible.
- Paired (same seed index; derived): `lg7_fp4_h2` - `na_fp4_c8_h2` = +0.0357 (per-seed +0.0395, +0.0329, +0.0317, +0.0389; 2.9x the larger sd 0.0123, below M2); `lg6_fp4_h2` - `lg7_fp4_h2` = +0.0611 (+0.0616, +0.0668, +0.0658, +0.0503; 5.0x, passes M2); `lg6_fp4_h2` - `na_fp4_c8_h2` = +0.0969 (11.2x, passes). Every seed has the same order na < lg7 < lg6, so the frontier ordering by bits is confirmed with 4 seeds.
- Against `dense_bf16` 2.2864 / `dense_nobias` 2.2891 / `qat_fp4` 2.2983 (decomp, 4 seeds, other snapshot; cross-run): na +0.0308 / +0.0281 / +0.0189; lg7 +0.0666 / +0.0639 / +0.0547; lg6 +0.1277 / +0.1250 / +0.1158. `lg6_fp4_h2` stays inside the O1 line (0.15) by 0.022, 3.8x its sd (cross-run reference).
- **Not covered:** the claim of R9 that the grid beats the counter at the same bits (lg6 vs `fp4c2_h2`, lg7 vs `fp4c4_h2`, lg8 vs `fp4c8_h2`) needs 4 seeds of the counter arms at 6.69 / 7.69 bits; those were not rerun, so O7's first question is still open. These 4-seed numbers only fix the grid's own levels and the lg-vs-na ordering.

## 3. Training-step speed (bench, eager vs compile)

Bench protocol and caveats first. **All GEMMs are fp32 simt sgemm (`cutlass_80_simt_sgemm`, `matmul_precision highest`, no autocast) for every arm, dense included.** That is the reference the earlier SPEED_NOTES estimates assumed (~60 TFLOPS), but it is not how dense would be trained in practice; with bf16/TF32 tensor cores the dense reference would get several times faster and every ratio below would grow (**[hypothesis]**, SPEED_NOTES: "TF32 would be several x less"). Eager rows come from `bench_std`; `+compile` / `+fastpack` rows from `bench_x`. The eager rows of the two runs agree within 0.9 % for every arm and width (dense, qat, na, lg6, lg8), so the two files are mutually consistent. `x dense eager` is the ratio to `dense_nobias` eager of the same d; `x dense+compile` is the ratio to the compiled dense (the fair reference for the `+compile` rows). `update` = time inside `apply_update_` (a subset of bwd), `decode` = `weight()`/`quantized_weight()` (a subset of fwd); `GPU kernel ms` = summed kernel time of one profiled step, `launches` = its kernel count. lg8 (k=8) has the same speed as lg6 within 0.2 % at every d (10.24 vs 10.22, 10.30 vs 10.31, 11.42 vs 11.42 ms), so the lg8 rows are listed for completeness only.

| d | arm / variant | ms/step | tok/s | x dense eager | x dense+compile | fwd | bwd | opt | update (in bwd) | decode (in fwd) | GPU kernel ms | launches |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | dense_nobias | 3.11 | 1,315,954 | 1.00x | 1.28x | 1.06 | 1.74 | 0.36 | - | - | 2.06 | 284 |
| 128 | dense_nobias+compile | 2.43 | 1,685,181 | 0.78x | 1.00x | 0.79 | 1.23 | 0.41 | - | - | 1.59 | 211 |
| 128 | qat_fp4 | 9.62 | 425,786 | 3.09x | 3.96x | 7.45 | 2.40 | 0.48 | - | 5.57 | 3.17 | 884 |
| 128 | qat_fp4+compile | 2.94 | 1,391,684 | 0.95x | 1.21x | 1.20 | 1.28 | 0.46 | - | - | 1.87 | 276 |
| 128 | tern_counter | 11.23 | 364,662 | 3.61x | 4.62x | 4.52 | 7.19 | 0.29 | 1.05 | - | 3.00 | 1036 |
| 128 | na_fp4_c8 | 25.50 | 160,616 | 8.19x | 10.49x | 7.36 | 19.49 | 0.29 | 16.06 | 3.64 | 5.67 | 2700 |
| 128 | na_fp4_c8+compile | 17.90 | 228,831 | 5.75x | 7.36x | 5.08 | 13.83 | 0.29 | 10.33 | 1.41 | 3.46 | 1324 |
| 128 | na_fp4_c8+fastpack | 21.64 | 189,297 | 6.95x | 8.90x | 5.57 | 17.43 | 0.29 | 14.10 | 3.52 | 4.97 | 2412 |
| 128 | lg6_fp4_h2 | 10.22 | 400,669 | 3.28x | 4.21x | 2.26 | 8.76 | 0.27 | 6.00 | 0.59 | 3.29 | 1116 |
| 128 | lg6_fp4_h2+compile | 7.49 | 547,164 | 2.41x | 3.08x | 2.27 | 6.03 | 0.28 | 3.19 | 0.58 | 2.37 | 572 |
| 128 | lg8_fp4_h2 | 10.24 | 400,150 | 3.29x | 4.21x | 2.30 | 8.81 | 0.28 | 6.02 | 0.60 | 3.30 | 1116 |
| 128 | lg8_fp4_h2+compile | 7.46 | 549,176 | 2.40x | 3.07x | 2.24 | 5.99 | 0.28 | 3.17 | 0.57 | 2.37 | 572 |
| 256 | dense_nobias | 3.50 | 1,170,499 | 1.00x | 1.21x | 1.21 | 2.12 | 0.17 | - | - | 3.43 | 289 |
| 256 | dense_nobias+compile | 2.90 | 1,410,542 | 0.83x | 1.00x | 1.04 | 1.70 | 0.17 | - | - | 2.78 | 216 |
| 256 | qat_fp4 | 9.31 | 439,935 | 2.66x | 3.21x | 7.32 | 2.39 | 0.29 | - | 5.42 | 4.76 | 901 |
| 256 | qat_fp4+compile | 3.26 | 1,258,232 | 0.93x | 1.12x | 1.28 | 1.69 | 0.28 | - | - | 3.08 | 281 |
| 256 | tern_counter | 11.94 | 342,925 | 3.41x | 4.11x | 4.56 | 7.62 | 0.30 | 0.82 | - | 4.65 | 1041 |
| 256 | na_fp4_c8 | 25.31 | 161,822 | 7.23x | 8.72x | 7.34 | 19.47 | 0.29 | 15.40 | 3.64 | 7.92 | 2717 |
| 256 | na_fp4_c8+compile | 17.81 | 229,924 | 5.09x | 6.13x | 5.13 | 14.03 | 0.30 | 9.88 | 1.42 | 5.41 | 1341 |
| 256 | na_fp4_c8+fastpack | 21.98 | 186,314 | 6.28x | 7.57x | 5.63 | 17.58 | 0.29 | 13.53 | 3.51 | 6.62 | 2429 |
| 256 | lg6_fp4_h2 | 10.31 | 397,469 | 2.94x | 3.55x | 2.27 | 8.90 | 0.28 | 5.44 | 0.46 | 4.83 | 1133 |
| 256 | lg6_fp4_h2+compile | 7.64 | 535,793 | 2.18x | 2.63x | 2.29 | 6.16 | 0.29 | 2.67 | 0.47 | 3.89 | 589 |
| 256 | lg8_fp4_h2 | 10.30 | 397,708 | 2.94x | 3.55x | 2.26 | 8.90 | 0.28 | 5.45 | 0.45 | 4.85 | 1133 |
| 256 | lg8_fp4_h2+compile | 7.65 | 535,588 | 2.19x | 2.63x | 2.28 | 6.15 | 0.29 | 2.62 | 0.45 | 3.89 | 589 |
| 512 | dense_nobias | 8.27 | 495,578 | 1.00x | 1.10x | 2.89 | 4.88 | 0.48 | - | - | 8.35 | 302 |
| 512 | dense_nobias+compile | 7.50 | 546,085 | 0.91x | 1.00x | 2.65 | 4.34 | 0.52 | - | - | 7.75 | 229 |
| 512 | qat_fp4 | 12.29 | 333,350 | 1.49x | 1.64x | 7.31 | 4.99 | 0.58 | - | 4.01 | 9.80 | 918 |
| 512 | qat_fp4+compile | 7.90 | 518,769 | 0.96x | 1.05x | 2.96 | 4.32 | 0.62 | - | - | 8.12 | 294 |
| 512 | tern_counter | 14.25 | 287,401 | 1.72x | 1.90x | 4.62 | 9.75 | 0.30 | 0.70 | - | 9.49 | 1054 |
| 512 | na_fp4_c8 | 28.40 | 144,206 | 3.44x | 3.79x | 8.92 | 20.41 | 0.29 | 14.51 | 3.50 | 16.15 | 2734 |
| 512 | na_fp4_c8+compile | 21.84 | 187,587 | 2.64x | 2.91x | 6.81 | 15.94 | 0.31 | 9.91 | 1.30 | 13.13 | 1358 |
| 512 | na_fp4_c8+fastpack | 24.49 | 167,262 | 2.96x | 3.26x | 6.79 | 18.60 | 0.30 | 12.18 | 3.49 | 12.64 | 2446 |
| 512 | lg6_fp4_h2 | 11.42 | 358,644 | 1.38x | 1.52x | 3.21 | 8.15 | 0.14 | 3.12 | 0.31 | 10.08 | 1150 |
| 512 | lg6_fp4_h2+compile | 9.18 | 445,986 | 1.11x | 1.22x | 3.23 | 5.91 | 0.14 | 0.95 | 0.32 | 8.67 | 606 |
| 512 | lg8_fp4_h2 | 11.42 | 358,583 | 1.38x | 1.52x | 3.21 | 8.16 | 0.14 | 3.12 | 0.31 | 10.08 | 1150 |
| 512 | lg8_fp4_h2+compile | 9.18 | 446,008 | 1.11x | 1.22x | 3.22 | 5.91 | 0.14 | 0.95 | 0.32 | 8.67 | 606 |

### Headline ratios [verified, derived from the table]

| arm | d=128 | d=256 | d=512 | vs dense+compile (same d) |
|---|---:|---:|---:|---|
| `qat_fp4` eager | 3.09x | 2.66x | 1.49x | 3.96x / 3.21x / 1.64x |
| `tern_counter` eager | 3.61x | 3.41x | 1.72x | 4.62x / 4.11x / 1.90x |
| `na_fp4_c8` eager | 8.19x | 7.23x | 3.44x | 10.49x / 8.72x / 3.79x |
| `lg6_fp4_h2` eager | 3.28x | 2.94x | 1.38x | 4.21x / 3.55x / 1.52x |
| `qat_fp4` + compile | 0.95x | 0.93x | 0.96x | 1.21x / 1.12x / 1.05x |
| `na_fp4_c8` + compile | 5.75x | 5.09x | 2.64x | 7.36x / 6.13x / 2.91x |
| `na_fp4_c8` + fastpack | 6.95x | 6.28x | 2.96x | 8.90x / 7.57x / 3.26x |
| `lg6_fp4_h2` + compile | 2.41x | 2.18x | **1.11x** | 3.08x / 2.63x / **1.22x** |
| `dense_nobias` + compile | 0.78x | 0.83x | 0.91x | 1.00x |

- **The slowdown shrinks with d [verified], but the absolute step time of the stateless arms barely moves:** lg6 10.22 -> 10.31 -> 11.42 ms, na 25.50 -> 25.31 -> 28.40 ms, while dense grows 3.11 -> 3.50 -> 8.27 ms. So the ratio falls because dense gets slower (its fp32 GEMM time is 0.93 / 2.11 / 6.07 ms), not because the update gets faster. **[hypothesis]** Once the update becomes bandwidth-bound (it already is at d=512, see below) both the update and the GEMMs scale with the parameter count and the ratio plateaus near 1 + update/GEMM time (about 1.4x for lg6 eager at d=512), not at 1.0x; a bf16/TF32 dense would raise that plateau.
- **Host-bound regime [verified numbers, interpretation hypothesis]:** wall time / kernel launches is 9.1-9.4 us for lg6 and na at d=128/256, 9.9 / 10.4 us at d=512, 10.3-11.5 us for qat/tern/dense at d=128/256 (dense at d=512, 27 us per launch, is GPU-bound). Wall time equals about 10 us per launch; the GPU is busy only 22 % / 31 % / 57 % (na) and 32 % / 47 % / 88 % (lg6) of the step at d=128 / 256 / 512 (kernel time / wall; dense is at 98-100 % from d=256). This matches the SPEED_NOTES estimate of ~10 us of host time per eager op (na ~2,200 launches -> 20-30 ms, lg ~1,000 -> ~10 ms): measured 2,700 launches -> 25.5 ms and 1,116 -> 10.2 ms at d=128.
- **lg vs na:** lg is 2.50x / 2.45x / 2.49x faster per step than na (25.50/10.22, 25.31/10.31, 28.40/11.42). SPEED_NOTES expected the lg/dense ratio to be about 2x smaller than the na/dense ratio; measured na/dense : lg/dense = 2.50 / 2.46 / 2.49. On the update alone (bwd `update` column) lg is 2.7x / 2.8x / 4.7x cheaper (6.00 vs 16.06, 5.44 vs 15.40, 3.12 vs 14.51 ms); SPEED_NOTES predicted ~3.9x from bytes (465 vs 1800 B/param), which is reached once the update is bandwidth-bound (d=512).
- **tern_counter (the fused-Triton-update reference) is not faster than eager lg:** its update is 1.05 / 0.82 / 0.70 ms, yet the step is 11.23 / 11.94 / 14.25 ms (2.8 ms above lg6 at d=512, 1.0 ms above at d=128; 1,036-1,054 launches vs 1,116-1,150). The bwd without the update is 6.1 ms at d=128 (7.19 - 1.05) against 1.74 for dense. So a fused update alone does not give dense-like speed in this harness; the remaining launches (decode cache refresh, autograd glue; **[hypothesis]**) matter at least as much.

### Phase breakdown (ms; `update` is inside bwd, `decode` inside fwd)

| d | arm | fwd | bwd | opt | update | decode | what dominates [verified from the table] |
|---:|---|---:|---:|---:|---:|---:|---|
| 128 | dense | 1.06 | 1.74 | 0.36 | - | - | bwd |
| 128 | qat_fp4 | 7.45 | 2.40 | 0.48 | - | 5.57 | fake-quant decode in fwd (75 % of fwd) |
| 128 | na_fp4_c8 | 7.36 | 19.49 | 0.29 | 16.06 | 3.64 | update = 63 % of the step |
| 128 | lg6 | 2.26 | 8.76 | 0.27 | 6.00 | 0.59 | update = 59 % of the step |
| 512 | dense | 2.89 | 4.88 | 0.48 | - | - | GEMMs |
| 512 | qat_fp4 | 7.31 | 4.99 | 0.58 | - | 4.01 | decode 33 % of the step |
| 512 | na_fp4_c8 | 8.92 | 20.41 | 0.29 | 14.51 | 3.50 | update 51 %, decode + unpack the rest of fwd |
| 512 | lg6 | 3.21 | 8.15 | 0.14 | 3.12 | 0.31 | update 27 % |

The na forward (7.36 / 8.92 ms) is much more than its `decode` (3.64 / 3.50): the remainder is `unpack_state` (the generic bit-matrix path), not counted in the `decode` column. At d=512 lg6 minus (update + decode) is 7.99 ms = 0.97x dense; at d=128 / 256 it is 3.64 / 4.41 ms = 1.17x / 1.26x; for na it is 1.86x / 1.79x / 1.26x (derived ceiling if update + decode were free; ignores the cost of any replacement kernel). That the lg remainder is above dense at small d (bwd without update 2.76 ms vs 1.74 ms at d=128) means the custom-function glue is itself launch-bound **[hypothesis]**.

### Top kernels (eager, one profiled step)

| d | arm | total | gemm | elementwise | copy/cast | reduce | index/gather | other/triton/rng | wall ms/step | GPU busy share |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | dense_nobias | 2.06 | 0.93 | 0.22 | 0.18 | 0.15 | 0.01 | 0.57 | 3.11 | 66% |
| 128 | qat_fp4 | 3.17 | 0.93 | 0.77 | 0.39 | 0.28 | 0.12 | 0.68 | 9.62 | 33% |
| 128 | tern_counter | 3.00 | 0.93 | 0.59 | 0.82 | 0.15 | 0.01 | 0.50 | 11.23 | 27% |
| 128 | na_fp4_c8 | 5.67 | 0.93 | 2.15 | 1.27 | 0.41 | 0.40 | 0.52 | 25.50 | 22% |
| 128 | lg6_fp4_h2 | 3.29 | 0.93 | 0.76 | 0.52 | 0.28 | 0.34 | 0.47 | 10.22 | 32% |
| 128 | na_fp4_c8+compile | 3.46 | 0.90 | 0.84 | 0.70 | 0.39 | 0.13 | 0.51 | 17.90 | 19% |
| 128 | lg6_fp4_h2+compile | 2.37 | 0.89 | 0.36 | 0.30 | 0.26 | 0.07 | 0.49 | 7.49 | 32% |
| 256 | dense_nobias | 3.43 | 2.11 | 0.33 | 0.27 | 0.17 | 0.01 | 0.53 | 3.50 | 98% |
| 256 | qat_fp4 | 4.76 | 2.22 | 0.94 | 0.51 | 0.29 | 0.14 | 0.66 | 9.31 | 51% |
| 256 | tern_counter | 4.65 | 2.21 | 0.71 | 1.01 | 0.17 | 0.01 | 0.53 | 11.94 | 39% |
| 256 | na_fp4_c8 | 7.92 | 2.22 | 2.64 | 1.51 | 0.59 | 0.40 | 0.55 | 25.31 | 31% |
| 256 | lg6_fp4_h2 | 4.83 | 2.13 | 0.93 | 0.64 | 0.29 | 0.35 | 0.48 | 10.31 | 47% |
| 256 | na_fp4_c8+compile | 5.41 | 2.13 | 1.13 | 0.89 | 0.57 | 0.15 | 0.54 | 17.81 | 30% |
| 256 | lg6_fp4_h2+compile | 3.89 | 2.11 | 0.48 | 0.42 | 0.29 | 0.07 | 0.52 | 7.64 | 51% |
| 512 | dense_nobias | 8.35 | 6.07 | 0.82 | 0.51 | 0.25 | 0.01 | 0.70 | 8.27 | 101% |
| 512 | qat_fp4 | 9.80 | 6.11 | 1.50 | 0.74 | 0.40 | 0.26 | 0.79 | 12.29 | 80% |
| 512 | tern_counter | 9.49 | 6.13 | 0.98 | 1.54 | 0.25 | 0.01 | 0.57 | 14.25 | 67% |
| 512 | na_fp4_c8 | 16.15 | 6.17 | 4.80 | 2.67 | 1.42 | 0.54 | 0.55 | 28.40 | 57% |
| 512 | lg6_fp4_h2 | 10.08 | 6.06 | 1.56 | 1.00 | 0.38 | 0.58 | 0.49 | 11.42 | 88% |
| 512 | na_fp4_c8+compile | 13.13 | 6.14 | 2.82 | 1.98 | 1.42 | 0.21 | 0.56 | 21.84 | 60% |
| 512 | lg6_fp4_h2+compile | 8.67 | 6.07 | 0.83 | 0.71 | 0.38 | 0.09 | 0.59 | 9.18 | 94% |

- **dense:** 284 / 289 / 302 launches; at d=128 the top items are `AdamW.step` (0.346 ms, 16.8 %), cutlass sgemm kernels (0.19 ms each) and `GammaBetaBackward` (layer-norm, 0.152 ms); from d=256 gemm is 61 % (2.11 of 3.43 ms), at d=512 73 % (6.07 of 8.35 ms).
- **na_fp4_c8:** 2,700 / 2,717 / 2,734 launches (about 9.5x dense's). Non-GEMM kernels are elementwise 2.15 / 2.64 / 4.80 ms, copy/cast 1.27 / 1.51 / 2.67 ms, reduce 0.41 / 0.59 / 1.42 ms, index/gather ~0.4-0.54 ms. Top single kernels at d=128: `direct_copy_kernel` casts (128 + 144 calls, 0.287 + 0.275 ms), `index_elementwise` gathers (96 calls, 0.269 ms), `AdamW.step`; at d=512 the first non-GEMM entries are `reduce_kernel<long>` (48 calls, 1.005 ms) and `vectorized_elementwise<long>` (32 calls, 0.889 ms), i.e. the int64 temporaries SPEED_NOTES flagged. Non-GEMM kernel time at d=512: 10.0 ms.
- **lg6_fp4_h2:** 1,116 / 1,133 / 1,150 launches. At d=128 the top items are `index_elementwise` gathers (96 calls, 0.273 ms, 8.3 %), `AdamW.step` (0.223 ms) and `CUDAFunctor_add` (176 calls, 0.204 ms); at d=512 the top three are sgemm (16.6 / 15.7 / 12.7 %), and the non-GEMM kernel time is 4.0 ms (elementwise 1.56, copy/cast 1.00, index/gather 0.58, reduce 0.38, other 0.49).
- The profiler's summed kernel time can exceed wall by a few percent (dense d=512: 8.35 vs 8.27 ms; profiling overhead).

### Estimate (SPEED_NOTES) vs measurement

| item | SPEED_NOTES estimate | measured |
|---|---|---|
| na update path | ~1800 B/param, 140 launches/layer, ideal 0.94 / 3.8 / 15.1 ms at 1.5 TB/s | `update` 16.06 / 15.40 / 14.51 ms (17x / 4.1x / 1.0x the ideal); launches 2,700 (~169 per block linear, 16 of them) |
| lg update path | ~465 B/param, ~60 launches/layer, ideal 0.24 / 0.98 / 3.9 ms | `update` 6.00 / 5.44 / 3.12 ms (25x / 5.6x / 0.8x) |
| launch-bound step | na ~20-30 ms, lg ~10 ms at d=128 | 25.5 ms, 10.2 ms |
| lg cheaper than na | ~4x (bytes), "2x smaller ratio" vs dense | 4.7x on the update at d=512, 2.5x on the step |
| `--fast-pack` | ~975 B/param, i.e. -46 % update bytes | step -15.0 % / -13.8 % / -14.3 % (-3.83 / -3.52 / -4.09 ms); `update` -12 % / -13 % / -16 %; launches -288 at every d; GPU kernel time -0.56 / -0.95 / -3.53 ms |
| compile | "does not cover rms_update, pack/unpack, `_sr_int`, `_clip_mask`; lg's compiled `move` is experimental" | na -29.7 % / -30.2 % / -23.6 % (25.47 -> 17.90, 25.50 -> 17.81, 28.58 -> 21.84 ms); lg6 -26.6 % / -26.5 % / -19.5 %; qat -70 % / -65 % / -36 %; dense -22 % / -17 % / -9.4 % |
| fused update kernel (end state) | "an na / lg step lands within ~10-30 % of dense AdamW"; d=512 update ~15 ms -> ~0.2-0.3 ms | not built. lg6 + `torch.compile` already reaches 1.11x dense eager / 1.22x dense+compile at d=512, with the update at 0.95 ms (from 3.12) |

- **[verified]** The launch-floor part of the estimate holds (about 10 us per op; 25.5 vs ~20-30 ms, 10.2 vs ~10 ms). The bandwidth part is consistent at d=512 (ratio 1.0x for na, 0.8x for lg6): the hand count of bytes is at most mildly too high, **[hypothesis]** because part of the temporaries stay in L2 or because the HBM bandwidth is above the 1.5 TB/s assumed; the GPU's bandwidth is not stated in the files.
- **fast-pack:** measured gain is about a third of the predicted byte saving at d=512 (GPU kernel time -3.53 ms vs 825 B/param x 12.6 M params / 1.5 TB/s = 6.9 ms predicted) and at d=128 the gain (-3.83 ms wall, -0.56 ms kernel time) is the 288 fewer launches (at ~9.4 us each about -2.7 ms), not bytes. **[hypothesis]** the int64 bit-matrix passes are cheaper than hand-counted, and the generic pack is not the single dominant cost SPEED_NOTES suggested (~46 % of the update bytes).
- **compile:** the compiled lg update (3.19 / 2.67 / 0.95 ms) is the nearest thing to the fused kernel measured here. At d=512 0.95 ms is about 113 B/param at 1.5 TB/s if it were bandwidth-bound (derived, ignores that it may not be), against SPEED_NOTES' ~18 B/param for a hand-fused kernel (0.15 ms): compile gets 3.12 -> 0.95 ms of the possible 3.12 -> ~0.15 ms. At d=128 / 256 the compiled lg step is still 2.41x / 2.18x dense eager (3.08x / 2.63x dense+compile) with 572 / 589 launches and the GPU 32 % / 51 % busy: the small-d regime remains launch-bound.
- Compile costs memory at small d: peak MB eager -> compile, dense 188 -> 291 (d=128), 361 -> 455 (d=256), 827 -> 808 (d=512); lg6 169 -> 292, 353 -> 471, 794 -> 881; na 171 -> 290, 370 -> 465, 857 -> 860.

## 4. Memory

State/+grad are analytic (bits/8 x block-linear params; `+grad` = state plus the resident gradient bytes, as in the harness); peak and resident are **measured** by the bench (CUDA memory above what was allocated before the model; eager rows of `bench_std`). `other_fp` = emb/pos/LN/head with AdamW (12 B/param): 1.53 / 3.05 / 6.11 MB at d=128 / 256 / 512 in the bench model (vocab 256). The last column is resident minus (state + other_fp).

| d | arm | bits | state (analytic) | +grad | peak (bench) | resident (bench) | peak - resident | resident - state - other_fp |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | dense_nobias | 96.00 | 9.00 | 12.00 | 188 | 31 | 157 | 20.3 |
| 128 | qat_fp4 | 96.00 | 9.00 | 12.00 | 174 | 15 | 160 | 4.0 |
| 128 | tern_counter | 6.56 | 0.62 | 0.87 | 166 | 3 | 162 | 1.0 |
| 128 | na_fp4_c8 | 8.69 | 0.81 | 1.06 | 171 | 4 | 167 | 1.4 |
| 128 | lg6_fp4_h2 | 6.69 | 0.63 | 0.88 | 169 | 6 | 163 | 4.2 |
| 128 | lg8_fp4_h2 | 8.69 | 0.81 | 1.06 | 169 | 6 | 163 | 4.0 |
| 256 | dense_nobias | 96.00 | 36.00 | 48.00 | 361 | 53 | 308 | 13.5 |
| 256 | qat_fp4 | 96.00 | 36.00 | 48.00 | 374 | 53 | 321 | 13.5 |
| 256 | tern_counter | 6.28 | 2.36 | 3.36 | 333 | 7 | 326 | 1.5 |
| 256 | na_fp4_c8 | 8.59 | 3.22 | 4.22 | 370 | 9 | 360 | 3.0 |
| 256 | lg6_fp4_h2 | 6.59 | 2.47 | 3.47 | 353 | 20 | 334 | 14.3 |
| 256 | lg8_fp4_h2 | 8.59 | 3.22 | 4.22 | 353 | 20 | 334 | 13.5 |
| 512 | dense_nobias | 96.00 | 144.00 | 192.00 | 827 | 211 | 616 | 60.4 |
| 512 | qat_fp4 | 96.00 | 144.00 | 192.00 | 895 | 205 | 690 | 54.4 |
| 512 | tern_counter | 6.14 | 9.21 | 13.21 | 673 | 18 | 655 | 2.6 |
| 512 | na_fp4_c8 | 8.55 | 12.82 | 16.82 | 857 | 27 | 830 | 8.6 |
| 512 | lg6_fp4_h2 | 6.55 | 9.82 | 13.82 | 794 | 72 | 722 | 56.6 |
| 512 | lg8_fp4_h2 | 8.55 | 12.82 | 16.82 | 794 | 72 | 722 | 53.6 |

### Measured vs analytic [verified]
- State (analytic) is 15x smaller than dense for lg6 (9.82 vs 144.00 MB at d=512; bits 6.55 vs 96), 15.6x for tern (9.21) and 11x for na (12.82). **Resident** memory follows less: dense 211, qat 205, na 27, lg6 72, tern 18 MB at d=512 (-87 %, -66 %, -91 % vs dense for na, lg6, tern).
- **Peak** barely moves: dense 188 / 361 / 827 MB, lg6 169 / 353 / 794, na 171 / 370 / 857, tern 166 / 333 / 673, qat 174 / 374 / 895. At d=512 lg6 saves 33 MB (-4 %), na costs +30 MB (+4 %), tern saves 154 MB (-19 %). The column "peak - resident" (activations + update temporaries) is 157 / 308 / 616 MB for dense and 163 / 334 / 722 for lg6, 167 / 360 / 830 for na, 162 / 326 / 655 for tern: it doubles per width doubling and is larger than dense by 106 MB (lg6) and 214 MB (na) at d=512, which is the update's temporaries (int64 intermediates etc.) eating the 134 MB state saving of lg6.
- The training runs (`q/` logs, vocab 202, with eval) agree: peak seed 0 at d=128 / 256 / 512: dense 338 / 529 / 990 MB, qat 341 / 542 / 1064, tern 334 / 501 / 841, na 339 / 538 / 1025, lg6 335 / 521 / 963.
- **lg's resident excess over its analytic state is about the fp32 scale cache:** resident - state - other_fp = 4.2 / 14.3 / 56.6 MB (lg6; 4.0 / 13.5 / 53.6 for lg8) against 4 B x block params = 3.0 / 12.0 / 48.0 MB. na has 1.4 / 3.0 / 8.6 MB and tern 1.0 / 1.5 / 2.6 MB. SPEED_NOTES lists `_scale_cache` (fp32 [out,in], never freed) and `_weight_cache` for lg; the numbers match one fp32 plane, so the weight cache is evidently not resident at measurement time **[hypothesis]**. The residual 1.2 / 2.3 / 8.6 MB after subtracting the cache is the uint8 index (1 B/param, 12 MB at d=512) minus the bit-packed accounting (9.82 MB) plus small items. **[hypothesis]** Removing the cache (decode the scale from the E4M3 byte in the kernel) would take lg6's resident at d=512 from 72 to about 24 MB; the effect on peak is at most the same 48 MB and unmeasured.
- The bench dense `resident - state - other_fp` excess (20.3 / 13.5 / 60.4 MB) is not interpreted (at d=512 it is about the 48 MB of dense gradients; the d=128 value is not explained).

### Equal-memory comparisons (derived from sections 1 and 4)

(a) **Equal analytic training state of the block linears** (val on enwik8, 2 seeds; the d=512 counter/grid rows have adam-lr 3e-3 fp parts except lg6 at 1e-3; dense at its trained lr: 3e-3 at d=128 / 256):

| compared | state MB (block linears) | val | Delta val | state ratio |
|---|---|---|---:|---:|
| `lg6` d=512 (1.9771) vs dense d=128 (2.0200) | 9.82 vs 9.00 | 1.9771 vs 2.0200 | **-0.0429** (-0.0337 with lg6 at adam-lr 3e-3) | 1.09x |
| `tern_counter` d=512 vs dense d=128 | 9.21 vs 9.00 | 1.9982 vs 2.0200 | -0.0218 | 1.02x |
| `na_fp4_c8` d=512 vs dense d=128 | 12.82 vs 9.00 | 1.9042 vs 2.0200 | -0.1158 | 1.42x |
| `na_fp4_c8` d=256 vs dense d=128 | 3.22 vs 9.00 | 1.9847 vs 2.0200 | **-0.0353** | 0.36x |
| `lg6` d=256 vs dense d=128 | 2.47 vs 9.00 | 2.0821 vs 2.0200 | +0.0621 | 0.27x |
| `na_fp4_c8` d=512 vs dense d=256 | 12.82 vs 36.00 | 1.9042 vs 1.8508 | +0.0534 | 0.36x |
| `lg6` d=512 vs dense d=256 | 9.82 vs 36.00 | 1.9771 vs 1.8508 | +0.1263 | 0.27x |
| `tern_counter` d=512 vs dense d=256 | 9.21 vs 36.00 | 1.9982 vs 1.8508 | +0.1474 | 0.26x |
| `lg6` d=512 vs `qat_fp4` d=128 (2.0491) | 9.82 vs 9.00 | 1.9771 vs 2.0491 | -0.0719 | 1.09x |

Reading [verified numbers]: with the same block-linear state bytes (+-9 %) a 4x-wider counter/grid model is better than the dense one by 0.02-0.04 bits (lg6, tern) or 0.12 (na at 1.4x the state); `na_fp4_c8` at d=256 beats dense d=128 with 2.8x less state (-0.035, though the na sd at d=256 is 0.035, so not detectable). At roughly one quarter to one third of the state of the next dense width, the quantised models lose 0.05 (na) to 0.15 (tern) bits. Note the strict whole-model state also contains emb/pos/LN/head in fp32 + AdamW: lg6 d=512 is 9.82 + 6.11 = 15.9 MB against dense d=128 9.00 + 1.53 = 10.5 MB (1.5x), so "equal state" here means block linears only, as in the task.

(b) **Equal peak or resident memory: dense wins at every size measured [verified].** At peak 794-857 MB, lg6 / na / tern at d=512 reach 1.9771 / 1.9042 / 1.9982, dense d=512 (827 MB peak) 1.7982; at ~360 MB peak dense d=256 reaches 1.8508, better than every d=512 stateless arm that needs 673-857 MB. In resident terms (72 MB lg6 d=512 vs 53 MB dense d=256) dense d=256 is still better by 0.126. The reason is the one above: at batch 32 x seq 128 the peak is activations and update temporaries (157 -> 616 MB in the "peak - resident" column) and the state saving is 134 MB (state) to 178 MB (state + grad) at d=512.

(c) **When the state saving would show [hypothesis, extrapolation of three points]:** dense state + grad grows 4x per width doubling (12 / 48 / 192 MB for the block linears), while the working set (peak - resident) doubled per width step (157 / 308 / 616). At d=1024 dense state + grad would be about 768 MB against a working set of about 1.2 GB, at d=2048 about 3.1 GB against about 2.5 GB, so the state term would dominate from d around 1024-2048. For lg/na to realise that, the update temporaries must stop scaling with the layer (eager temporaries grow with the largest weight tensor, 4x per doubling): that needs the fused update (section 5), and the fp32 caches must go. At d=512 and eager, no peak-memory advantage exists. The earlier reports' "storage-state win only" stands, and for the training peak it stands with the quantification above.

## 5. Conclusions

1. **Scaling [verified, 2 seeds]:** the quality gap of the stateless arms shrinks with width: lg6 -0.104, tern -0.180, qat -0.029 from d=128 to d=512 (against dense with a re-tuned lr at d=512); `na_fp4_c8` is inside its noise. The O1 second clause is met for lg6 and tern on this corpus. 4 seeds, a dense lr sweep at d=128 / 256 and a fp-part lr sweep at d=512 are needed before this goes beyond a direction.
2. **lr must be retuned with width [verified]:** AdamW 3e-3 trains d=128 / 256 and diverges at d=512; the RMS-normalised counter/grid update did not diverge at its lr 0.01 (and 3e-2 for tern). A fixed-lr scaling study of the dense reference would have shown a false advantage of 1.2-1.3 bits to every stateless arm at d=512.
3. **4-seed frontier at d=128 [verified]:** na_h2 2.3172, lg7 2.3530, lg6 2.4141; ordering holds in every seed.
4. **Speed [verified]:** eager mx6 steps are host/launch-bound at d<=256 (about 10 us per launch); the ratio to dense shrinks 3.3x -> 1.4x (lg6) and 8.2x -> 3.4x (na) from d=128 to 512 mostly because dense GEMM time grows. `torch.compile` brings lg6 to 1.11x dense eager at d=512 and `--fast-pack` takes 14-15 % off na. The fp32-GEMM reference flatters the ratios.
5. **Memory [verified]:** state 15x smaller (lg6), resident 3x smaller, peak unchanged at these sizes; the state advantage needs larger models, a fused update (bounded temporaries) and removal of lg's fp32 caches.
6. **Where the time goes [verified numbers]:** the update is 59-63 % of the lg6 / na step at d=128 and 27 % / 51 % at d=512; with the update and decode costs removed lg6 would sit at 0.97x-1.26x dense, na at 1.26x-1.86x.

## Limits (not verified or not run)

- **n=2 seeds for every scaling point** (M2 asks 4); several n=2 sds are small by luck (lg6 d=256 0.0003, na d=512 0.0023) and the d=512 dense/lg6 sds are large (0.1228 for the diverged dense). The na trend is explicitly not detectable.
- **adam-lr confounds.** Dense / qat at d=128 / 256 use 3e-3 without a sweep (3e-3 is demonstrably wrong at d=512, so it may also be sub-optimal at 256); the d=512 dense grid is {5e-4, 1e-3, 3e-3}, optimum unresolved; `na_fp4_c8` and `tern_counter` at d=512 have fp parts at 3e-3 only; the lg6 1e-3 vs 3e-3 comparison (0.0092, below its sd) is the only direct estimate of that confound. The counter/grid lr 0.01 (tern 0.03) was not retuned with d (M8): the shrinking gap could change with a per-d lr.
- One corpus (enwik8 20 M chars, 1.37 epochs: not data-limited, unlike tinyshakespeare), one depth (4 layers), 6000 steps, seq 128; the d=512 models are 13 M parameters. Nothing here says what happens at 100 M+ parameters or with longer training.
- The 4-seed block mixes two code snapshots (seeds 0, 1 old uncommitted working tree, seeds 2, 3 `e9830bc`); only the small spread between the halves supports treating them as equivalent. The tinyshakespeare model is 2 layers / 1 head, not the 4-layer scaling model.
- **Speed:** fp32 GEMMs (no autocast) for all arms; one GPU, one batch/seq, 100 timed steps; `bench_std` and `bench_x` run once each; the "alone on an idle GPU" statement comes from the task. No bf16/TF32 or fused-kernel result; the lg `+compile` path is the experimental one (SPEED_NOTES) and was not parity-checked here (M5), so its speed is a measurement, not a validated equivalent path.
- **Memory:** peak/resident are CUDA allocator figures above the pre-model allocation, for one fixed batch/seq; the "state" columns are accounting bits (lg's run-time index is a dense uint8 buffer plus fp32 caches, the bit-packed layout is the checkpoint format); the attribution of lg's resident excess to `_scale_cache` is a numerical match, not a measurement of the cache itself. The extrapolation in 4(c) is three points.
- The bench uses vocab 256 and the training logs vocab 202, so the activations/`other_fp` numbers of the two are not identical (peak 827 vs 990 MB for dense d=512 train vs bench).

## Open / next

1. **Fused Triton update for lg** (SPEED_NOTES section 3: one state, one SR draw, closed-form `G[idx]`, ~40-50 lines), with a statistical parity test (E[step] unbiased) and a same-seed reproducibility note; measure against the compiled path (0.95 ms at d=512) and the 0.15 ms ideal. Expected end state per the estimate: lg within 10-30 % of dense, which `lg6 + compile` already gets at d=512.
2. **Drop the fp32 `_scale_cache` / `_weight_cache`** (decode the E4M3 scale in the kernel); re-measure resident and peak (hypothesis: -48 MB resident at d=512).
3. **bf16 autocast (and a TF32 control) for all arms** in the bench and in a quality run, so that dense is not slowed by fp32 simt GEMMs and the ratios are the real ones; then re-read this report's slowdown trend.
4. **Larger d and longer runs** (d=768-1024, more steps / tokens) with 4 seeds; at the same time the memory crossover of 4(c) can be measured instead of extrapolated.
5. **Tune adam-lr per d** for the dense reference at d=128 / 256 / 512 (grid around 1e-3 .. 3e-3 plus 2e-3, 1.5e-3) and for the fp parts of the counter/grid arms at d=512 (1e-3 for `na_fp4_c8` and `tern_counter`); sweep the counter/grid lr with d.
6. 4 seeds of the counter arms at 6.69 / 7.69 bits (`fp4c2_h2`, `fp4c4_h2`) to settle R9 / O7's grid-vs-counter question, and `lg7`/`lg8` rows in the scaling grid.
