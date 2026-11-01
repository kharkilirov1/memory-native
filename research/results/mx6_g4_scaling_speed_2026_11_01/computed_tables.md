T1
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

T2
| arm | d=128 | d=256 | d=512 | change 128->512 | note |
|---|---:|---:|---:|---:|---|
| qat_fp4 | +0.0291 | +0.0155 | +0.0001 | -0.0289 | adam-lr 1e-3 at d=512 (same as the dense reference) |
| tern_counter | +0.3796 | +0.2987 | +0.2001 | -0.1796 | fp emb/LN/head at adam-lr 3e-3 at d=512 (confound) |
| na_fp4_c8 | +0.1431 | +0.1339 | +0.1061 | -0.0370 | fp emb/LN/head at adam-lr 3e-3 at d=512 (confound) |
| lg6_fp4_h2 | +0.2825 | +0.2313 | +0.1790 | -0.1035 | adam-lr 1e-3 at d=512 (same as dense) |
| lg6_fp4_h2 (adam-lr 3e-3 at d=512; dense ref still a1e-3) | +0.2825 | +0.2313 | +0.1882 | -0.0943 | unconfounded vs 3e-3 dense is impossible: dense diverged |

gap sd (sqrt(sd_a^2+sd_ref^2), n=2): qat_fp4: 0.0042 / 0.0105 / 0.0072; tern_counter: 0.0038 / 0.0161 / 0.0084; na_fp4_c8: 0.0253 / 0.0357 / 0.0026; lg6_fp4_h2: 0.0051 / 0.0059 / 0.0226

T2b relative gaps (arm - qat_fp4 at same d; qat d=512 = a1e-3)
tern_counter: +0.3506 / +0.2832 / +0.1999
na_fp4_c8: +0.1140 / +0.1184 / +0.1060
lg6_fp4_h2: +0.2534 / +0.2158 / +0.1789

d=512 dense: 3e-3 3.1658, 1e-3 1.7982, 5e-4 1.8563; qat: 3e-3 3.1691, 1e-3 1.7983
relative gap (% of dense val): qat_fp4: 1.4% / 0.8% / 0.0%; tern_counter: 18.8% / 16.1% / 11.1%; na_fp4_c8: 7.1% / 7.2% / 5.9%; lg6_fp4_h2: 14.0% / 12.5% / 10.0%

val change with d (128->256, 256->512):
dense_nobias: 2.0200 -> 1.8508 -> 1.7982  (-0.1692, -0.0526)
qat_fp4: 2.0491 -> 1.8663 -> 1.7983  (-0.1828, -0.0680)
tern_counter: 2.3996 -> 2.1495 -> 1.9982  (-0.2501, -0.1513)
na_fp4_c8: 2.1631 -> 1.9847 -> 1.9042  (-0.1784, -0.0804)
lg6_fp4_h2: 2.3025 -> 2.0821 -> 1.9771  (-0.2204, -0.1050)

T3 seed spread (|seed1-seed0|) per arm per d
dense_nobias: d128: sd 0.0035 (range 0.0050); d256: sd 0.0059 (range 0.0084); d512: sd 0.1228 (range 0.1737)
qat_fp4: d128: sd 0.0023 (range 0.0033); d256: sd 0.0086 (range 0.0122); d512: sd 0.0033 (range 0.0046)
tern_counter: d128: sd 0.0013 (range 0.0019); d256: sd 0.0149 (range 0.0211); d512: sd 0.0083 (range 0.0118)
na_fp4_c8: d128: sd 0.0250 (range 0.0354); d256: sd 0.0352 (range 0.0498); d512: sd 0.0023 (range 0.0033)
lg6_fp4_h2: d128: sd 0.0037 (range 0.0052); d256: sd 0.0003 (range 0.0004); d512: sd 0.0204 (range 0.0289)

T4
| arm | bits | seeds 0,1 mean (old report) | seeds 2,3 mean | per-seed val (s0, s1, s2, s3) | 4-seed mean | sd (n-1) | train | val-train | edge % |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| na_fp4_c8 (h2) | 8.688 | 2.3151 | 2.3194 | 2.3191, 2.3111, 2.3100, 2.3287 | **2.3172** | 0.0087 | 1.9952 | +0.3220 | 24.09 |
| lg6_fp4_h2 | 6.688 | 2.4155 | 2.4127 | 2.4202, 2.4108, 2.4075, 2.4179 | **2.4141** | 0.0059 | 2.1395 | +0.2747 | 16.09 |
| lg7_fp4_h2 | 7.688 | 2.3513 | 2.3546 | 2.3586, 2.3440, 2.3417, 2.3676 | **2.3530** | 0.0123 | 2.0484 | +0.3046 | 18.99 |

lg7_fp4_h2 - na_fp4_c8 (h2): +0.0357 per-seed [0.0395, 0.0329, 0.0317, 0.0389]; larger sd 0.0123; ratio 2.91
lg6_fp4_h2 - na_fp4_c8 (h2): +0.0969 per-seed [0.1011, 0.0997, 0.0975, 0.0892]; larger sd 0.0087; ratio 11.19
lg6_fp4_h2 - lg7_fp4_h2: +0.0611 per-seed [0.0616, 0.0668, 0.0658, 0.0503]; larger sd 0.0123; ratio 4.97
dense_bf16 2.2864, dense_nobias 2.2891, qat_fp4 2.2983 (decomp, 4 seeds, other snapshot)
na_fp4_c8 (h2): vs qat_fp4 2.2983 +0.0189; vs dense_nobias 2.2891 +0.0281; vs dense_bf16 2.2864 +0.0308
lg6_fp4_h2: vs qat_fp4 2.2983 +0.1158; vs dense_nobias 2.2891 +0.1250; vs dense_bf16 2.2864 +0.1277
lg7_fp4_h2: vs qat_fp4 2.2983 +0.0547; vs dense_nobias 2.2891 +0.0639; vs dense_bf16 2.2864 +0.0666

T5 speed
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

T5b eager repeatability (bench_std vs bench_x, same arm)
dense_nobias: d128: 3.11 vs 3.10 (-0.4%), d256: 3.50 vs 3.49 (-0.2%), d512: 8.27 vs 8.28 (+0.2%)
qat_fp4: d128: 9.62 vs 9.70 (+0.8%), d256: 9.31 vs 9.26 (-0.5%), d512: 12.29 vs 12.40 (+0.9%)
na_fp4_c8: d128: 25.50 vs 25.47 (-0.1%), d256: 25.31 vs 25.50 (+0.7%), d512: 28.40 vs 28.58 (+0.6%)
lg6_fp4_h2: d128: 10.22 vs 10.20 (-0.3%), d256: 10.31 vs 10.39 (+0.8%), d512: 11.42 vs 11.41 (-0.1%)
lg8_fp4_h2: d128: 10.24 vs 10.22 (-0.1%), d256: 10.30 vs 10.39 (+0.9%), d512: 11.42 vs 11.41 (-0.1%)

T6 kernel categories (ms of GPU kernel time per step, from the profile)
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

T7 top kernels (eager)
na_fp4_c8 d=128:
   0.287 ms 5.1% x128  void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso
   0.275 ms 4.8% x144  void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso
   0.269 ms 4.7% x96  void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:
   0.254 ms 4.5% x1  Optimizer.step#AdamW.step
   0.191 ms 3.4% x13  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nn_align1>(cutlass_80_simt_sgemm_12
na_fp4_c8 d=256:
   0.583 ms 7.4% x17  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(cutlass_80_simt_sgemm_12
   0.400 ms 5.0% x12  void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_nt_align1>(cutlass_80_simt_sgemm_64x
   0.384 ms 4.8% x8  void magma_sgemmEx_kernel<float, float, float, true, false, 6, 4, 6, 3, 4>(int, int, int, 
   0.297 ms 3.8% x48  void at::native::reduce_kernel<512, 1, at::native::ReduceOp<long, at::native::func_wrapper
   0.293 ms 3.7% x128  void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso
na_fp4_c8 d=512:
   1.684 ms 10.4% x17  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nt_align1>(cutlass_80_simt_sgemm_12
   1.646 ms 10.2% x12  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x256_8x4_tn_align1>(cutlass_80_simt_sgemm_1
   1.290 ms 8.0% x17  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(cutlass_80_simt_sgemm_12
   1.005 ms 6.2% x48  void at::native::reduce_kernel<512, 1, at::native::ReduceOp<long, at::native::func_wrapper
   0.889 ms 5.5% x32  void at::native::vectorized_elementwise_kernel<2, at::native::AUnaryFunctor<long, long, lo
lg6_fp4_h2 d=128:
   0.273 ms 8.3% x96  void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:
   0.223 ms 6.8% x1  Optimizer.step#AdamW.step
   0.204 ms 6.2% x176  void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std:
   0.192 ms 5.8% x13  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nn_align1>(cutlass_80_simt_sgemm_12
   0.181 ms 5.5% x9  void cutlass::Kernel2<cutlass_80_simt_sgemm_64x128_8x5_tn_align1>(cutlass_80_simt_sgemm_64
lg6_fp4_h2 d=256:
   0.556 ms 11.5% x17  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(cutlass_80_simt_sgemm_12
   0.394 ms 8.2% x12  void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_nt_align1>(cutlass_80_simt_sgemm_64x
   0.369 ms 7.6% x8  void magma_sgemmEx_kernel<float, float, float, true, false, 6, 4, 6, 3, 4>(int, int, int, 
   0.269 ms 5.6% x96  void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:
   0.259 ms 5.4% x13  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nt_align1>(cutlass_80_simt_sgemm_12
lg6_fp4_h2 d=512:
   1.674 ms 16.6% x17  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nt_align1>(cutlass_80_simt_sgemm_12
   1.579 ms 15.7% x12  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x256_8x4_tn_align1>(cutlass_80_simt_sgemm_1
   1.281 ms 12.7% x17  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(cutlass_80_simt_sgemm_12
   0.578 ms 5.7% x4  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>(cutlass_80_simt_sgemm_2
   0.546 ms 5.4% x4  void magma_sgemmEx_kernel<float, float, float, true, false, 6, 4, 6, 3, 4>(int, int, int, 

T8 SPEED_NOTES estimate vs measured: update time in bwd (ms) and B/param implied
na_fp4_c8 d=128: measured update 16.06 ms (+ decode 3.64); SPEED_NOTES ideal HBM time at 1.5 TB/s 0.94 ms; ratio measured/floor 17.0x
na_fp4_c8 d=256: measured update 15.40 ms (+ decode 3.64); SPEED_NOTES ideal HBM time at 1.5 TB/s 3.77 ms; ratio measured/floor 4.1x
na_fp4_c8 d=512: measured update 14.51 ms (+ decode 3.50); SPEED_NOTES ideal HBM time at 1.5 TB/s 15.10 ms; ratio measured/floor 1.0x
na_fp4_c8+fastpack d=128: measured update 14.10 ms (+ decode 3.52); SPEED_NOTES ideal HBM time at 1.5 TB/s 0.51 ms; ratio measured/floor 27.6x
na_fp4_c8+fastpack d=256: measured update 13.53 ms (+ decode 3.51); SPEED_NOTES ideal HBM time at 1.5 TB/s 2.04 ms; ratio measured/floor 6.6x
na_fp4_c8+fastpack d=512: measured update 12.18 ms (+ decode 3.49); SPEED_NOTES ideal HBM time at 1.5 TB/s 8.18 ms; ratio measured/floor 1.5x
lg6_fp4_h2 d=128: measured update 6.00 ms (+ decode 0.59); SPEED_NOTES ideal HBM time at 1.5 TB/s 0.24 ms; ratio measured/floor 24.6x
lg6_fp4_h2 d=256: measured update 5.44 ms (+ decode 0.46); SPEED_NOTES ideal HBM time at 1.5 TB/s 0.98 ms; ratio measured/floor 5.6x
lg6_fp4_h2 d=512: measured update 3.12 ms (+ decode 0.31); SPEED_NOTES ideal HBM time at 1.5 TB/s 3.90 ms; ratio measured/floor 0.8x

T9 memory (bench; MB)
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

T12 ms per profiled kernel launch (wall ms / launches) and ceiling if update+decode were free
d=128 dense_nobias: 11.0 us/launch; step 3.11; minus update 0.00 and decode 0.00 = 3.11 ms = 1.00x dense
d=128 qat_fp4: 10.9 us/launch; step 9.62; minus update 0.00 and decode 5.57 = 4.05 ms = 1.30x dense
d=128 tern_counter: 10.8 us/launch; step 11.23; minus update 1.05 and decode 0.00 = 10.18 ms = 3.27x dense
d=128 na_fp4_c8: 9.4 us/launch; step 25.50; minus update 16.06 and decode 3.64 = 5.80 ms = 1.86x dense
d=128 lg6_fp4_h2: 9.2 us/launch; step 10.22; minus update 6.00 and decode 0.59 = 3.64 ms = 1.17x dense
d=256 dense_nobias: 12.1 us/launch; step 3.50; minus update 0.00 and decode 0.00 = 3.50 ms = 1.00x dense
d=256 qat_fp4: 10.3 us/launch; step 9.31; minus update 0.00 and decode 5.42 = 3.89 ms = 1.11x dense
d=256 tern_counter: 11.5 us/launch; step 11.94; minus update 0.82 and decode 0.00 = 11.12 ms = 3.18x dense
d=256 na_fp4_c8: 9.3 us/launch; step 25.31; minus update 15.40 and decode 3.64 = 6.27 ms = 1.79x dense
d=256 lg6_fp4_h2: 9.1 us/launch; step 10.31; minus update 5.44 and decode 0.46 = 4.41 ms = 1.26x dense
d=512 dense_nobias: 27.4 us/launch; step 8.27; minus update 0.00 and decode 0.00 = 8.27 ms = 1.00x dense
d=512 qat_fp4: 13.4 us/launch; step 12.29; minus update 0.00 and decode 4.01 = 8.28 ms = 1.00x dense
d=512 tern_counter: 13.5 us/launch; step 14.25; minus update 0.70 and decode 0.00 = 13.55 ms = 1.64x dense
d=512 na_fp4_c8: 10.4 us/launch; step 28.40; minus update 14.51 and decode 3.50 = 10.39 ms = 1.26x dense
d=512 lg6_fp4_h2: 9.9 us/launch; step 11.42; minus update 3.12 and decode 0.31 = 7.99 ms = 0.97x dense

T13 fastpack effect (bench_x eager na vs fastpack)
d=128: step 25.47 -> 21.64 (-15.0%, -3.83 ms); fwd 7.31->5.57; bwd 19.47->17.43; update 16.00->14.10; launches 2700->2412; kernel ms 5.53->4.97
d=256: step 25.50 -> 21.98 (-13.8%, -3.52 ms); fwd 7.47->5.63; bwd 19.55->17.58; update 15.50->13.53; launches 2717->2429; kernel ms 7.57->6.62
d=512: step 28.58 -> 24.49 (-14.3%, -4.09 ms); fwd 8.87->6.79; bwd 20.55->18.60; update 14.57->12.18; launches 2734->2446; kernel ms 16.17->12.64
T14 compile effect (bench_x)
d=128 dense_nobias: 3.10 -> 2.43 ms (1.28x faster); launches 284->211
d=128 qat_fp4: 9.70 -> 2.94 ms (3.30x faster); launches 884->276
d=128 na_fp4_c8: 25.47 -> 17.90 ms (1.42x faster); launches 2700->1324
d=128 lg6_fp4_h2: 10.20 -> 7.49 ms (1.36x faster); launches 1116->572
d=256 dense_nobias: 3.49 -> 2.90 ms (1.20x faster); launches 289->216
d=256 qat_fp4: 9.26 -> 3.26 ms (2.85x faster); launches 901->281
d=256 na_fp4_c8: 25.50 -> 17.81 ms (1.43x faster); launches 2717->1341
d=256 lg6_fp4_h2: 10.39 -> 7.64 ms (1.36x faster); launches 1133->589
d=512 dense_nobias: 8.28 -> 7.50 ms (1.10x faster); launches 302->229
d=512 qat_fp4: 12.40 -> 7.90 ms (1.57x faster); launches 918->294
d=512 na_fp4_c8: 28.58 -> 21.84 ms (1.31x faster); launches 2734->1358
d=512 lg6_fp4_h2: 11.41 -> 9.18 ms (1.24x faster); launches 1150->606

fp32 scale cache size (4 B x block-linear params, MB): d128: 3.0, d256: 12.0, d512: 48.0
compile peaks: d128 dense_nobias+compile: 291 (eager 188); d128 lg6_fp4_h2+compile: 292 (eager 169); d128 na_fp4_c8+compile: 290 (eager 171); d256 dense_nobias+compile: 455 (eager 361); d256 lg6_fp4_h2+compile: 471 (eager 353); d256 na_fp4_c8+compile: 465 (eager 370); d512 dense_nobias+compile: 808 (eager 827); d512 lg6_fp4_h2+compile: 881 (eager 794); d512 na_fp4_c8+compile: 860 (eager 857)

T10 equal-state-memory
lg6_fp4_h2 d=512 (state 9.82, +grad 13.82, +other 15.93, resident 72, peak 794, val 1.9771) vs dense_nobias d=128 (state 9.00, +grad 12.00, +other 10.53, resident 31, peak 188, val 2.0200): dval -0.0429; state ratio 1.09
tern_counter d=512 (state 9.21, +grad 13.21, +other 15.32, resident 18, peak 673, val 1.9982) vs dense_nobias d=128 (state 9.00, +grad 12.00, +other 10.53, resident 31, peak 188, val 2.0200): dval -0.0218; state ratio 1.02
na_fp4_c8 d=512 (state 12.82, +grad 16.82, +other 18.93, resident 27, peak 857, val 1.9042) vs dense_nobias d=128 (state 9.00, +grad 12.00, +other 10.53, resident 31, peak 188, val 2.0200): dval -0.1158; state ratio 1.42
na_fp4_c8 d=512 (state 12.82, +grad 16.82, +other 18.93, resident 27, peak 857, val 1.9042) vs dense_nobias d=256 (state 36.00, +grad 48.00, +other 39.05, resident 53, peak 361, val 1.8508): dval +0.0534; state ratio 0.36
lg6_fp4_h2 d=512 (state 9.82, +grad 13.82, +other 15.93, resident 72, peak 794, val 1.9771) vs dense_nobias d=256 (state 36.00, +grad 48.00, +other 39.05, resident 53, peak 361, val 1.8508): dval +0.1263; state ratio 0.27
tern_counter d=512 (state 9.21, +grad 13.21, +other 15.32, resident 18, peak 673, val 1.9982) vs dense_nobias d=256 (state 36.00, +grad 48.00, +other 39.05, resident 53, peak 361, val 1.8508): dval +0.1474; state ratio 0.26
lg6_fp4_h2 d=256 (state 2.47, +grad 3.47, +other 5.53, resident 20, peak 353, val 2.0821) vs dense_nobias d=128 (state 9.00, +grad 12.00, +other 10.53, resident 31, peak 188, val 2.0200): dval +0.0621; state ratio 0.27
na_fp4_c8 d=256 (state 3.22, +grad 4.22, +other 6.28, resident 9, peak 370, val 1.9847) vs dense_nobias d=128 (state 9.00, +grad 12.00, +other 10.53, resident 31, peak 188, val 2.0200): dval -0.0353; state ratio 0.36
tern_counter d=256 (state 2.36, +grad 3.36, +other 5.41, resident 7, peak 333, val 2.1495) vs dense_nobias d=128 (state 9.00, +grad 12.00, +other 10.53, resident 31, peak 188, val 2.0200): dval +0.1295; state ratio 0.26
lg6_fp4_h2 d=512 (state 9.82, +grad 13.82, +other 15.93, resident 72, peak 794, val 1.9771) vs qat_fp4 d=128 (state 9.00, +grad 12.00, +other 10.53, resident 15, peak 174, val 2.0491): dval -0.0719; state ratio 1.09
na_fp4_c8 d=512 (state 12.82, +grad 16.82, +other 18.93, resident 27, peak 857, val 1.9042) vs qat_fp4 d=128 (state 9.00, +grad 12.00, +other 10.53, resident 15, peak 174, val 2.0491): dval -0.1448; state ratio 1.42

T11 train-run peak (from q logs)
dense_nobias d=128: state 9.0 +grad 12.0 peak 338 over 185
dense_nobias d=256: state 36.0 +grad 48.0 peak 529 over 376
dense_nobias d=512: state 144.0 +grad 192.0 peak 990 over 838
qat_fp4 d=128: state 9.0 +grad 12.0 peak 341 over 188
qat_fp4 d=256: state 36.0 +grad 48.0 peak 542 over 389
qat_fp4 d=512: state 144.0 +grad 192.0 peak 1064 over 912
tern_counter d=128: state 0.6 +grad 0.9 peak 334 over 182
tern_counter d=256: state 2.4 +grad 3.4 peak 501 over 348
tern_counter d=512: state 9.2 +grad 13.2 peak 841 over 689
na_fp4_c8 d=128: state 0.8 +grad 1.1 peak 339 over 187
na_fp4_c8 d=256: state 3.2 +grad 4.2 peak 538 over 385
na_fp4_c8 d=512: state 12.8 +grad 16.8 peak 1025 over 872
lg6_fp4_h2 d=128: state 0.6 +grad 0.9 peak 335 over 183
lg6_fp4_h2 d=256: state 2.5 +grad 3.5 peak 521 over 369
lg6_fp4_h2 d=512: state 9.8 +grad 13.8 peak 963 over 810