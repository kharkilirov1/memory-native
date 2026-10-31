# mx6 on G4: block-scale headroom and the latent grid, held-out - 2026-10-31

Raw evidence (17 unedited logs, one per arm, copied byte-for-byte from the Colab G4 session archive `q_logs.tar`): `mx6_g4_headroom_lg_2026_10_31/`.
Continues `mx6_g4_decomp_2026_10_31.md` (same harness, split, GPU, schedule). Numbers from that report are cited as "decomp" and come from an OLDER code snapshot (see Limits).

Every number below is copied from those logs or derived from them (marked "derived"). Statements are labelled **[verified]** (read off the logs, the job scripts or the code) or **[hypothesis]** (not tested by these runs).
sd = sample sd (n-1), the convention of the logs. All means were recomputed from the per-seed lines and agree with the logs' summary rows to 0.0001 (rounding); the tables below use the logs' own summary values.
"M2" = rule M2 of `docs/LESSONS.md`: an effect smaller than 3x the larger sd of the two arms is "not detectable". Every arm here has 2 seeds (M2 asks for 4), so the M2 column only says which gaps are large against the observed spread.

## TL;DR

1. **[verified]** Static block-scale headroom **h=2** is the main lever for full-E2M1 counters. C=8: 2.3724 -> **2.3151** (-0.0573, both seeds, 5.1x sd). C=2: 2.5415 -> 2.5022 (-0.0393, both seeds, below the M2 bar). h=4 (2.3298) is not better than h=2.
2. **[verified]** Dynamic scale at h=1 helps less, and nothing beats h=2: pressure growth 2.3458, learnable scale 1e-3 2.3511 (+0.188 bits), 3e-3 2.3771 (= base). h=2 combined with them was not run.
3. **[verified]** lr interacts with h: at h=2, lr 0.01 beats 0.005 (+0.0208 for 0.005); at h=1, lr 0.005 is slightly better than 0.01 (-0.0146) and 0.02 is worse (+0.0295). "lr 0.01 is optimal at C=8" holds only at h=2 and only against 0.005 (nothing above 0.01 was tried at h=2).
4. **[verified]** Latent grid (one k-bit index into a fine E2M1-aligned grid, no counter) with h=2: **2.4155 @ 6.69 / 2.3513 @ 7.69 / 2.3253 @ 8.69 bits**. It beats the counter at the same bits at 6.69 (-0.0867 vs C=2) and 7.69 (-0.0322 vs C=4, just over the M2 bar); at 8.69 it ties (counter ahead by 0.0102, under the M2 bar).
5. **[verified]** The old reduced ladder {0,.5,1,6} does not profit from h=2: 2.5661 vs 2.5449 for the same two seeds in the decomp run (+0.0212; cross-run baseline, see (f)).
6. **[derived]** The counter cost at C=8 against `qat_fp4` (2.2983, decomp) falls from +0.062 (decomp, h=1) / +0.074 (this batch, h=1) to **+0.017** at h=2. Best 8.69-bit arm is +0.026 above `dense_nobias` (2.2891) and +0.029 above `dense_bf16` (2.2864).

## Frontier: bits -> quality (held-out val, bits/char)

Bits = measured bits/param as printed by the harness (accounting bits of the persistent state, see Limits). Lower val is better. "decomp" rows are other runs of the same harness on an older snapshot.

| bits | best arm in this batch | val | vs `qat_fp4` 2.2983 | vs `dense_bf16` 2.2864 | previous best at this width (decomp) | change |
|---:|---|---:|---:|---:|---|---:|
| 6.69 | `lg6_fp4_h2` (latent grid k=6) | **2.4155** | +0.1172 | +0.1291 | `tern_counter` 2.4811 @ 6.56 bits (lr 3e-2, 4 seeds); mx6 `e2m1_c5` 2.5474 @ 6.69 (3 seeds) | -0.0656 (+0.126 bits) / -0.1319 |
| 7.69 | `lg7_fp4_h2` (latent grid k=7) | **2.3513** | +0.0530 | +0.0649 | `na_fp4_c4` 2.4362 (h=1, 2 seeds) | -0.0849 (cross-run) |
| 8.69 | `fp4c8_h2` (counter C=8, h=2) | **2.3151** | +0.0168 | +0.0287 | `na_fp4_c8` 2.3607 (h=1, 2 seeds) | -0.0456 (cross-run); -0.0573 vs same-batch control 2.3724 |

Everything else at these widths, same batch (val):

```
6.69 bits  lg6_fp4_h2 2.4155 < fp4c2_h2 2.5022 < fp4c2 (h=1) 2.5415 < [decomp e2m1_c5 2.5474] < e2m1c5_h2 2.5661
7.69 bits  lg7_fp4_h2 2.3513 < fp4c4_h2 2.3835                      [decomp na_fp4_c4 (h=1) 2.4362]
8.69 bits  fp4c8_h2 2.3151 < lg8_fp4_h2 2.3253 < fp4c8_h4 2.3298 < fp4c8_h2_lr0.005 2.3359 < fp4c8_press 2.3458
           < fp4c8_lr0.005 (h=1) 2.3578 < fp4c8_base 2.3724 < lg8_fp4 (h=1) 2.3977 < fp4c8_lr0.02 (h=1) 2.4019
8.88 bits  (learnable scale)  fp4c8_slr1e-3 2.3511, fp4c8_slr3e-3 2.3771
```

Step per extra bit (derived): latent grid h=2 6.69 -> 7.69: -0.0642; 7.69 -> 8.69: -0.0260. Counter h=2 (C=2 -> 4 -> 8): -0.1187, then -0.0684.
Reading, per **[hypothesis]**: returns diminish fast beyond 7.7 bits; the 8.69-bit counter and the 8.69-bit grid are within noise of each other. Not on this frontier by definition: `qat_fp4` / dense, whose training state is 96 bits/param (fp32 latent + AdamW) even though the deployed weight is about 4.5 bits (4-bit E2M1 + 0.5 bit of E4M3 scale per 16; derived).
Against the O1 kill line "dense + 0.15 bits/char at d=128" (LESSONS O1): `lg6_fp4_h2` is +0.129 vs `dense_bf16` at 6.69 bits (the best arm at 6.6-6.7 bits in the decomp report, `tern_counter`, was +0.195; mx6 `e2m1_c5` +0.261). The second clause of O1 (gap shrinking at d=256-512) is untested.

## Setup

| item | value |
|---|---|
| GPU | `NVIDIA RTX PRO 6000 Blackwell Server Edition` (log header). torch / triton versions are not printed in these logs |
| model | causal transformer, `d=128`, `seq=128`, `batch=32`, `vocab=65`, loss in bits/char |
| data | full tinyshakespeare, `tokens=1115394`; contiguous split: train = first 90 % (`train_tokens=1003854`), val = last 10 % (`val_tokens=111540`), `val_frac=0.1`, `epochs=24.48` |
| schedule | 6000 steps, cosine (warmup 0.02, min 0.1), `eval_every=500`, counter / latent `lr=0.01` unless stated, `adam_lr=0.003` for the dense parts |
| scale format | `scale_format=nvfp4` in every log (E4M3 byte per 16 + fp32 per tensor) |
| seeds | 0, 1 for every arm (n=2). `val` = eval at step 6000 |
| bits | the `bits=` printed per seed (harness accounting): code bits + 0.5 (E4M3 per 16) + row / tensor state. Code bits = ceil(log2 n_codes) for the counters (n_codes = 15 x (2C-1): C=2 -> 45, C=4 -> 105, C=8 -> 225; e2m1 C=5 -> 7 x 9 = 63) and k for the latent grid (`grid_points` 57/64, 127/128, 253/256, printed in the lg logs). 8.876 = 8.688 + 0.188 for the learnable-scale arms (derived) |
| code version | Colab ran the snapshot `g4e/mx.tar` (sha256 `7DFB7CF21D0DBEA2...`, written 2026-10-31 23:21): the working tree after the fp4-ladder / overflow-guard / scale-rebase edits plus the new `mx6/latent_grid.py` and `probes/lg_ab.py`. SHA-256 of `mx6/{layer,packed_na,latent_grid,update,pack}.py` and `probes/{e1_quality,lg_ab}.py` in the snapshot equal the working-tree files at the time of writing. Repo HEAD is `2299ee7`; those edits are uncommitted, so the runs are not reproducible from a commit |

**Arm flags come from the job scripts, not from the logs.** The log headers print `counter_wd`, `scale_lr`, `lr`, `scale_format` (and the lg logs print `headroom=`) but NOT `--headroom`, `--headroom-mode` or `--scale-growth` for the `e1_quality` arms. Commands (Colab, `/content/mx/mxfp4-native`, `PYTHONPATH=/content/mx/src:/content/mx/mxfp4-native`; scripts `claude-ctl\g4e\{jobs,h2,lg}.py`, not in the repo), all with `--d 128 --steps 6000 --sched cosine --eval-every 500 --log-curve --hist --seeds 0,1 --scale-format nvfp4`; the nine jobs of `jobs.py`, the four of `h2.py` and the four of `lg.py` were each launched as concurrent background processes (`Popen`, no wait) on the one GPU the header names, so the wall-times in the logs are not speed measurements:

| log | script | extra flags |
|---|---|---|
| fp4c8_base / lr0.005 / lr0.02 | jobs.py | `--arms na_fp4_c8 --lr 0.01 / 0.005 / 0.02` |
| fp4c8_slr1e-3 / slr3e-3 | jobs.py | `--arms na_fp4_c8 --lr 0.01 --scale-lr 1e-3 / 3e-3` |
| fp4c8_press | jobs.py | `--arms na_fp4_c8 --lr 0.01 --scale-growth pressure` |
| fp4c8_h2 / h4 | jobs.py | `--arms na_fp4_c8 --lr 0.01 --headroom-mode ladder --headroom 2 / 4` |
| fp4c2 | jobs.py | `--arms na_fp4_c2 --lr 0.01` |
| fp4c4_h2 / fp4c2_h2 / e2m1c5_h2 | h2.py | `--arms na_fp4_c4 / na_fp4_c2 / na_e2m1_c5 --lr 0.01 --headroom-mode ladder --headroom 2` |
| fp4c8_h2_lr0.005 | h2.py | `--arms na_fp4_c8 --lr 0.005 --headroom-mode ladder --headroom 2` |
| lg8_fp4 / lg8_fp4_h2 / lg7_fp4_h2 / lg6_fp4_h2 | lg.py | `probes/lg_ab.py ... --lr 0.01 --val-frac 0.1 --arms <name>` |

`--headroom-mode ladder --headroom h` sets the NVFP4 target block amax / scale = L/h with L = 6 (outer rung): h=1 puts the block max on the outer rung, h=2 on 3, h=4 on 1.5 (code: `layer.py _nvfp4_target`; `lg_ab.py` docstring for `_h<h>`). The harness default (`absolute` mode, nvfp4) already uses target L, i.e. **h=1**, so "h=1" arms and the decomp runs share the same init rule. Block scale is frozen after init except under `--scale-growth pressure` (per-block bump + per-tensor rebase, counters `grown` / `rebased`) and `--scale-lr` (counters `scale_moves` / `rebased`); latent-grid arms support neither.

Arms: `fp4cK` = `na_fp4_c<K>` = PackedNA on the full E2M1 ladder {0,.5,1,1.5,2,3,4,6}, counter depth K; `e2m1c5` = ladder {0,.5,1,6}, C=5 (the 6-bit mx6 word); `lgK_fp4[_h2]` = `LatentGridLinear` (`mx6/latent_grid.py`): the whole k-bit word is ONE index into the E2M1 values with each gap cut into m = (2^k - 1)//14 sub-steps (k=6 -> 57 points, 7 -> 127, 8 -> 253); deployed weight = round-to-nearest to E2M1 x block scale; stochastic-rounding update inside backward, no momentum, no counter.

## The arm table (held-out, every arm, n=2)

Rows are the logs' summary rows (mean over seeds 0 and 1); `gap = val - train`; `train` = loss on a fixed train subset. References for scale (decomp, 4 seeds): dense_bf16 2.2864 (train 1.9259), dense_nobias 2.2891 (train 1.9208), qat_fp4 2.2983 (train 1.9635), `tern_counter` 2.4811.

| arm | log | knob | bits | val mean | per-seed val (0, 1) | sd | train | gap |
|---|---|---|---:|---:|---|---:|---:|---:|
| fp4c8_base | `fp4c8_base.log` | C=8, h=1, lr 0.01 (control) | 8.688 | **2.3724** | 2.3644, 2.3805 | 0.0113 | 2.0664 | +0.3060 |
| fp4c8_lr0.005 | `fp4c8_lr0.005.log` | C=8, h=1, lr 0.005 | 8.688 | **2.3578** | 2.3553, 2.3603 | 0.0035 | 2.0603 | +0.2975 |
| fp4c8_lr0.02 | `fp4c8_lr0.02.log` | C=8, h=1, lr 0.02 | 8.688 | **2.4019** | 2.3991, 2.4048 | 0.0040 | 2.1155 | +0.2864 |
| fp4c8_press | `fp4c8_press.log` | C=8, h=1, `--scale-growth pressure` | 8.688 | **2.3458** | 2.3569, 2.3347 | 0.0157 | 2.0144 | +0.3314 |
| fp4c8_slr1e-3 | `fp4c8_slr1e-3.log` | C=8, h=1, `--scale-lr 1e-3` | 8.876 | **2.3511** | 2.3474, 2.3549 | 0.0053 | 2.0517 | +0.2995 |
| fp4c8_slr3e-3 | `fp4c8_slr3e-3.log` | C=8, h=1, `--scale-lr 3e-3` | 8.876 | **2.3771** | 2.3887, 2.3656 | 0.0163 | 2.0573 | +0.3198 |
| fp4c8_h2 | `fp4c8_h2.log` | C=8, **h=2**, lr 0.01 | 8.688 | **2.3151** | 2.3191, 2.3111 | 0.0057 | 1.9909 | +0.3242 |
| fp4c8_h4 | `fp4c8_h4.log` | C=8, h=4, lr 0.01 | 8.688 | **2.3298** | 2.3233, 2.3363 | 0.0092 | 1.9961 | +0.3337 |
| fp4c8_h2_lr0.005 | `fp4c8_h2_lr0.005.log` | C=8, h=2, lr 0.005 | 8.688 | **2.3359** | 2.3303, 2.3415 | 0.0079 | 2.0218 | +0.3141 |
| fp4c4_h2 | `fp4c4_h2.log` | C=4, h=2 | 7.688 | **2.3835** | 2.3867, 2.3804 | 0.0045 | 2.0766 | +0.3069 |
| fp4c2 | `fp4c2.log` | C=2, h=1 | 6.688 | **2.5415** | 2.5292, 2.5538 | 0.0174 | 2.3002 | +0.2413 |
| fp4c2_h2 | `fp4c2_h2.log` | C=2, h=2 | 6.688 | **2.5022** | 2.4978, 2.5065 | 0.0062 | 2.2406 | +0.2615 |
| e2m1c5_h2 | `e2m1c5_h2.log` | e2m1 ladder {0,.5,1,6}, C=5, h=2 | 6.688 | **2.5661** | 2.5663, 2.5659 | 0.0003 | 2.2899 | +0.2762 |
| lg8_fp4 | `lg8_fp4.log` | latent grid k=8, h=1 | 8.688 | **2.3977** | 2.4007, 2.3948 | 0.0042 | 2.1195 | +0.2782 |
| lg8_fp4_h2 | `lg8_fp4_h2.log` | latent grid k=8, h=2 | 8.688 | **2.3253** | 2.3286, 2.322 | 0.0047 | 2.0061 | +0.3193 |
| lg7_fp4_h2 | `lg7_fp4_h2.log` | latent grid k=7, h=2 | 7.688 | **2.3513** | 2.3586, 2.344 | 0.0103 | 2.0498 | +0.3015 |
| lg6_fp4_h2 | `lg6_fp4_h2.log` | latent grid k=6, h=2 | 6.688 | **2.4155** | 2.4202, 2.4108 | 0.0067 | 2.1411 | +0.2744 |

Telemetry (summary rows and per-seed lines of the same logs). `edge` = share of deployed weights on the outer rungs (+-6); `flip` = share of deployed weights whose rung changed over the last eval interval; `clip` = for the counter arms, share of updates clipped at an outer rung; for the lg arms, requested moves beyond the grid ends (not the same quantity; do not compare across the two families). `kurt` / `tail3` / `zero` = kurtosis, share of |w| > 3 rms_row (Gaussian 0.27 %) and share of exact zeros of the DEPLOYED weights of the 4 block linears (Gaussian kurt 3, uniform 1.8; decomp `qat_fp4`: kurt 3.07, zero 6.9 %, `na_fp4_c8` at h=1: 1.66 / 2.3 %). `grown` = cumulative block-scale bumps over the run, `rebased` = per-tensor scale doublings, `scale_moves` = E4M3 bytes moved by the learnable scale (all cumulative, summed over layers: `layer.py total_grown / total_rebased / total_scale_moves`).

| arm | edge % | flip % (last interval) | clip % | kurt | tail3 % | zero % | grown (s0 / s1) | rebased (s0 / s1) | scale_moves (s0 / s1) |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| fp4c8_base | 35.60 | 56.77 | 0.07 | 1.65 | 0.00 | 2.2 | 0 / 0 | - | - |
| fp4c8_lr0.005 | 30.18 | 54.57 | 0.03 | 1.79 | 0.00 | 2.7 | 0 / 0 | - | - |
| fp4c8_lr0.02 | 42.25 | 56.13 | 0.14 | 1.53 | 0.00 | 1.9 | 0 / 0 | - | - |
| fp4c8_press | 19.21 | 36.38 | 0.00 | 2.53 | 0.02 | 4.6 | 427514 / 428835 | 28 / 27 | - |
| fp4c8_slr1e-3 | 31.74 | 57.71 | 0.06 | 3.80 | 0.85 | 2.6 | 0 / 0 | 28 / 31 | 710714 / 711985 |
| fp4c8_slr3e-3 | 30.18 | 56.43 | 0.06 | 6.67 | 1.89 | 3.0 | 0 / 0 | 40 / 40 | 1921822 / 1924851 |
| fp4c8_h2 | 24.10 | 53.78 | 0.02 | 2.01 | 0.00 | 3.6 | 0 / 0 | - | - |
| fp4c8_h4 | 14.53 | 44.45 | 0.00 | 2.65 | 0.00 | 5.9 | 0 / 0 | - | - |
| fp4c8_h2_lr0.005 | 19.71 | 49.16 | 0.01 | 2.25 | 0.00 | 4.6 | 0 / 0 | - | - |
| fp4c4_h2 | 18.58 | 68.74 | 0.03 | 2.26 | 0.00 | 4.7 | 0 / 0 | - | - |
| fp4c2 | 19.03 | 81.38 | 0.15 | 2.24 | 0.00 | 4.7 | 0 / 0 | - | - |
| fp4c2_h2 | 16.05 | 77.75 | 0.05 | 2.43 | 0.00 | 5.6 | 0 / 0 | - | - |
| e2m1c5_h2 | 36.06 | 51.61 | 0.02 | 2.72 | 0.00 | 10.7 | 0 / 0 | - | - |
| lg8_fp4 | 41.89 | 44.03 | 2.31 | 1.50 | 0.00 | 1.8 | 0 / 0 | - | - |
| lg8_fp4_h2 | 27.69 | 40.48 | 1.39 | 1.79 | 0.00 | 2.8 | 0 / 0 | - | - |
| lg7_fp4_h2 | 18.97 | 56.04 | 1.83 | 1.96 | 0.00 | 3.3 | 0 / 0 | - | - |
| lg6_fp4_h2 | 16.12 | 69.98 | 2.66 | 2.22 | 0.00 | 3.5 | 0 / 0 | - | - |

Seed-0 signed-magnitude histograms of the deployed weights (share per rung, as logged; the lg logs print the histogram after RTN to E2M1):

```
fp4c8_base        -6:17.8%  -4:11.7%  -3:7.3%  -2:4.2%  -1.5:2.9%  -1:2.5%  -0.5:2.3%  +0:2.2%  +0.5:2.3%  +1:2.5%  +1.5:3.0%  +2:4.3%  +3:7.2%  +4:11.8%  +6:17.9%
fp4c8_lr0.005     -6:15.2%  -4:11.2%  -3:7.5%  -2:4.9%  -1.5:3.8%  -1:3.2%  -0.5:2.8%  +0:2.7%  +0.5:2.9%  +1:3.2%  +1.5:3.9%  +2:4.9%  +3:7.6%  +4:11.2%  +6:15.1%
fp4c8_lr0.02      -6:21.2%  -4:11.7%  -3:6.3%  -2:3.5%  -1.5:2.3%  -1:2.1%  -0.5:1.9%  +0:1.9%  +0.5:2.0%  +1:2.1%  +1.5:2.4%  +2:3.5%  +3:6.3%  +4:11.7%  +6:21.2%
fp4c8_press       -6:9.5%  -4:9.0%  -3:7.7%  -2:6.3%  -1.5:5.5%  -1:5.0%  -0.5:4.7%  +0:4.7%  +0.5:4.7%  +1:5.0%  +1.5:5.4%  +2:6.3%  +3:7.7%  +4:9.0%  +6:9.6%
fp4c8_slr1e-3     -6:15.9%  -4:11.3%  -3:7.4%  -2:4.7%  -1.5:3.5%  -1:3.0%  -0.5:2.7%  +0:2.6%  +0.5:2.7%  +1:3.0%  +1.5:3.5%  +2:4.7%  +3:7.5%  +4:11.4%  +6:16.0%
fp4c8_slr3e-3     -6:15.0%  -4:10.8%  -3:7.5%  -2:5.0%  -1.5:3.9%  -1:3.3%  -0.5:3.1%  +0:3.0%  +0.5:3.0%  +1:3.3%  +1.5:3.8%  +2:5.1%  +3:7.5%  +4:10.7%  +6:15.1%
fp4c8_h2          -6:12.0%  -4:10.3%  -3:7.7%  -2:5.7%  -1.5:4.7%  -1:4.0%  -0.5:3.7%  +0:3.7%  +0.5:3.7%  +1:4.1%  +1.5:4.7%  +2:5.7%  +3:7.7%  +4:10.2%  +6:12.2%
fp4c8_h4          -6:7.3%  -4:7.5%  -3:7.1%  -2:6.8%  -1.5:6.4%  -1:6.2%  -0.5:5.9%  +0:5.9%  +0.5:5.9%  +1:6.1%  +1.5:6.4%  +2:6.7%  +3:7.1%  +4:7.5%  +6:7.3%
fp4c8_h2_lr0.005  -6:9.8%  -4:9.0%  -3:7.4%  -2:6.3%  -1.5:5.5%  -1:5.0%  -0.5:4.6%  +0:4.6%  +0.5:4.7%  +1:5.0%  +1.5:5.5%  +2:6.2%  +3:7.5%  +4:9.0%  +6:9.9%
fp4c4_h2          -6:9.3%  -4:8.8%  -3:7.5%  -2:6.4%  -1.5:5.7%  -1:5.2%  -0.5:4.8%  +0:4.7%  +0.5:4.8%  +1:5.1%  +1.5:5.7%  +2:6.4%  +3:7.6%  +4:8.7%  +6:9.3%
fp4c2             -6:9.5%  -4:8.5%  -3:7.4%  -2:6.4%  -1.5:5.8%  -1:5.2%  -0.5:4.8%  +0:4.7%  +0.5:4.8%  +1:5.1%  +1.5:5.8%  +2:6.4%  +3:7.3%  +4:8.6%  +6:9.5%
fp4c2_h2          -6:8.1%  -4:7.7%  -3:7.1%  -2:6.6%  -1.5:6.1%  -1:5.9%  -0.5:5.7%  +0:5.6%  +0.5:5.6%  +1:5.9%  +1.5:6.3%  +2:6.7%  +3:7.2%  +4:7.7%  +6:8.0%
e2m1c5_h2         -6:18.1%  -1:14.9%  -0.5:11.6%  +0:10.8%  +0.5:11.6%  +1:15.0%  +6:18.0%
lg8_fp4           -6:20.9%  -4:13.4%  -3:5.5%  -2:3.1%  -1.5:2.2%  -1:2.0%  -0.5:1.9%  +0:1.8%  +0.5:1.9%  +1:2.0%  +1.5:2.2%  +2:3.2%  +3:5.6%  +4:13.5%  +6:21.0%
lg8_fp4_h2        -6:13.8%  -4:13.5%  -3:7.1%  -2:4.5%  -1.5:3.5%  -1:3.1%  -0.5:3.1%  +0:2.8%  +0.5:3.0%  +1:3.2%  +1.5:3.4%  +2:4.5%  +3:7.1%  +4:13.6%  +6:13.8%
lg7_fp4_h2        -6:9.4%  -4:13.0%  -3:8.4%  -2:5.8%  -1.5:4.5%  -1:3.8%  -0.5:3.4%  +0:3.3%  +0.5:3.4%  +1:3.7%  +1.5:4.5%  +2:5.8%  +3:8.5%  +4:13.0%  +6:9.5%
lg6_fp4_h2        -6:8.1%  -4:9.7%  -3:7.9%  -2:6.8%  -1.5:5.8%  -1:5.1%  -0.5:4.8%  +0:3.5%  +0.5:4.8%  +1:5.2%  +1.5:5.8%  +2:6.7%  +3:8.0%  +4:9.7%  +6:8.0%
```

## Paired differences (first minus second; negative = first is better)

"Larger sd" = the larger of the two arms' logged sds. M2 = difference >= 3x that sd. The last three rows compare with decomp runs on an older snapshot and are NOT same-batch.

| comparison | delta val | per-seed (0, 1) | larger sd | delta / sd | M2 |
|---|---:|---|---:|---:|---|
| fp4c8_h2 - fp4c8_base (headroom 2 vs 1, C=8) | -0.0573 | -0.0453, -0.0694 | 0.0113 | 5.1 | yes |
| fp4c8_h4 - fp4c8_base | -0.0426 | -0.0411, -0.0442 | 0.0113 | 3.8 | yes |
| fp4c8_h4 - fp4c8_h2 | +0.0147 | +0.0042, +0.0252 | 0.0092 | 1.6 | no |
| fp4c8_press - fp4c8_base | -0.0266 | -0.0075, -0.0458 | 0.0157 | 1.7 | no |
| fp4c8_slr1e-3 - fp4c8_base (+0.188 bits) | -0.0213 | -0.0170, -0.0256 | 0.0113 | 1.9 | no |
| fp4c8_slr3e-3 - fp4c8_base (+0.188 bits) | +0.0047 | +0.0243, -0.0149 | 0.0163 | 0.3 | no |
| fp4c8_h2 - fp4c8_press | -0.0307 | -0.0378, -0.0236 | 0.0157 | 2.0 | no (same sign, both seeds) |
| fp4c8_h2 - fp4c8_slr1e-3 | -0.0360 | -0.0283, -0.0438 | 0.0057 | 6.3 | yes |
| fp4c8_h2 - fp4c8_slr3e-3 | -0.0620 | -0.0696, -0.0545 | 0.0163 | 3.8 | yes |
| fp4c8_lr0.005 - fp4c8_base (h=1) | -0.0146 | -0.0091, -0.0202 | 0.0113 | 1.3 | no |
| fp4c8_lr0.02 - fp4c8_base (h=1) | +0.0295 | +0.0347, +0.0243 | 0.0113 | 2.6 | no (worse in both seeds) |
| fp4c8_h2_lr0.005 - fp4c8_h2 | +0.0208 | +0.0112, +0.0304 | 0.0079 | 2.6 | no (worse in both seeds) |
| fp4c8_h2_lr0.005 - fp4c8_lr0.005 (h=2 vs 1 at lr 0.005) | -0.0219 | -0.0250, -0.0188 | 0.0079 | 2.8 | no (better in both seeds) |
| fp4c2_h2 - fp4c2 (h=2 vs 1, C=2) | -0.0393 | -0.0314, -0.0473 | 0.0174 | 2.3 | no (better in both seeds) |
| lg8_fp4_h2 - lg8_fp4 (h=2 vs 1, k=8) | -0.0724 | -0.0721, -0.0728 | 0.0047 | 15.4 | yes |
| lg8_fp4 - fp4c8_base (grid vs counter, h=1) | +0.0253 | +0.0363, +0.0143 | 0.0113 | 2.2 | no (counter ahead in both) |
| lg8_fp4_h2 - fp4c8_h2 (grid vs counter, 8.69 bits) | +0.0102 | +0.0095, +0.0109 | 0.0057 | 1.8 | no (counter ahead in both) |
| lg7_fp4_h2 - fp4c4_h2 (grid vs counter, 7.69 bits) | -0.0322 | -0.0281, -0.0364 | 0.0103 | 3.1 | yes (barely) |
| lg6_fp4_h2 - fp4c2_h2 (grid vs counter, 6.69 bits) | -0.0867 | -0.0776, -0.0957 | 0.0067 | 12.9 | yes |
| e2m1c5_h2 - decomp `na_e2m1_c5` nvfp4, seeds 0-1 (2.5449) | +0.0212 | +0.0222, +0.0202 | 0.0011 (decomp) | n/a | cross-run |
| fp4c8_base - decomp `na_fp4_c8` (2.3607), same flags | +0.0117 | +0.0079, +0.0156 | - | - | cross-run drift reference |
| fp4c4_h2 - decomp `na_fp4_c4` (2.4362, h=1) | -0.0527 | -0.0439, -0.0615 | - | - | cross-run |

## Findings

**(a) Headroom h=2 is the main lever for full-E2M1 counters. [verified]** C=8: 2.3724 (h=1) -> 2.3151 (h=2), -0.0573, both seeds (-0.0453 / -0.0694), 5.1x the larger sd. C=2: 2.5415 -> 2.5022, -0.0393, both seeds, below the M2 bar because the h=1 arm has sd 0.0174. C=4 has no h=1 arm in this batch; against the decomp C=4 run (h=1, other snapshot) it is -0.0527, the same size as C=2 and C=8 (derived, cross-run). h=4 (2.3298) is 0.0147 behind h=2 (per-seed +0.0042 / +0.0252, 1.6x sd: not detectable) and still 0.0426 ahead of h=1. h=3 was not run, so the optimum in [2, 4] is not resolved. Early training is faster with headroom (seed 0, step 500: h=1 3.079, h=2 2.960, h=4 2.971); the lead persists to step 6000.
Shape telemetry moves in step (h=1 / 2 / 4): edge 35.60 / 24.10 / 14.53 %, kurtosis 1.65 / 2.01 / 2.65, zeros 2.2 / 3.6 / 5.9 %, clip 0.07 / 0.02 / 0.00 %, final-interval flips 56.8 / 53.8 / 44.5 %. Train falls more than val: train 2.0664 / 1.9909 / 1.9961, gap +0.306 / +0.324 / +0.334. h=4 is the most bell-shaped (kurtosis 2.65, `qat_fp4` 3.07 in decomp) and is not better than h=2, so bell-shape alone is not the objective.
**[hypothesis]** At h=1 the init puts every block max on the outer rung, weights that grow have nowhere to go and pile on +-6 (the U-shape of decomp finding (f)); h=2 leaves a factor 2. The edge / kurtosis shifts agree with this and the val gain follows, but h also halves the absolute grid step near typical weights at init (finer effective resolution). These two effects are not separated by any run here.

**(b) Dynamic scale at h=1 helps less than static headroom. [verified numbers, interpretation = hypothesis]** All three arms run at h=1 and none was combined with h=2.
- Pressure growth: 2.3458 (-0.0266 vs base; per-seed -0.0075 / -0.0458; sd 0.0157; M2: not detectable). `grown` 427514 / 428835 bumps and `rebased` 28 / 27 per-tensor doublings; edge 19.2 %, clip 0.00 %, kurtosis 2.53, final-interval flips 36.4 % (lowest of all counter arms). h=2 is ahead of it by 0.0307 in both seeds (2.0x sd: not M2-detectable).
- Learnable scale (rebase fix active, so the scale can now actually move: `scale_moves` ~0.71 M at 1e-3 and ~1.92 M at 3e-3, `rebased` 28-31 and 40): 1e-3 gives 2.3511 (-0.0213, not detectable) at +0.188 bits; 3e-3 gives 2.3771 (+0.0047, = base). The weights get heavy tails, not a bell: kurtosis 3.80 / 6.67 and tail3 0.85 / 1.89 % (Gaussian 0.27 %). h=2 beats both (-0.0360, -0.0620), M2-detectable.
- Relation to decomp (e) / (f): in decomp the learnable scale hurt on the reduced ladder without the rebase (e2m1 C=5, scale-lr 3e-3 / 1e-2: +0.040 / +0.063, 2 seeds). Here, on the fp4 ladder at C=8 with the rebase, 1e-3 is neutral to slightly better and 3e-3 neutral (different ladder and lr grid, so not a like-for-like change), and neither is better than h=2. Pressure growth was the "next experiment" of decomp (f): it reduces edge% (35.6 -> 19.2) and val, i.e. it supports the ceiling hypothesis, and it is still worse than the static h=2.
- **[hypothesis]** Pressure bumps a block only after its clip counter fills and only upward; the learnable scale adds a noisy per-block degree of freedom (heavy tails). Static headroom gives every block room from step 0. Not tested: h=2 + pressure, h=2 + learnable scale.

**(c) lr depends on h; "0.01 is optimal at C=8" is only half-verified. [verified]** h=2: lr 0.01 2.3151 < lr 0.005 2.3359 (+0.0208 for 0.005; worse in both seeds, 2.6x sd). h=1: lr 0.005 2.3578 < lr 0.01 2.3724 < lr 0.02 2.4019 (the first gap -0.0146 is not detectable; the second +0.0295 is worse in both seeds, 2.6x sd). Edge % at h=1 rises with lr: 30.18 (0.005) / 35.60 (0.01) / 42.25 (0.02). The gain from h=2 shrinks at the lower lr: -0.0219 at lr 0.005 vs -0.0573 at 0.01. So: at h=2 the best lr tried is 0.01, but lr > 0.01 and 0.0075 / 0.015 were not run at h=2, and at h=1 the lr optimum is at or below 0.01. **[hypothesis]** a lower lr moves fewer weights to the outer rung (edge), so the ceiling costs less and headroom gains less.

**(d) The latent grid matches the counter at 8.69 bits and beats it at 6.69 / 7.69 bits. [verified numbers; 2 seeds]**

| bits | grid (h=2) | counter (h=2) | grid - counter | M2 |
|---:|---|---|---:|---|
| 6.69 | `lg6_fp4_h2` 2.4155 (57 of 64 codes) | `fp4c2_h2` 2.5022 (45 of 64) | -0.0867 | yes (12.9x) |
| 7.69 | `lg7_fp4_h2` 2.3513 (127 of 128) | `fp4c4_h2` 2.3835 (105 of 128) | -0.0322 | yes, barely (3.1x) |
| 8.69 | `lg8_fp4_h2` 2.3253 (253 of 256) | `fp4c8_h2` 2.3151 (225 of 256) | +0.0102 | no (counter ahead in both seeds) |

At h=1, `lg8_fp4` 2.3977 is behind `fp4c8_base` 2.3724 by 0.0253 (counter ahead in both seeds, 2.2x sd); the grid gains more from h=2 (-0.0724, per-seed -0.0721 / -0.0728) than the counter (-0.0573). Same-batch cross-width check: `lg7_fp4_h2` (7.69 bits) 2.3513 vs `fp4c8_base` (8.69 bits, h=1) 2.3724 = -0.0213 (per-seed -0.0058 / -0.0365, not detectable): a 7.69-bit grid at h=2 is at least as good as the 8.69-bit counter at h=1. Grid telemetry: edge 41.89 / 27.69 / 18.97 / 16.12 % (lg8 h=1, lg8, lg7, lg6 at h=2), kurtosis 1.50 / 1.79 / 1.96 / 2.22, `grid_points` printed 253/256, 127/128, 57/64.
**[hypothesis]** The counter splits its word into rung x accumulator, so neighbouring rungs share overlapping accumulator ranges (redundant hysteresis, `latent_grid.py` docstring) and, at C=2, only 3 accumulator levels remain; the grid spends the whole word on one latent position and fills 89-99 % of the codes versus 70-88 %. At 8 bits the counter is deep enough that this stops mattering. No ablation at equal code occupancy was run.

**(e) The 8.69-bit counter at h=2 is within +0.017 of the fp32-latent QAT reference. [derived, cross-run reference]** `fp4c8_h2` 2.3151 vs `qat_fp4` 2.2983 (decomp, 4 seeds, lr 3e-3, fp32 latent + AdamW): +0.0168; vs `dense_nobias` +0.0260; `lg7_fp4_h2` +0.0530 / +0.0622 and `lg6_fp4_h2` +0.1172 / +0.1264. Train-side deficit is larger than the val-side one (derived): `fp4c8_h2` train 1.9909 vs `dense_nobias` 1.9208 = +0.0701 against +0.0260 on val, so the val gaps understate the capacity cost on this 1 MB, 24.5-epoch corpus (**hypothesis**, as decomp (i)).

**(f) The reduced ladder {0,.5,1,6} does not profit from h=2. [verified; baseline is cross-run]** `e2m1c5_h2` 2.5661 vs decomp `na_e2m1_c5` nvfp4 (= h=1), same seeds 0-1: 2.5449; +0.0212, +0.0222 / +0.0202 per seed. The baseline comes from the older snapshot, and the same-flags C=8 control drifted by +0.0117 in the same direction between snapshots, so "worse" is at most 2x the drift; "no gain, probably slightly worse" is what the logs support. There is no same-batch h=1 arm for this ladder. Same-batch at 6.69 bits and h=2, the full ladder wins clearly: `fp4c2_h2` 2.5022 vs `e2m1c5_h2` 2.5661 (-0.0639; per-seed -0.0685 / -0.0594; the e2m1 arm has sd 0.0003). At h=1 the two are not distinguishable (`fp4c2` 2.5415 vs decomp 2.5474, cross-run). **[hypothesis]** On {0,.5,1,6} h=2 puts the block max at 3, below the 3.5 midpoint towards rung 6: the large weights of every block start on rung 1, and the ladder has no rung in between to use the extra room.

## Limits (not verified or not run)

- **n=2 seeds for every arm.** M2 (4 seeds) is not met anywhere; n=2 sds are themselves unreliable (`fp4c2` h=1 sd 0.0174 vs `fp4c2_h2` 0.0062; `e2m1c5_h2` 0.0003). Verdicts marked "no" in the paired table (h=4 vs h=2, pressure / learnable scale vs base and vs h=2, the lr effects, C=2 h=2 vs h=1, grid vs counter at 8.69 bits) are "not detectable", even where both seeds agree in sign.
- One model size (d=128), one 1 MB corpus, 24.5 epochs; dense overfits (gap +0.36) so quantisation partly regularises. Nothing here says how any gap moves at d=256-512 or on more data.
- **Cross-run baselines.** Decomp rows (`qat_fp4`, `dense_*`, `tern_counter`, `na_e2m1_c5`, `na_fp4_c4`, `na_fp4_c8`) come from an older snapshot (its `na_fp4_*` arms used a Colab-only `LADDERS["fp4"]` edit). The same-flags C=8 control moved by +0.0117 (both seeds) between snapshots; cause not isolated (code edits, GPU nondeterminism, stochastic rounding). Cross-run differences below ~0.02 are not interpretable.
- **Code is uncommitted** (HEAD `2299ee7`); the exact files are in `claude-ctl\g4e\mx.tar` / `src.tar`, outside the repo. Arm flags are in the job scripts (outside the repo), not in the log headers.
- **Bits are accounting bits** of the persistent packed state, as printed by the harness; the dense / qat references' 96 bits is training width. The latent grid is eager torch only (no fused kernel, no `enable_compile`, not exercised by any speed test), and at run time its index sits in a dense uint8 buffer; the bit-packed layout (`pack_state()`) is what `persistent_bytes()` counts. These logs do not show a bit-packed round trip, speed or peak memory for the grid. No speed or memory claim follows from this report.
- Latent-grid arms keep the block scale frozen and support neither `--scale-lr` nor `--scale-growth`; h is their only scale lever. Pressure growth and the learnable scale were only run on the counter, at h=1.
- Not run: h=3; h=2 x lr {0.0075, 0.015, 0.02}; h=2 + pressure / learnable scale; C=4 at h=1 in this batch; C=9; grid k=4 (plain stateless SR on E2M1), k=5, k>=9; h between 1 and 2 for the grid; the e2m1 ladder at h=1 on the current snapshot; any dense / qat lr sweep (AdamW 3e-3 unswept, LESSONS M8).
- The wall-times in the logs (69-70 s per seed for lg, 283-495 s for the other na arms, 589-940 s for the press / scale-lr arms) come from concurrently launched jobs on one GPU and are not speed measurements.
- `best` (minimum over evals) is in the logs but not used; `val` is the step-6000 eval. It is below the final value for six arms (`e2m1c5_h2`, `fp4c2`, `fp4c2_h2`, `fp4c4_h2`, `fp4c8_slr3e-3`, `lg6_fp4_h2`); the largest gap is `e2m1c5_h2` 2.5489 vs 2.5661.

## Open / next

1. 4 seeds for the frontier arms (`lg6/7/8_fp4_h2`, `fp4c2/4/8_h2`) and a same-snapshot h=1 control for C=4 and for the e2m1 ladder.
2. Resolve h and lr jointly: h in {1.5, 2, 3} x lr {0.0075, 0.01, 0.015, 0.02}; h=2 combined with `--scale-growth pressure` and with `--scale-lr`.
3. Bottom of the grid frontier: k=4 (15 points = stateless SR on E2M1, ~4.7 bits) and k=5 (29 points); top: k=9..10 and counter C=9 / 12 (int16 code word) to see where returns stop.
4. Before any speed or memory claim: a fused / compiled latent-grid path with a fail-closed parity check (LESSONS M5), then `probes/bench_gpu.py`-style timing.
5. d=256-512 and more data (O1 second clause); dense / qat lr sweeps.
