# mx6 latent grid: fewer bits, int4 momentum, momentum speed cost (G4, 2026-11-03)

Commit `12b8672` (int4 momentum buffer). One Colab G4 cell (`colab_cell.py`), its full stdout in `colab_cell_stdout.txt`
(the per-run logs lived on the VM and were not kept; the summary lines below are copied from the stdout).
Hardware: NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB).

## Quality: d512 enwik8 (same setup as `../mx6_g4_momentum_2026_11_01.md`)

4 layers, 3000 steps, batch 128x128, bf16 + compile, cosine, NVFP4 scales, mu 0.9, counter lr 0.03 unless noted.
Held-out val bits/char. All 10 runs ran in parallel on one GPU.

| arm | momentum buffer | B/param (buffer) | seed 0 | seed 1 |
|---|---|---|---|---|
| dense_nobias AdamW 1e-3 | AdamW m+v | 12-16 (master+m+v) | 1.7598 | (1.752 on 11-01) |
| lg6_fp4_h2 | fp8 | 1.006 | (1.769 on 11-01) | 1.7746 |
| **lg6_fp4_h2** | **int4** | **0.562** | **1.7692** | **1.7701** |
| lg5_fp4_h2 | fp8 | 1.006 | 1.7875 | 1.7883 |
| lg5_fp4_h2, lr 0.05 | fp8 | 1.006 | 1.7870 | |
| lg5_fp4_h2 | int4 | 0.562 | 1.7915 | 1.7985 |
| lg4_fp4_h2 | fp8 | 1.006 | 1.8298 | |

Reading:
- **int4 momentum = fp8 within seed noise at 6 bits**: 1.7692 / 1.7701 vs fp8 1.769 / 1.7746 (2 seeds each).
  The buffer drops from 1.0 to 0.56 B/param. lg6 + int4 momentum: gap to this session's AdamW +0.010.
- **5 bits**: +0.028 vs AdamW (1.6 %), same with lr 0.05 -> not an lr problem; int4 momentum adds ~0.007 at 5 bits.
- **4 bits** (15-point E2M1 + headroom 2): +0.070.
- Training state of the block linears (idx k bits + 0.5 bit E4M3 block scale + row RMS + momentum):
  lg6 + int4 ~ 6.5 + 4.5 = 11 bits = 1.38 B/param; lg5 + int4 ~ 1.26 B/param; AdamW 12-16 B/param.

## Speed / memory: 1.21B (d2048, 24 layers), batch 8x512, no grad checkpoint, `train_1b.py --probe-batch`

25 steps per arm (first = compile, excluded), arms run one after another, GPU otherwise idle.

| arm | ms/step | peak GB |
|---|---|---|
| dense (bf16, fused AdamW, compile) | 142 | 27.1 |
| lg6_fp4_h2, no momentum | 109 | 8.2 |
| lg6 + bf16 momentum | 113 | 10.4 |
| lg6 + fp8 momentum | 181 | 9.6 |
| lg6 + int4 momentum | 169 | 9.0 |

- bf16 momentum (one in-place lerp) is almost free (+4 %). fp8 / int4 momentum are **+55..65 %**: unfused per-chunk torch ops
  (dequant, EMA, amax, stochastic rounding, requant) after the fused Triton update. That is the next engineering target:
  fold load-dequant / EMA / SR-requant into `_lg_update_kernel` (it already streams the row of g twice) -> expected ~bf16 cost.
- 24 timed steps is a short window: treat ms/step as +-5 %.
