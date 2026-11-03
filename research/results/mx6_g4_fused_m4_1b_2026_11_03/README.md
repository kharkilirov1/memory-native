# mx6: fused int4 momentum kernel + first 1.21B run with momentum (G4, 2026-11-03)

Commit `4e5cc8d`. One Colab G4 cell (`colab_cell.py`), full stdout in `colab_cell_stdout.txt` (per-run logs stayed on the VM).

## Fused kernel

`lg_update_(..., momentum=(mbuf, mscale, mu, seed2))`: the int4 EMA (dequant, `t = mu*m + (1-mu)*g`, group-32 amax, bf16
scale rounded up, stochastic rounding, nibble pack) runs inside the existing row pass of `_lg_update_kernel`; the RMS
normaliser and the index step see the exact fp32 `t`. `tests/test_lg_fused_m4.py` (vs the unfused kernel fed `t` with the same
seed; buffer unbiasedness; module path never calls the torch `ema_step_`) passed on the G4 (CUDA): 23 passed incl. momentum tests.

## Speed / memory, 1.21B (d2048, 24 layers), batch 8x512, no grad checkpoint, 25 steps (first excluded)

| arm | ms/step | peak GB | source |
|---|---|---|---|
| dense (bf16, fused AdamW, compile) | 142 | 27.1 | this run (same as 11-03 q5) |
| lg6_fp4_h2, no momentum | 110 | 8.2 | this run |
| lg6 + int4 momentum, unfused torch ops | 169 | 9.0 | `../mx6_g4_bits_momentum_2026_11_03/` (commit 12b8672) |
| **lg6 + int4 momentum, fused** | **114** | **8.8** | this run |

The in-session "unfused" A/B is INVALID: the cell set `MX6_FUSE_M4=0` before `cd`, so the variable never reached python and both
rows measured the fused path (114 / 114). The unfused figure is the previous session's (before the kernel existed).

## Quality d512 enwik8 (setup of `../mx6_g4_momentum_2026_11_01.md`, AdamW 1.760 / 1.752)

| arm | val |
|---|---|
| lg6 int4 momentum, **fused**, lr 0.03, seed 0 | 1.7724 (unfused: 1.7692 / 1.7701) |
| lg6 int4, mu 0.95 | 1.8258 |
| lg5 int4, mu 0.95 | 1.8227 |
| lg5 int4, headroom 3 | 1.8022 |
| lg5 int4, headroom 1.5 | 1.7981 |
| lg5 int4, lr 0.02 | 1.8020 |
| (lg5 int4 h2 lr 0.03, 11-03 q5) | (1.7915 / 1.7985) |

5 bits: none of headroom / lr / mu helps; mu 0.95 hurts at both 5 and 6 bits. 5 bits stays ~+0.03 behind AdamW.

## 1.21B TinyStories (byte-level, batch 128x512, 5000-step cosine schedule, stopped at 1000; same config as 11-01)

| step | dense AdamW (11-01) | lg6 no momentum, ctr lr 3e-3 (11-01) | **lg6 + fused int4 momentum, ctr lr 1e-2** |
|---|---|---|---|
| 500 val bpb | 1.376 | 3.275 | **2.070** |
| 1000 val bpb | 0.933 | 2.942 | **1.117** |
| ms/step (batch 128, grad ckpt) | ~1938 | 1924 | 1915 |
| peak GB | 32.2 | 18.8 | 19.5 |

- Momentum cuts the early plateau (train loss 3.24 at step 250, 2.19 at 500, 1.42 at 750) and closes most of the gap at 1000
  (2.942 -> 1.117 vs dense 0.933), but **+0.18 bpb remains** at step 1000. Not closed at 1B yet (closed at d512).
- One run, one counter lr (1e-2 = 3x the no-momentum lr, the d512 rule). Untested: lr 3e-2, longer training (the gap may keep
  shrinking or not), Adam lr for the FP params of the lg arm (1e-4, same as dense).
- Fused momentum costs nothing measurable in step time at batch 128 (1915 vs 1924 ms).
