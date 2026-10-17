# Fused counter-update kernel — GPU validation

The counter backward's hot path is the per-weight update (RMS row-stats → stochastic-rounding
tick → carry/remainder → re-encode). In stock PyTorch that's ~15 small ops over the full
`[out,in]` matrix; on a 2048×2048 layer it costs **9.45 ms/update** on a T4 — the dominant term
in the backward once the GEMM is cuBLAS.

`memory_native.fused_update.triton_counter_update` collapses the whole thing into **one launch**
(one program per output row: pass 1 reduces the row stats, pass 2 ticks all four packed lanes and
writes the 3 packed bytes). The stochastic rounding is *deterministic* — a hash of
`seed ^ elem_index` (the engine's `cc_hash_u32`), not `torch.rand` — so the update is bit-for-bit
reproducible and the CPU reference `counter_update_hashsr` is exactly what the kernel computes.

## Correctness (T4, vs the deterministic-SR reference)

| shape (C) | mismatching weights | max divergence | scale/v |
|---|---|---|---|
| 24×64 (C=11)    | 0 / 1,536         | —          | match |
| 512×2048 (C=11) | 1 / 1,048,576     | 1 quantum  | match |
| 256×4096 (C=8)  | 110 / 1,048,576   | 1 quantum  | match |
| 2048×2048 (C=11)| 320 / 4,194,304   | 1 quantum  | match |

Not bit-identical, and that's expected and benign: the kernel reduces row stats (`g_sq`,
`grad_s`) in `BLOCK_I` chunks while torch reduces in one pass, so `s_new`/`denom` differ by
~1e-7. That feeds every element's rounding threshold and flips it for an O(1) handful of weights
out of millions — each moved by **exactly one counter quantum (1/C)**, the same unbiased noise
stochastic rounding already injects. < 0.1 % of weights, ≤ 1 quantum each.

## Speed (T4, 2048×2048)

The update in isolation:

```
torch  update   9.452 ms/update
triton update   0.206 ms/update      ->  x45.9
```

End-to-end forward+backward step (B·T = 4096), with the kernel wired into the live backward vs.
forced onto the torch tile path:

```
forced-torch update   32.865 ms/step
fused-kernel update    26.138 ms/step  ->  x1.26
```

The update was ~28 % of the step, so collapsing it 45.9× lifts the whole step 1.26×; the
remaining time is the three cuBLAS GEMMs (forward, grad_x, grad_w), which the kernel doesn't
touch. (Earlier experiments confirmed hand-written Triton matmuls *lose* to cuBLAS, so the GEMMs
stay on torch — see SUMMARY.md.)

## Integration

`PackedRMSCounterLinear._fused_update` plugs the kernel into the live backward. It fires only when
its preconditions hold (CUDA + triton, `use_rms`, `pulse_mode="direct"`, no `local_grad_clip`,
whole-matrix untiled) and otherwise returns `False` so the caller runs the torch tile path —
CPU, tiled, clipped, and ternary-pulse configs are unaffected. A per-call counter (`_sr_step`)
seeds the deterministic SR so successive updates don't share a rounding pattern. GPU-confirmed:
one backward advances `_sr_step` (kernel path taken) and a teacher fits through the wired update.

Validated bit-quantified + benchmarked on Kaggle T4 (`mn-kernel-test`).
