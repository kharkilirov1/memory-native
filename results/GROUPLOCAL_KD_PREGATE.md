# Row-vs-group stats_scope — KD pre-gate on real donor blocks (CPU)

Date: 2026-07-30. Script: `scripts/grouplocal_kd_pregate.py`. Design doc:
`results/GROUP_LOCAL_UPDATE.md` (the group-local update is the one-pass fused-kernel
enabler; this gate asks whether the numerics change costs recovery quality).

## Protocol

Qwen2.5-0.5B fp32 truncated to blocks 0-1 (+ final norm), teacher = the fp truncation.
Student: body linears PTQ-converted (`counter_packed`, group=128, classic cheap config —
both arms share the identical start; solver fanciness is irrelevant for a relative gate).
The group arm is a buffer-exact clone of the row student (state/scale/perm copied, v zero
in both scopes). KD: 120 steps, batch 1x512 WikiText-2 train stream, IDENTICAL data order
and hash-SR seeds in both arms; fp params frozen (the ONLY difference is the counter
update rule); lr=0.002 const, clip=1.0. Metric: held-out (validation) hidden-state MSE,
4 windows x 512.

## Result — group scope WINS at every checkpoint, gap widening monotonically

| step | row eval MSE | group eval MSE |
|---:|---:|---:|
| 0 (warm) | 12.146 | 12.146 |
| 30 | 10.999 | 10.514 |
| 60 | 10.676 | 9.782 |
| 90 | 10.548 | 8.986 |
| 120 | 10.217 | **8.542** |

Final: row −15.9% from warm, **group −29.7% from warm; group/row = 0.836**.

## Reading

- The fused-kernel enabler is not a quality trade-off at this scale — it is a quality
  WIN. Consistent with the mechanism argument (denominator over 128 weights is
  finer-grained adaptivity than the row, EMA smooths the noisier estimate) and with the
  decimation witness (staggered/finer normalization behaves better than hot uniform).
- Both arms share data, seeds, start, and frozen fp slice, so the difference is the
  optimizer statistic geometry alone. Wall-clock per step differed (CPU contention with a
  concurrent job during the row arm) — irrelevant to the comparison: identical steps/data.

## Caveats / next

- 2 blocks, 120 steps, single-domain stream, constant lr: a pre-gate, not a recovery
  claim. The full-scale gate stays as designed (GPU session): recovery from the s2i2
  start, row vs group, strict alpha=0 curves, mixed corpus, cosine schedule.
- Given this margin, the GPU session should ALSO run the fused-kernel [L3] benchmark arm
  and the quanta-parity gate first (results/GROUP_LOCAL_UPDATE.md), then the KD gate can
  use the fused kernel directly — speed and quality land together.
