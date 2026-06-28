# M2 — 2:4 group-counter: structured sparsity works; error-feedback rotation unproven

Make the visible weight N:M (2:4) structured-sparse: within each contiguous group of 4 along the
reduction dim, only the 2 with the largest accumulated evidence `|t·C + c|` are nonzero in the
forward weight (forward + grad_x use the masked weight). The masked weights are NOT pruned — the
update ticks **all** counters (error-feedback past the mask), so a masked weight keeps accumulating
and can flip back into the visible top-2. A stored mask + hysteresis bonus keeps the set from
thrashing. The premise (plan M2): error-feedback rotation beats pruning because nothing dies.

## Witness — `scripts/group_counter_witness.py`

Dense Gaussian teacher recovery (n=64, N=256, C=11, 700 steps), target var 0.125.

| arm | MSE | gap% vs unstructured | ever-visible | visible-set churn |
|---|---|---|---|---|
| unstructured counter | 0.20503 | — | — | — |
| 2:4 error-fb hyst=0 | 0.23614 | +15.2 | 100.0% | 1,177,574 |
| 2:4 error-fb hyst=1 | 0.21811 | +6.4 | 100.0% | 213,556 |
| 2:4 error-fb hyst=2 | 0.17267 | −15.8 | 50.0% | 0 |
| 2:4 error-fb hyst=4 | 0.17267 | −15.8 | 50.0% | 0 |
| 2:4 pruning (frozen) | 0.16961 | −17.3 | 50.0% | 0 |

## Verdict

**PASS — 2:4 structured sparsity does not break recovery.** A 2:4-masked visible weight learns the
teacher at least as well as the unstructured counter (best 2:4 MSE 0.170 vs 0.205). The visible
weight can be hardware-sparse (the Ampere/Hopper sparse Tensor-Core pattern) without losing the
counter's capacity on this task. The structural unit tests confirm exactly 2 of every 4 are visible
and grad_x uses the masked weight.

**NOT SHOWN — error-feedback rotation beating pruning.** This is the honest negative. There is a hard
tension between *keeping masked weights alive* and *stability*:
- low hysteresis (0–1): masked weights accumulate fast and constantly displace the visible ones →
  the 2:4 set **thrashes** (churn ~1.2M over 700 steps) → MSE *worse* than pruning;
- high hysteresis (≥2): the set **freezes** (churn 0, only 50% of weights ever visible) → behaviour
  becomes *identical to pruning* (MSE 0.173 vs 0.170).

There is no setting on this task where rotation *helps*. A separate discovery probe — a teacher whose
important weights start **masked** — was inconclusive: error-fb (hyst=1) did recover 96% of the
teacher's support (rotation genuinely found the right weights), but no arm fit that teacher below the
zero-predictor, so it isn't a clean comparison. The reason rotation buys nothing here is structural:
a **stationary** teacher needs no specific 2:4 pattern *discovered* — the initial support is as good
as any. Error-feedback's advantage should appear only when the optimal support is unknown and must be
learned, with a rotation rule that neither thrashes nor freezes. That regime is **not demonstrated on
these CPU teacher tasks**; it needs a real model/task and a smarter evidence/stickiness rule.

## What this means for the stack
- 2:4 is **viable as the forward/grad_x speed lever** (the M-STACK base): it trains fine, so the
  Ampere/Hopper sparse-Tensor-Core path (cuSPARSELt, GPU stage 2) is worth pursuing.
- The "evidence keeps masked weights alive → better than pruning" story is **not** a proven benefit;
  treat 2:4 as structured pruning that *happens to be re-activatable*, default to a stable hysteresis,
  and revisit rotation only with a task that needs support discovery.

## Tests — `tests/test_group_counter.py` (7, pass)
2:4 structural (exactly 2/4 visible), masked-weights-accumulate (error-fb) vs pruning-freezes,
hysteresis-reduces-churn, grad_x-consistent-with-masked-forward, fits-a-target. Default OFF / opt-in.
