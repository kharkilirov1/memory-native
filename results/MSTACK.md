# M-STACK — do the per-step levers compose? Mechanically yes, but at a quality cost

`StackCounterLinear` combines two independently-witnessed per-step levers in one weight: a **2:4
group-counter base** (M2 — visible weight is top-2 of every 4, the forward/grad_x lever) plus a
**low-rank slow-fast residual `A@Bᵀ`** (M3 — the wgrad-frequency lever), with the base frozen
between merges and the residual folded into the 2:4 base every K steps.

## Witness — `scripts/mstack_witness.py`

Tiny GPT, every linear (q,k,v,proj,fc,fc2) swapped between arms; same data/steps.
Config kept small to finish on CPU (the 2:4 base does a Python decode+mask+update per linear per
step): d=96, 2 layers, 400 steps.

| arm | val-loss | gap vs dense |
|---|---|---|
| dense (fp, AdamW) | 1.3046 | — |
| counter (RMSCounterLinear) | 2.0743 | +59% |
| **stack (2:4 + slow-fast)** | **2.7262** | **+109%** |

## Verdict — composes mechanically, but costs quality (honest negative at this scale)

**They compose mechanically:** the stack trains (no divergence/NaN), the 2:4 base stays structurally
intact, merges fire on schedule, and `A,B` are the only fp params — the two levers run together
without breaking. (On an isolated teacher the stacked layer reaches MSE 0.19, on par with a plain
counter — `test_stack_linear`.)

**But the composition costs quality:** in the full GPT the stack lands **+31% above the plain
counter** (2.73 vs 2.07). That is expected and mechanistic — 2:4 makes only half the weights visible
per layer (capacity loss), and the merge folds an fp residual through ternary *and then* a 2:4 mask
(doubly lossy). The two levers' losses add rather than cancel.

**Caveats that bound the negative:**
- Short budget: at 400 steps even the *plain* counter is +59% behind AdamW-dense (counter converges
  slower than fp — same effect seen in M4's counter-dense). So absolute gaps here are convergence-
  dominated, not capacity-dominated; the *relative* stack-vs-counter +31% is the meaningful number.
- The whole point of 2:4 is a **per-step speedup** that is **not measured here** — it needs the
  cuSPARSELt 2:4 sparse-Tensor-Core kernel (M2 stage-2), which is unbuilt. On the dense PyTorch
  fallback the stack is only slower, so there is no speed upside to weigh the quality cost against.

**Conclusion:** the levers do not conflict catastrophically, but they do not compose *for free* —
2:4 + slow-fast is measurably worse than a plain counter at this scale. The speed/quality tradeoff
that would justify the stack **cannot be closed without the sparse-Tensor-Core kernel and a real-
scale run**; until then, the stack is a validated *composition mechanism*, not a recommended path.

## Update — merge consistency fix (review round 2)

The witness table above PREDATES a fix: the merge was folding the UNMASKED base weight while the
forward used the 2:4-MASKED weight, so the residual A,B (trained against the masked forward) were
folded into a fuller base and re-masked — wasting residual capacity. Fixed to fold the masked
`(s·T)·vis + A@Bᵀ`. On a teacher this dropped MSE 0.188 → **0.113** (~40%). The GPT witness (+31% vs
counter) was measured before the fix, so it is now a **pessimistic upper bound**; a re-run on GPU
(queued) will give the corrected gap. The qualitative conclusion (composes, but 2:4 + lossy merge
cost quality vs plain counter; speedup kernel-gated) stands.

## Module / tests
`src/memory_native/stack_linear.py`; `tests/test_stack_linear.py` (5 pass): only-A,B-are-fp,
base-stays-2:4, merge-fires-on-schedule, rank-0-is-plain-group-counter, recovers-a-teacher.
