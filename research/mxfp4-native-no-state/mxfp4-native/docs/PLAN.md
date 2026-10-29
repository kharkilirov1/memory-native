# Plan: what gets built, in what order, and what would make us stop

Design rule for this project: **no component is built before the number it is supposed to
move has been measured.** Each phase ends with a decision, not a deliverable.

## Phase 0 — study (done)

- [x] Prior art: ECO, MXFP4 pretraining, NVFP4 recipes, BitNet, torchao. → PRIOR_ART.md
- [x] Format spec and accounting. → FORMAT.md
- [x] Update rule and its failure modes. → UPDATE.md
- [x] Kernel survey + probe on the target GPU. → KERNELS.md

Decision gate: none. This phase exists so the rest is not improvised.

## Phase 1 — reference implementation (pure torch) — **done**

Smallest thing that can be wrong:

- `mx6/pack.py` — encode/decode a `[out, in]` matrix to the 25-byte-per-32 layout.
- `mx6/layer.py` — `MX6Linear` with a custom autograd `Function`: forward uses `v` only;
  backward computes `grad_w`, applies the updater in place, returns `grad_x`.
- `mx6/reference.py` — slow, obvious, branch-free implementation of the updater.
- Tests:
  - round-trip `decode(encode(w))` == reference MXFP4 rounding (bit-exact);
  - `persistent_bytes()` == `ceil(N/32)*25 + 4*out_features`;
  - forward is bit-identical when the residual plane is zeroed;
  - scale renormalisation across a power of two is exact;
  - one-forward-per-backward guard fires on reuse.

Exit criteria: all five invariants pass; no CUDA needed. **Met:** 16 tests green
(pytest), including the two bugs the invariants caught (a scale that doubled because
log2 is not exact on powers of two, and a scale that re-derived itself on a zero-gradient
step and rewrote every code in the block).

## Phase 2 — does the update actually train? — **partly done, see PHASE2.md**

Synthetic teacher first (cheap, isolates the quantiser), then a tiny char-LM.

| run | baseline | what it answers |
|---|---|---|
| dense BF16 + AdamW | — | the reference loss |
| MX6, SR only (no feedback) | same tokens | the cost of removing the momentum buffer |
| **MX6, SR + 2-bit feedback** | SR only | is the 2-bit residual doing the work |
| MX6 + unbiased estimator (Agrusa) | SR | cheaper-bias alternative |

Acceptance (must all hold before any kernel work):

1. feedback run beats SR-only by a clear margin on the synthetic teacher;
2. on the char-LM, MX6 is within the NVFP4 literature's typical gap to BF16 at equal tokens;
3. stationary error does not grow as `lr` decays (ECO's `1/lr` check);
4. `kappa` shows no monotone drift over 10k steps.

Kill criterion: if 3 or 4 fail and no variant fixes them, the "no optimizer state" claim is
wrong for this format and the project should fall back to a 2-bit residual **plus** a
quantised momentum (i.e., become ECO-shaped) rather than pretend.

## Phase 3 — kernels and the actual speed number

- Updater as a single Triton launch; compare against the eager chain on peak and time.
- GEMM: benchmark `tl.dot_scaled` (emulated here) vs `scaled_mm` vs cuBLASLt on
  `sm_120`. If none is native, record "memory-only on consumer HW" honestly.
- Peak-memory harness with the same method used for the old project
  (fresh process / `max_memory_allocated`, base captured before model construction).

Exit criteria: a table of {time, peak B/param} for {dense, MX6 fp32-updater, MX6 fused}
at 1536²/B64 and 4096²/B512.

## Phase 4 — scale

Only after 1–3: 30–100M parameter LM, then compare the state-vs-loss Pareto against the
published ECO and NVFP4 numbers. This is the only result worth publishing.

## Explicitly out of scope

converter, salient channel, act-order permutation, PTQ solver, recovery finetune,
LUT-GEMM, int8 path. If any of them turns out to be needed, that is a design failure to
discuss, not a task to add.

## Repository layout (target)

```
mx6/
  pack.py          bit layout, encode/decode, accounting
  layer.py         autograd Function, in-backward update
  reference.py     slow oracle for the updater
  update.py        the maths, framework-free, testable
  kernels/         Triton: updater, matmul wrappers
tests/             invariants first, numerics second
probes/            one-off measurements that justify a decision
docs/              this specification
```
