# Memory-Native Acceleration Deep Iterations v2

Date: 2026-06-27

This pass follows the project protocol: claims are tied to code diffs, existing T4 logs, or an
explicit next witness.  The goal is not only to make the method smaller, but to move it onto the
same hardware path as fast dense training.

## STATE — verified from the repo before this pass

- Test suite baseline: `59 passed, 9 skipped` on CPU with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.
- Current fast path on T4 for a 2048x2048 layer, M=4096:
  - `grad_w = grad_out.T @ x`: 7.336 ms.
  - fused counter transition: 0.419 ms.
  - strict from-IO update: 7385.822 ms, but +0.00 MiB dense-gradient peak.
- Therefore the strict from-IO kernel is a memory witness, not a speed path.
- `cache_mode=int8` already removes a large part of forward decode tax: 10.604 ms -> 7.705 ms;
  honest int8 row-scale forward was 5.176 ms in the T4 log.
- The code already contains `rms_mode={exact,lagged,proxy}`, `scale_rebase={eager,lazy}`, fused QKV,
  reversible anchors, update decimation, and a fused Triton transition kernel.

## Structural diagnosis

The exact counter update is not yet GEMM-epilogue-native because it is row-coupled:

\[
G = \Delta^T X,
\quad
r_o^2 = \frac{1}{K}\sum_i G_{oi}^2,
\quad
\nabla s_o = \frac{1}{\sqrt K}\sum_i G_{oi}T_{oi}.
\]

A standard high-throughput GEMM epilogue naturally sees an accumulator tile, not the whole row.
Therefore an exact-eager update wants two global row reductions before every element can be ticked:

\[
q_{oi}^{k+1}=\Phi(q_{oi}^k, G_{oi}^k, r_o^k, s_o^{k+1}).
\]

The fastest future path is to make the optimizer update epilogue-compatible:

\[
q_{oi}^{k+1}=\Phi(q_{oi}^k, G_{oi}^k, \hat r_o^k, \hat s_o^k),
\]

where \(\hat r_o^k,\hat s_o^k\) are either lagged, proxy, or periodically refreshed row statistics.
Then the GEMM accumulator fragment can be consumed immediately by the finite-state transition.

---

# Iteration 1 — Practical bridge: tiled cuBLAS gradient + fused tile transition

## Idea

Do not choose between:

1. full `grad_w` + fast fused update, and
2. strict from-IO + slow hand-written GEMM.

Use a tile buffer:

\[
G_{B} = \Delta_B^T X,
\qquad B = \{lo,\dots,hi-1\},
\]

where \(G_B\in\mathbb R^{R\times K}\).  The peak gradient buffer is only:

\[
4RK\ \text{bytes}
\]

instead of:

\[
4NK\ \text{bytes}.
\]

For \(N=K=2048\):

| tile rows R | transient grad tile |
|---:|---:|
| 64 | 0.5 MiB |
| 128 | 1.0 MiB |
| 256 | 2.0 MiB |
| 512 | 4.0 MiB |
| 2048 | 16.0 MiB |

This still uses cuBLAS/torch matmul for the correlation, so it should be much closer to the fast
path than strict from-IO.

## Code diff in this pass

`PackedRMSCounterLinear._fused_update(lo, hi, grad_w)` now accepts row slices.  Before this pass the
Triton kernel only fired for `lo == 0 and hi == out_features`, which meant `tile_rows>0` fell back to
the slow torch transition.  Now:

```python
triton_counter_update(self.state[lo:hi], self.scale[lo:hi], self.v[lo:hi], grad_w.contiguous(), ...)
```

and only the touched rows of the derived visible cache are refreshed:

```python
t_new, _ = self._decode_rows(lo, hi)
self._refresh_t_cache(lo, hi, t_new)
```

## Witness added

- `scripts/tiled_update_frontier.py`: GPU benchmark for the Pareto curve `tile_rows -> time/peak`.
- `tests/test_tiled_fused_update.py`: CUDA/Triton validation that the row-slice fused path refreshes
  only touched cache rows. It skips cleanly on CPU.

## Expected high-value command

```bash
PYTHONPATH=src python scripts/tiled_update_frontier.py \
  --d 2048 --M 4096 --tiles 64,128,256,512,1024,2048 --device cuda
```

This is the next executable witness to run on a T4/A10/A100/H100.

---

# Iteration 2 — FORGE-compatible counter update: move transition into GEMM epilogue

## Core theorem/claim

If the preconditioner used by the tick is already known at GEMM epilogue time, the counter update is
an elementwise epilogue over the GEMM accumulator:

\[
G_{oi} = \sum_m \Delta_{mo}X_{mi},
\]

\[
q_{oi}^{+}=T_{\text{counter}}(q_{oi},G_{oi}; d_o,s_o,\xi_{oi}).
\]

This is exactly the class of update that can be fused into the GEMM output path.  The problem is not
that counter training is impossible to fuse; the problem is the exact-eager row statistics.

## Required mode

Use:

```text
rms_mode="lagged"
scale_rebase="lazy"
```

or:

```text
rms_mode="proxy"
scale_rebase="lazy"
```

Then the tick does not require the current step's full-row reduction before each element update.
The row statistics can be refreshed asynchronously or periodically:

\[
v_{k+1}=\beta v_k+(1-\beta)\tilde r_k^2.
\]

The main update consumes the current GEMM fragment and writes the new packed state directly.

## Why this is faster

The current strict from-IO kernel is slow because it implements the GEMM itself.  A CUTLASS/Triton
matmul mainloop should compute \(G\) using Tensor Cores; the finite-state transition belongs in the
epilogue.  This preserves GEMM throughput and removes the dense `grad_w` store.

## Next implementation target

A CUDA/CUTLASS extension, not pure Python:

```text
counter_gemm_update_epilogue(
    A = grad_out^T,
    B = x,
    state6_packed,
    scale,
    v_lagged,
    optional T_cache,
    mode={lagged,proxy},
)
```

The epilogue consumes accumulator fragments and stores packed state, not `G`.

---

# Iteration 3 — Reuse already-saved activation codes for int8 correlation

Current `int8_correlation(delta, x)` re-quantizes both `delta` and `x`, and the T4 log shows it can
lose to fp32 cuBLAS at d=2048:

```text
grad_w fp32  go^T@x       7.336 ms
grad_w int8 correlation   8.210 ms
```

This does not refute int8 update.  It says the quantization staging is wrong.

If `act_save_bits=8`, the forward already stores \(Q(X)\) and a row scale.  The update should reuse
that saved representation rather than quantizing `x` again.  The mathematical contract remains:

\[
\mathbb E[Q(X)\mid X]=X,
\quad
\mathbb E[Q(\Delta)^TQ(X)\mid X,\Delta]=\Delta^TX.
\]

The correct future API is:

```python
SavedCounterActivation.as_int8_update_view(kind="column_or_block_scaled")
```

not a fresh `quantize_int8_cols(x)` call inside the update.

## Constraint

Forward wants row-scale activation quantization for \(XT^T\), while update wants a scale that factors
across \(\Delta^TX\).  Therefore the saved activation format should be block-scaled, not only row-
scaled or column-scaled:

\[
X_{m i} \approx a_{b_m,b_i}\,q_{mi}
\]

with small blocks such as 16x64 or 32x64.  Then both forward and update can reuse the same codes with
acceptable epilogue scale handling.

---

# Iteration 4 — Visible cache should be maintained inside update epilogue

The current safe fix refreshes `T_cache` after the fused kernel by decoding the touched rows.  This
is correct but still pays a decode pass after every update.

The finite-state transition already computes:

\[
t_{old}, t_{new}.
\]

Therefore the update kernel should write both:

```text
state6_packed[row, group]
T_cache[row, col] = t_new   if cache_mode != none
```

inside the same lane where it writes the new code.  This turns cache maintenance into essentially a
free epilogue side-store.  If cache memory is a concern, only write where:

\[
t_{new}\neq t_{old}.
\]

The project already tracks flip-rate; that same statistic can predict whether this branch is cheap.

---

# Iteration 5 — Update decimation should become residual accumulation, not just skip

Current decimation uses:

```text
skip update for r-1 steps, then apply lr *= r
```

This is first-order correct when gradients are slowly varying, but it throws away information from
skipped steps.  A stronger variant uses row-level residual accumulation:

\[
\bar\Delta_o \leftarrow \lambda\bar\Delta_o + \Delta_o,
\qquad
\bar X \leftarrow \mu\bar X + X_{summary},
\]

or simpler:

\[
E_o \leftarrow E_o + \|\Delta_o\|_2^2\mathbb E[X^2]
\]

for row statistics, then fires the full update when the flip-rate clock triggers.  This keeps the
cheap statistics hot while avoiding the expensive \(\Delta^TX\) correlation every step.

A good next witness is a teacher recovery comparing:

```text
no decimation
skip+lr*r
skip+row-stat residual
skip+low-rank gradient sketch
```

---

# Iteration 6 — Use ternary algebra, not int8 algebra, for the visible-weight GEMMs

The visible weight is not arbitrary int8.  It is ternary:

\[
T_{oi}\in\{-1,0,1\}.
\]

Forward and grad_x are therefore signed masked sums:

\[
Y_{mo}=s_o\left(\sum_{i:T_{oi}=+1}X_{mi}-\sum_{i:T_{oi}=-1}X_{mi}\right).
\]

A specialized ternary GEMM can pack signs and masks separately:

```text
P = bitmask(T == +1)
N = bitmask(T == -1)
Y = pop/sum(X where P) - pop/sum(X where N)
```

For fp activations this is a gather-reduction problem; for int8 activations it can become bitmask-
gated dot products.  This is not a quick PyTorch win, but it is the path that inference systems for
ternary LLMs point toward: do not treat ternary weights as generic int8 forever.

---

# Fork log

- Rejected: keep optimizing strict from-IO Triton GEMM. Reason: current T4 log shows ~860x slower
  than cuBLAS+fused; re-enter only if the implementation becomes a real Tensor-Core mainloop.
- Rejected: make 5-bit state the active training default. Reason: previous ablations showed C=11/63
  states is the active-training sweet spot; re-enter for frozen/stable layers only.
- Rejected: always use int8 update correlation. Reason: current staging re-quantizes and loses to
  fp32 cuBLAS; re-enter when activation codes are reused or block-scaled.
- Rejected: always use O(1) reversible mode. Reason: anchors give speed for a small memory spend;
  re-enter O(1) when VRAM is the hard bottleneck.

# Three weakest points / self-audit

1. The row-slice fused update is not GPU-run here; CPU can only compile/skip it.  The executable GPU
   witness is `scripts/tiled_update_frontier.py`.
2. CUTLASS epilogue fusion is a design-level conclusion here, not implemented in this PyTorch-only
   pass.  It needs a CUDA extension.
3. Reusing activation codes for update needs a redesigned saved-activation format; row-scale forward
   and column-scale update have different factorization requirements.

# NEXT — one falsifiable action

Run on the same T4 used for the logs:

```bash
PYTHONPATH=src python scripts/tiled_update_frontier.py \
  --d 2048 --M 4096 --tiles 64,128,256,512,1024,2048 --device cuda \
  | tee results/gpu_tiled_update_frontier_T4.log
```

Success criterion:

```text
At least one tile size R <= 512 gives < 1.5x full-gw+fused time while using <= 4 MiB gradient tile.
```

If confirmed, make `tile_rows=512` or `tile_rows=auto(memory_budget)` the recommended low-peak fast
mode.  If refuted, move directly to CUTLASS epilogue fusion with lagged/lazy RMS.
