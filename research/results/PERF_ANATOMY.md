# Performance anatomy — what slows the counter+reversible step down, and why

Honest, mechanism-level breakdown of where wall-clock goes vs a plain dense fp16 model, why each
piece costs time, what we already do about it, and what remains. The unifying fact: the method is
**memory-bandwidth-bound, not compute-bound** — it trades cheap (sub-byte) arithmetic for extra data
movement, so `wall-clock != MACs`.

## Baseline: a dense fp16 step (what we are compared against)

1. **forward:** per layer `Y = X·Wᵀ` — one big GEMM, weights already in GEMM-ready fp16; activations
   are **stored**.
2. **backward:** `grad_x = Δ·W` (GEMM) + `grad_w = Δᵀ·X` (GEMM), reading the stored activations.
3. **optimizer:** `w -= lr·(…)` — one fused kernel (Adam: a few fused ops). Cheap.

Everything is **compute-bound**: time is in three efficient GEMMs, weights are streamed from memory
exactly once. Our method adds four sources of **data movement** on top. In decreasing impact:

---

## 1. The "second forward" — reversible recompute (largest add)

**What we do.** To avoid storing activations (O(1) activation memory), a reversible block
reconstructs its input from its output in backward via the exact inverse, then re-runs the local
forward (with grad) to get gradients:

```
forward:  y1 = x1 + F(x2);   y2 = x2 + G(y1)
inverse:  x2 = y2 − G(y1);   x1 = y1 − F(x2)     ← one extra F and G eval
recompute: z = block(reconstructed_x) with grad   ← another forward
```

**Why it costs.** Per block, backward evaluates F and G **twice** (inverse + recompute) instead of
zero extra. So a step goes from `forward(1) + backward(2)` to `forward(1) + [inverse + recompute(≈1)
+ backward(2)]` — roughly **+33–50% wall-clock**. This is a pure **compute** add (extra F/G GEMMs).

**Mitigation.** Anchors: store an activation every K blocks and recompute *from the anchor* (ordinary
checkpointing, no inverse, exact) → recompute count drops to O(L/A).

**Measured (the spare-memory→speed knob; Blackwell, ReversibleGPT d=512, 8 layers, int8, fused_qkv):**

| anchor_every | tok/s | peak GiB | train loss |
|---|---|---|---|
| 0 (pure reversible) | 36485 | 0.43 | 2.763 |
| **2** | **49190 (+35%)** | 0.54 | 2.760 |
| 4 | 50271 (+38%) | 0.71 | 2.763 |
| 8 (whole-chain checkpoint) | 50811 (+39%) | 1.08 | 2.761 |

The inverse pass is ~1 of the ~2 extra F/G evals; anchors drop it. **anchor_every=2 buys +35% tok/s
for +0.11 GiB** — pure reversible (the minimum-memory extreme) leaves ~a third of throughput on the
table for almost no memory saving. Loss is identical (anchors are exact) → free speed given any
memory headroom. Sweet spot 2–4; anchor=8 adds +1% for 2× the extra memory. (`run_rev` arm.)

**Lever decomposition (anchor=4 fixed, d=512, Blackwell; MIG slice has ~±10-15% run-to-run noise):**

| config | tok/s | note |
|---|---|---|
| fqkv=1 int8=1 | 44766 | all on |
| fqkv=1 **int8=0** | **61506** | int8 OFF → **+37%** |
| fqkv=0 int8=1 | 41801 | fqkv off (~within noise of int8=1) |

**int8 is net-NEGATIVE at d=512** (the quantize epilogue costs more than the GEMM saving on narrow
layers) — it only pays at d≥768 (×2.05 fwd), as the d-sweep in `ACCELERATION.md` first showed. So
the fast config is **d-dependent**: small models → anchors + int8 OFF; large (d≥768 / 1B) → anchors +
int8 ON. fused_qkv is ~neutral at this size (gap < noise). Best d=512 (anchor 2-4 + int8 off) ≈ 60k
vs 36.5k for pure-reversible+int8-on = **+65%**, loss unchanged.

**Two levers that did NOT add speed at d=512 (honest negatives; same arm, baseline noise ~12%):**
- **decimation** (`decimate_updates`): 61292 tok/s vs baseline 55–61k → within noise. It is a
  *late-training* lever — the skip period only grows once a layer's flip-rate falls; over 200 warmup
  steps almost every layer is still active (period=1, no skips). Loss unaffected. Re-check on long
  runs, not a short-run win.
- **decode-prologue** (`counter_triton`, decode fused into the forward GEMM): 50393 tok/s, *below*
  the baseline band → not faster at d=512. The custom triton decode-matmul doesn't beat
  packed-decode + cuBLAS on narrow layers (kernel overhead); may pay at larger d. Loss preserved.

**Consolidated speed recipe (witnessed):** the one dominant lever is **anchors=2-4 (+35%)**; **int8 is
d-dependent** (OFF below ~768, ON above); fused_qkv / decimation / decode-prologue are
neutral-to-marginal at d=512. Best d=512 ≈ 60k vs 36.5k pure-reversible+int8 = **+65%**, loss
unchanged. At 1B (d=2048) the recipe is anchors=2 + int8 ON.

**Residual / note.** Anchors reduce but do not remove it. It is **opt-in**: it is the price of the
activation-memory lever. With enough memory you turn reversible off and there is no second forward.

---

## 2. The counter update — a memory-bound automaton instead of `w -= lr·g`

**What we do per weight** (instead of SGD's single op):

```
decode 6-bit state → (t, c)                              # unpack 4 codes from 3 bytes
g_sq = mean(grad²); v = β·v + (1−β)·g_sq; denom = √v      # RMS row-stat
grad_s = (grad·t)/√fan_in; s_new = s − lr_s·grad_s        # learn the per-row scale
ticks = −lr·grad_eff·(C/s_new)
cc = stochastic_round(c_rebased + ticks)                  # stochastic rounding
carry = trunc(cc/C); remainder = cc − carry·C             # carry / residual
t_new = clamp(t + carry, −1, 1)
encode(t_new, remainder) → 6-bit; pack                    # back to packed state
```

~15 ops.

**Why it costs — two reasons.**
- **More arithmetic** than SGD (one fused op) or Adam (a few).
- **It is memory-bound.** Per weight it reads packed state (1 B), grad_w (4 B), scale, v; writes new
  state, scale, v — ~10+ bytes moved for a handful of flops. In **naïve PyTorch** those 15 ops are
  **15 separate kernel launches**, each reading/writing the whole tensor → **15× the traffic** +
  launch overhead.

**Mitigation.** The fused Triton kernel collapses all 15 into **one launch**: one read (state+grad),
one write. Measured **×17.6 on the update** in isolation, **×1.26 on the full step** (i.e. the update
was a meaningful fraction of the step). Decimation skips the update on near-stable layers.

**Residual.** Even fused it is an extra memory pass over the weights every step; Adam is also fused
but cheaper (no decode/encode/SR logic).

---

## 3. Decode/encode around the GEMM — sub-byte helps memory, not the matmul

**What we do.** The weight lives **packed at 6 bits**. To run the forward GEMM you must **decode** it:
unpack the codes, take the visible ternary `t`, multiply by the per-row scale → a usable weight.

**Why it costs.**
- Extra work + traffic vs a weight already in GEMM dtype.
- **Key point:** decode to **fp16** and the matmul runs at the **same speed as dense** — no win. The
  sub-byte storage saves **memory, not the GEMM**. To speed the GEMM you need **int8 Tensor Cores**
  (the ×2 from int8) — but then every step pays **activation quantization** (and Amdahl bites: the
  step becomes **quant/epilogue-bound**, not GEMM-bound).

**Mitigation.** The derived T-cache (`cache_mode`): decode **once**, keep the visible `t` in a
GEMM-friendly dtype, reuse it across forward AND grad_x, refresh only on flips. int8 forward ×2.05.

**Residual.** Cache maintenance + the int8 quant epilogue around the GEMM — traffic dense doesn't pay.

---

## 4. (Not an add, a clarification) grad_w = Δᵀ·X

To tick the counters we need `grad_w = Δᵀ·X`. **Dense computes this too** (it is the ordinary weight
gradient), so it is **not extra** vs dense. The subtlety: forming it densely materializes an
`[out,in]` gradient (memory); `update_from_io` forms it **in-kernel** (a memory win, not a speed one).

---

## Why it all adds up: memory-bandwidth-bound, not compute-bound

The method **trades compute for data movement**: little arithmetic (sub-byte weights) but a lot of
decode + state + grad + gather (memory-FFN) — **bytes, not flops**. On a GPU the small per-op passes
are **bandwidth-limited**, not ALU-limited. Hence:

- **wall-clock ≠ MACs** — memory-FFN has ~5× fewer MACs yet is *slower* than dense (random-access to
  keys dominates).
- **Amdahl on T4:** the GEMM is **no longer the bottleneck** → lowering GEMM bits (int4) buys almost
  nothing; the cost is **around** the GEMM (quant, epilogue, recompute, decode, gather).
- Measured anchor: `counter_rms` ≈ **0.89× dense** at s512 (≈12% slower) **without** reversible — that
  12% is decode + update overhead; reversible adds the recompute on top.

## What this implies for speed

Because we are bandwidth-bound, speedups come from **removing data movement**, not faster math:
- keep state/grad **in registers** (fused kernels — done, ×17.6 on the update);
- **presaved activation** (don't re-quantize X/Δ for the update);
- **anchors** (less recompute);
- do **not** chase low-bit GEMM (Amdahl says it's the wrong target on T4).

## One-line summary

Three slowdowns, in decreasing order: **(1) the second forward from reversible (+~33%, opt-in)**,
**(2) the memory-bound counter update**, **(3) decode + small ops around the GEMM** — all because the
method is **bandwidth-bound, not compute-bound**: we move data, we don't compute. That is exactly why
raw per-step speed doesn't improve from "fewer bits" — the bottleneck isn't the GEMM.
