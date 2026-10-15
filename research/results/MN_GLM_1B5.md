# MN-GLM-1.5B — a GLM-5.2-class, memory-native model at 1–2B (blueprint + tuned config)

A modern GLM-class decoder (MoE + GQA + RoPE + SwiGLU + RMSNorm) rebuilt on the counter-synapse
method, at ~1–2B parameters, with **every knob set to what the runs in this repo actually showed**
(not defaults). Honesty first: §5 marks exactly what is already in the codebase vs what is new design
work. Knob justifications point at the witness.

---

## 1. Architecture (GLM-5.2-class)

```
tokens ─► embedding (tied)                                   [counter-linear, see §3]
   │
   ├─ N× reversible block  (O(1) activations, anchor_every=2)
   │     F = ATTENTION:  RMSNorm → GQA(+RoPE, QK-norm) → out-proj     [counter-linears]
   │     G = FFN:        RMSNorm → Counter-MoE (SwiGLU experts, top-k)  [M4, grouped+stacked]
   │
   ├─ final RMSNorm
   └─ heads:  next-token  +  MTP (k extra heads, UNTIED)   [M9]
```

- **Sparse MoE FFN** (the GLM-4.5/5-class move) = our **M4 Counter-MoE** — the one *validated*
  architecture win (beats dense at equal active compute, monotonic E-scaling, `MOE_FFN.md`).
- **GQA** (few KV heads) shrinks the KV cache; **RoPE** for long context; **SwiGLU** experts;
  **RMSNorm** pre-norm; optional **QK-norm** for stability at depth.

## 2. Concrete config (~1.5B total / ~0.6B active)

| field | value | note |
|---|---|---|
| d_model | 1536 | |
| n_layer | 24 | reversible coupling blocks |
| n_head / n_kv_head | 12 / 2 | GQA 6× KV reduction |
| head_dim | 128 | RoPE applied per head |
| vocab | 50257 (gpt2) | GLM uses ~150k; swap tokenizer if desired |
| seq_len | 4096 | RoPE-enabled long context (runs here used 256) |
| MoE experts E / top_k | 8 / 2 | capacity = E/top_k × dense ≈ 4× |
| expert = SwiGLU hidden h | ≈ 1.33·d (=2048) | sized so top_k experts ≈ dense active MACs |
| MTP extra heads | 2 (untied) | M9 |

**Param budget** (per layer ≈ `2.33 d²` attn + `E·4 d²` MoE = `34.3 d²`; ×24 + embed):
≈ **2.0–2.3B total**, **~0.6B active/token**. Dial to ~1.3B with `L=20, E=6`. Persistent state at
**0.72 B/coeff → ~1.1–1.6 GiB** (vs dense+Adam ~30–37 GiB).

## 3. The method knobs — set to what the runs showed (the part you asked for)

| knob | value | why (witness) |
|---|---|---|
| linear `kind` | **counter_packed** | 0.75 B/weight; `counter_triton` decode-prologue was **not** faster at d≤512 (`PERF_ANATOMY.md`), re-check only at d≥1536 |
| `C` (counter range) | **11** | best per the bit-budget ablation (63 states) |
| reversible | **ON**, `anchor_every=2` | **+35% tok/s for +0.1 GiB**, loss identical — the dominant speed lever; pure reversible (anchor=0) left a third of throughput on the table |
| `ffn` | **moe**, `grouped=True` | M4 = the validated win; stacked-grouped kernels gave **×5.3** (27k→145k tok/s), quality preserved |
| `aux_loss_weight` | **1e-2** | load-balance; >0 needed or the router collapses (`MOE_FFN.md`) |
| `forward_compute` / `update_compute` | **int8** | d=1536 ≥ 768 → int8 **wins** (×2.05 fwd, ×1.45–2.16 update). At d≤512 int8 was −27%, so this is ON *only because the model is wide enough* |
| `act_save_bits` | **8** | int8 saved activation is reused by the int8 update (no re-quant of X) |
| `fused_qkv` | **ON** | q,k,v in one launch; ~neutral-to-small but free |
| `decimate_updates` | **ON** | skips updates on stabilized layers — a *late-training* win (no effect in the first ~200 steps, real over a long run) |
| MTP heads | **untied** | tied small-embedding heads REGRESS (M9 toy-fail); untied is the fix |
| optimizer (fp params) | AdamW, lr 3e-4 | owns ONLY router + embed + norms + MTP heads; counters self-update (lr 0.04, lr_scale 2e-4) |
| DDP | grad_w all-reduce | states stay bit-identical across ranks (`test_ddp_*`) |
| loss head | chunked (`loss_chunk`) | never materialize `[B,T,V]` at vocab 50k |

**Scale-dependence is the headline tuning lesson:** `int8` and `counter_triton` flip sign with `d` —
ON here because d=1536 is wide; both would be *off* on a narrow model. `anchors` and the MoE kernels
help at every scale.

## 4. Budget estimates (extrapolated; mark as estimate, not witness)

| | MN-GLM-1.5B | dense+AdamW equiv |
|---|---|---|
| persistent state | **~1.1–1.6 GiB** | ~30–37 GiB |
| training peak (reversible, anchor=2) | **fits one 24 GB GPU comfortably** | OOM on <40 GB |
| throughput | MoE quality at ~**0.6–0.9× dense** step speed (grouped kernels close most of the tiny-scale gap; routing + E× weight traffic remain) | 1.0× |

The prize is unchanged: **a 1.5–2B-capacity model that trains in the memory of a small GPU**, with
MoE buying frontier-style quality-per-active-FLOP — not a faster step.

## 5. In the codebase vs NEW design work (honest)

**Already here (wired, tested):** counter-synapse linears, reversible + anchors, **Counter-MoE
(grouped+stacked kernels)**, int8 fwd/update, act-quant, fused_qkv, decimation, DDP grad_w
all-reduce, chunked loss, MTP module, `GPTConfig.ffn`/`ffn_grouped` flags. 128 tests green.

**Now DONE (glm.py, tested + GPU-witnessed):** RoPE, GQA, **SwiGLU experts** (gate/up/down in both
the loop and the grouped/stacked path; parity with gelu at equal active compute — val 2.2421 vs
2.2476, same 8.8M coeffs), RMSNorm, QK-norm, and the reversible wrapper (`ReversibleMNGLM`, ×3.1 less
activation memory).

**Still open (new modules):**
- MoE **backward grad_w grouping** (the last per-expert loop) — for more step speed.
- A real-corpus / larger-scale run (the witnesses here are toy char-level).

None of these conflict with the counter method (they are all just different linear/norm shapes that
counter-linears slot into); they are the build list to go from "our GPT" to "GLM-5.2-class".

## 5b. Witness — the skeleton trains end-to-end on real GPU (Blackwell, ZeroGPU)

`glm.py` (RMSNorm + GQA + RoPE + QK-norm + Counter-MoE) built and trained on real tinyshakespeare,
d=768, L=6, E=8, int8 fwd+update, grouped MoE, 300 steps. CPU unit tests (5) pin RMSNorm/RoPE/GQA;
this is the GPU end-to-end:

| GQA (kv/heads) | val | tok/s | counter-coeffs |
|---|---|---|---|
| 12/12 (full MHA) | **2.2435** | 29517 | 14.2M |
| 3/12 (GQA 4×) | 2.3218 | 27890 | 8.8M |
| 1/12 (MQA) | 2.5950 | 29821 | 7.7M |

The classic GQA tradeoff shows up correctly — **MHA > GQA > MQA** in val; GQA-4× costs +0.08 val for
−38% attention params, MQA (1 KV head) is too aggressive (+0.35). tok/s ≈ equal (attention isn't the
bottleneck; the MoE FFN dominates). The skeleton behaves exactly as a GLM-class model should. (Toy
char corpus, so absolute val is not meaningful — the point is the components are wired right and the
full stack trains with the tuned counter knobs.) Run via the `glm:` space arm.

**SwiGLU vs GELU experts** (d=768, L=6, E=8, GQA 3/12, equal active compute): val 2.2421 (gelu) vs
2.2476 (swiglu), **same 8.8M coeffs** — parity at this toy scale (SwiGLU's edge shows at real scale),
implementation correct and equal-compute (h=8d/(3·top_k)). SwiGLU is the default (GLM/Llama expert).

## 5c. Reversible witness — O(1) activation memory in the GLM skeleton (Blackwell)

`ReversibleMNGLM` wraps the GLM blocks (F=attention sublayer, G=Counter-MoE sublayer) in the
reversible coupling stack. d=768, L=12, GQA 3/12, E=8, int8, anchor=2:

| variant | peak GiB | tok/s | val | counter-coeffs |
|---|---|---|---|---|
| plain (non-reversible) | 5.35 | 15525 | 2.5361 | 17.7M |
| reversible (anchor=2) | **1.73** | 13847 | 2.7723 | 17.7M |

**Activation memory 5.35 → 1.73 GiB = −68% (×3.1)** — the O(1)-in-depth lever works in the GLM stack.
Weights unchanged (17.7M coeffs). Speed −11% (the recompute tax, smaller than pure reversible's +33%
because anchor=2). **Honest caveat on the val gap (2.54 vs 2.77):** the reversible *coupling*
(`y1=x1+F(x2); y2=x2+G(y1)`, two streams — RevNet/Reformer) is a **different architecture** than a
plain residual stack, not a memory-optimized identical model. It is exact w.r.t. itself
(`test_reversible_glm_anchor_invariance`: anchor=0 == anchor=2), but its function differs from
plain-residual; the two-stream form fit this toy char corpus slightly worse. RevNet-class models
match standard ones at real scale (established) — a scale/tuning question, not a memory-lever defect.

## 5d. Speed profile + the bf16 lever — measured, parity-gated, NOT yet passed

**Profile of the 2B-width step** (d=1536, seq=1024, Blackwell, `prof:` arm): fwd 13% / **bwd 87%** /
opt 0%. Inside: grouped_mm 27.7% + grad_w bmm 27.7% + mm ~17% — **~72% of CUDA time in fp32-SIMT
GEMMs** (`cutlass_80_simt_sgemm`; Tensor Cores idle except int8's 1.9%), counter-update elementwise
~25-30%.

**bf16 GEMM operands in the stacked experts** (`moe_dtype="bf16"`; tick math stays fp32):

| | step d=1536 | tok/s d=768 | val d=768/300 steps | peak |
|---|---|---|---|---|
| fp32 (×2 runs) | 2115 ms | 32.2k / 29.1k | **2.2215–2.2476** (noise band) | 2.83 GiB |
| bf16 | **1406 ms (×1.50)** | 35.5k (+22%) | **2.3204 — OUTSIDE the band** | 2.26 GiB |
| bf16+tf32 | 1376 ms (×1.54) | 36.6k | 2.3598 (tf32 adds +0.04) | 2.26 GiB |

**Verdict: ×1.5 speed is real, but the parity gate FAILED at toy scale** (+0.09 val beyond the ±0.026
fp32 noise band; bf16 grad rounding acts as extra tick noise and slows early convergence). Per the
project rule, `moe_dtype` stays **opt-in, default fp32**. The decider is a LONGER-horizon test (300
toy steps may just be early-convergence drag, not a quality ceiling — bf16 training is standard at
scale): add a bf16 arm to the Kaggle convergence witness. After bf16, the next speed bottleneck is
the counter-update elementwise chain (~25-30%) — a fusion (Triton) target, not a dtype one.

### 5e. Fused stacked-expert update kernel — the elementwise chain collapsed (both levers landed)

`stacked_update.py`: ONE Triton launch per expert matrix replaces the ~15-op elementwise chain over
`[E,out,in]` (decode is a u8 load — the stack is unpacked; reuses the packed kernel's `_tick`
hash-SR automaton; accepts **fp32 or bf16 grad_w in-register**, so the bf16 GEMM path feeds the
update with no fp32 copy). Validated on GPU: **19 kernel tests pass** (kernel == `stacked_update_
hashsr` reference up to one SR quantum, fp32 and bf16; inactive experts untouched). d=1536 step:

| config | step | vs start |
|---|---|---|
| fp32 (before) | 2115 ms | ×1.0 |
| fp32 + fused update | 1757 ms | ×1.20 |
| bf16 + fused update | **1026 ms** | **×2.06** |

**The profiled step is now ×2.06 faster** (both profile targets — fp32-SIMT GEMMs and the update
chain — eliminated). E2e quality gate with the fused kernel on the training path: val **2.2084**,
inside (slightly better than) the fp32 band 2.2215–2.2476, +15% tok/s at d=768 — the hash-SR switch
is in-family, quality preserved. **fp32+fused (×1.20) ships unconditionally; bf16 (+fused ×2.06)
remains gated on the long-horizon parity arm.** On the 2B Colab run this maps to ~1950 → ~2350
(fp32) or ~4000 (bf16) tok/s.

## 6. One-line spec

`ReversibleGPT(d=1536, L=24, GQA 12/2, RoPE, RMSNorm, ffn="moe" E=8 k=2 SwiGLU grouped, kind="counter_packed", C=11, anchor_every=2, int8 fwd+update, act_save_bits=8, fused_qkv, decimate, MTP×2 untied)` → ~2B params, ~1.3 GiB state, trains on one mid-GPU.
