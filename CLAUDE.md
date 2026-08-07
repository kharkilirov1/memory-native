# memory-native — project notes

Auto-loaded context. Read this first to resume work cold.

## What this repo is

`memory-native` — finite-state **counter synapses** + **reversible activations** for
memory-efficient training, pure PyTorch (CPU/CUDA, no custom engine), plus an MLX/Metal
port for Apple silicon (`src/memory_native_mlx`).

**What the method saves (the whole point):** it is NOT a weight compressor — it *eliminates
the training-state pools*. For a Gemma-4-E2B-shaped model (35 layers, d=2048, vocab 262K,
seq 2048, batch 8):

| pool | BF16 AdamW | counter |
|---|---:|---:|
| persistent (weights) | 4.28 | 2.24 |
| grad | 4.28 | **0** |
| optim (Adam m/v/master) | **25.68** | **0** |
| activations | 13.12 | **0.47** |
| **total** | **47.4 GiB** | **2.7 GiB** |

The dominant wins: `optim → 0` (no Adam moments, no fp master), `grad → 0`, reversible
activations (~28×). This matters most for finetuning, which is dominated by exactly the
optim + activation pools the method zeroes.

## Key source files

- `src/memory_native/counter.py` — counter layers; carry/saturation via `_carry_resolve`
  (single source of truth — group layers import it).
- `packed.py` — 6-bit storage; `fused_update.py` — row-scale Triton update (hash-SR,
  clip folded into the RMS denominator).
- `group_scale_packed.py` / `group_scale_counter.py` / `group_scale_kernels.py` — the
  group-128 act-ordered packed format: trainable group scales, salient channel (exact fp16
  overrides, frozen base), kernel modes `gemm` (decode+cuBLAS, default via `auto`),
  `triton` (decode-in-GEMM + strict O(out·groups)-scratch update), `torch` (reference).
- `donor/qwen.py` — HF donor loader/swap; `donor/ptq.py` — the calibrated PTQ solver
  (GPTQ-style group ternary v3: act-order, post-sweep s↔t alternation with a monotone
  Hessian gate, `scale_refit="align"` exact joint solve, `grid="itf"` asymmetric grid,
  `salient_first` BiLLM-style split, `salient_scope="layer"` global top-K budget,
  `solve_group_state` = the shared one-layer deploy solve,
  `collect_hessians_guided` = GuidedQuant loss-weighted H via `hessian_weighting`).
- `donor/streaming.py` — block-sequential streaming conversion (peak memory set by block
  width, not model size; MoE-capable; resumable via manifest).
- `donor/asym.py` — GPTAQ-style cascade calibration (`calibration="asym"`): sequential
  two-tower collection of H_q = X_qᵀX_q and G = X_qᵀX_fp, residual-form target
  w̃ = w + (H_q+λI)⁻¹(G−H_q)w solved by the unchanged v3 solver under H_q.
- `recovery/distill.py` + `recovery/runtime.py` — KD recovery finetune, resumable runner
  helpers (strict α=0 evaluation, RNG capture, counter-structure restore).
- `glm.py`, `moe_ffn.py`, `reversible.py`, `budget.py` — receiver architecture, MoE,
  reversible blocks, 4-pool memory model.

## Where the work stands

1. **Conversion + swap + recovery pipeline: DONE.** Qwen2.5-0.5B/1.5B donors convert,
   warm-start, and recover. Mixed multilingual corpus (EN/RU/code/math) is REQUIRED —
   a narrow corpus recovers only its own domain and forgets the rest (measured).
2. **Best trained 1.5B floor so far:** EN 71.9 / RU 35.3 / code 12.8 / math 48.8
   (fp baseline 11.6 / 9.2 / 3.0 / 6.7). The wall ordering is data → LR schedule → data.
   Constant-LR plateau = noise ball; cosine decay to ~1e-4 is the recipe. From a good PTQ
   start, begin the counter schedule LOW (cosine 0.002 → 1e-4) — a hot lr damages it.
3. **PTQ solver chain (no training), 1.5B EN warm PPL:** naive 575k → optimal 187k →
   GPTQ row 17.5k → group-128 5.6k → v3 solver ~0.9k-class. Full-chain layerwise gate:
   −22.3% rel. H-error over v3 base. Rotations (QuaRot-style) are measurably HARMFUL for
   the ternary grid (pinned by a documented-negative test); salient/outlier ISOLATION is
   the correct direction.
4. **PTQ-start + recovery compounds:** EN ~75 after 400 steps (~100× less compute than
   the naive path to the same neighborhood).
5. **Kernels (T4-gated):** gemm-mode forward/grad_x 4–14× over decode-in-GEMM with
   bit-exact parity; slim dense update ~950× over the strict from-IO update at M=4096
   (strict is 30–46 s/layer there — unrunnable); step outlook A100 ~2.5–4 s at B8×T512.
6. **Next planned step:** a short A100 baseline (~5 min) to pick STEPS/batch, then the
   full recovery run from the v3-solver start (~3000 steps fits the remaining budget).
7. **Solver upgrades (2026-07-22, implemented + unit-gated, awaiting A100 warm-PPL
   gates):** `calibration=asym` (GPTAQ-class cascade objective; residual-form target —
   the naive Tikhonov-to-zero form measurably LOST 4.3× network-KL, pinned by a test),
   `salient_scope=layer` (global fp16 budget, same bpw), `hessian_weighting=end_loss`
   (GuidedQuant g=1; collect with [1, seq] batches on CPU — the LM backward graph is
   the memory hog). Gate protocol fix: the 2048-token layerwise gate overfits (base
   train/eval rel-err gap 4–18×; an ICD flip post-pass "won" −20…−49% on train-H and
   LOST +2…+12% on held-out H) — collect a second probe blob with OFFSET and pass
   EVAL_CALIB to the layerwise witness; judge arms on BOTH numbers.
8. **CPU network witnesses (1.5B, blocks 0-1 quantized, KL vs fp on held-out):**
   `salient_scope=layer` carries to held-out (q_proj −20%, k_proj −26% rel H-err;
   attention rows are heterogeneous, MLP neutral) — safe to enable. `asym` at FULL
   strength loses on a SHALLOW cascade (KL 0.392–0.441 vs classic 0.360 @2048tok;
   the true cascade signal there is tiny), `asym_strength=0.5` already beats classic
   on top-1 (77.39 vs 76.90) and CE (3.0985 vs 3.1094) at 8192 tok. Deploy gate on
   A100 = FULL quantization + 524k calibration, arms classic vs asym(c7, s=0.5/1.0);
   chunk=1 is the cleanest objective but costs 2×n_layers calibration passes.
9. **Deploy-scale warm gate DONE (Colab G4, STEPS=0, 524k calib, rebuilt 12M mix —
   results/solver_v3_salient_scope_asym_gate_colab.md):** `salient_scope=layer` WINS
   (aggregate 3.792→3.687; en 77.9→68.0, ru 108→88.7, code 15.0→12.7, math +3%) —
   ENABLE for the next recovery run. asym(s=0.5, c7): ru 108→51.1 (−53%!) but the
   other domains revert/worsen (aggregate 3.746) — cascade correction is real and
   largest where distortion is worst; tune s∈0.25–0.4 / deep-blocks-only next.
   classic arm reproduced the ladder (en 77.9 ≈ historical 74.6 on a fresh val slice)
   — the solve_group_state refactor is non-regressive at deploy scale.
10. **ASYM_STRENGTH sweep DONE — s=0.15 is a STEP CHANGE (same gate):** curve
   0/0.08/0.15/0.20/0.25/0.35/0.5 → metric 3.687/3.202/**3.156**/3.184/3.216/3.320/
   3.746, smooth, unimodal, minimum at 0.15. At s=0.15 EVERY domain beats every other
   config: en 46.75 (classic 77.9, −40%), ru 42.1 (−61%), code 8.28, math 16.85,
   science 35.12, instruct 17.38. Warm EN 46.75 is BELOW the best TRAINED strict
   checkpoint of the previous campaign (47.4 after 6000 KD steps) — solver-only now
   clears the post-recovery bar. Mechanism: tempered target keeps the cascade
   correction inside what the ternary+salient grid can absorb before snapping.
11. **Capacity+iteration sweep DONE (packs 1-2): the s-curve bottleneck IS capacity.**
   `ASYM_PASSES` implemented (multi-pass asym: re-collect on the quantized net,
   re-solve from ORIGINAL weights w0; fp tower + w0 captured once — pinned by test;
   NOTE pass 2 is a bit-exact no-op at chunk=1 on a pure chain, signal comes from
   intra-chunk staleness). Results (layer+asym s=0.15): salient 1/2/3% → metric
   3.156/2.950/2.795 (no saturation; +0.48 bpw per 1%); 2 passes at 2% → **2.899**;
   guided-H standalone 3.687→3.631 (RU-heavy gain, second-order vs asym).
   **NEW DEPLOY DEFAULT (same-format budget): SALIENT_SCOPE=layer CALIBRATION=asym
   ASYM_STRENGTH=0.15 ASYM_PASSES=2 SALIENT_FIRST=0.02** (en 35.6, metric 2.899);
   quality option SALIENT_FIRST=0.03 (en 31.7, metric 2.795, ~3.1-3.6 bpw total).
   Session total vs production solver: 3.792 → 2.795 (~2.7× lower mean PPL), zero
   training. Full sweep tables: results/solver_v3_salient_scope_asym_gate_colab.md.
12. **Recovery run from the s2i2 start DONE (6000 steps, G4):** strict alpha=0 final
   (= best, monotone) **en 30.06, ru 22.39, metric 2.6779** (warm 2.899 → trained
   2.678); vs previous campaign best en 47.4 / ru 65.9. Step ≈ 0.9 s (2× salient
   strict channel; was 0.33 s at 1%) — budget ≈19 units for solve+6k steps.
   Checkpoint NOT persisted (no Drive grant) — rerun is seeded/reproducible.
13. **MoE donor conversion DONE (`efb9879`).** transformers 5.x keeps Mixtral/Qwen3-MoE
   experts as STACKED PARAMETERS (`mlp.experts.gate_up_proj` [E, 2*inter, hidden]), not
   `nn.Linear` — the old target walk found ZERO of them and `ptq_warm_start` reported
   SUCCESS having converted attention only, silently leaving ~87-95% of the weights
   (measured 87% on a synthetic Mixtral). That silent partial conversion now raises.
   Adds stacked + legacy discovery, per-expert Hessians over exactly the routed tokens,
   dead-expert fallback to the data-free solve, rank-deficiency warnings, MoE round-trip
   in `restore_counter_structure`. Router stays fp32. `asym` on MoE WORKS via
   teacher-forced routing (below). NOT supported, with the real reason:
   `hessian_weighting=end_loss` on MoE is NOT a wiring gap — it weights H rows by
   `g_n = mean_o (dL/dy)^2` and so needs dL/dy per target linear, but a stacked-expert
   module builds its gate/up/down intermediates INSIDE the HF forward where a hook
   cannot reach them (recomputing does not help: a recomputed tensor is outside the
   autograd graph). It needs the expert forward re-expressed under our own graph.
   Batched expert solving: MEASURED AND DROPPED. The premise was that ~44 700 python
   solve calls would be dominated by per-call overhead — they are not. On this CPU a V3
   expert costs 107.7 s (gate_up 4096x7168 = 84.1 s, down 7168x2048 = 23.6 s) while the
   per-call floor is 157 ms, i.e. **0.3%**; batching would remove 1.3 h out of 446
   CPU-hours for the full 14 906 experts. The cost is the sweep's linear algebra, so the
   real lever is moving the sweep to GPU (or cutting refine iters), not grouping calls.
   Corollary: DeepSeek-class MoE is not CPU-convertible at all — it needs a GPU.
   **Teacher-forced routing is MEASURED, not speculative** (scratch probe, synthetic
   Mixtral 8 experts / top-2, q tower = fp weights + 15% noise): quantization diverts
   **10.9% of tokens at layer 0 and 25.0% at layer 1** to different experts, so without
   forcing, `G = X_q^T X_fp` for those tokens pairs unrelated activations. The fix works:
   a forward hook that returns the fp tower's `router_indices` (the router returns
   `(logits, scores, indices)`; the expert INPUT depends on indices only — scores apply
   after, so forcing indices alone is sufficient and minimal) makes the expert token
   assignments identical in both towers. IMPLEMENTED: `collect_asym_stats` takes
   `moe_targets_q/moe_targets_fp/routers`, forces the q tower's routers to the fp
   indices and accumulates H_q/G per expert over the aligned streams; expert slices whose
   streams still disagree in shape are SKIPPED rather than paired as noise.
14. **Streaming conversion DONE (`315d5de`, `857411a`) — `donor/streaming.py`.**
   Block-sequential: embeddings → activation buffer, then per block materialize weights
   lazily from the safetensors shards → accumulate H → solve → swap → re-run → write the
   block's state to disk → free. Peak memory is set by the block width, not the model
   size. Measured on the real Qwen2.5-1.5B (peak RSS): in-memory **16.23 GiB** (6.2 GB
   model + 9.85 GiB of H) vs streaming **3.66 GiB** — 4.4×, and flat in depth.
   `cascade=True` (default) calibrates each block on the ALREADY-CONVERTED previous one
   (the asym error-cascade signal, for free); `cascade=False` reproduces in-memory
   semantics bit-exactly and is one pass cheaper. Incremental output + manifest resume;
   finished blocks are REPLAYED on resume (skipping them would feed the next block
   unconverted activations). MoE streams too: transformers stores experts one module per
   expert and stacks them at load time without exposing the mapping, so the reader
   reassembles it — `gate_up_proj[e] = cat([w1, w3])`, `down_proj[e] = w2`, verified
   bit-exact against `from_pretrained`. Unknown layouts raise.
   CPU cost (measured primitives, ~285 GFLOPS on this box): 1.5B classic ≈ 5 h, full
   deploy config ≈ 20 h; 70B needs a GPU (4090 ≈ 3.5 h, A100 ≈ 2 h for the deploy config).
   **End-to-end witness on the REAL Qwen2.5-1.5B** (not synthetic): all 28 blocks
   streamed on CPU, state written per block, then reloaded through the normal
   `restore_counter_structure` — 196 counter linears restored, 0 missing / 0 unexpected
   keys, forward finite, top-5 after "The capital of France is" = the/not/a/,/also
   (solver-only, no recovery finetune). Counter state on disk: **937.1 MiB** of
   6-bit packed `.state`, **1130.6 MiB** with scales + salient + perm.
   CORRECTION: this line used to read "63.0 MiB", measured with a filter matching
   `counter.state`. Only the 84 BIAS-carrying attention linears nest under
   `CounterLinearWithBias` and get that `.counter.` infix; the 112 bias-free ones
   (MLP + o_proj) end in plain `.state` and were silently skipped — 15x too low.
   Match `.state`, never `counter.state`.
15. **gemma-4-12B FULLY CONVERTED on this CPU box (48/48 blocks, 0 errors) —
   results/gemma4_12b_full_conversion.md.** Peak resident **1.86 GiB** vs 22.3 GiB of
   donor weights; output 9.2 GB counter state; ~13 min/block solve, ~45 s/block replay;
   the resume was exercised for real (run died at block 46, replay 0–45 + solve 46–47
   lost nothing). Platform fixes that made it survive, each with the real cause pinned:
   (a) batched-LU `torch.linalg.solve` intermittently corrupts pivots on this MKL
   (SLASWP flood → wrong-pivot RuntimeError / access violation / silently wrong
   scales) — the align matrix is Gram-PSD, so the solve is now Cholesky with per-row
   fallback; (b) safetensors mmap reads intermittently access-violate on Windows
   (even 118 MiB tensors) — the reader now uses plain file I/O + `frombuffer` for
   everything the header describes; (c) gemma-4's last layer per attention type WRITES
   `shared_kv_states` — `_run_block` passes a fresh dict (sound only because
   `num_kv_shared_layers=0`; donors with real KV sharing are refused); (d) the
   computed-buffer probe is 1 layer × 2048 vocab (buffers are vocab/depth-independent,
   verified bit-identical) and embeddings load only the calibration-touched rows
   (gather+scale == module output exactly; dropping `embed_scale` errs by 6.22).
   NOTE: full-attention layers (5, 11, …, 47) have NO `v_proj` — they reuse K as V
   (`value_states = key_states`), so 6 targets there is correct, not a skip.
   Restore witness + PPL/KL gate vs fp: NOT yet run (first restore attempt died
   silently; restored model needs ~11–12 GiB resident).

16. **Group-local update + one-launch fused kernel (CPU-gated, GPU gates pending) —
   results/GROUP_LOCAL_UPDATE.md.** `stats_scope="group"` on `PackedGroupScaleCounterLinear`
   puts EVERY update statistic on the storage geometry (v `[out, n_groups]`; RMS denom and
   clip over the 128-group instead of the row). That removes the FUSION_PLAN-lever-#1
   blocker by construction: the tick of [o,p] depends only on its group's grad slice, so
   `triton_group_counter_update_fused` computes the correlation with `tl.dot` tiles held
   in REGISTERS over all of M and runs the full automaton (stats→v→clip→scale→SR→repack)
   in the epilogue — one launch, no [out,in] grad_w, no fp32 casts, no scratch. CPU
   witnesses green (tests/test_grouplocal_update.py): locality (perturb one gw element →
   only its group changes; row scope provably couples the row), degeneracy (group==K is
   BIT-identical to row math — g² must reduce via `view(...).square().mean(-1)`;
   scatter_add drifts an ULP and that class of drift tips SR), teacher recovery at
   clip=1.0, salient frozen, checkpoint round-trip (row↔group refuse to cross-load —
   different optimizer, on purpose). GPU GATES RUN (2026-07-30, Kaggle T4 —
   results/GPU_GATE_T4_GROUPLOCAL.md): quanta parity ESSENTIALLY EXACT (7e-08…9e-07 code
   mismatch, scale ≤6e-08), CUDA suites green, peak memory the lowest of all update
   paths (152-435 MiB vs dense 244-570). Speed after the x_perm+BLOCK_M=64 pass:
   0.4-0.8× dense (was 0.03-0.2× with in-kernel gather — H3 pinned at kernel scale;
   fp16-vs-bf16 closed NEGATIVE — the residual ~2× is the G-fold go re-read + serial M
   loop, i.e. Stage-2 restructure). Deploy arithmetic already works through decimation
   (grid over 1/4 groups → ~4× → faster than dense at lower memory, plus the KD-pregate
   quality edge); the grid restriction LANDED and MEASURED (v7): fused+dec4 =
   **1.7-3.0× FASTER than dense** at lower peak memory, dec-parity exact vs the masked
   reference. FULL-MODEL KD GATE RUN (0.5B, 24 blocks, T4, 300 steps —
   results/GROUPLOCAL_KD_FULLMODEL_GATE.md): run 1 at a single lr REVERSED the 2-block
   pre-gate (row 134 vs group 189 ppl) — an lr artifact: group's finer denominator
   halves its optimal lr. Run 2 lr-matched: **group@1e-3 ppl 112.4 beats row's best
   134.1 (−16%); dec4@2e-3 ties row's best at 1/4 update FLOPs; dec4 lr×4 is the
   documented hot-lr failure (264.8).** Deploy candidate: group scope + fused + lr at
   HALF the row recipe + decimation=4 as the speed option; final promotion gate = the
   s2i2-start mixed-corpus cosine run at 1.5B with the lr grid re-centered. Row scope
   stays the default until that run.

17. **Group-local follow-through (2026-07-30, all CPU): decimation lever + KD pre-gate WIN
   + restore machinery proven + standardized eval.** (a) `active_groups` on the
   group-local reference: decimation is the full math restricted to a subset (inactive
   groups BIT-untouched, active groups BIT-equal to the full update — pinned by test;
   only well-defined under group scope). Witness (results/DECIMATION_WITNESS.md): dec4 +
   lr×4 beats EVERY full arm on every seed (1.6–5.4× lower final mse) at 0.25× update
   FLOPs — full-lr×4 crosses fast then bounces into the noise ball, dec4 settles ~5×
   lower (staggered noise injection at the same integrated signal). (b) KD pre-gate on
   REAL donor blocks (results/GROUPLOCAL_KD_PREGATE.md): Qwen2.5-0.5B blocks 0-1,
   identical PTQ start, frozen fp slice, 120 KD steps — **group scope wins at every
   checkpoint, final 8.54 vs 10.22 held-out MSE (−29.7% vs −15.9% from warm)**; the
   fused-kernel enabler is a quality WIN at this scale, not a trade-off. (c)
   `scripts/restore_witness.py` (results/RESTORE_WITNESS.md): restore WITHOUT the donor
   ever fully resident (meta skeleton + shard materialization + probe-path computed
   buffers + re-tie; streamed fp reference for PPL/KL; faulthandler+RSS) — proven
   end-to-end on a real streamed 0.5B conversion (24/24 blocks, 0.14 GiB conversion
   peak, 2.89 GiB restore peak, 0 meta left); targets the pending 12B gate of item 15
   (12B projection ~12-13 GiB vs ~44 GiB naive). (d) `scripts/eval_wikitext_ppl.py`:
   GPTQ-protocol WikiText-2 PPL for fp/streamed/ckpt models — the external-comparability
   harness (custom val slices cannot sit next to published GPTQ/AWQ/AQLM tables).
   `ptq_warm_start` now passes `stats_scope` through to packed layers (was silently
   dropped by the counter_kw filter).

18. **PRODUCTION GATE 1.5B DONE (Kaggle 2xT4, results/PROD_GATE_15B.md) — recipe
   SWITCHED to group+dec4.** Three arms, identical v3-layer start (warm 3.2992,
   solver-determinism witnessed at 1.5B), same 12M mixed pilot corpus (bit-exact bins),
   production loop (homotopy, feature-KD, fp AdamW tail, cosine), 1500 steps B2x512:
   row@0.002 → 2.7120; group@0.001 → 2.7262; **dec4@0.002 → 2.7107 (ties row, best
   code/science, at 1/4 update FLOPs)**. Plain group did NOT beat row at the production
   loop (the 0.5B −16% win is regime-sensitive: one lr point, homotopy interaction) —
   per the pre-registered rule row keeps the CLASS defaults; the PRODUCTION RECIPE is
   now `STATS_SCOPE=group DECIMATION=4` + fused kernel + unchanged row lr schedule
   (quality parity + 1.7-3.0x faster update at lower memory). dec4 homotopy phase is
   turbulent (3.30→3.49 by step 600) before the cosine collapse — consider starting
   decimation after the homotopy hold. Also productionized: `SALIENT_REFIT=align`
   (solver post-pass, held-out-gated, off by default). Debug ledger pinned in the gate
   doc: B4 OOMs a 16 GiB T4 at the first KD step (fp32 logits) → B2 +
   expandable_segments; kernel_sources of an ERROR version do not mount; FOUR kw-filter
   sites (2x ptq.py, 2x runtime.py) must pass stats_scope/decimation — all fixed.
   Inference-export arithmetic (bare ternary, no counter residual): body bits ≈ 1.6-2.0
   + 0.125 scales + 0.64 salient@2% → a 284B DeepSeek-V4-Flash-class MoE ≈ 81 GiB
   deploy pack vs 529 GiB bf16; export utility (state→t, drop c, repack) not yet
   written.

19. **gemma-12B cached-KD on 2xT4 (results/GEMMA12B_CACHED_KD.md): infrastructure
   PROVEN, recipe FAILED the gate.** New and witnessed at 12B: 2-GPU model-parallel
   split (layers n/2.. → cuda:1, embed/norm/head+tie on cuda:0, pre-hook movers,
   Triton launches now device-pinned — the multi-GPU fix), reentrant grad-ckpt
   compatible with the eager-only guard (no-grad first pass = plain path; recompute
   builds the Function once), FREEZE_EMBED (AdamW moments for the tied 1.0B embedding
   = 8.5 GiB — a T4 killer), slim best-only ckpt (full 12B state_dict 13-15 GiB blows
   Kaggle's 20 GiB output cap; drop frozen fp + salient/perm/v, stage via /kaggle/tmp).
   WHY 2 GPUs are load-bearing: 12B counter buffers are ~11.7 GiB RESIDENT (state 8.2
   + salient 1.3 + salient perm int64 1.7 + scales/v 0.5) + frozen embed 2.0 → v3/v4
   OOMed in the first forward at ~14.2/14.3 GiB. Cost: 31-34 s/step (no ckpt) / 46-53
   (ckpt), eval ~19 min per 12k tok x 6 domains. QUALITY: **warm R1 solver-only 12B =
   metric 3.2585** (en 46.4/ru 53.7/code 6.8 — same class as the 1.5B prod-gate warm
   3.2992: the conversion chain stands at 12B). Cached-KD training DEGRADED it
   monotonically (250: 5.60, 500: 5.59, 750 strict: 11.59); signature = kd falls to
   ~3-8 while alpha anneals then jumps to ~22-25 at alpha=0 → c absorbed the objective,
   t drifted. The 1.5B dec4-turbulence-then-collapse precedent does NOT transfer at
   this depth/compression. Next arms (need Saturday quota): lr 0.001/0.0005, earlier+
   longer anneal (hold 0.1/end 0.6), dec1 control, 0.5B-at-24-blocks cached-loop
   sanity. Best 12B artifact remains the WARM conversion state
   (`mn-gemma12b-counter-state` v2).

## Gotchas (hard-won, keep)

- Counter layers are **eager-only**: exactly one forward per backward; wrap measurement
  forwards in `torch.no_grad()` or the reuse guard fires.
- `to_empty()` allocates UNINITIALIZED memory. Rotary `inv_freq` is a COMPUTED buffer
  that no checkpoint stores, so materializing a skeleton block-by-block left RoPE running
  on garbage — finite numbers, silently corrupted calibration. Streaming rebuilds rotary
  from config and `_materialize` raises on any tensor the checkpoint lacks. Caught only
  because the streaming tests demand bit-exact agreement with the in-memory path.
- The installed `memory_native` may resolve to a DIFFERENT checkout
  (`Desktop\memory-native`). Run with `PYTHONPATH=<this repo>/src` and check
  `memory_native.__file__` before trusting any result.
- `TEMP` is a Windows system variable — never use it as an env knob name (`GEN_TEMP` in
  `scripts/infer_counter_cpu.py` exists for exactly that reason).
- Packed kinds need `in_features % 4 == 0`; strict Triton group update needs a
  power-of-two group size (guard rejects others before launch).
- Counter layers are bias-free; the swap preserves donor bias via `CounterLinearWithBias`.
- `weight_flips` does NOT increment on fused/Triton paths — use the decoded-sample
  telemetry (`flip_rate_alt`, `observe_flip_sample`) instead.
- Recovering a full-precision donor's outputs is a NETWORK-level effect (composed layers +
  distillation). A single layer does not self-improve from its own optimum — do not write
  that test.
- Evaluate and select checkpoints at strict ternary `alpha=0`; homotopy (`alpha>0`) PPL is
  a diagnostic, not a deployable number.
- Salient channel: base (t, c) is zero and FROZEN at salient entries; checkpoints with
  salient load into fresh layers (buffers resize on load).
- torch.compile of the update chain: `dynamic=False` only (dynamic emits slower code);
  SR stays eager to preserve the RNG stream.

## Setup

```bash
pip install -e .            # torch>=2.1, numpy
pip install -e .[donor]     # + transformers/safetensors/accelerate for donor work
pip install pytest
python -m pytest tests/ -q  # CUDA/Triton tests skip on CPU-only boxes
```

Recovery runs live in `scripts/` (`run_ptq_recovery.py` is the resumable v3 runner;
env-driven: MODEL, DATA_DIR, CKPT_DIR, STEPS, GRID, SALIENT_FIRST, …). Results and
measurement protocols are under `results/` — treat them as the evidence record.
