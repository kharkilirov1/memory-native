# mn-solver v2 — design from the Bonsai investigation (2026-07-16)

## What we actually learned from the primary sources (whitepaper read in full, 24 pp)

1. **Their solver is NOT published.** The whitepaper discloses the FORMAT (ternary/binary
   g128, fp16 scale per 128 weights — exactly what our group-128 probe implemented) and the
   deployment stack, but the "representation transformation" is "built on proprietary Caltech
   intellectual property — a mathematically grounded framework". No algorithm, no calibration
   details, no compute budget.
2. **They never claim "no retraining".** The "они просто перевели и всё" reading is NOT
   supported by the text: the paper says "starts from an off-the-shelf pretrained model and
   moves it into a binary/ternary representation ... preserving its behavior". "Preserving
   behavior" is compatible with (and likely includes) a behavior-recovery stage.
3. **The bibliography contains ZERO PTQ-solver citations** — no GPTQ, QuIP#, AQLM, BiLLM,
   OmniQuant. Only BitNet (as the contrast: "requires pretraining from scratch") and HQQ
   (used for the 4-bit vision tower only). They position against BitNet, not against PTQ.
4. Ternary g128 retention: 94.6% avg over 15 benchmarks (thinking mode) on a 27B.

## Verdict on "is the gap our implementation of their method?"

We did not (and cannot) implement *their method* — nobody outside PrismML can. We implemented
their *format* with the best open solver (GPTQ + group-128). The measured chain on our 1.5B:

naive 575k -> optimal 187k -> GPTQ(row) 17.5k -> GPTQ(g128) 5.6k (EN PPL; fp = 11.6)

Open literature (as of early 2026) supports a strong prior: **no published no-training PTQ
reaches anything like 90-95% retention below 2 bpw at ANY scale.** Best open results:
- BiLLM (~1.08 bpw, PTQ-only): PPL several-x worse than fp even at 70B; far worse at 7B.
- QuIP# / AQLM: usable from ~2 bpw AND both rely on calibration-time optimization
  (fine-tuning scales/codebooks); AQLM's best results include block-wise + end-to-end
  calibration training.
- OneBit (1 bpw): explicitly knowledge distillation — i.e., training.
- OmniQuant W2g128: learnable clipping/transform parameters optimized by gradient descent
  on calibration data — again optimization, only over quantization parameters.

Hence the most probable explanation of Bonsai's numbers: **their transformation includes a
substantial behavior-recovery/QAT-style stage** (they never deny it), plus 27B redundancy.
Our project's recovery-distillation IS the open-world equivalent of that stage — with the
difference that our format stays *trainable* (counters), theirs is inference-only.

## mn-solver v2 — the plan (strongest open ingredients, staged)

### Stage A: PTQ initialization (no training) — upgrades over today's gptq_group_ternary
A1. **act-order for the group solver** (implemented in this pass): process columns in
    descending diag(H) order with groups formed in the PERMUTED order (AutoGPTQ g_idx
    style). Fixes the RU regression our no-order group version showed (12k -> 20k).
A2. **Alternating s<->t refinement** (implemented in this pass): after each group's sweep,
    refit the row scale by least squares against the achieved ternary support
    (s = <w,t>/<t,t>), then re-round once. ARB-LLM-style alternation, 1 cheap iteration.
A3. **Foldable rotations (QuaRot-style)** [next]: random Hadamard on the residual-stream
    dimension folded into adjacent weights (computation-invariant), online Hadamard for
    down_proj inputs. Kills activation/weight outliers before quantization — the single
    biggest known lever for sub-2-bit PTQ. Probe implementation: apply mathematically,
    measure PPL (foldability is established by QuaRot/SpinQuant, so the probe is honest).
A4. **Salient residual binarization (BiLLM-style)** [next]: top Hessian-salient rows get a
    second binary component within an explicit bpw budget; keeps avg bpw honest.

### Stage B: behavior recovery (what their "transformation" most plausibly hides)
Our existing counter-recovery distillation, now started from the Stage-A state
(ptq_warm_start already wired). Combined recipe measured next: does GPTQ-start + recovery
reach a lower floor than the naive-start floor (EN 72 / RU 35 / code 13 / math 49)?

### Calibrated expectations (orderings only — absolute-PPL forecasts retired after 4 misses)
- A1+A2 > current g128 on all domains (must fix RU specifically).
- A3 expected to be the largest single Stage-A jump (literature-consistent), still NOT to
  usable-without-training territory on a 1.5B.
- Stage A+B together is the only path we expect to approach "retention"-style numbers, and
  the fair comparison target at our scale is recovery-from-PTQ-start vs recovery-from-naive.
