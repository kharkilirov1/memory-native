# 1.5B end-to-end: v3-full solver -> 6000-step recovery -> benchmark retention

Date: 2026-07-19. Hardware: Colab G4 (Blackwell-class, 95.6 GB VRAM). Code: main @ bb0005a.
Donor: Qwen/Qwen2.5-1.5B (bf16 teacher). Corpus: mix_v2, six domains
(en45/ru20/code15/math10/science5/instruct5), 150M train tokens, teacher-verified
(en 11.6 / ru 9.2 / code 3.0 / math 4.5 / science 5.9 / instruct 5.1).

Recipe: gptq_group solver (itf grid + align refit + salient 1% exact-original channel +
in-sweep refit, refine_iters=2, calib 128xB8xT512), packed group counters (group=128, C=11),
counter cosine 2e-3 -> 1e-4, residual homotopy hold 20% then cosine to 0 at 90%,
KD(T=2) + 0.3 CE + 0.05 feature KD, fp tail AdamW 1e-4 -> 1e-5. 6000 steps at B8xT512,
0.33 s/step (~35 min, ~6-7 compute units total).

## Perplexity (strict ternary alpha=0 unless noted)

| domain | fp teacher | warm (solver only) | final strict | final homotopy diag |
|---|---:|---:|---:|---:|
| en | 11.6 | 74.6 | **47.4** | 46.3 (a=0.006 @5200) |
| ru | 9.2 | 123.5 | 65.9 | 64.1 |
| code | 3.0 | 13.8 | **9.1** | 9.0 |
| math | 4.5 | 24.3 | **17.5** | 17.1 |
| science | 5.9 | 69.8 | **35.7** | 34.x |

- Solver-only warm EN 74.6 matches the best TRAINED floor of the previous campaign
  (71.9, reached via 36k+20k steps): the ladder is naive 575k -> v2 895 -> v3-full 74.6.
- The anneal exam PASSED: strict merged with the homotopy curve (~3% gap at alpha->0);
  the final step is the best checkpoint (strict metric 3.2514, monotone best).
- vs previous best floor: EN 71.9 -> 47.4, code 12.8 -> 9.1, math 48.8 -> 17.5.
  RU regressed (35.3 -> 65.9): mix share cut to 20% + only 6000 steps — the known lever
  for the next run, not a mystery.
- flip_alt stayed 0.0000 for all 6000 steps. RESOLVED post-hoc (incident 2 below): the
  counter self-update path never engaged — this run trained the FP TAIL ONLY. The frozen
  edge=0.0210 (constant to 4 decimals) was correctly read as the signature of a
  non-updating variable, not of a converged one. Consequences: (a) every number in this
  file is a LOWER BOUND for the method — EN 74.6 -> 47.4 was achieved by norms/biases/
  embeddings alone on top of the solver state; (b) the "solver sets the skeleton" and
  "the LR window was right" readings from the first draft of this report are RETRACTED
  as unsupported; the counters/scales channel has not been measured at all yet.

## Benchmark retention (lm-eval, 500 samples/task, acc_norm where defined)

| task | student (strict ternary) | fp teacher | retention |
|---|---:|---:|---:|
| arc_easy | 0.440 | 0.710 | 62.0% |
| arc_challenge | 0.230 | 0.432 | 53.2% |
| hellaswag | 0.428 | 0.600 | 71.3% |
| winogrande | 0.536 | 0.656 | 81.7% |
| piqa | 0.612 | 0.776 | 78.9% |
| **average** | | | **70.8%** |

First true retention number of the project: **~70.8% at ~1.7-2.2 bpw strict ternary on a
1.5B donor, with ~2 bits/weight of TRAINABLE state and 35 minutes of recovery.** The
easy/commonsense tasks hold 71-82%; deep reasoning (arc_challenge) degrades most (53%) —
the expected pattern. Context: 1.5B is close to the worst case for ternary (low
redundancy); the known scale-up levers are a longer run (6k -> 20-30k steps is still
cheap on this hardware), an RU/data rebalance, larger C, and a 7B-class donor.

## Session incident log (kept honest)

**Incident 2 (caught by review of this very report): the counter path never trained.**
The v3 runner lost the `student.train()` call during the consolidation rewrite;
`from_pretrained` returns the model in eval mode and `evaluate_at_alpha` faithfully
RESTORES eval, so all 6000 steps ran with training=False. In eval mode the packed
counter layer takes the plain matmul branch (no autograd.Function), so `_update_from_io`
was never called: counters AND group scales were frozen; only the fp tail learned via
ordinary autograd. Every observation matched and none was investigated hard enough at
the time: flip_alt = 0.0000 (rationalized as sub-quantum physics), edge frozen to 4
decimals, sr_step = 0 in the live diagnostics, rel_a0 = 0.524 unchanged. The reviewer's
argument — "a constant edge is the handwriting of a variable that is not being updated" —
identified it before the code did. Fixes: `student.train()` before the loop, plus an
engagement guard (RuntimeError if max(sr_step) == 0 after the first logging window).
Lesson recorded: a green loss curve is not evidence that the intended path ran.

**Incident 1.** The first launch of this run produced warm PPL ~1.3M: the salient channel stored
s2*sign(w) of the feedback-ADJUSTED block, which act-order tail inflation amplified
catastrophically on real 1.5B layers (tiny/0.5B gates never triggered it). Diagnosed
live (teacher-on-corpus check -> layer stats -> solver ablation), fixed in bb0005a
(salient override = exact original weight), verified on a real layer (full-chain
H-err 0.00125 vs base 0.11803), rerun clean. Weight-relative error proved to be a
misleading gate metric; the H-weighted output error is the one that tracks reality.

Checkpoints (latest + best) and the full eval log: Drive mn_recovery_v3f;
corpus tarball: Drive mn_corpus_v2.tar.gz.
