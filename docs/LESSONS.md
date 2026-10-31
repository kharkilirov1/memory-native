# Cross-line lessons ledger

Canonical, shared by all three lines. Read before changing training, schedules, formats or benches.
Format follows "Lessons Ledger" (M16) of `C:\Users\Kharki\.agent-prompt-stack\cognitive-protocol-v8.md`:
one lesson per entry, one-line summary on top, update an entry instead of duplicating, delete lessons that turn out wrong.

**Lines.**
- A = main project `memory-native-training` (ternary counter layers; root `CLAUDE.md`).
- B = branch `research/mxfp4-native-no-state`, package `research/mxfp4-native-no-state/mxfp4-native` (mx6).
- C = MotifCL, `C:\Users\Kharki\Desktop\motifcl_main_wt` (ternary 15M pretrain on RX 580).

**Evidence status.** Paths are relative to this repo unless prefixed `MotifCL:` (= `C:\Users\Kharki\Desktop\motifcl_main_wt\`).
`file:N` = line opened and text matched on 2026-10-31 (cite-check only; numbers are as that file reports them, not re-run).
- `verified-2026-10-31` = measured or observed by an agent in the 2026-10-31 session (tool output, logs in
  `research/results/mx6_g4_2026_10_31/`) and/or code and tests opened and matched.
- `witnessed-2026-10-31` = observed in session tool output, no file artifact to cite.
- `claimed in <file>` = reported by that file only; nothing re-run.
- `unverified` = taken from a task note, not checked.
- Quality numbers measured before the held-out fix (E1 report, quality / rescale tables of `mx6_g4_2026_10_31.md`, g4b logs) are memorisation (T14) and are marked SUPERSEDED where cited. Held-out evidence: `research/results/mx6_g4_decomp_2026_10_31.md`, raw logs in `research/results/mx6_g4_decomp_2026_10_31/{heldout,decomposition,memorization_superseded}/`.
- Newest held-out evidence (block-scale headroom, latent grid; 2 seeds per arm, arm flags from the Colab job scripts, not the log headers): `research/results/mx6_g4_headroom_lg_2026_10_31.md`, raw logs in `research/results/mx6_g4_headroom_lg_2026_10_31/`. It was produced by uncommitted code (HEAD `2299ee7` + working tree) and is cross-run against the decomp numbers (older snapshot; the same-flags C=8 control moved by +0.012 between them).

Entry IDs: R = recipe, T = trap, M = measurement rule, O = open question, S = results snapshot.

---

## 1. Recipes that work

### R1. Counter lr must decay (cosine); a constant lr is a noise ball plateau
- **Why it matters:** stochastic-rounding counters keep flipping weights even where the mean gradient is zero, so a constant lr orbits the optimum in a ball of radius ~lr. 79k steps (C) and 17k steps (A) were spent on a flat shelf. lr acts as a temperature.
- **Evidence:** A: `scripts/run_tail.py:85-87` (cosine `lr_for`); `research/results/recovery_15b_main.md:20-23` (7 evals, no trend), `:74` (const 0.008 control reproduces plateau, EN 84-94), `:82` (0.002 breaks it), `:109-128` (cosine 0.008 -> 1e-4 over 20k steps: EN 84.3 -> 71.9). C: `MotifCL: docs/TERNARY15M_PRETRAIN_RESULTS.md:75-84` (plateau 11k-90k, 2.00 -> 1.87; cooldown descent ~3x faster), `:79` ("lr acts as a temperature"). Status: claimed in those files (cite-checked 2026-10-31).
- **Applies to:** A, C; B untested (see T11, O1).
- **Rule for agents:** never run a long counter-layer job at constant lr. Use warmup + cosine to ~10% of peak. If a curve is flat, check the lr schedule before blaming the format or data.

### R2. Known-good schedule values (starting points, not optima)
- **Why it matters:** saves a sweep; these are the values the existing results were produced with.
- **Evidence:** A recovery: counter lr 2e-3 -> 1e-4, fp lr 1e-4 -> 1e-5 (`scripts/run_ptq_recovery.py:113-116`, env defaults). A cosine tail from a plateau: 0.008 -> 1e-4 (`research/results/recovery_15b_main.md:109`). C pretrain from scratch: counter cosine 4e-3 -> 4e-4 (min 10%), warmup 2% = 3200 steps; FP params AdamW cosine 6e-3 -> 6e-4 (`MotifCL: docs/TERNARY15M_PRETRAIN_RESULTS.md:39-40`). Status: claimed in those files (cite-checked 2026-10-31).
- **Applies to:** A, C.
- **Rule for agents:** copy the line's own recipe when staying inside that line. Do not transplant lr values across lines or model scales without a short sweep (M1).

### R3. Warm start: begin the schedule low, skip the hot phase
- **Why it matters:** the lr tuned to resurrect a bad start overheats a good one. From a GPTQ start constant 0.008 reached EN 74.8 at step 400 and then climbed back to 97.7 by step 2000.
- **Evidence:** `research/results/recovery_15b_main.md:224-250` (STAGE B witness; `:246` "noise ball is WIDER than the distance from the GPTQ start to the optimum", `:250` use cosine 0.002 -> 1e-4). Status: claimed in recovery_15b_main.md (cite-checked 2026-10-31).
- **Applies to:** A; plausibly B/C when initialised from a quantised or pretrained start (untested).
- **Rule for agents:** when initialising from PTQ or a pretrained checkpoint, start at the low end (about 2e-3 for A) and decay. Never reuse the from-scratch peak lr.

### R4. Telemetry triad (flip_rate, counter_edge, PPL) separates three different stalls
- **Why it matters:** "lr decayed and PPL stopped moving" has three causes with three different fixes, and naive reading confuses them.
  - PPL falls and flips continue -> constant-lr noise ball (decay lr).
  - PPL flat, flips frozen, saturation low -> lr below one quantum (schedule artefact, not a wall).
  - PPL flat, flips continue, saturation high -> accumulator ceiling.
- **Evidence:** `research/results/recovery_15b_main.md:36-51` (diagnosis and decision table), `:125-128` (measured: flip_rate 0.192 -> 0.004 as lr 0.008 -> 1e-4; counter_edge flat 0.081-0.082, so ceiling refuted at 1.5B). Status: claimed in recovery_15b_main.md (cite-checked 2026-10-31).
- **Applies to:** A; C should log the same triad; B has `track_clip` telemetry (`tests/test_saturation.py:73`) but no flip-rate equivalent was checked.
- **Rule for agents:** every counter-layer run logs {PPL or loss, true flip fraction, counter_edge}. Do not call a plateau a "ceiling" or a "wall" without all three (and see T1 for the flip-rate pitfall).

### R5. Decimation dec4 with lr x4 (group scope)
- **Why it matters:** a quarter of the update FLOPs, and in the CPU witness it settled lower than every full-update arm: full lr x4 crosses fast then bounces into the noise ball; dec4 settles about 5x lower (staggered noise injection at the same integrated signal).
- **Evidence:** `research/notes/DEVELOPMENT_LOG.md:267-273` (dec4 + lr x4 beats every full arm on every seed, 1.6-5.4x lower final mse at 0.25x update FLOPs); `research/results/DECIMATION_WITNESS.md:29-31` (table rows). Scope limit: root `CLAUDE.md:19-20` (the 1.7-3.0x is an isolated update-kernel speedup, not end-to-end training speed). Status: claimed in DEVELOPMENT_LOG.md (CPU, group-local reference).
- **Applies to:** A only.
- **Rule for agents:** quote this as a CPU update-math result at 0.25x update FLOPs, never as a training-speed or end-to-end quality claim. Decimation is only defined under group scope.

### R6. Fused decode-in-GEMM: bf16 tensor cores, scalar where-chain decode, no global gather
- **Why it matters:** the bf16 rewrite is 1.19-4.42x faster than unfused decode+matmul and 2.3-7.6x faster than the old fp32 mode, but still only 0.23-0.50x of a dense bf16 GEMM on an already-decoded weight. It is a memory-footprint tool (weights stay at ~5-6 bits), not a speed win.
- **Evidence:** `research/mxfp4-native-no-state/mxfp4-native/mx6/triton_kernel.py:6-16` (modes), `:30-35` (no per-element global gather; scale converted once per 32-block); `research/results/mx6_g4_2026_10_31.md:58-63` (timings, sm_120). Status: verified-2026-10-31 (G4 logs `research/results/mx6_g4_2026_10_31/s2.log`; forward GEMM only, end-to-end step speed not measured, `mx6_g4_2026_10_31.md:223-224`).
- **Applies to:** B; A/C if they ever write a decode-in-GEMM kernel.
- **Rule for agents:** start from the bf16 mode with an exact-in-bf16 decode (`triton_kernel.py:18-28`). Claim memory savings, not speed vs dense.

### R7. Mechanism tests with null twins instead of single-seed quality tests
- **Why it matters:** a test that asserts a short quality ordering is a lottery (T12). A mechanism test (drift, symmetry, reachability) with a deliberately broken twin proves both that the property holds and that the check can fail.
- **Evidence:** `research/mxfp4-native-no-state/mxfp4-native/tests/test_transformer_block.py:5-6` (every property has a null twin); `tests/test_ladder_drift.py:92` (legacy `gap_mode="min"` must still drift); `tests/test_saturation.py:98` (legacy headroom: outer rung unreachable). Status: verified-2026-10-31 (files opened; G4 run `mx6_g4_2026_10_31/s1.log`: 162 passed, 1 failed, the failure being the since-removed `test_charlm`).
- **Applies to:** B; A/C as a testing pattern.
- **Rule for agents:** for each claimed mechanism, write the test and its negative control in the same file. Do not add tests that depend on one seed beating another at a few hundred steps.

### R8. Block-scale headroom h=2 for nvfp4 counters on the full E2M1 ladder
- **Why it matters:** at the default h=1 the init puts every block max on the outer rung and the trained weights pile onto +-6 (edge 35.6%, kurtosis 1.65). Putting the block max at 3 (`--headroom-mode ladder --headroom 2`) is the largest single gain found for the counter: C=8 2.3724 -> 2.3151 (-0.0573, both seeds, 5.1x sd), C=2 2.5415 -> 2.5022 (-0.0393, both seeds, below the M2 bar), C=4 -0.053 against an older-snapshot run. The counter cost at C=8 against `qat_fp4` (2.2983) falls from +0.074 to +0.017. h=4 (2.3298) is not better than h=2 (not M2-detectable); h=3 was not run. Dynamic scale at h=1 helps less: pressure growth 2.3458, learnable scale 2.3511 (1e-3, +0.19 bits) and 2.3771 (3e-3). The lr optimum moves with h: at h=2 lr 0.01 beats 0.005 (+0.0208), at h=1 lr 0.005 beats 0.01 (-0.0146) and 0.02 is worse (+0.0295). The reduced ladder {0,.5,1,6} does not profit (2.5661 vs 2.5449 at h=1, cross-run, +0.021).
- **Evidence:** `research/results/mx6_g4_headroom_lg_2026_10_31.md` (findings (a)-(c), (f), paired-difference table); logs `research/results/mx6_g4_headroom_lg_2026_10_31/{fp4c8_base,fp4c8_h2,fp4c8_h4,fp4c8_press,fp4c8_slr1e-3,fp4c8_slr3e-3,fp4c8_lr0.005,fp4c8_lr0.02,fp4c8_h2_lr0.005,fp4c2,fp4c2_h2,e2m1c5_h2}.log`; code `research/mxfp4-native-no-state/mxfp4-native/mx6/layer.py` `_nvfp4_target` (target block amax/s = L/h; the harness default is h=1). Status: verified-2026-10-31 for the numbers (2 seeds; only the C=8 headroom effect clears the M2 bar); "room to grow" as the mechanism is a hypothesis (h also halves the absolute grid step near typical weights at init); uncommitted code.
- **Applies to:** B (nvfp4 scale, full E2M1 ladder).
- **Rule for agents:** start nvfp4 counter runs on the full E2M1 ladder at `--headroom-mode ladder --headroom 2` and re-sweep lr after any headroom change (M8). Do not use pressure growth or the learnable scale as a substitute, and do not carry h=2 over to the {0,.5,1,6} ladder.

### R9. Latent grid (one k-bit index, no counter) beats the counter at 7 bits and below
- **Why it matters:** the counter splits the code word into rung x accumulator (neighbouring rungs have overlapping accumulator ranges). The latent grid (`mx6/latent_grid.py`) uses the whole word as one position on a fine E2M1-aligned grid (k=6: 57 points, k=7: 127, k=8: 253), deploys round-to-nearest E2M1 x block scale and updates by stochastic rounding inside backward. At h=2, held-out, 2 seeds: 6.69 bits 2.4155 (counter C=2 2.5022, -0.0867; `tern_counter` 2.4811 at 6.56 bits; mx6 `e2m1_c5` 2.5474), 7.69 bits 2.3513 (counter C=4 2.3835, -0.0322, 3.1x sd), 8.69 bits 2.3253 (counter C=8 2.3151: tie, counter ahead by 0.0102, not M2-detectable). At 6.69 bits it is +0.129 above `dense_bf16`, the first 6-bit-class arm inside the O1 line.
- **Evidence:** `research/results/mx6_g4_headroom_lg_2026_10_31.md` (frontier table, finding (d)); logs `research/results/mx6_g4_headroom_lg_2026_10_31/{lg6_fp4_h2,lg7_fp4_h2,lg8_fp4_h2,lg8_fp4}.log`; `research/mxfp4-native-no-state/mxfp4-native/mx6/latent_grid.py` (docstring: format and exact-expectation update), `probes/lg_ab.py` (arms `lg<k>_fp4[_h<h>]`). Status: verified-2026-10-31 for the numbers; the mechanism (redundant hysteresis; code occupancy 89-99% vs 70-88%) is a hypothesis; eager torch only, no fused kernel, speed and memory not measured; the run-time index is a dense uint8 buffer, so 6.69 / 7.69 bits are accounting bits of the bit-packed layout; uncommitted code.
- **Applies to:** B.
- **Rule for agents:** for a word of 7 bits or less use the latent grid at h=2, and compare any grid width with the counter at the same measured bits and the same h. Do not quote a speed or memory benefit until a fused path passes a fail-closed parity check (M5).

---

## 2. Traps found

### T1. `weight_flips` does not increment on fused/Triton paths; flip-rate telemetry read 0.0000
- **Why it matters:** the dead counter made "flips frozen" look true at every lr including the hot one, and led to a correct prediction being wrongly retracted. Fixed by diffing decoded ternary state across the interval.
- **Evidence:** root `CLAUDE.md:56-57` and `:113`; `research/results/recovery_15b_main.md:92-96` (bug and fix), `:137` ("the retraction was the error"). Status: claimed in those files (cite-checked 2026-10-31).
- **Applies to:** A (any fused/Triton path); C if it logs a flip counter from a fused kernel.
- **Rule for agents:** on fused paths use the decoded-sample telemetry (`flip_rate_alt`, `observe_flip_sample`). Validate any new instrument with a forced state change before trusting a zero.

### T2. Non-uniform ladders: `gap_mode="min"` ticks bias weights outward
- **Why it matters:** the tick is inversely proportional to the local gap; normalising by the minimum gap makes inward moves easy and outward moves hard, so weights pile onto the outer rungs. On G4 `min` put 36-43% of weights on each outer rung; `directional` about 13-22%. For e2m1 it also cut seed sd from 0.34 / 0.22 to 0.02, but it made wide_c3 worse (2.4307 -> 2.5234), so it is not a free win.
- **Evidence:** `research/mxfp4-native-no-state/mxfp4-native/mx6/packed_na.py:98-101` (`gap_mode`, default `directional`), `:412` (dispatch); `tests/test_ladder_drift.py:5-8,61,76,92`; `research/results/mx6_g4_2026_10_31.md:94-105` (histograms), `:119-121` (reading). Status: verified-2026-10-31 (G4 `quality.log`, code and tests opened).
- **Applies to:** B.
- **Rule for agents:** for any ladder with unequal rung spacing, run the drift test and print the rung histogram before comparing arms. Treat a result with >30% mass on the outer rung as a dynamics bug, not a format property.

### T3. Rescale rounding: `bucketize` sends exact midpoint ties to the lower rung (sign asymmetry)
- **Why it matters:** after a power-of-two scale refresh every +-0.5 rung lands exactly on a midpoint, so +0.5 -> 0 while -0.5 stays: a one-sided drain. Fix is `PackedNA.round_idx` (ties away from zero, odd map), used by `refresh_scale_`, `quantize`, `scale_bytes_mse`.
- **Evidence:** `mx6/packed_na.py:174-185` (`round_idx`); `tests/test_rescale_symmetry.py:43`; `research/results/mx6_g4_2026_10_31.md:175-185`. Status: verified-2026-10-31 for the cause and the fix on CPU; G4 `--rescale` runs predate the fix and were not re-run (`research/results/mx6_g4_2026_10_31.md:226-228`).
- **Applies to:** B; any line that rounds on a grid whose midpoints are representable.
- **Rule for agents:** a rounding function used after any rescale must satisfy f(-a) = -f(a); test with exact ties, not random floats.

### T4. `scale_headroom=1.5` is not ladder-aware: the outer rung is unreachable after a refresh
- **Why it matters:** a refresh maps a block max to <= 1.5*s, below the outer-rung threshold (midpoints 1.75 wide, 3.5 e2m1). On G4 the headroom rescale path was harmful (validation about 3.48, the bigram floor). `headroom_mode="ladder"` with h=1.0 keeps the outer rungs; the CPU control showed rung preservation and symmetry only, not a quality gain.
- **Evidence:** `mx6/layer.py:458-462,477,535-541` (`headroom_mode`); `tests/test_saturation.py:92-98` (null twin: legacy gives 0 outer fraction); `research/results/mx6_g4_2026_10_31.md:169-173,231-238`. Status: verified-2026-10-31 (code, test and report opened); GPU re-run with the fix not done.
- **Update 2026-10-31 (headroom sweep, held-out):** on the full E2M1 ladder with nvfp4 scale, h=1 (outer rung reachable) is itself the weak point: h=2 gains -0.057 at C=8 (R8). The tie and headroom fixes are in the code under test; the G4 rescale-refresh experiment (O5) is still not run. Evidence: `research/results/mx6_g4_headroom_lg_2026_10_31.md`. Status: verified-2026-10-31 (2 seeds).
- **Applies to:** B.
- **Rule for agents:** express headroom relative to the ladder maximum. Re-run any rescale experiment only after T3 and T4 are both in the code under test.

### T5. A report's "root cause" can be superseded by the next experiment
- **Why it matters:** the E1 report concluded "block scale frozen at initialisation" explained 80-84% of weights on the outer rung. The G4 histograms (same day, later run) show the same pile-up under `gap_mode="min"` (72-87% of weights on the two outer rungs together, derived) and 27-44% under `directional`, i.e. the min-gap drift (T2) was a large part of it. The E1 report also records that tracking the scale made the arms collapse to zero or the bigram floor (3.44-3.63), and that the MSE scale rule did not change the histogram.
- **Evidence:** `research/results/mx6_e1_quality_2026_10_31.md:41,45` (claim) vs `research/results/mx6_g4_2026_10_31.md:97-104` (histograms, summed over the two outer rungs). Status: verified-2026-10-31 (both files opened); the split between the two causes was not isolated by a controlled run.
- **Update 2026-10-31 (held-out decomposition):** every histogram in this entry comes from memorisation-harness runs (T14). The frozen-scale cause is back as an untested hypothesis for the full-E2M1 ladder (T17); the min-gap drift part stands (T2).
- **Applies to:** B; the habit applies to all lines.
- **Rule for agents:** when citing a root cause, cite the newest report that touches the same histogram. Mark older diagnoses "superseded" in place.

### T6. Fused Triton kernel with fp32-ieee `tl.dot` and a global LUT gather ran at 0.06x of bf16
- **Why it matters:** correct (15/15 parity) but 21.44 ms vs 1.30 ms per step on A100. The bf16 rewrite (R6) fixed the kernel; the dense path still wins on speed because a ~16 MB weight is cheaper to materialise once than to re-decode per output tile.
- **Evidence:** `research/results/mx6_gpu_2026_10_30.md:22` (0.06x row), `:52-60` (parity, cause, verdict); `mx6/triton_kernel.py:15-16,30-35` (legacy mode, gather removed). Status: verified-2026-10-31 for the old numbers (A100 report, raw harness `probes/bench_gpu.py`); the exact "fp32-ieee + gather" attribution is from the session witness and the kernel docstring, not from a separate ablation.
- **Applies to:** B.
- **Rule for agents:** before writing a fused kernel, state the arithmetic-intensity argument; require bf16 tensor cores and no per-element gather; benchmark against the dense decoded GEMM, not against the unfused path alone.

### T7. Commit messages quoted speed numbers the reports later disowned
- **Why it matters:** commit `2afdc3b` says "0.62x -> 0.73x BF16"; the A100 report measured 0.23x (eager NA6) and 0.33x (with `enable_compile`), a different hardware and bench, and states the 0.62x/0.73x figures "must not be quoted". Persistent state (6.26 bits/param vs 96) is the real win; training peak and speed were worse until compile.
- **Evidence:** `git log` (read-only) shows `2afdc3b 2026-10-30 perf: torch.compile integration - 0.62x -> 0.73x BF16`; `research/results/mx6_gpu_2026_10_30.md:14-23` (table), `:34-36` (disclaimer). Status: verified-2026-10-31 (git log and report opened).
- **Applies to:** B; all lines.
- **Rule for agents:** trust reports that ship raw logs over commit messages and over docs written before the last correctness change. Never copy a speed ratio into a new doc without its hardware, bench and date.

### T8. Bits-per-parameter accounting: count all state, and keep units straight
- **Why it matters:** an earlier mx6 draft compared 6.27 bits with 12 bytes and claimed 1.9x; the corrected figure is 6.26 vs 96 bits (15x). In A, 6 bits is the counter and accumulator only, not scales, RMS statistics, salient metadata, caches, the FP tail and its optimizer, activations or transient `grad_w`.
- **Evidence:** `research/results/mx6_gpu_2026_10_30.md:27-31` (unit error corrected); root `CLAUDE.md:12-14`, `:45-46` (measured 0.906 B/param persistent, 1.002 B/param resident on a deployed layer). Status: claimed in those files (cite-checked 2026-10-31).
- **Applies to:** A, B, C.
- **Rule for agents:** report persistent, resident and peak separately, in the same unit, with the pre-hook that asserts measured == format-expected (`probes/e1_quality.py`, per `research/results/mx6_e1_quality_2026_10_31.md:3-4`).

### T9. [SUPERSEDED by T14 and S6] mx6 quality numbers of 2026-10-31 were produced under a constant lr (the known-bad regime)
- **Why it matters:** E1 and the G4 quality runs used counter lr 0.01 and AdamW 3e-3 with no schedule; the quantized-vs-dense gap of ~0.34-0.37 bits/char may be a noise-ball artefact (R1), so it must not be read as a format limit.
- **Evidence:** `research/results/mx6_g4_2026_10_31.md:68-70` (lr 0.01), `research/results/mx6_e1_quality_2026_10_31.md:6-9` (setup, counter lr 0.01); `probes/e1_quality.py:364-366` (`--sched` default `const (legacy)`; a cosine option is being added by a concurrent edit, so line numbers may shift). Status: verified-2026-10-31.
- **Update 2026-10-31 (later run, supersedes the above):** those runs were also memorisation (T14). Re-run with `--sched cosine` on a held-out split: the 6-bit mx6 arm (`na_e2m1_c5`, nvfp4) is still +0.261 behind dense (2.5474 vs 2.2864, 3 seeds), so the gap is not a constant-lr noise-ball artefact; the decomposition (T15, M7) puts most of it in the reduced ladder. Do not quote the 0.34-0.37 figure. Evidence: `research/results/mx6_g4_decomp_2026_10_31.md`, `research/results/mx6_g4_decomp_2026_10_31/heldout/{dense,mx_nvfp4}.log`. Status: verified-2026-10-31.
- **Applies to:** B.
- **Rule for agents:** any mx6 quality comparison must state its schedule. Re-run the E1 matrix with `--sched cosine` before drawing conclusions about ladders, C or the gap (O1).

### T10. `to_empty()` allocates uninitialised memory: computed buffers (RoPE `inv_freq`) silently became garbage
- **Why it matters:** finite numbers, silently corrupted calibration; caught only because streaming tests demanded bit-exact agreement with the in-memory path.
- **Evidence:** root `CLAUDE.md:31-37`. Status: claimed in CLAUDE.md (cite-checked 2026-10-31).
- **Applies to:** A; any line that builds skeleton models.
- **Rule for agents:** rebuild computed buffers from config; raise on any tensor the checkpoint lacks; keep bit-exact parity tests against the in-memory path.

### T11. Counter layers update inside backward: no gradient accumulation, one forward per backward
- **Why it matters:** effective batch cannot be raised by accumulation. C trained at 4096 tokens/step, 16x smaller than the BitNet reference, which plausibly raises its floor (per-update noise about 4x higher converts into spurious flips).
- **Evidence:** root `CLAUDE.md:31-32` (eager-only, one forward per backward); `MotifCL: docs/TERNARY15M_PRETRAIN_RESULTS.md:38,90-96`. Status: claimed in those files (cite-checked 2026-10-31); the "batch is the cause" part is a hypothesis (O3).
- **Applies to:** A, C; B (update happens inside backward, `research/results/mx6_gpu_2026_10_30.md:9-10`).
- **Rule for agents:** wrap measurement forwards in `torch.no_grad()`. To get a larger batch, enlarge the micro-batch or add an accumulate-then-update mode; plain accumulation silently does 16 updates.

### T12. Single-seed short quality tests are lotteries
- **Why it matters:** `test_charlm` asserted "5-bit beats SR-only at 200 steps" (2.5472 vs 2.5324 failed on CPU and G4) and has been removed. The E1 claim "wide_c5 is worse" held at 2 seeds (2.587 vs 2.455) and reversed at 4 seeds (2.4085 vs 2.4203, spread 0.028-0.059). SHOOTOUT.md's own single-seed noise estimate is ~0.04.
- **Evidence:** `research/results/mx6_g4_2026_10_31.md:18-21`; `research/results/mx6_e1_quality_2026_10_31.md:22-25`; `research/results/SHOOTOUT.md:27`; `grep charlm` in `tests/test_packed_na.py` returns nothing (the removal is an uncommitted working-tree change on 2026-10-31, `git diff --stat` shows 63 lines deleted from that file). Status: verified-2026-10-31.
- **Applies to:** A, B, C.
- **Rule for agents:** a 2-seed difference smaller than 3x the seed sd is not a finding. See M2.

### T13. Windows box: venv and temp-directory quirks
- **Why it matters:** wastes sessions on environment errors that look like code bugs.
- **Evidence:** root `CLAUDE.md:42-44,88-92` (the workspace `.venv` is uv-managed, no pip; use `uv pip install --python .venv\Scripts\python.exe ...`). Status: verified-2026-10-31 for no-pip (file). witnessed-2026-10-31 for: TEMP is a system variable, and the venv python cannot write `%LOCALAPPDATA%` (no file artifact).
- **Applies to:** A, B (same `.venv`); C builds with CMake/Ninja (`MotifCL: .hermes.md`).
- **Rule for agents:** use `uv pip`, not `python -m pip`; set output and cache paths explicitly to a writable directory; confirm the imported package path before measuring (`python -c "import memory_native; print(memory_native.__file__)"`, `CLAUDE.md:38`).


### T14. `e1_quality` had no held-out validation: every earlier mx6 quality number is memorisation
- **Why it matters:** "val" was the loss on the training text (200k chars, about 123 epochs at 6000 steps). Dense reached 1.5074 there; on a held-out 10% split of the full corpus (24.5 epochs of 1.0M train chars) the same arm gets 2.2864 with train 1.9259 (gap +0.36). Affected: the E1 report, the quality and rescale tables of `mx6_g4_2026_10_31.md`, the g4b 6000-step runs. Fixed: `--val-frac 0.1` is now the default (contiguous split; the train sampler never reaches val); `--val-frac 0 --chars 200000` reproduces the legacy numbers.
- **Evidence:** `research/mxfp4-native-no-state/mxfp4-native/probes/e1_quality.py:13-16` (docstring), `split_ids` (~`:180`; the file is under concurrent edit, lines may shift); `research/results/mx6_g4_decomp_2026_10_31/memorization_superseded/*.log` (header `vocab=62 tokens=200000`, no train / val tokens) vs `.../heldout/dense.log` (`train_tokens=1003854 val_tokens=111540`); `research/results/mx6_g4_decomp_2026_10_31.md` ("old, memorisation harness"). Status: verified-2026-10-31 (logs and code opened); that the 1500-step G4 logs in `mx6_g4_2026_10_31/` used the legacy harness is inferred (those logs hold summaries only).
- **Applies to:** B (any probe built on `e1_quality`); the habit applies to A and C.
- **Rule for agents:** a quality benchmark prints `train_tokens / val_tokens / epochs` and `gap = val - train` in its header and asserts train and val are disjoint. Never quote a pre-2026-10-31 e1 number without the word "memorisation".

### T15. [PARTLY SUPERSEDED by R9: true for the counter layout only] The 6-bit codeword forced a reduced ladder that discarded FP4's advantage
- **Why it matters:** 64 codes = 7 signed values x 9 accumulator levels (C=5) = 63, so mx6 uses `{0, .5, 1, 6}`; full E2M1 has 15 signed values and would leave at most 4 accumulator levels in 6 bits (derived). With an fp32 latent + AdamW + STE (QAT) on the same harness, full E2M1 costs +0.009 vs `dense_nobias` (2.2983 vs 2.2891), the reduced ladder +0.140 (2.4291), ternary +0.142 (2.4314 at lr 1e-2; +0.157 at 3e-3). So most of the 6-bit mx6 gap (+0.258 vs dense_nobias) is the ladder, not the counter (+0.118). At 8 bits (C=8) full E2M1 + counter reaches 2.3607 (+0.07).
- **Evidence:** `research/results/mx6_g4_decomp_2026_10_31.md` (decomposition table, findings (c), (d)); logs `research/results/mx6_g4_decomp_2026_10_31/decomposition/{dense_nobias,qat_3e-3,qat_1e-2,na_fp4_c8}.log`. Status: verified-2026-10-31 for the numbers (4 seeds for the qat arms, 2 for `na_fp4_c8`); the code-count arithmetic is derived.
- **Update 2026-10-31 (latent grid):** the ladder restriction comes from the counter's code layout, not from the 6-bit word. A latent grid stores full-E2M1 values in 6 bits (57 grid points) and reaches 2.4155 held-out at 6.69 bits, versus 2.5474 for the 6-bit counter on {0,.5,1,6} (-0.13) and 2.4811 for `tern_counter` at 6.56 bits (R9). With headroom 2 the full-ladder counter at C=2 gets 2.5022 (R8). Evidence: `research/results/mx6_g4_headroom_lg_2026_10_31.md`. Status: verified-2026-10-31 (2 seeds).
- **Applies to:** B.
- **Rule for agents:** choose the ladder first and the word width second; measure the format with an fp latent (M7) before tuning the counter. Do not pick a ladder to fit a bit budget without that measurement.

### T16. Code overflow when n_codes exceeds the storage type (silent wrap)
- **Why it matters:** n_codes = n_signed_values x (2C - 1). The harness printed `ceil(log2 n_codes)` bits but the Colab snapshot stored codes as uint8. Full-E2M1 ladder at C=16: 15 x 31 = 465 > 256; the run printed 9.688 bits, the seed-0 histogram has no positive rung (`-6` 57.0%) and val is 3.1171 (vs 2.3607 at C=8). With uint8 the largest C for a 15-value ladder is 9 (derived). The "measured == expected bits" pre-hook did not catch it: it checks ideal bits, not the storage width.
- **Evidence:** `research/results/mx6_g4_decomp_2026_10_31/decomposition/na_fp4_c16.log`; `research/results/mx6_g4_decomp_2026_10_31.md` finding (g); Colab snapshot `mx6/packed_na.py:190,287,404,415` (`.to(torch.uint8)`; snapshot = `claude-ctl\g4d\mx.tar`). The working tree has since gained `PackedNA.code_dtype` (int16 beyond 256 codes) and a `MAX_CODES` ValueError guard (uncommitted, edited by another agent; not run on GPU). Status: verified-2026-10-31 for the log and the code reading; the wrap as the sole cause was not isolated by an ablation.
- **Applies to:** B.
- **Rule for agents:** the constructor must raise or widen the dtype when n_codes > capacity, and a round-trip test must show every rung present. Read the rung histogram of any new ladder / C before reading its loss.

### T17. nvfp4 init puts the largest block at byte 126, so the scale cannot grow without a per-tensor rebase
- **Why it matters:** the E4M3 block-scale byte is clamped to [1, 126] (126 = 448); a block already there cannot grow, and `rebase_tscale` (tensor scale x2, value-preserving) is the only headroom. It fires automatically only under `scale_growth="pressure"` (off by default); in every 2026-10-31 held-out run `grown=0`, i.e. the scale was frozen. A learnable block scale (`--scale-lr`) cannot grow such a block either, and it did not help (+0.040 / +0.063, 2 seeds). Observed signature: `na_fp4_c8` deployed weights are U-shaped (kurtosis 1.66, 35% on the two outer rungs) while `qat_fp4` weights are bell-shaped (kurtosis 3.07). **Hypothesis, not verified:** the frozen / ceilinged scale causes the U-shape and part of the counter cost (+0.062 at C=8). No scale-byte statistic was logged.
- **Evidence:** `research/results/mx6_g4_decomp_2026_10_31.md` findings (e), (f); Colab snapshot `mx6/packed_na.py:297,334,344`, `mx6/layer.py:656-659`; logs `.../decomposition/{na_fp4_c8,qat_3e-3,na_slr_3e-3,na_slr_1e-2}.log`. Status: weight shape verified; cause = hypothesis; "init puts the max block at byte 126" is from the session note and the code comment, not from a logged statistic.
- **Update 2026-10-31 (headroom sweep):** the scale experiments were run. Static headroom h=2 at C=8: edge 35.6 -> 24.1 %, kurtosis 1.65 -> 2.01, val 2.3724 -> 2.3151; pressure growth (about 428k block bumps, 27-28 per-tensor rebases): edge 19.2 %, val 2.3458; learnable scale with the rebase fix (0.7-1.9 M byte moves): heavy tails (kurtosis 3.8 / 6.7), val 2.3511 / 2.3771. The ceiling hypothesis is now supported (a scale with room gives fewer outer-rung weights and a lower loss; counter cost at C=8 against `qat_fp4` +0.074 -> +0.017 at h=2) but "room to grow" is not separated from "finer grid step at init". See R8. Evidence: `research/results/mx6_g4_headroom_lg_2026_10_31.md`. Status: verified-2026-10-31 (2 seeds).
- **Applies to:** B.
- **Rule for agents:** log the block-byte histogram and the share of blocks at byte 126; run `na_fp4_c8` with `--scale-growth pressure` before drawing any conclusion about counter cost.


---

## 3. Measurement rules

### M1. Same harness, hardware, seeds and schedule for every arm, dense included; never compare numbers across lines or benches
- **Why it matters:** the same counter-vs-AdamW question gives opposite signs in two A benches (SHOOTOUT: counter 2.5161 vs AdamW 2.5672 at 400 steps, single seed, T4; README width sweep at d=512: counter 2.6390 vs AdamW 2.6184, 200 steps). C's reference ran on an L40S for ~50 min; C ran 48 h on an RX 580. Dense bf16 is 2.0394 (T4, 2 seeds) in the E1 report and 2.0332 (G4, 4 seeds) in the G4 report (both memorisation numbers, T14; held-out dense on G4 is 2.2864, S6).
- **Evidence:** `research/results/SHOOTOUT.md:8-14,25-29`; `research/results/README.md:19`; `MotifCL: docs/TERNARY15M_PRETRAIN_RESULTS.md:9-12`; `research/results/mx6_e1_quality_2026_10_31.md:6,18`; `research/results/mx6_g4_2026_10_31.md:69-70`. Status: claimed in those files (cite-checked 2026-10-31).
- **Applies to:** A, B, C.
- **Rule for agents:** a comparison table has one harness, one GPU, one data slice, one schedule. A number from another line or bench is context, labelled with its source, never a row in the table.

### M2. At least 4 seeds on both sides; report mean, sd and the per-seed list
- **Why it matters:** see T12. E1 flipped a conclusion between 2 and 4 seeds; the mx6 e2m1 arm had one seed at 3.13 among three at ~2.5.
- **Evidence:** `research/results/mx6_e1_quality_2026_10_31.md:22-25,28`; `research/results/mx6_g4_2026_10_31.md:81-92` (per-seed lists). Status: verified-2026-10-31.
- **Applies to:** A, B, C.
- **Rule for agents:** if the effect is smaller than 3x the larger sd, write "no detectable difference". Do not tune a hyperparameter (C, lr, headroom) on such evidence.

### M3. Label inherited numbers and keep slices apart
- **Why it matters:** Qwen 1.5B v3f2 is EN 34.41 / RU 30.34, while EN 47.4 / RU 65.9 were v3f FP-tail-only; s2i2 EN 30.06 / RU 22.39 has no persisted checkpoint and used different corpus slices, so it cannot be ranked against v3f2 or the PTQ sweep. The 1.21B report lacks its primary raw log.
- **Evidence:** root `CLAUDE.md:15-18,24-27`. Status: claimed in CLAUDE.md (cite-checked 2026-10-31).
- **Applies to:** A; the rule applies to all lines.
- **Rule for agents:** every inherited number carries (source file, run id, slice, schedule). Without same-slice evaluation, do not rank.

### M4. Evaluate and select checkpoints at strict alpha = 0
- **Why it matters:** homotopy (alpha > 0) PPL is not the deployed model's PPL.
- **Evidence:** root `CLAUDE.md:61`. Status: claimed in CLAUDE.md (cite-checked 2026-10-31).
- **Applies to:** A (homotopy runs); C evaluates hard ternary (`MotifCL: docs/TERNARY15M_PRETRAIN_RESULTS.md:21-22`).
- **Rule for agents:** report the strict-state metric as the headline; any homotopy number is labelled as such.

### M5. Fail-closed parity before timing
- **Why it matters:** a fast kernel that is wrong is worse than a slow one; the G4 bench refuses to print a time until every mode matches the reference.
- **Evidence:** `research/mxfp4-native-no-state/mxfp4-native/probes/bench_fused_kernel.py:2,40-52,121`; `research/results/mx6_g4_2026_10_31.md:28`. Status: verified-2026-10-31.
- **Applies to:** B; any benchmark of a rewritten kernel.
- **Rule for agents:** put an assertion on correctness in front of every timing loop; exit non-zero on mismatch.

### M6. Raw log or it did not happen; mark derived numbers; freeze forecasts before the run and score them after
- **Why it matters:** the G4 report copies every number from `mx6_g4_2026_10_31/*.log` and marks ratios as "derived". The A recovery log froze forecasts before each run and scored them, finding a stable optimism bias on lr-decay magnitude (predicted ~30-40% off the plateau, got ~19%; stepped tail predicted 15-30%, got 12%).
- **Evidence:** `research/results/mx6_g4_2026_10_31.md:3-5`; `research/results/recovery_15b_main.md:27-30,53,131-134`. Status: verified-2026-10-31 for the G4 logs (files present: s1, s2, quality, rescale); claimed in recovery_15b_main.md for the forecast scores.
- **Applies to:** A, B, C.
- **Rule for agents:** write the expected number down before launching. When forecasting the gain from lr decay on a counter format, discount your first estimate.

### M7. Decompose gaps with fp-latent (QAT / STE) references and a dense arm that uses the same update rule
- **Why it matters:** "dense bf16 + AdamW vs counter arm" confounds the FORMAT (ladder), the OPTIMIZER (momentum, decay, bias) and the COUNTER (accumulator, stochastic rounding, frozen scale). One-factor arms separate them: `dense_nobias` (bias), `dense_rms` (the RMS-normalised update on fp32 weights), `qat_*` (fp32 latent + AdamW, quantised forward), then the counter arm. 2026-10-31, val bits/char vs `dense_nobias` 2.2891: bias -0.003 (noise); RMS update -0.019 (no cost); format +0.009 (full E2M1) / +0.140 (`{0,.5,1,6}`) / +0.142 (ternary); counter +0.062 (fp4 C=8) / +0.118 (e2m1 C=5) / +0.050 (ternary).
- **Evidence:** `research/results/mx6_g4_decomp_2026_10_31.md` (decomposition table); arms defined in `probes/e1_quality.py` docstring ("Gap-decomposition arms"). Status: verified-2026-10-31. Caveat: the AdamW reference ran at one lr (0.003), so "RMS -0.019" reads as "no cost", not "RMS is better" (M8).
- **Applies to:** A, B, C.
- **Rule for agents:** a claim "format X is N behind dense" shows the chain dense_nobias -> same-update-rule dense -> qat at the same ladder -> counter, 4 seeds per step, and names the step that carries the gap.

### M8. Sweep the lr per arm; never inherit a default (tern optimum was 3e-2 here)
- **Why it matters:** `tern_counter` at its `cli.py` default 3e-3 gives 2.6479 (sd 0.0495); the optimum on this harness is 3e-2, 2.4811 (-0.167), with 1e-2 = 2.5479, 1e-1 = 2.5584, 3e-1 = 2.8176. `qat_fp4` goes from 2.2983 (lr 3e-3) to 2.4450 with sd 0.1344 (1e-2); `dense_rms` at 3e-2 has one seed at 2.4849. Not swept: AdamW 3e-3 for the dense and qat references, and mx6 lr 0.01 (only 0.02 / 0.03 tried, one seed, worse).
- **Evidence:** `research/results/mx6_g4_decomp_2026_10_31/heldout/tern_*.log`, `.../decomposition/{qat_3e-3,qat_1e-2,dense_rms_*}.log`; report finding (h) and Limits. Status: verified-2026-10-31.
- **Update 2026-10-31 (headroom sweep):** the lr optimum depends on other knobs. `na_fp4_c8`: at h=1 lr 0.005 2.3578 < 0.01 2.3724 < 0.02 2.4019; at h=2 lr 0.01 2.3151 < 0.005 2.3359 (lr above 0.01 not run at h=2). Re-sweep lr after any change of init, headroom or scale rule. Evidence: `research/results/mx6_g4_headroom_lg_2026_10_31.md` finding (c). Status: verified-2026-10-31 (2 seeds; most differences are under the M2 bar).
- **Applies to:** A, B, C.
- **Rule for agents:** at least three lrs on a half-decade grid around the default for every arm, same schedule, at least 2 seeds; report the whole sweep and mark the optimum. Never compare a tuned arm with an untuned default without saying so.


---

## 4. Open questions and kill criteria

Kill criteria marked "task note" were set by the project owner's session on 2026-10-31. Criteria marked "proposed" are suggestions for the owner to confirm; they are not agreed.

### O1. Does a cosine schedule close the mx6 gap to dense?
- **Why it matters:** [SUPERSEDED 2026-10-31: the figures below are memorisation, T14; see the Update] all mx6 quality data is at constant lr (T9). Current gap: best quantized arms 2.3689-2.4020 bits/char vs dense 2.0332, about 0.34-0.37.
- **Evidence:** `research/results/mx6_g4_2026_10_31.md:123-125`; `probes/e1_quality.py:364-368` (`--sched cosine` being added). Status: partly tested 2026-10-31, see the Update.
- **Update 2026-10-31 (held-out, cosine, d=128, 6000 steps):** the gap persists. `na_e2m1_c5` nvfp4 (6.69 bits) 2.5474 vs dense_bf16 2.2864 = +0.261 (3 seeds); `tern_counter` (lr 3e-2, 6.56 bits) +0.195; `na_fp4_c8` (8.69 bits, 2 seeds) +0.074, below the "dense + 0.15" line. Not tested: whether the gap shrinks at d=256-512. By S6 / T15 most of the 6-bit gap is the ladder. Evidence: `research/results/mx6_g4_decomp_2026_10_31.md`, `.../heldout/`, `.../decomposition/`. Status: verified-2026-10-31.
- **Update 2026-10-31 (headroom + latent grid, held-out, d=128, 2 seeds):** the frontier moved. Against `dense_bf16` 2.2864: 8.69 bits `fp4c8_h2` 2.3151 (+0.029), 7.69 bits `lg7_fp4_h2` 2.3513 (+0.065), 6.69 bits `lg6_fp4_h2` 2.4155 (+0.129). The best 6.6-6.7-bit arms of the previous update were +0.195 (`tern_counter`) and +0.261 (`na_e2m1_c5`). The first clause of the kill criterion (best arm above dense + 0.15) therefore no longer holds at 6.69 bits for the latent grid; the second clause (gap not shrinking at d=256-512) is untested, and the grid has no fused kernel yet. Evidence: `research/results/mx6_g4_headroom_lg_2026_10_31.md` (frontier table). Status: verified-2026-10-31 (2 seeds, uncommitted code).
- **Applies to:** B.
- **Rule for agents / kill criterion (task note):** re-run with cosine plus the scale fixes (T3, T4). Kill mx6 if the best arm stays above dense + 0.15 bits/char at d=128 and the gap is not shrinking at d=256-512.

### O2. How many counter bits are needed for parity? (C sweep; accumulator ceiling at other scales)
- **Why it matters:** A refuted an accumulator ceiling at 1.5B (counter_edge flat 0.081-0.082), whereas C names the bounded uint8 accumulator (C=11, ~6 bits) as the most plausible reason its floor sits at 1.8018 vs the reference's 1.6074. mx6 saw no C effect between 3 and 5 at constant lr (T12), which says little under the noise ball.
- **Evidence:** `research/results/recovery_15b_main.md:128,39`; `MotifCL: docs/TERNARY15M_PRETRAIN_RESULTS.md:97-102`; `research/results/mx6_e1_quality_2026_10_31.md:22-25`. Status: not tested under a decaying schedule (superseded in part, see the Update).
- **Update 2026-10-31:** under cosine on a held-out split, on the full-E2M1 ladder the counter cost halves from C=4 (+0.138 vs `qat_fp4`) to C=8 (+0.062), 2 seeds; C=16 broke on a storage overflow (T16). For the 6-bit e2m1 word C is capped at 5 by the width. Still untested: 4 seeds, C = 9, and C under a moving scale (T17). Evidence: `research/results/mx6_g4_decomp_2026_10_31/decomposition/na_fp4_c{4,8,16}.log`. Status: verified-2026-10-31.
- **Update 2026-10-31 (headroom 2, 2 seeds):** with `--headroom 2` the fp4 counter gives C=2 2.5022 (6.69 bits), C=4 2.3835 (7.69), C=8 2.3151 (8.69): -0.1187 and -0.0684 per added bit, so accumulator depth is still a lever up to C=8 (far above the seed sd). At 7 bits and below the latent grid (R9) sidesteps the question: 6.69 bits 2.4155, 7.69 bits 2.3513. Still untested: C=9, 4 seeds, C=4 at h=1 on the same snapshot. Evidence: `research/results/mx6_g4_headroom_lg_2026_10_31.md`. Status: verified-2026-10-31.
- **Applies to:** A, B, C.
- **Rule for agents / kill criterion (proposed):** run the C sweep only under cosine, 4 seeds, with the R4 triad. If higher C does not move the final loss by more than the seed sd, drop accumulator depth as a lever and stop tuning it.

### O3. Larger effective batch without gradient accumulation
- **Why it matters:** updates happen in backward (T11); C is at 4096 tokens/step vs 65 536 for the reference. Candidates listed by C: chunked vocabulary loss with BATCH=32, tied head, an accumulate-then-update mode in `CounterStateLinear`.
- **Evidence:** `MotifCL: docs/TERNARY15M_PRETRAIN_RESULTS.md:90-96,136-145`. Status: not tested.
- **Applies to:** A, C.
- **Rule for agents / kill criterion (proposed):** if a 4x batch does not lower the plateau by more than seed spread, batch noise is not the cause; stop investing there and move to O2.

### O4. Fallback: ECO-style momentum error feedback
- **Why it matters:** an alternative if the schedule and scale fixes do not close the gap in B.
- **Evidence:** arXiv 2601.22101. Status: unverified (reference taken from a task note; the paper was not opened).
- **Applies to:** B (fallback).
- **Rule for agents / kill criterion (proposed):** open and read the paper before using it; only start after O1 fails its criterion.

### O5. Rescale (periodic scale refresh) on GPU after the tie and headroom fixes
- **Why it matters:** the G4 rescale runs predate T3 and T4; whether refresh helps quality at all is unknown. CPU control showed symmetry and rung preservation only.
- **Evidence:** `research/results/mx6_g4_2026_10_31.md:226-228,236-238`. Status: not run.
- **Applies to:** B.
- **Rule for agents / kill criterion (proposed):** 4 seeds, `headroom_mode="ladder"`, tie fix, cosine. If refresh does not beat no-refresh by more than the seed sd, drop it.

### O6. End-to-end mx6 training-step speed on sm_120, and the fused update kernel
- **Why it matters:** only the forward GEMM was timed on G4; on A100 the update math is 53% of a step and the next lever of size.
- **Evidence:** `research/results/mx6_g4_2026_10_31.md:223-224`; `research/results/mx6_gpu_2026_10_30.md:63-73`. Status: not measured.
- **Applies to:** B.
- **Rule for agents:** do not quote any end-to-end speed ratio for mx6 on G4 until it is measured with `probes/bench_gpu.py` and a fail-closed parity check.

### O7. Does the latent grid hold up with 4 seeds, at d=256-512, and in a fused kernel?
- **Why it matters:** R9 rests on 2 seeds at d=128 (the 7.69-bit gain over the counter clears the M2 bar only barely, 3.1x), in eager torch; there is no fused or compiled path and no speed or memory measurement.
- **Evidence:** `research/results/mx6_g4_headroom_lg_2026_10_31.md` (Limits); `mx6/latent_grid.py` (eager only; the bit-packed `pack_state()` is not used at run time). Status: not tested.
- **Applies to:** B.
- **Rule for agents / kill criterion (proposed):** 4 seeds of `lg7_fp4_h2` vs `fp4c4_h2` (and `lg6_fp4_h2` vs `fp4c2_h2`); drop the grid at 7 bits or less if the gap is under 3x the larger sd. Before any speed or memory claim: a fused decode-in-GEMM or compiled update with a fail-closed parity check (M5), timed against the dense decoded GEMM as in R6.

### O8. Headroom x lr x scale dynamics, and the width range of grid and counter
- **Why it matters:** only h = 1, 2, 4 were tried; the lr optimum moves with h (M8); pressure growth and the learnable scale were run only at h=1; grid widths k=4, 5, 9 or more, C=9, and grid h between 1 and 2 are untested.
- **Evidence:** `research/results/mx6_g4_headroom_lg_2026_10_31.md` (Limits, Open / next). Status: not tested.
- **Applies to:** B.
- **Rule for agents / kill criterion (proposed):** h in {1.5, 2, 3} x lr {0.0075, 0.01, 0.015, 0.02} at C=8 and at k=7, 4 seeds, cosine; add h=2 + pressure and h=2 + learnable scale. Drop a lever if it does not beat plain h=2 by more than the larger sd.

---

## 5. Results snapshot (claimed vs verified)

Context only. Do not rank across rows (M1).

| ID | Line | Result | Status |
|---|---|---|---|
| S1 | A | Counter vs AdamW, short single-seed runs: SHOOTOUT counter 2.5161 vs AdamW 2.5672 (T4, s512, 400 steps, memory peak 0.99 vs 1.19 GiB); width sweep d=512 counter 2.6390 vs AdamW 2.6184, d=768 2.93 vs 3.08. The file itself calls the quality spread noise-level. | claimed in `research/results/SHOOTOUT.md:8-14,25-29`, `research/results/README.md:19` |
| S2 | A | Qwen2.5-1.5B 6-bit counter recovery: plateau EN ~87 / RU ~46 / code ~19 / math ~65 at constant lr 0.008; cosine tail best EN 71.9 / RU 35.3 / code 12.8 / math 48.8 (fp teacher 11.6 / 9.2 / 3.0 / 6.7). Different recipe from the v3f2 numbers in M3. | claimed in `research/results/recovery_15b_main.md:20-23,120` |
| S3 | C | Ternary 15M from scratch on one RX 580: best val loss 1.8018 (ppl 6.06) at step 155 000, final 1.8209; 160 000 steps, 655 M tokens, 48 h. Reference (BitNet-style STE, L40S): 1.6074 hard ternary, 1.5970 latent. Batch is 16x smaller than the reference; there is no same-hardware Adam baseline. | claimed in `MotifCL: docs/TERNARY15M_PRETRAIN_RESULTS.md:3,18-24,90-93` |
| S4 | B | **SUPERSEDED by S6 (memorisation, val == training data, T14).** mx6 on G4, d=128, 1500 steps, 4 seeds, constant lr 0.01: dense bf16 2.0332 +/- 0.0073; best quantized 2.3689 (wide_c5, `min`, high variance), 2.3943 (wide_c5, `directional`), 2.4020 (e2m1_c5, `directional`). | verified-2026-10-31 (logs `research/results/mx6_g4_2026_10_31/quality.log`, `rescale.log`; report `mx6_g4_2026_10_31.md:69-70,123-125`) |
| S5 | B | mx6 persistent state 6.26 bits/param vs 96 for bf16 + AdamW (15x); training peak 168 MB with `enable_compile()` vs 178 MB dense; speed 0.33x of dense bf16 on A100. | claimed in `research/results/mx6_gpu_2026_10_30.md:27-34` (A100, 2-layer stack, not a transformer) |
| S6 | B | mx6 gap decomposition, HELD-OUT (val = last 10% of full tinyshakespeare), G4, d=128, 6000 steps, cosine, val bits/char (bits/param): dense_bf16 2.2864 (96), dense_nobias 2.2891, dense_rms lr 1e-2 2.2702; qat_fp4 2.2983, qat_e2m1lad 2.4291, qat_tern 2.4314 (best lr, 4 seeds); `na_fp4_c8` 2.3607 (8.69, 2 seeds); `tern_counter` lr 3e-2 2.4811 (6.56, 4 seeds); `na_e2m1_c5` nvfp4 2.5474 (6.69, 3 seeds). Full arm table in the report. The `na_fp4_c8`, `tern_counter` and `na_e2m1_c5` rows are no longer the best arms at their widths: see S8. | verified-2026-10-31 (report `research/results/mx6_g4_decomp_2026_10_31.md`, logs `research/results/mx6_g4_decomp_2026_10_31/{heldout,decomposition}/`) |
| S7 | B | Old harness (val == training data, 200k chars, 6000 steps, kept for the record, DO NOT USE): dense 1.5074 (cosine), `na_e2m1_c5` nvfp4 2.1511, e8m0 2.1932; the apparent gap to dense (+0.64) is 2.5x the held-out gap (+0.26) (derived; different corpus slice). | verified-2026-10-31 (logs `research/results/mx6_g4_decomp_2026_10_31/memorization_superseded/`) |
| S8 | B | Frontier with block-scale headroom and the latent grid, HELD-OUT, G4, d=128, 6000 steps, cosine, 2 seeds, val bits/char (measured bits/param): 6.69 bits `lg6_fp4_h2` 2.4155 (counter C=2: h=2 2.5022, h=1 2.5415); 7.69 bits `lg7_fp4_h2` 2.3513 (counter C=4 h=2 2.3835); 8.69 bits `fp4c8_h2` 2.3151 (grid k=8 h=2 2.3253; counter h=1 control 2.3724; h=4 2.3298; pressure growth 2.3458; learnable scale 2.3511 at 8.88 bits). References from S6: dense_bf16 2.2864, qat_fp4 2.2983, `tern_counter` 2.4811 at 6.56 bits, `na_e2m1_c5` 2.5474 at 6.69 bits. The latent grid is eager-only (no kernel, no speed or memory number). | verified-2026-10-31 (report `research/results/mx6_g4_headroom_lg_2026_10_31.md`, logs `research/results/mx6_g4_headroom_lg_2026_10_31/`; flags from the Colab job scripts; uncommitted code) |
