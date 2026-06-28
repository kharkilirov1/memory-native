# Verification plan — results & deployment roadmap (Phase 1 + 2)

Status of every method from `METHODS_VERIFICATION_PLAN.md`, with the rule that nothing counts
without a real witness. Each row links its module / witness / results doc. Honest negatives are
stated as plainly as the passes.

## Scoreboard

| # | method | what it attacks | verdict | witness |
|---|---|---|---|---|
| M1 | CounterMemoryFFN | FLOPs-to-loss (retrieval FFN) | **PARTIAL** — matches dense at ~5× less active compute; capacity-scaling inconclusive on toy data (overfits); resists overfitting | `MEMORY_FFN.md` |
| M2 | 2:4 group-counter | per-step (fwd/grad_x) | **PARTIAL** — 2:4 sparsity learns fine (≤ unstructured); error-feedback rotation NOT shown to beat pruning | `GROUP_COUNTER.md` |
| M3 | slow-fast `sT+ABᵀ` | per-step (wgrad freq) | **PASS** — base-update cut 8–32× at no accuracy cost; standalone ceiling ~1.43× | `SLOWFAST.md` |
| M4 | Counter-MoE FFN | FLOPs-to-loss (experts) | **PASS (cleanest)** — beats dense at E=8/16, monotonic capacity scaling, no routing collapse | `MOE_FFN.md` |
| M5 | int8 forward | per-step | **PASS** (prior) — ×2.05 isolated, T4-verified | `ACCELERATION.md` |
| M6 | int4-wgrad | per-step (wgrad) | **math PASS / HW rejected on T4** (Amdahl) | `INT4_WGRAD.md` |
| M-STACK | 2:4 + slow-fast compose | per-step (integration) | **PARTIAL** — compose mechanically (train, no divergence) but at a quality cost (+31% vs plain counter); speedup unmeasured (kernel-gated) | `MSTACK.md` |
| M9 | Multi-Token Prediction | tokens-to-loss | **IMPLEMENTED / toy-scale FAIL** — n_pred=1 exact parity, unit-correct (8 tests); but toy witness REGRESSES (tied small-embedding heads pull the next-token head: +0.57 at n_pred=2, +1.13 at n_pred=4). Real gains need scale + untied heads | `mtp.py` |
| M10 | Mixture-of-Depths | FLOPs/token | **IMPLEMENTED** — top-capacity routing, capacity=1.0 exact parity, skipped tokens bit-identical; witness on GPU | `mod.py` |
| — | Triton/CUDA kernels | (correctness) | **PASS on T4** — all kernel tests execute & pass (closed the CI gap) | `GPU_KERNEL_VERIFICATION.md` |
| M8 | prototype-stat | wgrad | not attempted — stays behind its bias-gate (biased; M1 is strictly better) | — |
| M11 | int4-IMMA kernel | M6 realization | Blackwell-only — not buildable/runnable on T4; spec only | — |
| M12 | Metal port | energy | different runtime — out of scope for this environment; spec only | — |

## Queued for the new Kaggle GPU account (`scratchpad/build_gpu_realdata.py <username>`)
One T4 job runs everything that needs proof: full pytest (kernels + new tests), the **real-data
scaling gate** (`realdata_scaling.py` — M1/M4 on real tinyshakespeare, the experiment toy data
couldn't be), and the M9/M10/M-STACK training witnesses (too slow on CPU). This closes the open
proofs in one run.

## The honest shape of the result

**What is proven and deployable today** (this is the *core method*, already run at 1B on 2×T4 and
now GPU-verified + review-fixed): counter synapses, reversible/anchors, int8 forward, the fused
Triton update kernel, DDP. The `memory-native` package ships these.

**What is mechanism-validated but NOT yet shown to help at scale** (the new levers): M1, M2, M3, M4.
Each works and is correct; none has beaten a strong baseline on *real data at scale*. The toy
CPU/teacher witnesses cannot test the thing that matters (FLOPs-to-loss on real tokens) — the
synthetic corpus overfits before capacity pays off (M1 scaling re-test: dense val 0.96 → 1.80 as
steps grow). So these are research-validated, not deployment-ready.

**Two clean wins to build on:** M4 (Counter-MoE) shows monotonic capacity scaling and beats dense
already at toy scale — the strongest architecture lever. M3 (slow-fast) cleanly cuts base-update
frequency. **Two honest negatives:** M2's error-feedback rotation doesn't beat plain pruning (no
task here needs support-discovery), and M1's capacity-scaling is inconclusive on toy data.

## Deployment roadmap (what "внедрение" actually requires)

The core method is deployed. Deploying the *new levers* is gated, in order:

1. **Real-data scaling gate** (the missing experiment): run M1 + M4 on a real corpus (FineWeb /
   tinyshakespeare, many tokens) with a dense baseline, on GPU. This is the experiment the toy data
   could not be — it decides whether the FLOPs-to-loss levers actually pay off. M4 is the front-runner.
2. **M-STACK per-step gate** (needs a kernel): the composability witness here shows the levers train
   together, but the *speedup* requires the cuSPARSELt 2:4 kernel (M2-stage-2) — unbuilt. Without it
   the per-step number is theoretical. Build the sparse-Tensor-Core path, then measure per-step on T4.
3. **A/B at scale = deployment**: wire the winning lever(s) into `scripts/fineweb_1b_2xt4.py` and run
   head-to-head against the current 1B baseline (checkpoint + loss curve already in hand). A real run
   where the lever helps IS the deployment.

Phase 3 (int4-IMMA on Blackwell, native Metal) is hardware-gated and out of scope for this
environment.

## Test status
93 tests pass on CPU (10 GPU-gated skipped); all kernel tests pass on T4. New since the review:
M1 (5), M2 (—), M3 (3), M4 (6), M-STACK (5), DDP (2), reversible exactness (3).
