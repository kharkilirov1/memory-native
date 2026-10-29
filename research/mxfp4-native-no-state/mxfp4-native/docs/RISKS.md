# Risks: no-master training at 100M+, prior art and failure modes

Survey note, 2026-10-28, branch `mxfp4-native`. Current result (`docs/PHASE2.md` Round 4): 461k-param
char-LM, 3000 steps, MXFP4 values + E8M0 scale + 6-bit integer accumulator, no fp32 master, no Adam
moments = 2.3775 bits/char vs best-tuned Adam-QAT 2.4146 (3/3 seeds), 10.25 bits/param vs 100.

## 1. Prior art that trains without a master or a separate optimizer state

| method | state kept | scale | what broke / signature |
|---|---|---|---|
| 1-bit SGD [3] | **fp32 weights and momentum**; only gradients compressed | speech DNNs, 2014 | not master-free — communication only; without EF, accuracy drops |
| ECO [1] | no master; quantization error injected into momentum | 30-800M Transformers, Gemma-3 1B, 2.1B MoE (FP8), 16B MoE INT4 FT | naive master removal has stationary error ∝ 1/lr (grows as the schedule decays); error injection gives a constant-radius neighborhood |
| QAR / "Beyond Shadow Weights" [2] | no shadow weight; updates the quantization code directly | post-training of LLMs | shadow-weight QAT explanation fails: endpoint loss improves while shadow loss can grow; residual-phase "crossing" and signal-imbalance model; comparable-or-better than QAT at lower memory |
| BitNet b1.58 2B4T [4] | ternary weights; report describes the deployed format, **not** the training master policy | 2B params, 4T tokens | none reported at that scale; "no master" is unverified from the accessible report |
| MXFP4 pretraining [6] | BF16 master; FP4 in Fprop/Dgrad/Wgrad | Llama-3.1-8B, C4 | **Wgrad quantization drives divergence**; SR and random Hadamard do not rescue it, deterministic Hadamard does. We quantize weights only, their safe subset |
| MX sweep [5] | block-scaled formats, ~1000 LMs, 2e17-4.8e19 FLOPs | to LLM compute scale | **sharp stochastic loss instabilities, worse at larger compute**; mechanism: multiplicative gradient bias from quantized layer-norm affine params and a few activations. We keep those in fp, which removes the mechanism but not the evidence that scale brings instabilities |

Every published master-free method still keeps a *memory of past gradients* (ECO's momentum, QAR's
code state); we do too (the 6-bit accumulator), so the mechanism is precedented. The 6-bit
in-weight accumulator at 100M+ is not.

## 2. Failure modes of integer/low-precision accumulators

| mode | mechanism | how to detect it |
|---|---|---|
| drift / random walk | SR noise plus carried unit steps ⇒ the weight norm wanders; stationary error grows with steps, and as 1/lr when the schedule decays [1] | weight norm vs fp32-master control; block MSE vs master vs step; val-gap slope over 10k+ steps |
| limit cycle | carries fire +1/-1 alternately on adjacent codes; weights revisit the same code | carry-sign autocorrelation lag 1-4; `cycle_frac` (fraction returning to the same code within k steps); SR is the known cure for truncation stagnation [15] |
| rail / dead units | weights pinned at ±6 (or the headroom level), carries blocked; zero-gradient units stuck at the smallest code | `rail_frac`, `blocked_frac`, fraction with `abs(idx) in {0,7}` per layer |
| scale ratchet | sticky band wider than the target band lets amax drift to the top rung before rescale | scale-flip rate per block per 100 steps; histogram of `amax/s` (should sit near h, not 6) — `docs/SCALES.md` S1 |
| stale low-precision state | a quantized EMA stops moving (nominal update rounds back) | fraction of unchanged codes over 200 steps while the statistic moved; resets are the documented fix [11] |
| late divergence | MX instabilities are stochastic and worsen with compute [5]; Wgrad FP4 diverges first when quantized [6] | loss-spike frequency, gradient-norm outliers; keep Wgrad fp32 (we do) |

## 3. Scale transfer: does the small-model win survive?

- **Quantization-induced degradation shrinks with model size and with undertraining**: over 1500+
  checkpoints, larger models or fewer tokens show less QiD, while smaller well-trained models lose
  significantly; the projection is that >100T-token models may quantize poorly [7]. This favors us
  going 461k → 100M on a short budget and disfavors long-budget runs.
- **MX training instabilities grow with compute scale** [5], and MXFP4 Wgrad diverges at 8B [6]; both
  are activation/gradient-path failures, which we avoid by keeping the gradient path fp32 — but our
  setting has then never been tested at that scale.
- **Where low-bit state has been taken**: 8-bit Adam to 1.5B [12], 4-bit AdamW state 130M-2.7B [13],
  ECO to 2.1B [1], BitNet to 2B [4], NVFP4 recipes to 8B/30B. No published no-master method above
  ~2.1B and no 6-bit in-weight accumulator at all.
- **Width is our near-term unknown**: per-32 E8M0 and h=1.5 were tuned at d=128; at d=768+ weight
  tails and outliers change (massive activations are a large-model phenomenon [14]) and the fp tail
  that masks errors today (65k of 459k params) shrinks below 1%.

## 4. Packing the 6-bit accumulator

Budget: 32 params = 16 B values + 24 B accumulator (32x6 bit) + 1 B scale = **41 B / 32 = 10.25
b/param**. Three practical facts:

1. **41 B/block is not 4-byte aligned**, so blocks after the first cannot use `u32`/16 B vector
   loads. Fix: separate planes — value plane 16 B/block and acc plane 24 B/block (40 B = 10 u32,
   4-aligned) with the 1 B scale stream separate; both hot planes then have clean 4-byte strides.
2. **Contiguous 6-bit fields straddle 32-bit words**: field i is at bit `6i`, word `6i>>5`, shift
   `6i&31`; 28/32 need one shift+mask, 4/32 (starts 30, 60, 90, 150 mod 32) need a funnel shift
   across two words. Cheap ALU-wise, but unaligned 2/4/8/16 B accesses are split by the hardware and
   Triton only vectorises when offsets are hinted aligned [9,10].
3. **The split-plane layout is the systems precedent**: FP6-LLM stores 6-bit weights as a 4-bit low
   plane plus a 2-bit high plane to use INT4 tensor cores and avoid bitfield extraction [8]; GGML's
   Q6_K does the same. Copy it: low nibbles 16 B/block + high 2-bit plane 8 B/block = the same 24 B,
   byte-aligned, no straddling.

The accumulator plane is **not read by the forward matmul** (only values + scale are), so the packing
penalty is confined to the updater, not the GEMM or the memory-peak claim. Measure the updater in
isolation before believing 10.25 bits is free.

## 5. What is not published

- No master-free method above ~2.1B (ECO) [1]; **no 6-bit in-weight accumulator at any scale**.
- No scaling curve for integer-accumulator training: drift/limit-cycle rates vs width, depth or
  tokens are unmeasured, and no drift analysis over >10^12 tokens for an SR-quantized weight state
  exists (closest: quantized-state staleness/resets [11]).
- BitNet's training-time master policy is not stated in the accessible technical report [4].
- No fix for MX stochastic instabilities other than changing precision mid-training (hybrid) [5],
  which would abandon the 4-bit weight claim.
- Packed 6-bit **training state** has no precedent (FP6-LLM's packing is inference weights [8]);
  our 41 B/block layout and alignment fix are our construction.
- The h=1.5-2 knee, the 33-state knee and the 3/3-seed win are single-corpus, single-model
  measurements, not published results.

## 6. Ranked: the 3 most likely ways this fails at 100M+

| # | risk | why it is likely | cheapest experiment |
|---|---|---|---|
| 1 | **Long-horizon drift / stationary error** | 3000 steps is ~10^-3 of a real run; the accumulator has no decay or leak, SR adds a random walk, and ECO's radius bound needs a decaying lr [1] | extend the existing 461k run to 30k-100k steps, fixed lr and decayed lr, logging weight norm, block MSE vs master and gap vs step. Hours on CPU, no new code |
| 2 | **Width/outlier shift breaks the tuned scale** | h=1.5 and per-32 E8M0 were tuned at d=128; at 100M weight tails change [14], the fp tail stops masking, and one bad block clips | one width step d=256/384 (~2-6M params), same corpus/steps, logging per-layer `rail_frac`, kurtosis, `amax/s` and block MSE; sweep h in {1.5,2,3} |
| 3 | **Baseline/regime artifact** | Adam's lr was swept at 3000 steps only for the 461k model; our lr is coupled to C and width; QiD falls for undertrained larger models [7], flattering whichever arm quantizes more | at the scaled size: {our rule, Adam-QAT} x {3 lr values}, equal tokens, >=2 seeds, plus a 3x-token point to see the gap trajectory |

Packing is a systems risk (memory claim, updater throughput), not a quality one; settle it with a
byte-layout test and an isolated updater benchmark before the scaled run.

## References

[1] Nikdan et al., *ECO: Quantized Training without Full-Precision Master Weights* — https://arxiv.org/abs/2601.22101 · [2] Xu et al., *Beyond Shadow Weights* — https://optimization-online.org/2026/09/beyond-shadow-weights-quantization-aware-training-as-quantized-endpoint-descent/
[3] Seide et al., *1-bit SGD* — https://www.isca-archive.org/interspeech_2014/seide14_interspeech.pdf · [4] Ma et al., *BitNet b1.58 2B4T* — https://arxiv.org/abs/2504.12285
[5] Su et al., *Training Instabilities in Microscaling Formats* — https://arxiv.org/abs/2506.20752 · [6] Cim et al., *Pretraining LLMs with MXFP4* — https://arxiv.org/abs/2605.09825
[7] Ouyang et al., *Low-Bit Quantization Favors Undertrained LLMs* — https://aclanthology.org/2025.acl-long.1555/ · [8] Xia et al., *FP6-LLM* — https://arxiv.org/abs/2401.14112
[9] CUDA C++ Programming Guide, *Device Memory Accesses* — https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses · [10] Triton `tl.multiple_of` — https://triton-lang.org/main/python-api/generated/triton.language.multiple_of.html
[11] Topollai, Choromanska, *Optimizer-State Staleness and Resets* — https://arxiv.org/abs/2603.16731 · [12] Dettmers et al., *8-bit Optimizers* — https://arxiv.org/abs/2110.02861
[13] Li et al., *4-bit AdamW Optimizer-State Quantization* — https://arxiv.org/abs/2610.12444 · [14] Sun et al., *Massive Activations in LLMs* — https://arxiv.org/abs/2402.17762
[15] Gupta et al., *Deep Learning with Limited Numerical Precision* — https://arxiv.org/abs/1502.02551
