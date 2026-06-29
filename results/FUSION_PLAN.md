# Kernel-fusion plan — confirm the round-trip-collapse levers (status per proposal)

The speed bottleneck is data movement *around* the GEMM, not the GEMM (Amdahl; see `PERF_ANATOMY.md`).
The levers below collapse intermediate-tensor HBM round-trips toward the irreducible floor (read
packed W, read activations, write packed state). This doc records, per lever, **what is confirmed vs
what is still a hypothesis needing a kernel/GPU** — under the project rule "witness or it doesn't
count". CPU confirmations: `scripts/fusion_invariants.py`.

## Irreducible traffic floor (per step)
Must happen: read packed W (6-bit), read activations, write new packed state. Everything else —
grad_w in HBM, the decoded weight, intermediate v/scale — can in principle stay in SRAM/registers.
The levers aim at this floor.

## Levers (math unchanged → bit-exact, unless noted)

### #1 — Epilogue fusion: counter-update inside the grad_w GEMM, grad_w never hits HBM
Removes the fattest traffic (write+read the `[out,in]` grad_w, 4 B/weight). Extends our existing
`update_from_io` (grad_w in-kernel) to run the full automaton in the matmul epilogue from the
accumulator.
- **Structural obstacle (was unstated):** the RMS denom needs `g_sq = mean_in(grad²)` — a full-ROW
  reduction — before ticking, which fights a per-tile epilogue.
- **CONFIRMED enabler:** in `rms_mode="lagged"` the state-write is **per-element** — perturbing
  `grad_w[0,5]` changes only `state[0,5]` (1 byte); in `rms_mode="exact"` it changes the whole row
  (7 bytes, full-row reduction). So **lagged unblocks the epilogue fusion; exact blocks it.**
  (`fusion_invariants.py`, CONFIRMED.) The v-EMA update still needs a row reduction but emits only
  O(out) values → fold into the GEMM's split-K reduction or a cheap separate pass.
- **Still hypothesis (needs kernel + Nsight Compute):** that the per-weight automaton (decode/SR/
  encode) in the epilogue does not spill registers / tank GEMM occupancy. This is the real risk.
- Verdict: **architecturally unblocked (lagged), bit-exact; occupancy is the open kernel question.**

### #2 — Prologue fusion: decode in the GEMM mainloop (mixed-input), decoded weight never in HBM
- **CONFIRMED:** the forward depends only on the visible ternary `t`, not the counter `c`
  (`fusion_invariants.py`, max|Δ|=0). So the forward mixed-input is **ternary × activation**
  (BitNet/Marlin-ternary class — kernels of this class exist), NOT exotic 6-bit. The 6-bit packing
  only matters on the update GEMM. Forward prologue-fusion is more tractable than "6-bit mixed-input".
- **Still hypothesis (needs kernel):** a custom ternary+per-row-scale mainloop unpack at our layout.
- Verdict: **enabler confirmed (forward is ternary), bit-exact; kernel work remains, but standard-ish.**

### #3 — Overlap recompute (compute-bound) with counter-update (bandwidth-bound) across layers
- **NOT confirmable on CPU.** Pure scheduling; the update of block i is independent of block i−1's
  reconstruction, so the parallelism exists in principle.
- **Needs GPU + Nsight Systems:** on T4 (40 SMs) the recompute may saturate the SMs, leaving nothing
  to overlap; default-stream serialization is a trap. Realistic only on larger GPUs.
- Verdict: **GPU-gated; deprioritized for T4.**

### #4 — Cross-step persistent T-cache + flip-patching
- **Already CONFIRMED by the suite:** the derived T-cache is a persistent buffer refreshed only on
  flipped rows and stays bit-identical to a fresh decode across update steps
  (`tests/test_cache.py::test_cache_tracks_state_through_updates`). Extending "don't rebuild across
  steps, patch only flips" is the same mechanism → amortized decode ≈ 0 with decimation. Bit-exact.
- Verdict: **confirmed + lowest-risk; the cheapest first realization.**

### #5 — Lower-precision grad/recompute path (fp8/bf16) — CHANGES NUMERICS
- **Split the lever (correction):**
  - fp8 on **grad_w**: relatively safe (SR-tolerant correlation). Halves that pass's bytes.
  - fp8 on the **recompute-forward + reversible**: **dangerous** — reversible needs the recompute to
    match the original forward for correct gradients (the inverse is exact in fp32; fp8 recompute
    would diverge from the bf16 original → wrong grads, not "in-SR noise"). Keep recompute at the
    original precision.
- **NOT bit-exact → requires a loss/accuracy parity witness** before adoption (the original author's
  own caveat, agreed).
- Verdict: **fp8-grad_w only, parity-gated; fp8-recompute rejected on the reversible path.**

## The ceiling — honest, regime-dependent
- **Big-batch training = compute-bound GEMM.** #1–#4 remove the overhead *around* the GEMM → approach
  **parity with dense** (same GEMM, same FLOPs). The only GEMM speedup is int8 (×2, already shipped).
  So the training prize is **"<1 byte/weight at NO speed penalty"**, not a new multiplier.
- **Small-batch / inference = memory-bound.** 6-bit weight = ~8× less weight traffic than fp16 → can
  be genuinely **faster than dense** (BitNet-inference territory). This is where a real speedup lives.

## Implementability ranking (for our constraints: T4, no kernel team)
1. **#4** persistent cache + flip-patch — confirmed, bit-exact, cheapest.
2. **#1 (lagged only)** — biggest traffic win; occupancy risk → Nsight.
3. **#2 (ternary forward prologue)** — standard-ish, bit-exact.
4. **#5 (grad_w fp8 only)** — parity-gated.
5. **#3 overlap** — GPU-gated, T4-unlikely.

## Witnesses to run when a kernel exists
- #1/#2: `diff(packed state) == 0` after a step (proves bit-exact) + Δ step-time + Nsight occupancy.
- #3: Nsight Systems timeline showing co-resident kernels + Δ step-time.
- #5: loss/accuracy curve vs baseline over N steps (parity).
