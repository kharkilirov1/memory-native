"""mn-solver v3 CONSOLIDATED local witness (CPU-friendly): layerwise output-error gating
of the consolidated solver (agent v3 refine cycle + Stage-A ingredients) on the real
local donor (Qwen2.5-0.5B, WikiText-2 train calibration). NO training anywhere.

Metric: relative H-weighted layer output error  sum_l (w-q)^T H_l (w-q) / sum_l w^T H_l w
over SAMPLED decoder layers (0 and 23). Exactly the objective the solver optimizes.

Arms (consolidated API; v2 numbers are the Stage-A finetune-branch reference for
context, not rerun here):
  v3_base     agent v3 default: refine_iters=2, scale_refit='hdiag'
  v3_hesscd   their greedy H-metric CD refit
  v3_align    A7 exact joint sym re-solve as the refit mode
  v3_itf      A5 asymmetric grid + align refit (itf scales)
  v3_salient  A4.1 salient-first split on the v3 cycle
  v3_full     itf + align + salient_first (the Stage-A chain on the v3 cycle)

Env: CALIB (path to the probe .pt), SALIENT (0.01), ARMS, OUT (json).
"""
import json
import os
import time

import torch

from memory_native.donor.ptq import gptq_group_ternary

CALIB = os.environ.get("CALIB", "/tmp/solver_v3_calib.pt")
SALIENT = float(os.environ.get("SALIENT", "0.01"))
OUT = os.environ.get("OUT", "results/solver_v3_consolidated_layerwise.json")
ARMS = os.environ.get(
    "ARMS", "v3_base,v3_hesscd,v3_align,v3_itf,v3_salient,v3_full"
).split(",")

torch.set_num_threads(os.cpu_count())
blob = torch.load(CALIB, weights_only=False)
W, H = blob["W"], blob["H"]
targets = list(W.keys())
print(f"consolidated layerwise witness: {blob['model']} layers={blob['layers']} "
      f"calib={blob['calib_batches']}x{blob['batch']}x{blob['seq']} "
      f"({len(targets)} linears)", flush=True)

ARM_KW = {
    "v3_base": dict(),
    "v3_hesscd": dict(scale_refit="hessian_cd"),
    "v3_align": dict(scale_refit="align"),
    "v3_itf": dict(grid="itf", scale_refit="align"),
    "v3_salient": dict(salient_first=SALIENT),
    "v3_full": dict(grid="itf", scale_refit="align", salient_first=SALIENT),
}

den = sum(float((W[p] @ H[p] * W[p]).sum()) for p in targets)
results = {}
if os.path.exists(OUT):
    results.update(json.load(open(OUT)))
results["_denominator"] = den
results["_config"] = {k: blob[k] for k in ("model", "layers", "seq", "batch", "calib_batches")}

for arm in ARMS:
    if arm in results:
        print(f"[{arm:10s}] cached: rel_err={results[arm]['rel_err']:.5f}", flush=True)
        continue
    t0 = time.perf_counter()
    num = 0.0
    for p in targets:
        q, _, _ = gptq_group_ternary(W[p], H[p], group=128, **ARM_KW.get(arm, {}))
        d = W[p] - q
        num += float((d @ H[p] * d).sum())
    rel = num / den
    results[arm] = {"rel_err": rel, "seconds": round(time.perf_counter() - t0, 1)}
    print(f"[{arm:10s}] rel_err={rel:.5f}  ({time.perf_counter()-t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=1)

print("\n=== solver v3 CONSOLIDATED layerwise gate (calibration H, sampled layers) ===",
      flush=True)
for arm in ARMS:
    r = results[arm]
    print(f"{arm:10s} rel_err={r['rel_err']:.5f}  ({r['seconds']:.0f}s)", flush=True)
