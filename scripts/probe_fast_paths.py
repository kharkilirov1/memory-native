#!/usr/bin/env python
"""A/B probe for the packed counter layer implementations (single layer + model level).

Verifies the new dense-decode path is numerically identical to the torch decode, then times
forward / backward / update separately and counts CUDA launches per step, so the model-level
gain can be attributed to a specific component instead of guessed.

    python scripts/probe_fast_paths.py --out-dir /tmp/probe
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory_native.counter import RMSCounterLinear  # noqa: E402
from memory_native.fast_decode import PackedDecodeCounterLinear, decode_dense_triton  # noqa: E402
from memory_native.packed import PackedRMSCounterLinear  # noqa: E402
from memory_native.triton_counter import TritonCounterLinear  # noqa: E402


def bench(fn, warmup=5, iters=20, reps=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) / iters)
    return statistics.median(out)


def count_launches(fn) -> int:
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    evts = prof.key_averages()
    for e in evts:
        if e.key == "cudaLaunchKernel":
            return e.count
    return -1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shapes", default="256x256,1024x256")
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--C", type=int, default=8)
    ap.add_argument("--out-dir", default=str(ROOT / "research/results/speed_recovery_2026_10_18"))
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        print("needs CUDA")
        return 0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"device": torch.cuda.get_device_name(0), "tokens": args.tokens, "layers": {}}

    classes = [("torch_rms", RMSCounterLinear), ("packed_torch", PackedRMSCounterLinear),
               ("triton_packed", TritonCounterLinear), ("decode_cublas", PackedDecodeCounterLinear)]
    shapes = []
    for spec in args.shapes.split(","):
        o, i = spec.lower().split("x")
        shapes.append((int(o), int(i)))

    for out_f, in_f in shapes:
        x = torch.randn(args.tokens, in_f, device="cuda")
        go = torch.randn(args.tokens, out_f, device="cuda")
        gw = (go.t().float() @ x.float()).contiguous()
        ref_layer = None
        for name, cls in classes:
            torch.manual_seed(0)
            layer = cls(in_f, out_f, C=args.C, lr=3e-3, lr_scale=2e-4).cuda().train()
            entry = {"shape": [out_f, in_f], "fused_update_available": bool(
                layer._fused_update(0, out_f, gw.clone()))}
            # correctness of the new decode kernel vs the torch dense weight
            if name == "decode_cublas":
                w_triton = decode_dense_triton(layer.state, layer.scale, C=layer.C,
                                               out_features=out_f, in_features=in_f)
                w_torch = layer._dense_weight(torch.float32)
                entry["decode_max_abs_err"] = float((w_triton - w_torch).abs().max())
                entry["decode_bitwise_equal"] = bool(torch.equal(w_triton, w_torch))
                y = layer(x)
                y_ref = x @ w_torch.t()
                entry["forward_max_abs_err"] = float((y - y_ref).abs().max() / y_ref.abs().max())
                del w_triton
                # reference forward from the torch layer for the same weights
                if ref_layer is None:
                    ref_layer = layer
            entry["fwd_ms"] = bench(lambda: layer._forward_matmul(x))
            entry["bwd_gradx_ms"] = bench(lambda: layer._backward_grad_x(go))
            entry["update_ms"] = bench(lambda: layer.apply_update_from_grad_w(gw))

            def one_step():
                yy = layer(x)
                yy.backward(go)

            entry["step_ms"] = bench(one_step)
            entry["launches_per_step"] = count_launches(one_step)
            results["layers"][f"{name}_{out_f}x{in_f}"] = entry
            print(f"{name:15s} {out_f}x{in_f}: fwd {entry['fwd_ms']:7.3f} | gradx "
                  f"{entry['bwd_gradx_ms']:7.3f} | upd {entry['update_ms']:7.3f} | step "
                  f"{entry['step_ms']:7.3f} ms | launches {entry['launches_per_step']:5d}"
                  + (f" | decode err {entry.get('decode_max_abs_err', float('nan')):.2e}"
                     if "decode_max_abs_err" in entry else ""))
            del layer
            torch.cuda.empty_cache()
        del ref_layer

    with open(out_dir / "fast_path_probe.json", "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"artifact: {out_dir/'fast_path_probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
