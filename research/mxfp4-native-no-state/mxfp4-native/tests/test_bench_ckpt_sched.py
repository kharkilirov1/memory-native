"""bench_train_step: --grad-checkpoint for lg arms on the native custom-op path, --lr-sched cosine, the "1b" preset and the
analytic memory estimate (CPU plumbing; the real timings / memory are measured on the GPU)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "probes"))

from mx6.latent_grid import HAS_LG_OPS                                 # noqa: E402

needs_ops = pytest.mark.skipif(not HAS_LG_OPS, reason="custom ops unavailable")


def _bench():
    spec = importlib.util.spec_from_file_location("bench_train_step_ckpt", ROOT / "probes" / "bench_train_step.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _e1():
    import lg_ab
    e1 = lg_ab.install(lg_ab.load_e1(register=False))
    e1.DEV, e1.SCALE_FORMAT, e1.HEADROOM_MODE, e1.HEADROOM = "cpu", "nvfp4", "ladder", 2.0
    return e1


def _ns(**kw):
    base = dict(lr=0.01, tern_lr=3e-2, adam_lr=3e-3, headroom=2.0, layers=2, heads=0, batch=2, seq=16, warmup=1,
                steps=3, phase_steps=0, no_profile=True, topk=3, fused="off", seed=0, precision="fp32",
                adam_impl="foreach", lg_native="on")
    base.update(kw)
    return SimpleNamespace(**base)


@needs_ops
def test_lg_native_arm_runs_with_grad_checkpoint_and_updates_once_per_layer_per_step(monkeypatch):
    import mx6.latent_grid as LG
    bench, e1 = _bench(), _e1()
    calls = []
    orig = LG._op_update_
    monkeypatch.setattr(LG, "_op_update_", lambda *a, **k: (calls.append(1), orig(*a, **k))[1])
    r = bench.bench_one("lg6_fp4_h2", 32, "eager", _ns(grad_checkpoint=True), e1, 64, "cpu")
    assert r["error"] is None, r.get("traceback")
    assert r["checkpointed_blocks"] == 2 and r["lg_ckpt_op_path"] is True and r["finite"]
    # warmup 1 + timed 3 + final 1 = 5 steps x (2 blocks x 4 linears)
    assert len(calls) == 5 * 8, len(calls)


def test_checkpoint_still_refused_for_function_path_arms():
    bench, e1 = _bench(), _e1()
    for arm, kw in (("na_fp4_c8", {}), ("lg6_fp4_h2", {"lg_native": "off"}), ("lg6_fp4_h2", {"lg_native": "auto"})):
        r = bench.bench_one(arm, 32, "eager", _ns(grad_checkpoint=True, **kw), e1, 64, "cpu")
        assert r["error"] and "grad-checkpoint refused" in r["error"] and "one-forward-per-backward" in r["error"], (arm, kw)


@needs_ops
def test_lr_sched_cosine_changes_lr_every_step_inside_the_step(monkeypatch):
    bench, e1 = _bench(), _e1()
    seen = []
    orig = e1.set_counter_lr
    monkeypatch.setattr(e1, "set_counter_lr", lambda model, lr, scale_frac=None: (seen.append(lr), orig(model, lr, scale_frac))[1])
    r = bench.bench_one("lg6_fp4_h2", 32, "eager", _ns(lr_sched="cosine", steps=6, warmup=2), e1, 64, "cpu")
    assert r["error"] is None and r["lr_sched"] == "cosine"
    assert len(seen) == 2 + 6 + 1                        # warmup + timed + final loss step (the update-count probe is CUDA-only)
    peak = seen.index(max(seen))
    assert all(a > b for a, b in zip(seen[peak:], seen[peak + 1:])), seen        # strictly decreasing after the peak
    c = bench.bench_one("lg6_fp4_h2", 32, "eager", _ns(), e1, 64, "cpu")
    assert c["lr_sched"] == "const"


def test_estimate_memory_orders_and_1b_preset():
    bench = _bench()
    args = SimpleNamespace(preset="1b", d=None, layers=None, heads=None, seq=None, batch=None)
    bench.resolve_shape(args)
    assert (args.d, args.layers, args.heads, args.seq, args.batch) == ("2048", 24, 32, 512, 8)
    n = bench.param_count(2048, 24, 256)
    assert 1.2e9 < n < 1.3e9
    dense = bench.estimate_memory("dense_nobias", 2048, 24, 8, 512, 256, "bf16", False)
    lg = bench.estimate_memory("lg6_fp4_h2", 2048, 24, 8, 512, 256, "bf16", False)
    lgc = bench.estimate_memory("lg6_fp4_h2", 2048, 24, 8, 512, 256, "bf16", True)
    assert lg["state_mb"] < dense["state_mb"] / 8                          # ~1.07 B/param vs 16 B/param
    assert lgc["act_mb"] < lg["act_mb"] / 5 and lgc["total_mb"] < lg["total_mb"]
    assert bench.estimate_memory("na_fp4_c8", 128, 2, 8, 64, 256, "bf16", False) is None


def test_estimate_only_prints_and_runs_nothing(capsys):
    bench = _bench()
    rc = bench.main(["--preset", "1b", "--arms", "dense_strong,lg6_fp4_h2,lg6_fp4_h2_nvfp4fwd", "--estimate-only",
                     "--device", "cpu", "--grad-checkpoint"])
    out = capsys.readouterr().out
    assert rc == 0 and out.count("  est d=2048") == 3 and "grad-checkpoint" in out and "ms/step" not in out
