"""bench_train_step --precision / --adam-impl / --preset / --grad-checkpoint / dense_strong and e1_quality --precision.
CPU, tiny shapes. bf16 autocast on CPU is skipped (with a message) if this torch cannot do it."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "probes"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bench = _load("bench_train_step", ROOT / "probes" / "bench_train_step.py")
import lg_ab                                                     # noqa: E402


def _bf16_cpu_ok() -> bool:
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            torch.randn(4, 4) @ torch.randn(4, 4)
        return True
    except Exception:
        return False


needs_bf16 = pytest.mark.skipif(not _bf16_cpu_ok(), reason="bf16 autocast matmul unavailable on CPU in this torch")


def _e1():
    e1 = lg_ab.install(lg_ab.load_e1(register=False))
    e1.DEV, e1.SCALE_FORMAT, e1.HEADROOM_MODE, e1.HEADROOM = "cpu", "nvfp4", "ladder", 2.0
    return e1


def _ns(**kw):
    base = dict(lr=0.01, tern_lr=3e-2, adam_lr=3e-3, headroom=2.0, layers=1, heads=0, batch=2, seq=16, warmup=1,
                steps=2, phase_steps=0, no_profile=True, topk=3, fused="off", seed=0)
    base.update(kw)
    return SimpleNamespace(**base)


# ---- presets / params -------------------------------------------------------------------------------------
def test_presets_shapes_and_exact_param_counts():
    assert set(bench.PRESETS) == {"tiny", "d512", "d1024", "d2048", "1b"}
    for name, (d, layers, heads, seq, batch) in bench.PRESETS.items():
        assert d % heads == 0 and seq <= 512 and batch >= 1
        n = bench.param_count(d, layers, 256)
        assert n == layers * (12 * d * d + 4 * d) + 2 * 256 * d + 512 * d + 2 * d
    assert bench.PRESETS["d1024"][:3] == (1024, 8, 16) and bench.PRESETS["d2048"][:3] == (2048, 16, 32)
    assert bench.PRESETS["1b"][:3] == (2048, 24, 32) and bench.PRESETS["1b"][3:] == (512, 8)
    assert 1.15e9 < bench.param_count(2048, 24, 256) < 1.25e9


def test_param_count_equals_built_model():
    e1 = _e1()
    for d, layers in ((32, 1), (64, 3)):
        m = e1.CharLM("dense_nobias", 256, d=d, blocks=layers, heads=1)
        assert sum(p.numel() for p in m.parameters()) == bench.param_count(d, layers, 256)


def test_resolve_shape_defaults_and_overrides():
    ns = SimpleNamespace(preset=None, d=None, layers=None, heads=None, seq=None, batch=None)
    bench.resolve_shape(ns)
    assert (ns.d, ns.layers, ns.heads, ns.seq, ns.batch) == ("128,256,512", 4, 0, 128, 32)       # legacy defaults
    ns = SimpleNamespace(preset="d1024", d=None, layers=None, heads=None, seq=None, batch=4)
    bench.resolve_shape(ns)
    assert (ns.d, ns.layers, ns.heads, ns.seq, ns.batch) == ("1024", 8, 16, 512, 4)
    ns = SimpleNamespace(preset="1b", d="64", layers=2, heads=None, seq=None, batch=None)
    bench.resolve_shape(ns)
    assert (ns.d, ns.layers, ns.heads) == ("64", 2, 32)


# ---- precision / adam / checkpoint ------------------------------------------------------------------------
@needs_bf16
@pytest.mark.parametrize("arm", ["dense_nobias", "qat_fp4", "na_fp4_c8", "lg6_fp4_h2", "tern_counter"])
def test_every_arm_runs_under_bf16_autocast_with_fused_adam(arm):
    e1 = _e1()
    r = bench.bench_one(arm, 32, "eager", _ns(precision="bf16", adam_impl="fused"), e1, 64, "cpu")
    if arm == "tern_counter" and r["error"] and "memory_native" in r["error"]:
        pytest.skip(r["error"])
    assert r["error"] is None, r.get("traceback")
    assert r["finite"] and r["precision"] == "bf16" and r["adam_impl"] == "fused" and r["model_tflops"] > 0
    assert (r["fp32_islands"] > 0) == (arm not in ("dense_nobias", "qat_fp4"))        # counter linears are fp32 islands


@needs_bf16
def test_dense_strong_is_bf16_fused_and_matches_dense_nobias_params():
    e1 = _e1()
    r = bench.bench_one("dense_strong", 32, "eager", _ns(), e1, 64, "cpu")          # args say fp32/foreach: overridden
    assert r["error"] is None, r.get("traceback")
    assert r["precision"] == "bf16" and r["adam_impl"] == "fused" and r["label"] == "dense_strong"
    d = bench.bench_one("dense_nobias", 32, "eager", _ns(), e1, 64, "cpu")
    assert d["precision"] == "fp32" and d["n_total"] == r["n_total"] == bench.param_count(32, 1, 64)


@needs_bf16
def test_bf16_loss_close_to_fp32_at_init():
    e1 = _e1()
    a = bench.bench_one("dense_nobias", 32, "eager", _ns(steps=1, warmup=0), e1, 64, "cpu")
    b = bench.bench_one("dense_nobias", 32, "eager", _ns(steps=1, warmup=0, precision="bf16"), e1, 64, "cpu")
    assert abs(a["final_loss"] - b["final_loss"]) < 0.25


def test_grad_checkpoint_dense_ok_and_counter_arms_refused():
    e1 = _e1()
    r = bench.bench_one("dense_nobias", 32, "eager", _ns(grad_checkpoint=True), e1, 64, "cpu")
    assert r["error"] is None and r["finite"] and r["checkpointed_blocks"] == 1
    for arm in ("na_fp4_c8", "lg6_fp4_h2"):
        r = bench.bench_one(arm, 32, "eager", _ns(grad_checkpoint=True), e1, 64, "cpu")
        assert r["error"] and "grad-checkpoint refused" in r["error"] and "one-forward-per-backward" in r["error"]


def test_grad_checkpoint_gradients_equal_plain_backward():
    e1 = _e1()
    def grads(ckpt):
        torch.manual_seed(0)
        m = e1.CharLM("dense_nobias", 64, d=32, blocks=2, heads=2)
        if ckpt:
            bench.enable_grad_checkpoint(m)
        x = torch.randint(0, 64, (2, 16))
        m(x).square().mean().backward()
        return [p.grad.clone() for p in m.parameters()]
    for g0, g1 in zip(grads(False), grads(True)):
        torch.testing.assert_close(g0, g1)


def test_main_cli_flags_presets_and_json(tmp_path):
    out = tmp_path / "b" / "bench"
    rc = bench.main(["--preset", "tiny", "--layers", "1", "--seq", "16", "--batch", "2", "--arms", "dense_nobias",
                     "--steps", "2", "--warmup", "1", "--phase-steps", "0", "--no-profile", "--adam-impl", "fused",
                     "--out", str(out)])
    assert rc == 0
    js = json.loads(Path(str(out) + ".json").read_text())
    assert js["meta"]["preset"] == "tiny" and js["meta"]["widths"] == [128] and js["meta"]["adam_impl"] == "fused"
    assert js["meta"]["params"]["128"] == bench.param_count(128, 1, 256)
    r = js["results"][0]
    assert r["error"] is None and r["heads"] == 2 and r["d"] == 128 and r["adam_impl"] == "fused"
    md = Path(str(out) + ".md").read_text()
    assert "model TFLOPS" in md and "vs dense_strong" in md and "preset tiny" in md


@needs_bf16
def test_main_bf16_strong_runs_eager_fallback_and_reports_vs_strong(tmp_path, monkeypatch):
    # torch.compile needs a C++ toolchain on CPU: stub it to identity so the plumbing (not inductor) is tested
    monkeypatch.setattr(bench.torch, "compile", lambda m, **k: m)
    out = tmp_path / "b" / "bench"
    rc = bench.main(["--preset", "tiny", "--layers", "1", "--seq", "16", "--batch", "2", "--steps", "2", "--warmup", "1",
                     "--phase-steps", "0", "--no-profile", "--precision", "bf16",
                     "--arms", "dense_strong,dense_nobias,na_fp4_c8", "--out", str(out)])
    assert rc == 0
    res = json.loads(Path(str(out) + ".json").read_text())["results"]
    assert [r["label"] for r in res] == ["dense_strong", "dense_nobias", "na_fp4_c8"]
    assert all(r["error"] is None and r["precision"] == "bf16" for r in res), res
    assert "1.00x" in Path(str(out) + ".md").read_text()                                   # dense_strong vs itself


# ---- e1_quality --precision -------------------------------------------------------------------------------
@needs_bf16
@pytest.mark.parametrize("arm", ["dense_nobias", "na_fp4_c8"])
def test_e1_run_arm_precision_bf16_trains_and_default_is_fp32(arm):
    e1 = _e1()
    ids = torch.randint(0, 24, (4000,))
    kw = dict(arm=arm, seed=0, d=32, steps=4, batch=4, seq=16, ids=ids, vocab=24, lr=0.01, adam_lr=3e-3, eval_every=2,
              blocks=1, heads=2)
    r32 = e1.run_arm(**kw)
    r16 = e1.run_arm(**kw, precision="bf16")
    assert r32["val"] == pytest.approx(r16["val"], abs=0.5) and r16["val"] == r16["val"]
    # default == explicit fp32, bit for bit
    assert e1.run_arm(**kw, precision="fp32")["val"] == r32["val"]


def test_e1_cli_has_precision_flag():
    src = (ROOT / "probes" / "e1_quality.py").read_text(encoding="utf-8")
    assert '"--precision", default="fp32"' in src and "precision=args.precision" in src
