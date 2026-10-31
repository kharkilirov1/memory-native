"""probes/e1_quality.py: lr schedule + flip/edge telemetry (CPU, tiny model)."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("e1_quality", ROOT / "probes" / "e1_quality.py")
e1 = importlib.util.module_from_spec(_spec)
sys.modules["e1_quality"] = e1
_spec.loader.exec_module(e1)

ARM = "na_wide_c3"
VOCAB, D = 24, 64


def test_cosine_formula():
    base, total, wf, mf = 0.01, 1000, 0.02, 0.1
    w = 20                                                   # round(0.02 * 1000)
    lo = mf * base
    f = lambda s: e1.sched_lr("cosine", base, s, total, wf, mf)
    assert f(0) == pytest.approx(base / w)                   # first warmup step
    assert f(w - 1) == pytest.approx(base)                   # last warmup step reaches lr
    assert f(w) == pytest.approx(base)                       # cosine starts at lr
    mid = w + (total - w) // 2
    assert f(mid) == pytest.approx(lo + 0.5 * (base - lo) * (1 + math.cos(math.pi * 0.5)))
    assert f(mid) == pytest.approx((base + lo) / 2)
    assert f(total) == pytest.approx(lo)                     # floor = lr_min_frac * lr
    assert f(total - 1) > lo
    vals = [f(s) for s in range(w, total)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))        # monotone decay


def test_const_unchanged():
    assert all(e1.sched_lr("const", 0.01, s, 500) == 0.01 for s in (0, 1, 10, 250, 499, 500))


def _batch():
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, VOCAB, (8, 32), generator=g)
    return x, torch.roll(x, -1, dims=1)


def _train(lr: float, steps: int = 6):
    torch.manual_seed(0)
    model = e1.CharLM(ARM, VOCAB, d=D, blocks=1, lr=lr)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    x, y = _batch()
    e1.set_counter_lr(model, lr)
    before = e1.snapshot_values(model)
    for _ in range(steps):
        loss = torch.nn.functional.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return e1.flip_edge(before, e1.snapshot_values(model))


def test_flip_rate_zero_when_lr_zero():
    flip, edge = _train(lr=0.0)
    assert flip == 0.0                                       # null twin: nothing moves
    assert 0.0 <= edge <= 1.0


def test_flip_rate_positive_when_lr_positive():
    flip, edge = _train(lr=0.05)
    assert 0.0 < flip <= 1.0
    assert 0.0 <= edge <= 1.0


def test_set_counter_lr_reaches_every_layer():
    model = e1.CharLM(ARM, VOCAB, d=D, blocks=2, lr=0.01)
    e1.set_counter_lr(model, 0.123)
    lrs = [m.lr for m in model.modules() if hasattr(m, "apply_update_")]
    assert len(lrs) == 8 and all(v == 0.123 for v in lrs)


def _run(sched: str, arm: str = ARM, steps: int = 4):
    ids = torch.randint(0, VOCAB, (4000,))
    return e1.run_arm(arm, 0, D, steps, 4, 32, ids, VOCAB, lr=0.01, adam_lr=3e-3,
                      sched=sched, eval_every=2)


def test_run_arm_const_and_cosine_curves():
    r = _run("const")
    assert [c[2] for c in r["curve"]] == [0.01, 0.01]        # const: lr never touched
    assert [c[0] for c in r["curve"]] == [2, 4]
    assert r["flip"][2] >= 0.0 and 0.0 <= r["edge"] <= 1.0
    r = _run("cosine")
    assert r["curve"][-1][2] < 0.01                          # cosine: decayed by the end


def test_dense_arm_has_no_flip_telemetry():
    r = _run("cosine", arm="dense_bf16")
    assert math.isnan(r["edge"]) and math.isnan(r["flip"][2])
