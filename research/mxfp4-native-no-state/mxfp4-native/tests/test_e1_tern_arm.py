"""probes/e1_quality.py: the main project's ternary counter arms (tern_counter*), CPU, tiny model.

Needs `memory_native` importable (PYTHONPATH=<dir containing memory_native>, or a repo checkout
with <ancestor>/src/memory_native); the module is skipped otherwise.
"""
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

try:
    e1._counter_cls()
except ImportError:                                           # pragma: no cover
    pytest.skip("memory_native not importable", allow_module_level=True)

ARM = "tern_counter"
VOCAB, D = 24, 64


def _batch():
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, VOCAB, (8, 32), generator=g)
    return x, torch.roll(x, -1, dims=1)


def _tern_layers(model):
    return [m for m in model.modules() if e1.is_tern(m)]


def _step(model, opt, x, y):
    loss = torch.nn.functional.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss.detach())


def test_forward_backward_runs_and_state_moves():
    torch.manual_seed(0)
    model = e1.CharLM(ARM, VOCAB, d=D, blocks=1, lr=0.05)
    assert len(_tern_layers(model)) == 4                      # qkv, proj, fc, out
    assert all(not hasattr(m, "bias") or m.bias is None for m in _tern_layers(model))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    x, y = _batch()
    before = e1.snapshot_values(model)
    for _ in range(6):
        assert math.isfinite(_step(model, opt, x, y))
    flip, edge = e1.flip_edge(before, e1.snapshot_values(model))
    assert 0.0 < flip <= 1.0
    assert 0.0 <= edge <= 1.0


def test_flip_edge_ternary_definition():
    torch.manual_seed(0)
    model = e1.CharLM(ARM, VOCAB, d=D, blocks=1, lr=0.0)
    snap = e1.snapshot_values(model)
    assert all(nv == -1 for _, nv in snap)                    # ternary marker
    flip, edge = e1.flip_edge(snap, e1.snapshot_values(model))
    assert flip == 0.0                                        # null twin
    assert edge == pytest.approx(2 / 3, abs=0.05)             # uniform {-1,0,1} init: 2/3 nonzero
    vals, fr = e1.magnitude_histogram(model)
    assert vals == [-1, 0, 1] and sum(fr) == pytest.approx(1.0)


def test_one_forward_per_backward_guard_not_tripped_and_still_armed():
    torch.manual_seed(0)
    model = e1.CharLM(ARM, VOCAB, d=D, blocks=1, lr=0.01)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    x, y = _batch()
    ids = torch.randint(0, VOCAB, (4000,))
    _step(model, opt, x, y)
    e1.evaluate(model, ids, seq=32, iters=2)                  # eval forwards run under no_grad
    _step(model, opt, x, y)                                   # no "reused before backward" error
    model(x)                                                  # forward without backward ...
    with pytest.raises(RuntimeError):                         # ... the guard is real
        model(x)


def test_lr_schedule_reaches_layer_every_step():
    torch.manual_seed(0)
    model = e1.CharLM(ARM, VOCAB, d=D, blocks=2, lr=0.003)
    layers = _tern_layers(model)
    assert len(layers) == 8
    e1.set_counter_lr(model, 0.123)
    assert all(m.lr == 0.123 for m in layers)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    x, y = _batch()
    total, base = 10, 0.003
    for s in range(total):
        want = e1.sched_lr("cosine", base, s, total)
        e1.set_counter_lr(model, want)
        _step(model, opt, x, y)
        assert all(m.lr == pytest.approx(want) for m in layers)
    assert want < base


def test_tern_lr_flag_routes_to_tern_arms_only(monkeypatch, tmp_path):
    data = tmp_path / "t.txt"
    data.write_text("abcdefgh" * 400, encoding="utf-8")
    seen = {}

    def fake_run_arm(arm, seed, d, steps, batch, seq, ids, vocab, lr, adam_lr, **kw):
        seen[arm] = lr
        return {"arm": arm, "seed": seed, "bits": 1.0, "val": 1.0, "sec": 0.0, "hist": None,
                "clip": float("nan"), "grown": 0, "curve": [(1, 1.0, lr, 0.0, 0.0, float("nan"))],
                "best": 1.0, "flip": (0.0, 0.0, 0.0), "edge": 0.0, "train": 1.0, "gap": 0.0}

    monkeypatch.setattr(e1, "run_arm", fake_run_arm)
    e1.main(["--d", "32", "--steps", "2", "--seeds", "0", "--lr", "0.01", "--tern-lr", "0.02",
             "--arms", "tern_counter,tern_counter_int4,na_e2m1_c5", "--data", str(data)])
    assert seen == {"tern_counter": 0.02, "tern_counter_int4": 0.02, "na_e2m1_c5": 0.01}


def test_bits_measured_equals_expected_and_arm_variants():
    for arm, bits in (("tern_counter", 0), ("tern_counter_noact", 0), ("tern_counter_int4", 4)):
        model = e1.CharLM(arm, VOCAB, d=D, blocks=1)
        assert all(m.act_save_bits == bits and m.C == e1.TERN_C for m in _tern_layers(model))
        mb, eb = e1.measured_bits(model, arm), e1.expected_bits(model, arm)
        assert abs(mb - eb) < 0.02 and 6.0 < mb < 8.0         # 6-bit code + 12 B/row fp32 stats


def test_run_arm_end_to_end_cosine():
    ids = torch.randint(0, VOCAB, (4000,))
    r = e1.run_arm(ARM, 0, D, 4, 4, 32, ids, VOCAB, lr=0.003, adam_lr=3e-3,
                   sched="cosine", eval_every=2)
    assert [c[0] for c in r["curve"]] == [2, 4]
    assert r["curve"][-1][2] < 0.003                          # cosine decayed the counter lr
    assert math.isnan(r["clip"]) and 0.0 <= r["edge"] <= 1.0 and math.isfinite(r["val"])


# ---- held-out validation split (applies to every arm; tern arm used as the CPU carrier) ----------
def test_split_is_contiguous_and_disjoint():
    ids = torch.arange(1000)
    tr, va = e1.split_ids(ids, 0.1)
    assert len(tr) == 900 and len(va) == 100 and tr.max() < va.min()
    assert torch.equal(torch.cat([tr, va]), ids)
    seq = 32
    # train sampler: starts in [0, len(tr) - seq - 1), x/y read up to start + seq  ->  < len(tr)
    assert (len(tr) - seq - 2) + seq < len(tr)
    t0, v0 = e1.split_ids(ids, 0.0)
    assert t0 is ids and v0 is ids                           # legacy: val == train


def test_train_batches_never_touch_val_tokens(monkeypatch):
    seen = {"train": [], "val": [], "trainloss": []}
    orig = e1.CharLM.forward

    def spy(self, idx):
        (seen["train"] if self.training else seen["val"]).append((int(idx.min()), int(idx.max())))
        return orig(self, idx)

    monkeypatch.setattr(e1.CharLM, "forward", spy)
    g = torch.Generator().manual_seed(0)
    train = torch.randint(0, 12, (3000,), generator=g)       # tokens 0..11
    val = torch.randint(12, 24, (1000,), generator=g)        # tokens 12..23 (disjoint alphabet)
    r = e1.run_arm(ARM, 0, D, 4, 4, 32, train, VOCAB, lr=0.003, adam_lr=3e-3, eval_every=2,
                   val_ids=val)
    assert all(hi < 12 for _, hi in seen["train"])            # training only sampled train
    assert any(lo >= 12 for lo, _ in seen["val"])             # val eval read val tokens only ...
    assert any(hi < 12 for _, hi in seen["val"])              # ... plus the train-loss estimate
    assert math.isfinite(r["train"]) and r["gap"] == pytest.approx(r["val"] - r["train"])


def test_val_frac_zero_is_legacy_equivalent():
    ids = torch.randint(0, VOCAB, (4000,))
    kw = dict(lr=0.003, adam_lr=3e-3, sched="cosine", eval_every=2)
    a = e1.run_arm(ARM, 0, D, 4, 4, 32, ids, VOCAB, **kw)               # legacy call
    b = e1.run_arm(ARM, 0, D, 4, 4, 32, ids, VOCAB, val_ids=ids, **kw)  # what --val-frac 0 passes
    assert a["val"] == b["val"] == a["train"] and a["gap"] == 0.0
    assert [c[1] for c in a["curve"]] == [c[1] for c in b["curve"]]


def test_header_fields_and_val_frac_zero_main(monkeypatch, tmp_path, capsys):
    data = tmp_path / "t.txt"
    data.write_text("abcdefgh" * 500, encoding="utf-8")        # 4000 chars
    captured = []

    def fake_run_arm(arm, seed, d, steps, batch, seq, ids, vocab, lr, adam_lr, val_ids=None, **kw):
        captured.append((len(ids), None if val_ids is None else len(val_ids), val_ids is ids))
        return {"arm": arm, "seed": seed, "bits": 1.0, "val": 1.0, "sec": 0.0, "hist": None,
                "clip": float("nan"), "grown": 0, "curve": [(1, 1.0, lr, 0.0, 0.0, float("nan"), 1.0)],
                "best": 1.0, "flip": (0.0, 0.0, 0.0), "edge": 0.0, "train": 1.0, "gap": 0.0}

    monkeypatch.setattr(e1, "run_arm", fake_run_arm)
    base = ["--d", "32", "--steps", "10", "--batch", "4", "--seq", "8", "--seeds", "0",
            "--arms", "na_e2m1_c5", "--data", str(data)]
    e1.main(base)                                              # defaults: whole file, val-frac 0.1
    out = capsys.readouterr().out
    assert "train_tokens=3600" in out and "val_tokens=400" in out and "val_frac=0.1" in out
    assert f"epochs={10 * 4 * 8 / 3600:.2f}" in out and captured[-1][:2] == (3600, 400)
    e1.main(base + ["--chars", "2000", "--val-frac", "0"])     # legacy: train-as-val
    out = capsys.readouterr().out
    assert "train_tokens=2000" in out and "val_tokens=0" in out and "LEGACY" in out
    assert captured[-1] == (2000, 2000, True)
