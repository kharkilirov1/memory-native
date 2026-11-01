"""probes/train_1b.py on CPU with a tiny model and a synthetic byte file instead of TinyStories."""
from __future__ import annotations

import importlib.util
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "probes"))

import mx6.latent_grid as LG                                           # noqa: E402

spec = importlib.util.spec_from_file_location("train_1b", ROOT / "probes" / "train_1b.py")
T = importlib.util.module_from_spec(spec)
sys.modules["train_1b"] = T
spec.loader.exec_module(T)

needs_ops = pytest.mark.skipif(not LG.HAS_LG_OPS, reason="custom ops unavailable")
TINY = ["--d", "32", "--layers", "2", "--heads", "2", "--seq", "16", "--batch", "4", "--precision", "fp32",
        "--no-compile", "--eval-windows", "4", "--eval-batches", "2", "--log-every", "2"]


@pytest.fixture
def corpus(tmp_path):
    rng = np.random.default_rng(0)
    # structured bytes (a repeating pattern + noise) so that loss can go down; all < 256
    base = np.tile(np.arange(97, 123, dtype=np.uint8), 400)
    noise = rng.integers(0, 256, size=base.size).astype(np.uint8)
    arr = np.where(rng.random(base.size) < 0.9, base, noise).astype(np.uint8)
    tr, va = tmp_path / "train.txt", tmp_path / "valid.txt"
    tr.write_bytes(arr.tobytes())
    va.write_bytes(arr[:3000].tobytes())
    return tr, va


def argv(corpus, out, *extra):
    tr, va = corpus
    return TINY + ["--train-file", str(tr), "--val-file", str(va), "--out", str(out)] + list(extra)


# в”Ђв”Ђ data в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

def test_memmap_windows_correct_and_seeded(tmp_path):
    n, seq = 1000, 16
    raw = (np.arange(n) * 7 % 251).astype(np.uint8)
    p = tmp_path / "f.bin"
    p.write_bytes(raw.tobytes())
    mm = T.open_corpus(p)
    assert isinstance(mm, np.memmap) and mm.dtype == np.uint8 and len(mm) == n
    x, y = T.get_batch(mm, 3, 5, 8, seq, "cpu")
    assert x.shape == y.shape == (8, seq) and x.dtype == torch.long
    st = T.batch_starts(n, 3, 5, 8, seq)
    for i, s in enumerate(st):
        assert x[i].tolist() == raw[s:s + seq].tolist()
        assert y[i].tolist() == raw[s + 1:s + seq + 1].tolist()
    x2, _ = T.get_batch(mm, 3, 5, 8, seq, "cpu")
    assert torch.equal(x, x2)                                       # same (seed, step) -> same batch
    assert not torch.equal(x, T.get_batch(mm, 3, 6, 8, seq, "cpu")[0])
    assert not torch.equal(x, T.get_batch(mm, 4, 5, 8, seq, "cpu")[0])
    for step in range(200):                                         # never reads past the end
        s = T.batch_starts(n, 0, step, 64, seq)
        assert s.min() >= 0 and s.max() + seq + 1 <= n
    tight = T.batch_starts(seq + 2, 0, 0, 50, seq)                  # smallest legal file: starts 0 and 1 only
    assert set(tight.tolist()) <= {0, 1}
    vs = T.val_starts(n, seq, 4, 8)
    assert vs.shape == (4, 8) and np.array_equal(vs, T.val_starts(n, seq, 4, 8))


# в”Ђв”Ђ atomic download в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

def test_ensure_file_atomic_and_cleanup(tmp_path):
    dest = tmp_path / "d" / "x.txt"
    calls = []

    def fetch(url, part):
        calls.append(url)
        assert Path(part).name == "x.txt.part"
        Path(part).write_bytes(b"hello")
        assert not dest.exists()                                    # final path appears only via the rename
    assert T.ensure_file(dest, "u", fetch) == dest
    assert dest.read_bytes() == b"hello" and calls == ["u"]
    assert sorted(p.name for p in dest.parent.iterdir()) == ["x.txt"]       # no .part / .lock left
    T.ensure_file(dest, "u", fetch)
    assert calls == ["u"]                                           # existing file: no fetch


def test_ensure_file_failure_leaves_nothing(tmp_path):
    dest = tmp_path / "x.txt"

    def bad(url, part):
        Path(part).write_bytes(b"par")
        raise OSError("boom")
    with pytest.raises(OSError):
        T.ensure_file(dest, "u", bad)
    assert not dest.exists() and list(tmp_path.iterdir()) == []


def test_ensure_file_concurrent_single_download(tmp_path):
    dest = tmp_path / "x.txt"
    calls, errs = [], []

    def fetch(url, part):
        calls.append(1)
        time.sleep(0.3)
        Path(part).write_bytes(b"data")

    def worker():
        try:
            T.ensure_file(dest, "u", fetch, poll=0.02)
            assert dest.read_bytes() == b"data"
        except BaseException as e:                                   # noqa: BLE001
            errs.append(e)
    ts = [threading.Thread(target=worker) for _ in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs and len(calls) == 1
    assert [p.name for p in tmp_path.iterdir()] == ["x.txt"]


def test_ensure_file_breaks_stale_lock(tmp_path):
    dest = tmp_path / "x.txt"
    lock = tmp_path / "x.txt.lock"
    lock.write_text("dead")
    old = time.time() - 10_000
    import os
    os.utime(lock, (old, old))

    def fetch(url, part):
        Path(part).write_bytes(b"ok")
    T.ensure_file(dest, "u", fetch, poll=0.01, stale_s=60)
    assert dest.read_bytes() == b"ok" and not lock.exists()


# в”Ђв”Ђ checkpoint / resume в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

def _states_equal(a, b):
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k


@pytest.mark.parametrize("arm", ["dense", pytest.param("lg6_fp4_h2", marks=needs_ops)])
def test_resume_is_bit_exact(arm, corpus, tmp_path):
    common = ["--arm", arm, "--steps", "8", "--eval-every", "4", "--ckpt-every", "4"]
    full = []
    r = T.run_train(T.build_parser().parse_args(argv(corpus, tmp_path / "A", *common)), record=full)
    assert len(full) == 8 and np.isfinite(full).all() and not r["diverged"]
    pre = []                                                         # "preemption" after step 4, then resume
    T.run_train(T.build_parser().parse_args(argv(corpus, tmp_path / "B", *common, "--stop-after", "4")), record=pre)
    assert pre == full[:4]
    ck4 = torch.load(tmp_path / "B" / "ckpt.pt", map_location="cpu", weights_only=True)
    assert ck4["step"] == 4
    if arm.startswith("lg"):
        assert any(k.endswith("v_rms") for k in ck4["model"]) and any(k.endswith(".idx") for k in ck4["model"])
        assert any(k.endswith("_scale") for k in ck4["model"])
    post = []
    T.run_train(T.build_parser().parse_args(argv(corpus, tmp_path / "B", *common, "--resume")), record=post)
    assert post == full[4:], (post, full[4:])                        # exactly the same losses, bit for bit
    a = torch.load(tmp_path / "A" / "ckpt.pt", map_location="cpu", weights_only=True)
    b = torch.load(tmp_path / "B" / "ckpt.pt", map_location="cpu", weights_only=True)
    assert a["step"] == b["step"] == 8
    _states_equal(a["model"], b["model"])
    assert not list((tmp_path / "B").glob("*.tmp"))                 # atomic write leaves no temp file
    lines = [__import__("json").loads(s) for s in (tmp_path / "B" / "train.jsonl").read_text().splitlines()]
    assert any(d["type"] == "eval" for d in lines) and any(d["type"] == "train" for d in lines)
    assert (tmp_path / "A" / "result.json").exists()


@needs_ops
@pytest.mark.parametrize("mdt", ["bf16", "fp8"])
def test_momentum_resume_is_bit_exact(mdt, corpus, tmp_path):
    common = ["--arm", "lg6_fp4_h2", "--steps", "8", "--eval-every", "4", "--ckpt-every", "4",
              "--momentum", "0.9", "--momentum-dtype", mdt, "--lr", "0.03"]
    full = []
    T.run_train(T.build_parser().parse_args(argv(corpus, tmp_path / "A", *common)), record=full)
    assert len(full) == 8 and np.isfinite(full).all()
    T.run_train(T.build_parser().parse_args(argv(corpus, tmp_path / "B", *common, "--stop-after", "4")))
    ck4 = torch.load(tmp_path / "B" / "ckpt.pt", map_location="cpu", weights_only=True)
    assert "lg_momentum" in ck4 and any(b.float().abs().sum() > 0 for b, _ in ck4["lg_momentum"])
    post = []
    T.run_train(T.build_parser().parse_args(argv(corpus, tmp_path / "B", *common, "--resume")), record=post)
    assert post == full[4:], (post, full[4:])
    bad = argv(corpus, tmp_path / "B", "--arm", "lg6_fp4_h2", "--steps", "8", "--eval-every", "4", "--ckpt-every", "4",
               "--lr", "0.03", "--resume")                       # momentum dropped -> refused
    with pytest.raises(ValueError, match="momentum"):
        T.run_train(T.build_parser().parse_args(bad))


def test_resume_rejects_changed_settings(corpus, tmp_path):
    common = ["--arm", "dense", "--steps", "4", "--ckpt-every", "2", "--eval-every", "100"]
    T.run_train(T.build_parser().parse_args(argv(corpus, tmp_path / "A", *common, "--stop-after", "2")))
    bad = argv(corpus, tmp_path / "A", "--arm", "dense", "--steps", "6", "--ckpt-every", "2", "--eval-every", "100", "--resume")
    with pytest.raises(ValueError, match="different settings"):
        T.run_train(T.build_parser().parse_args(bad))


@needs_ops
def test_lg_and_dense_train_and_loss_decreases(corpus, tmp_path):
    for arm, extra in (("dense", ["--adam-lr", "3e-3"]), ("lg6_fp4_h2", ["--adam-lr", "3e-3", "--lr", "0.02"])):
        rec = []
        T.run_train(T.build_parser().parse_args(argv(corpus, tmp_path / arm, "--arm", arm, "--steps", "40",
                                                      "--eval-every", "40", "--ckpt-every", "40", *extra)), record=rec)
        assert np.mean(rec[-5:]) < np.mean(rec[:5]), (arm, rec)


# в”Ђв”Ђ probe modes в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

def test_lr_probe_dense_and_lg(corpus, tmp_path, capsys):
    res = T.run_lr_probe(T.build_parser().parse_args(argv(
        corpus, tmp_path / "o", "--lr-probe", "--arm", "dense", "--probe-steps", "6", "--lrs", "1e-4,3e-3")))
    assert [r["lr"] for r in res["rows"]] == [1e-4, 3e-3] and res["best"] in (1e-4, 3e-3)
    assert all(np.isfinite(r["val_bpb"]) for r in res["rows"])
    assert "BEST: --adam-lr" in capsys.readouterr().out
    assert (tmp_path / "o" / "lr_probe_dense.json").exists()
    if LG.HAS_LG_OPS:
        res = T.run_lr_probe(T.build_parser().parse_args(argv(
            corpus, tmp_path / "o", "--lr-probe", "--arm", "lg6_fp4_h2", "--probe-steps", "6", "--lrs", "0.003,0.02")))
        assert res["flag"] == "--lr" and res["adam_lr_fixed"] == 6e-4
        assert "BEST: --lr" in capsys.readouterr().out


def test_probe_batch_runs(corpus, tmp_path, capsys):
    arms = "dense,lg6_fp4_h2" if LG.HAS_LG_OPS else "dense"
    res = T.run_probe_batch(T.build_parser().parse_args(argv(
        corpus, tmp_path / "o", "--probe-batch", "--arm", arms, "--batches", "2,4")))
    assert res["largest_common_batch"] == 4
    assert all(r["ok"] and r["ms_step"] > 0 for r in res["results"].values())
    assert "largest batch that fits ALL arms: 4" in capsys.readouterr().out
    assert (tmp_path / "o" / "probe_batch.json").exists()


def test_prepare_data_with_mock_fetch(tmp_path, capsys):
    fetched = []

    def fake(url, part):
        fetched.append(url.rsplit("/", 1)[1])
        Path(part).write_bytes(b"Once upon a time.\n" * 10)
    args = T.build_parser().parse_args(["--prepare-data", "--data-dir", str(tmp_path / "data")])
    T.ensure_data(args, fake)
    assert sorted(fetched) == sorted(T.DATA_FILES.values())
    assert all(u for u in fetched) and (tmp_path / "data" / T.DATA_FILES["train"]).exists()
    assert T.run_prepare(args) == 0                                  # files exist now: no network
    out = capsys.readouterr().out
    assert "train:" in out and "valid:" in out
