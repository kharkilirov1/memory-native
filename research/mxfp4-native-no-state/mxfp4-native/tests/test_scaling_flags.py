"""Scaling-study plumbing (CPU, tiny models, NO network): byte-level / enwik8 data loading, --layers / --heads,
parameter counts, memory accounting, lg_ab pass-through, and probes/bench_train_step.py."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import zipfile
from pathlib import Path

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


e1 = _load("e1_quality", ROOT / "probes" / "e1_quality.py")
bench = _load("bench_train_step", ROOT / "probes" / "bench_train_step.py")
import lg_ab                                                     # noqa: E402  (probes/lg_ab.py)

VOCAB, D = 24, 32


def _all_bytes_file(path: Path, n: int = 6000) -> bytes:
    """n bytes cycling through every byte value 0..255 (incl. \\r, \\n, 0x80+, which are invalid utf-8)."""
    raw = bytes(i % 256 for i in range(n))
    path.write_bytes(raw)
    return raw


# ---- data: latin-1 (byte-level) loading, encoding, enwik8 download logic -----------------------------------
def test_latin1_is_byte_level_without_newline_translation(tmp_path):
    p = tmp_path / "bytes.bin"
    raw = _all_bytes_file(p)
    text = e1.load_text(p, None, "latin-1")
    assert len(text) == len(raw) and text.encode("latin-1") == raw           # 1 byte = 1 char, \r\n untouched
    assert e1.load_text(p, 100, "latin-1") == raw[:100].decode("latin-1")     # only the head is read
    ids, chars = e1.encode_text(text)
    assert len(chars) == 256 and ids.dtype == torch.long and int(ids.max()) == 255
    assert torch.equal(ids, torch.tensor(list(raw)))                          # sorted code points == byte values


def test_utf8_default_is_the_legacy_text_mode(tmp_path):
    p = tmp_path / "t.txt"
    p.write_bytes("héllo\r\nwörld ☃".encode("utf-8"))
    assert e1.load_text(p) == p.read_text(encoding="utf-8")                   # universal newlines, as before
    assert e1.load_text(p, 3) == p.read_text(encoding="utf-8")[:3]
    assert len(e1.load_text(p, None, "latin-1")) == len(p.read_bytes())       # latin-1 = the raw bytes
    assert e1.load_corpus("tinyshakespeare", p)[1] == "utf-8"
    assert e1.load_corpus("tinyshakespeare", p, None, "latin-1")[1] == "latin-1"


def test_encode_text_equals_the_legacy_comprehension():
    text = "héllo wörld ☃ \U0001F600 abc\nxyz" * 5
    chars = sorted(set(text))
    legacy = torch.tensor([{c: i for i, c in enumerate(chars)}[c] for c in text], dtype=torch.long)
    ids, got = e1.encode_text(text)
    assert got == chars and torch.equal(ids, legacy)
    with pytest.raises(ValueError):
        e1.encode_text("")


def _zip_fetch(payload: bytes, calls: list, fail_first: bool = False):
    def fetch(url, dest):
        calls.append(url)
        if fail_first and len(calls) == 1:
            raise OSError("primary mirror down")
        with zipfile.ZipFile(dest, "w") as zf:
            zf.writestr("enwik8", payload)
    return fetch


def test_ensure_enwik8_downloads_unzips_and_falls_back_to_the_mirror(tmp_path, capsys):
    payload = bytes(range(256)) * 10
    calls: list = []
    path = tmp_path / "data" / "enwik8"
    out = e1.ensure_enwik8(path, urls=("http://a/enwik8.zip", "https://b/enwik8.zip"),
                           fetch=_zip_fetch(payload, calls, fail_first=True))
    assert out == path and path.read_bytes() == payload
    assert calls == ["http://a/enwik8.zip", "https://b/enwik8.zip"]            # primary failed -> mirror
    assert sorted(x.name for x in path.parent.iterdir()) == ["enwik8"]          # zip and .part cleaned up
    assert "expected 100000000" in capsys.readouterr().out                      # wrong size is reported, not fatal
    e1.ensure_enwik8(path, fetch=lambda *a: pytest.fail("existing file must not be downloaded again"))


def test_ensure_enwik8_all_mirrors_fail_and_bad_zip(tmp_path):
    def bad(url, dest):
        Path(dest).write_bytes(b"this is not a zip")
    with pytest.raises(RuntimeError, match="could not download enwik8") as ei:
        e1.ensure_enwik8(tmp_path / "enwik8", urls=("http://a", "http://b"), fetch=bad)
    assert "BadZipFile" in str(ei.value) and not list(tmp_path.iterdir())       # nothing half-written is left behind


def test_default_urls_and_dataset_paths():
    assert e1.ENWIK8_URLS == ("http://mattmahoney.net/dc/enwik8.zip", "https://data.deepai.org/enwik8.zip")
    assert e1.default_data_path("enwik8").name == "enwik8" and e1.default_data_path("enwik8").parent.name == "data"
    assert e1.default_data_path("tinyshakespeare").name == "tinyshakespeare.txt"
    with pytest.raises(ValueError):
        e1.load_corpus("nope", Path("x"))


# ---- model size: --layers / --heads, parameter counts -------------------------------------------------------
def test_single_head_is_the_legacy_attention_and_multi_head_matches_sdpa():
    torch.manual_seed(0)
    blk = e1.Block("dense_nobias", D, 0.01, heads=1)
    x = torch.randn(2, 8, D)
    h = blk.ln1(x)
    q, k, v = blk.qkv(h).chunk(3, dim=-1)
    att = ((q @ k.transpose(-2, -1)) / math.sqrt(D)).masked_fill(torch.triu(torch.ones(8, 8, dtype=torch.bool), 1),
                                                                   float("-inf")).softmax(-1)
    y = x + blk.proj(att @ v)
    ref = y + blk.out(torch.nn.functional.gelu(blk.fc(blk.ln2(y))))
    assert torch.equal(blk(x), ref)                                              # the exact legacy op sequence
    mh = e1.Block("dense_nobias", D, 0.01, heads=4)
    mh.load_state_dict(blk.state_dict())
    q, k, v = (z.reshape(2, 8, 4, D // 4).transpose(1, 2) for z in mh.qkv(mh.ln1(x)).chunk(3, dim=-1))
    sdpa = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(2, 8, D)
    y = x + mh.proj(sdpa)
    assert torch.allclose(mh(x), y + mh.out(torch.nn.functional.gelu(mh.fc(mh.ln2(y)))), atol=1e-5)
    assert not torch.allclose(mh(x), blk(x), atol=1e-3)                          # and it IS a different model
    with pytest.raises(ValueError):
        e1.Block("dense_nobias", D, 0.01, heads=3)


@pytest.mark.parametrize("d", [128, 256, 512])
@pytest.mark.parametrize("blocks", [2, 4])
def test_param_counts_formula(d, blocks):
    vocab = 200
    pc = e1.analytic_params(d, blocks, vocab)
    assert pc["lin"] == blocks * 12 * d * d                                       # qkv 3d^2 + proj d^2 + fc 4d^2 + out 4d^2
    assert pc["other"] == 2 * vocab * d + 512 * d + (2 * blocks + 1) * 2 * d
    assert pc["total"] == pc["lin"] + pc["other"]
    if d == 128:                                                                  # actual models agree (bias-free arms)
        for arm in ("dense_nobias", "na_e2m1_c5", "qat_fp4"):
            assert e1.count_params(e1.CharLM(arm, vocab, d=d, blocks=blocks, heads=2)) == pc
        assert e1.count_params(e1.CharLM("dense_bf16", vocab, d=d, blocks=blocks))["other"] == pc["other"] + 9 * d * blocks


def test_param_counts_known_values():
    assert [e1.analytic_params(d, 2, 256)["lin"] for d in (128, 256, 512)] == [393_216, 1_572_864, 6_291_456]
    assert [e1.analytic_params(d, 4, 256)["lin"] for d in (128, 256, 512)] == [786_432, 3_145_728, 12_582_912]


def test_memory_accounting_formulas():
    m = e1.CharLM("dense_nobias", VOCAB, d=D, blocks=2)
    a = e1.memory_accounting(m, "dense_nobias", e1.measured_bits(m, "dense_nobias"))
    n = 2 * 12 * D * D
    assert a["n_lin"] == n and a["state_bytes"] == 12 * n and a["grad_bytes"] == 4 * n     # AdamW: all grads resident
    assert a["other_state_bytes"] == 12 * a["n_other"]
    m = e1.CharLM("na_e2m1_c5", VOCAB, d=D, blocks=2)
    bits = e1.measured_bits(m, "na_e2m1_c5")
    a = e1.memory_accounting(m, "na_e2m1_c5", bits)
    assert a["state_bytes"] == pytest.approx(bits / 8 * n) and a["grad_bytes"] == 4 * 4 * D * D   # one layer (fc / out)
    assert a["state_bytes"] < 0.1 * 12 * n


def test_run_arm_reports_params_memory_and_takes_layers_heads():
    ids = torch.randint(0, VOCAB, (3000,), generator=torch.Generator().manual_seed(0))
    r = e1.run_arm("dense_nobias", 0, D, 2, 4, 16, ids, VOCAB, lr=0.01, adam_lr=3e-3, eval_every=1, blocks=1, heads=2)
    assert r["n_lin"] == 12 * D * D and r["n_total"] == e1.analytic_params(D, 1, VOCAB)["total"]
    assert r["state_mb"] == pytest.approx(12 * 12 * D * D / 2**20)
    assert r["state_grad_mb"] == pytest.approx(16 * 12 * D * D / 2**20)
    assert math.isnan(r["peak_mem_mb"]) and math.isnan(r["peak_mem_delta_mb"])             # CPU
    r2 = e1.run_arm("dense_nobias", 0, D, 2, 4, 16, ids, VOCAB, lr=0.01, adam_lr=3e-3, eval_every=1)   # defaults
    assert r2["n_lin"] == 2 * 12 * D * D                                                   # 2 blocks, 1 head
    assert r["val"] != r2["val"]


def test_run_arm_cuda_memory_bookkeeping_with_faked_cuda_calls(monkeypatch):
    """The CUDA-only branch of run_arm (never exercised on a CPU box) with torch.cuda faked: peak is the max over the
    eval windows, minus the int8 flip/edge snapshot (1 B per counter weight), and the stats are reset at the right times."""
    calls, peaks = [], iter([5_000_000, 9_000_000])
    monkeypatch.setattr(e1, "_cuda_mem", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: calls.append("sync"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: calls.append("reset"))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 1000)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: next(peaks))
    ids = torch.randint(0, VOCAB, (3000,), generator=torch.Generator().manual_seed(0))
    r = e1.run_arm("na_e2m1_c5", 0, D, 2, 4, 16, ids, VOCAB, lr=0.01, adam_lr=3e-3, eval_every=1, blocks=1)
    snap = 12 * D * D                                            # int8 snapshot of the 4 block linears, bytes
    assert r["peak_mem_mb"] == pytest.approx((9_000_000 - snap) / 2**20)
    assert r["peak_mem_delta_mb"] == pytest.approx((9_000_000 - snap - 1000) / 2**20)
    assert calls == ["sync", "reset", "reset", "reset", "reset"]    # run start, after the initial snapshot, after each eval


# ---- CLI --------------------------------------------------------------------------------------------------
def _main_args(path, *extra):
    return ["--d", "32", "--steps", "2", "--batch", "2", "--seq", "16", "--seeds", "0", "--eval-every", "1",
            "--data", str(path), *extra]


def test_main_enwik8_byte_level_header_and_tables(tmp_path, capsys):
    p = tmp_path / "enwik8"
    _all_bytes_file(p, 30000)                                  # exists -> nothing is downloaded
    assert e1.main(_main_args(p, "--dataset", "enwik8", "--chars", "20000", "--layers", "1", "--heads", "2",
                              "--arms", "dense_nobias,na_e2m1_c5")) == 0
    out = capsys.readouterr().out
    head = out.split("\n  ")[0]
    assert "vocab=256" in head and "tokens=20000" in head and "epochs=" in head
    assert "dataset=enwik8 encoding=latin-1 (byte-level)" in out and "chars=20000 (--chars 20000)" in out
    pc = e1.analytic_params(32, 1, 256)
    assert f"model: d=32 layers=1 heads=2 head_dim=16 mlp=4d seq=16 | params total={pc['total']:,} " \
           f"block_linears(counter layers)={pc['lin']:,} other(emb/pos/LN/head)={pc['other']:,}" in out
    assert "memory per arm, MB" in out and "      memory: params total=" in out
    assert out.count("dense_nobias") >= 3


def test_main_defaults_unchanged_and_data_encoding_override(tmp_path, capsys):
    p = tmp_path / "t.txt"
    p.write_bytes(("héllo wörld ☃ " * 400).encode("utf-8"))
    base = _main_args(p, "--arms", "dense_nobias")
    e1.main(base)
    out_utf8 = capsys.readouterr().out
    e1.main(base + ["--data-encoding", "latin-1"])
    out_l1 = capsys.readouterr().out
    assert "encoding=utf-8" in out_utf8 and "model: d=32 layers=2 heads=1 head_dim=32" in out_utf8
    v_utf8 = int(out_utf8.split("vocab=")[1].split()[0])
    v_l1 = int(out_l1.split("vocab=")[1].split()[0])
    assert v_l1 > v_utf8 and "tokens=" in out_l1                  # bytes: the multi-byte chars become several symbols


def test_main_rejects_bad_heads_and_seq(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("abcdefgh" * 400, encoding="utf-8")
    for bad in (["--heads", "3"], ["--heads", "0"], ["--layers", "0"], ["--seq", "1024"]):
        with pytest.raises(SystemExit):
            e1.main(_main_args(p, "--arms", "dense_nobias", *bad))


def test_main_forwards_layers_heads_to_run_arm(tmp_path, monkeypatch):
    p = tmp_path / "t.txt"
    p.write_text("abcdefgh" * 400, encoding="utf-8")
    seen = {}

    def fake_run_arm(arm, seed, d, steps, batch, seq, ids, vocab, lr, adam_lr, **kw):
        seen.update(kw)
        return {"arm": arm, "seed": seed, "bits": 1.0, "val": 1.0, "sec": 0.0, "hist": None, "clip": float("nan"),
                "grown": 0, "curve": [(1, 1.0, lr, 0.0, 0.0, float("nan"))], "best": 1.0, "flip": (0.0, 0.0, 0.0),
                "edge": 0.0, "train": 1.0, "gap": 0.0}

    monkeypatch.setattr(e1, "run_arm", fake_run_arm)
    e1.main(_main_args(p, "--arms", "dense_nobias", "--layers", "3", "--heads", "4"))
    assert seen["blocks"] == 3 and seen["heads"] == 4


def test_lg_ab_passes_the_new_flags_through(tmp_path, capsys):
    p = tmp_path / "enwik8"
    _all_bytes_file(p, 20000)
    assert lg_ab.main(_main_args(p, "--dataset", "enwik8", "--layers", "1", "--heads", "2", "--scale-format", "nvfp4",
                                 "--arms", "lg6_fp4_h2,na_fp4_c8", "--lr", "0.01")) == 0
    out = capsys.readouterr().out
    assert "encoding=latin-1 (byte-level)" in out and "layers=1 heads=2" in out
    assert "lg_ab: lg6_fp4_h2 -> LatentGridLinear" in out and "memory per arm, MB" in out
    assert " lg6_fp4_h2" in out and "na_fp4_c8" in out


# ---- probes/bench_train_step.py ---------------------------------------------------------------------------
def _bench(tmp_path, *extra):
    out = tmp_path / "b" / "bench"
    rc = bench.main(["--d", "32", "--steps", "2", "--warmup", "1", "--layers", "1", "--phase-steps", "2",
                     "--out", str(out), *extra])
    return rc, json.loads(Path(str(out) + ".json").read_text()), Path(str(out) + ".md").read_text()


def test_bench_cpu_plumbing_all_arm_kinds_json_and_markdown(tmp_path):
    arms = ["dense_nobias", "qat_fp4", "na_fp4_c8", "lg6_fp4_h2"]
    rc, js, md = _bench(tmp_path, "--arms", ",".join(arms), "--fast-pack")
    assert rc == 0 and js["meta"]["arms"] == arms
    assert "NaN" not in Path(str(tmp_path / "b" / "bench") + ".json").read_text()          # strict JSON (nan -> null)
    res = js["results"]
    assert [r["label"] for r in res] == arms[:3] + ["na_fp4_c8+fastpack", "lg6_fp4_h2"]
    for r in res:
        assert r["error"] is None, r
        assert r["finite"] and r["ms_step"] > 0 and r["tokens_per_s"] > 0 and r["n_lin"] == 12 * 32 * 32
        assert r["fwd_ms"] > 0 and r["bwd_ms"] > 0 and r["opt_ms"] >= 0
        assert r["profile"]["top"] and r["profile"]["unit"] == "cpu_ops"
    byarm = {r["label"]: r for r in res}
    assert byarm["na_fp4_c8"]["update_ms"] > 0 and byarm["lg6_fp4_h2"]["update_ms"] > 0       # apply_update_ was timed
    assert byarm["dense_nobias"]["update_ms"] is None and byarm["qat_fp4"]["decode_ms"] > 0
    assert byarm["dense_nobias"]["state_mb"] == pytest.approx(12 * 12 * 32 * 32 / 2**20)
    assert "| arm | d | heads |" in md and "na_fp4_c8+fastpack" in md and "#### lg6_fp4_h2 d=32" in md


def test_bench_is_fail_soft_per_arm_and_reports_missing_compile_paths(tmp_path):
    rc, js, md = _bench(tmp_path, "--arms", "bogus_arm,dense_nobias", "--no-profile", "--phase-steps", "0")
    assert rc == 0                                                        # one arm survived -> exit 0
    bad, good = js["results"]
    assert bad["arm"] == "bogus_arm" and "ValueError" in bad["error"] and good["error"] is None
    assert "FAILED" in md
    ns = bench.argparse.Namespace(lr=0.01, tern_lr=3e-2, adam_lr=3e-3, headroom=2.0, layers=1, heads=0, batch=2,
                                  seq=16, warmup=1, steps=1, phase_steps=0, no_profile=True, topk=3, fused="off",
                                  seed=0)
    e1b = lg_ab.install(lg_ab.load_e1(register=False))
    e1b.DEV = "cpu"
    r = bench.bench_one("tern_counter", 32, "compile", ns, e1b, 64, "cpu")
    assert "no compile path" in r["error"]
    r = bench.bench_one("lg6_fp4_h2", 32, "fastpack", ns, e1b, 64, "cpu")
    assert "only applies to na_" in r["error"]


def test_bench_fast_pack_is_bit_exact_inside_a_real_update():
    e1b = lg_ab.install(lg_ab.load_e1(register=False))
    e1b.DEV, e1b.SCALE_FORMAT, e1b.HEADROOM_MODE, e1b.HEADROOM = "cpu", "nvfp4", "ladder", 2.0

    def run(fast: bool):
        torch.manual_seed(3)
        cm = bench.fast_pack_hook() if fast else bench.contextlib.nullcontext()
        with cm:
            lin = e1b.make_linear("na_fp4_c8", 64, 96, 0.05)
            for step in range(3):
                x = torch.randn(8, 64, requires_grad=True)
                lin(x).square().mean().backward()
            return lin._state.clone(), lin.codes.clone()

    (s0, c0), (s1, c1) = run(False), run(True)
    assert torch.equal(s0, s1) and torch.equal(c0, c1) and s0.dtype == torch.uint8
    from mx6.packed_na import PackedNA
    assert PackedNA.pack_state.__name__ == "pack_state"                  # the stock functions are restored afterwards


def test_summarize_events_picks_device_kernels_only_and_handles_old_attribute_names():
    from types import SimpleNamespace as NS
    DT = torch.autograd.DeviceType
    rows = [NS(key="sm80_xmma_gemm_f32f32", count=4, device_type=DT.CUDA, self_device_time_total=6000.0),
            NS(key="void at::native::vectorized_elementwise_kernel<4>", count=30, device_type=DT.CUDA,
               self_device_time_total=1500.0),
            NS(key="void at::native::distribution_elementwise_grid_stride_kernel", count=8, device_type=DT.CUDA,
               self_device_time_total=500.0),
            NS(key="old_torch_kernel", count=1, device_type=DT.CUDA, self_cuda_time_total=250.0),   # pre-2.4 attribute
            NS(key="aten::mm", count=4, device_type=DT.CPU, self_device_time_total=0.0, self_cpu_time_total=900.0)]
    s = bench.summarize_events(rows, True, 3)
    assert s["unit"] == "cuda_kernels" and s["launches"] == 4 + 30 + 8 + 1 and len(s["top"]) == 3
    assert s["top"][0]["name"].startswith("sm80_xmma_gemm") and s["top"][0]["ms"] == pytest.approx(6.0)
    assert s["total_ms"] == pytest.approx(8.25) and s["top"][0]["pct"] == pytest.approx(100 * 6.0 / 8.25)
    assert list(s["categories"])[:3] == ["gemm", "elementwise", "rng"] and "aten::mm" not in str(s["top"])
    c = bench.summarize_events(rows, False, 2)                       # CPU: ops by self CPU time
    assert c["unit"] == "cpu_ops" and c["top"][0]["name"] == "aten::mm"
    # a torch that labels kernels differently: no CUDA-typed rows -> second chance without the device filter
    odd = [NS(key="k1", count=2, device_type=DT.CPU, self_device_time_total=400.0)]
    assert bench.summarize_events(odd, True, 5)["top"][0]["name"] == "k1"


def test_bench_one_cuda_branch_with_faked_events_and_memory(monkeypatch):
    """The CUDA code path of bench_one (events, memory, phases, spans, profiler errors are fail-soft) on a CPU box."""
    class FakeEvent:
        clock = 0.0

        def __init__(self, enable_timing=False):
            self.ts = None

        def record(self, *a):
            FakeEvent.clock += 1.0
            self.ts = FakeEvent.clock

        def elapsed_time(self, other):
            return other.ts - self.ts

    mem = {"alloc": 4096, "peak": 4096 + 123_456}
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: mem["alloc"])
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: mem["peak"])
    e1b = lg_ab.install(lg_ab.load_e1(register=False))
    e1b.DEV, e1b.SCALE_FORMAT, e1b.HEADROOM_MODE, e1b.HEADROOM = "cpu", "nvfp4", "ladder", 2.0
    ns = bench.argparse.Namespace(lr=0.01, tern_lr=3e-2, adam_lr=3e-3, headroom=2.0, layers=1, heads=0, batch=2, seq=16,
                                  warmup=1, steps=3, phase_steps=2, no_profile=True, topk=3, fused="off", seed=0)
    for arm in ("dense_nobias", "na_fp4_c8", "lg6_fp4_h2"):
        r = bench.bench_one(arm, 32, "eager", ns, e1b, 64, "cpu", cuda=True)
        assert r["error"] is None, r.get("traceback")
        assert r["peak_mem_mb"] == pytest.approx(123_456 / 2**20) and r["resident_mb"] == pytest.approx(0.0)
        assert r["ms_step"] > 0 and r["ms_step_ev_mean"] == pytest.approx(1.0)       # fake events tick 1 per record
        assert r["fwd_ms"] >= 1.0 and r["bwd_ms"] >= 1.0 and r["opt_ms"] >= 1.0      # +1 per wrapped span call
    assert r["update_ms"] > 0 and r["decode_ms"] > 0                                  # spans use the same clock


def test_spans_wrapper_restores_methods_and_classify_kernel():
    m = e1.CharLM("lg6_fp4_h2" if False else "dense_nobias", VOCAB, d=D, blocks=1)
    clock = bench.Clock(False)
    sp = bench.Spans(clock)
    lin = m.blocks[0].qkv
    assert sp.wrap(lin, "forward", "fwd") and "forward" in lin.__dict__
    lin(torch.randn(2, D))
    assert sp.total_ms("fwd") > 0
    sp.restore()
    assert "forward" not in lin.__dict__ and not sp.wrap(lin, "weight", "x")        # Parameter: not callable, not wrapped
    cls = bench.classify_kernel
    assert cls("sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize64x64x8") == "gemm"
    assert cls("void at::native::vectorized_elementwise_kernel<4, ...>") == "elementwise"
    assert cls("triton_poi_fused_add_mul_0") == "triton" and cls("void at::native::(anonymous namespace)::distribution_elementwise_grid_stride_kernel") == "rng"
    assert cls("void at::native::index_elementwise_kernel<128, 4>") == "index/gather"
