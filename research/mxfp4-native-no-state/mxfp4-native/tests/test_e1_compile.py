"""e1_quality / lg_ab --compile: lg arms compile fullgraph over the custom ops with ONE graph for the whole cosine schedule,
dense / qat compile plainly, na_* / tern_* are refused with a message; eval is eager and never updates; lg_ab parses the
*_nvfp4fwd arm. CPU, backend aot_eager."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "probes"))

import lg_ab                                                           # noqa: E402
from mx6.latent_grid import HAS_LG_OPS, LatentGridLinear               # noqa: E402

pytest.importorskip("torch._dynamo")
needs_ops = pytest.mark.skipif(not HAS_LG_OPS, reason="custom ops unavailable")


@pytest.fixture
def e1():
    mod = lg_ab.install(lg_ab.load_e1(register=False))
    mod.DEV = "cpu"
    mod.SCALE_FORMAT, mod.HEADROOM_MODE, mod.HEADROOM = "nvfp4", "ladder", 2.0
    mod.SDPA = True
    return mod


def ids(n=4000, vocab=11):
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, vocab, (n,), generator=g)


def run(e1, arm, **kw):
    return e1.run_arm(arm, 0, 64, 6, 4, 32, ids(), 11, 0.01, 3e-3, sched="cosine", eval_every=3, val_ids=ids(1500),
                      blocks=2, heads=2, precision=kw.pop("precision", "bf16"), **kw)


@needs_ops
def test_lg_arm_compiles_to_one_graph_over_a_cosine_schedule_bf16_sdpa(e1):
    r = run(e1, "lg6_fp4_h2", compile=True, compile_backend="aot_eager")
    assert r["compile"]["graphs"] == 1 and r["compile"]["graph_breaks"] == 0 and r["compile"]["kind"] == "lg"
    assert r["val"] == r["val"] and r["curve"][-1][2] != r["curve"][0][2]          # finite, lr really changed between evals


@needs_ops
def test_nvfp4fwd_arm_is_parsed_and_compiles(e1):
    assert lg_ab.parse_lg("lg6_fp4_h2_nvfp4fwd") == (6, 2.0, False) and lg_ab.lg_gemm("lg6_fp4_h2_nvfp4fwd") == "nvfp4_scaled_mm"
    assert lg_ab.lg_gemm("lg6_fp4_h2") == "cublas_bf16"
    lay = e1.make_linear("lg6_fp4_h2_nvfp4fwd", 64, 64, 0.01)
    assert isinstance(lay, LatentGridLinear) and lay.gemm == "nvfp4_scaled_mm"
    r = run(e1, "lg6_fp4_h2_nvfp4fwd", compile=True, compile_backend="aot_eager")      # CPU: the fp4 GEMM falls back to bf16
    assert r["compile"]["graphs"] == 1


def test_dense_arm_compiles(e1):
    r = run(e1, "dense_nobias", compile=True, compile_backend="aot_eager")
    assert r["compile"]["kind"] == "plain" and r["compile"]["graphs"] >= 1 and r["val"] == r["val"]


@pytest.mark.parametrize("arm", ["na_fp4_c8", "tern_counter", "dense_rms"])
def test_unsupported_arms_are_refused_with_a_message(e1, arm):
    assert "not supported" in e1.compile_refusal(arm) and arm in e1.compile_refusal(arm)
    with pytest.raises(ValueError, match="--compile is not supported"):
        e1.compile_model(torch.nn.Linear(2, 2), arm)
    with pytest.raises(SystemExit):                                                  # argparse error from main()
        e1.main(["--arms", arm, "--compile", "--steps", "1"])


@needs_ops
def test_eval_is_eager_and_does_not_update(e1):
    torch.manual_seed(0)
    model, _ = e1.build_model_opt("lg6_fp4_h2", 11, 64, 0.01, 3e-3, 2, 2)
    fwd, _ = e1.compile_model(model, "lg6_fp4_h2", "aot_eager")
    idx0 = [m.idx.clone() for m in model.modules() if isinstance(m, LatentGridLinear)]
    v = e1.evaluate(model, ids(1500), seq=32, iters=2, precision="bf16")
    assert v == v and all(torch.equal(a, m.idx) for a, m in zip(idx0, (m for m in model.modules() if isinstance(m, LatentGridLinear))))
    assert model.training                                                            # evaluate() restores train mode
    # null twin: a training forward/backward through the compiled model DOES update
    x = ids(200)[:64].reshape(2, 32)
    e1.fwd_logits(fwd, x, "bf16").float().square().mean().backward()
    assert any(not torch.equal(a, m.idx) for a, m in zip(idx0, (m for m in model.modules() if isinstance(m, LatentGridLinear))))
