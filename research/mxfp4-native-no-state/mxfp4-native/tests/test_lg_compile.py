"""Regional torch.compile of models with LatentGridLinear (mx6/latent_grid.py: torch.ops.mx6.lg_linear[_bwd],
prepare_lg_for_compile) and --sdpa attention parity.

CPU (backend `aot_eager`: dynamo + AOTAutograd + functionalization, no C++ compiler needed): the op path is bit-identical
to the autograd.Function path in eager; a compiled model has ONE graph and ZERO graph breaks (fullgraph=True would raise),
and the in-backward update runs EXACTLY once per step (index / RMS state bit-equal to eager after several steps; a missing or
doubled update would change them). GPU tests (inductor, Triton update kernel; skipped without CUDA + Triton) additionally count
the real update-kernel launches per step and compare the loss with eager within bf16 tolerance.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mx6.latent_grid as LG                                           # noqa: E402
from mx6 import lg_update_kernel as K                                  # noqa: E402
from mx6.latent_grid import LatentGridLinear, prepare_lg_for_compile   # noqa: E402

GPU = pytest.mark.skipif(not (torch.cuda.is_available() and K.HAS_TRITON), reason="needs CUDA + Triton")
needs_ops = pytest.mark.skipif(not LG.HAS_LG_OPS, reason="torch.library.register_autograd / register_fake missing")
dynamo = pytest.importorskip("torch._dynamo")


class Net(nn.Module):
    def __init__(self, dev="cpu", native=True):
        super().__init__()
        self.emb = nn.Embedding(20, 64)
        self.ln = nn.LayerNorm(64)
        self.a = LatentGridLinear(64, 128, k=8, lr=0.05, native=native)
        self.b = LatentGridLinear(128, 64, k=8, lr=0.05, native=native)
        self.head = nn.Linear(64, 20, bias=False)
        self.to(dev)

    def forward(self, t):
        h = self.emb(t)
        h = h + self.b(F.gelu(self.a(self.ln(h))))
        return self.head(h)


def train(compiled, mode="op", backend="aot_eager", steps=3, dev="cpu", use_op=None, dtype=None):
    torch.manual_seed(0)
    net = Net(dev)
    if compiled:
        prepare_lg_for_compile(net, mode)
        f = torch.compile(net, backend=backend, fullgraph=(mode == "op"), dynamic=False)
    else:
        if use_op:
            for m in (net.a, net.b):
                m.use_custom_op = True
        f = net
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    torch.manual_seed(3)
    losses = []
    for _ in range(steps):
        t = torch.randint(0, 20, (4, 8), device=dev)
        if dtype is None:
            logits = f(t)
        else:
            with torch.autocast(dev, dtype=dtype):
                logits = f(t)
        loss = F.cross_entropy(logits.float().reshape(-1, 20), t.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return net, losses


@needs_ops
def test_op_path_is_bit_identical_to_function_path_in_eager():
    ne, le = train(False, use_op=False)
    no, lo = train(False, use_op=True)
    assert le == lo
    for l1, l2 in ((ne.a, no.a), (ne.b, no.b)):
        assert torch.equal(l1.idx, l2.idx) and torch.equal(l1.v_rms, l2.v_rms)
    assert not torch.equal(Net().a.idx, ne.a.idx)                              # null twin: the update really moved indices
    assert float(no.a.last_clip_frac) == pytest.approx(float(ne.a.last_clip_frac))


@needs_ops
def test_compiled_one_graph_no_breaks_and_exactly_one_update_per_step():
    dynamo.reset()
    dynamo.utils.counters.clear()
    ne, le = train(False)
    nc, lc = train(True, "op")                                                 # fullgraph=True: any break would raise
    c = dynamo.utils.counters
    assert sum(c["graph_break"].values()) == 0 and c["stats"]["unique_graphs"] >= 1
    assert le == lc
    for l1, l2 in ((ne.a, nc.a), (ne.b, nc.b)):
        assert torch.equal(l1.idx, l2.idx) and torch.equal(l1.v_rms, l2.v_rms), "update count differs from eager"
    init = Net()
    assert (nc.a.idx != init.a.idx).any()                                      # null twin: not a no-op


@needs_ops
def test_update_runs_even_if_input_needs_no_grad_and_only_in_training_grad_mode():
    torch.manual_seed(0)
    lay = LatentGridLinear(64, 32, k=8, lr=0.05, native=True)
    lay.use_custom_op = True
    idx0 = lay.idx.clone()
    x = torch.randn(10, 64)                                                    # requires_grad = False
    with torch.no_grad():
        lay(x)
    lay.eval()
    lay(x)
    assert torch.equal(lay.idx, idx0)                                          # no update outside training grad mode
    lay.train()
    lay(x).square().sum().backward()                                           # the token makes autograd see the layer
    assert not torch.equal(lay.idx, idx0)


@needs_ops
def test_compiled_layer_without_input_grad_still_updates():
    torch.manual_seed(0)
    lay = LatentGridLinear(64, 32, k=8, lr=0.05, native=True)
    prepare_lg_for_compile(lay, "op")
    f = torch.compile(lay, backend="aot_eager", fullgraph=True, dynamic=False)
    idx0 = lay.idx.clone()
    f(torch.randn(10, 64)).square().sum().backward()
    assert not torch.equal(lay.idx, idx0)
    assert lay.last_clip_frac is not None


def test_disable_mode_has_graph_breaks_but_same_numbers():
    dynamo.reset()
    dynamo.utils.counters.clear()
    ne, le = train(False)
    nd, ld = train(True, "disable")
    assert dynamo.utils.counters["graph_break"], "expected breaks around the disabled lg forwards"
    assert le == ld and torch.equal(ne.a.idx, nd.a.idx)


def test_prepare_rejects_unknown_mode():
    with pytest.raises(ValueError):
        prepare_lg_for_compile(Net(), "nope")


@GPU
@needs_ops
def test_gpu_inductor_compile_matches_eager_and_launches_one_update_per_layer_per_step(monkeypatch):
    calls = []
    orig = LG.lg_update_
    monkeypatch.setattr(LG, "lg_update_", lambda *a, **k: (calls.append(1), orig(*a, **k))[1])

    idx_init = []

    def run(compiled):
        torch.manual_seed(0)
        net = Net("cuda", native="auto")
        assert net.a.native_enabled()
        idx_init.append(net.a.idx.clone())
        if compiled:
            prepare_lg_for_compile(net, "op")
        f = torch.compile(net, fullgraph=True, dynamic=False) if compiled else net
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        torch.manual_seed(3)
        out = []
        for _ in range(4):
            t = torch.randint(0, 20, (16, 32), device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = f(t)
            loss = F.cross_entropy(logits.float().reshape(-1, 20), t.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            out.append(float(loss.detach()))
        return net, out

    ne, le = run(False)
    n_eager = len(calls)
    calls.clear()
    nc, lc = run(True)
    assert n_eager == 8 and len(calls) == 8, (n_eager, len(calls))             # 2 layers x 4 steps, never 0 / 2x
    assert lc[0] == pytest.approx(le[0], rel=3e-2) and all(abs(a - b) < 0.3 for a, b in zip(le, lc))
    assert (nc.a.idx != idx_init[1]).any()                                     # null twin: the update moved something


@GPU
@needs_ops
def test_gpu_op_path_eager_matches_function_path_loss():
    torch.manual_seed(0)
    a = Net("cuda", native="auto")
    b = Net("cuda", native="auto")
    b.load_state_dict(a.state_dict())
    for m in (b.a, b.b):
        m.use_custom_op = True
    t = torch.randint(0, 20, (16, 32), device="cuda")
    outs = []
    for net in (a, b):
        torch.manual_seed(11)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = net(t)
        outs.append(y.float())
        y.float().square().mean().backward()
    assert torch.equal(outs[0], outs[1])
    assert torch.equal(a.a.idx, b.a.idx)                                       # same Philox seed stream, same kernels


# ── SDPA vs naive attention ───────────────────────────────────────────────────

def _load_e1():
    spec = importlib.util.spec_from_file_location("e1_quality_sdpa_test", ROOT / "probes" / "e1_quality.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("t", [8, 37])
def test_sdpa_attention_matches_naive_attention(heads, t):
    e1 = _load_e1()
    torch.manual_seed(0)
    blk = e1.Block("dense_nobias", 64, 0.01, heads)
    x = torch.randn(3, t, 64, requires_grad=True)
    e1.SDPA = False
    y0 = blk(x)
    g0 = torch.autograd.grad(y0.square().sum(), x)[0]
    e1.SDPA = True
    y1 = blk(x)
    g1 = torch.autograd.grad(y1.square().sum(), x)[0]
    assert torch.allclose(y0, y1, atol=1e-5, rtol=1e-4) and torch.allclose(g0, g1, atol=1e-4, rtol=1e-3)
    # null twin: a non-causal SDPA would differ -> the parity above really tests causality
    q = torch.randn(1, heads, t, 16)
    assert not torch.allclose(F.scaled_dot_product_attention(q, q, q, is_causal=True),
                              F.scaled_dot_product_attention(q, q, q, is_causal=False), atol=1e-3)


@GPU
def test_gpu_sdpa_matches_naive_in_bf16_autocast():
    e1 = _load_e1()
    torch.manual_seed(0)
    blk = e1.Block("dense_nobias", 128, 0.01, 2).cuda()
    x = torch.randn(4, 64, 128, device="cuda")
    outs = []
    for flag in (False, True):
        e1.SDPA = flag
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs.append(blk(x).float())
    assert torch.allclose(outs[0], outs[1], atol=5e-2, rtol=5e-2)
