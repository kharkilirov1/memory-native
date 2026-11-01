"""LatentGridLinear custom-op path: (1) a scheduled lr is DATA (0-dim fp32 buffer read by the update), so torch.compile sees
ONE graph over a cosine schedule and the update runs exactly once per step; (2) non-reentrant gradient checkpointing is
safe on the op path (forward recompute has no side effect, the update lives only in the backward op) and still refused on
the autograd.Function path. CPU: backend aot_eager. GPU tests (Triton kernel, inductor) are skipped without CUDA + Triton."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mx6.latent_grid as LG                                           # noqa: E402
from mx6 import lg_update_kernel as K                                  # noqa: E402
from mx6.latent_grid import LatentGridLinear, prepare_lg_for_compile   # noqa: E402

GPU = pytest.mark.skipif(not (torch.cuda.is_available() and K.HAS_TRITON), reason="needs CUDA + Triton")
needs_ops = pytest.mark.skipif(not LG.HAS_LG_OPS, reason="torch.library.register_autograd / register_fake missing")
dynamo = pytest.importorskip("torch._dynamo")


class Net(nn.Module):
    def __init__(self, dev="cpu", native=True, ckpt=False, wd=0.0):
        super().__init__()
        self.ckpt = ckpt
        self.emb = nn.Embedding(20, 64)
        self.ln = nn.LayerNorm(64)
        self.a = LatentGridLinear(64, 128, k=8, lr=0.05, native=native, weight_decay=wd)
        self.b = LatentGridLinear(128, 64, k=8, lr=0.05, native=native, weight_decay=wd)
        self.head = nn.Linear(64, 20, bias=False)
        self.to(dev)

    def block(self, h):
        return h + self.b(F.gelu(self.a(self.ln(h))))

    def forward(self, t):
        h = self.emb(t)
        h = checkpoint(self.block, h, use_reentrant=False) if self.ckpt else self.block(h)
        return self.head(h)


def cosine(step, total, base=0.05, lo=0.005):
    return lo + 0.5 * (base - lo) * (1 + math.cos(math.pi * step / total))


def train(steps, lrs=None, compiled=False, ckpt=False, use_op=True, backend="aot_eager", dev="cpu", wd=0.0, native=True):
    torch.manual_seed(0)
    net = Net(dev, native=native, ckpt=ckpt, wd=wd)
    if compiled:
        prepare_lg_for_compile(net, "op")
        f = torch.compile(net, backend=backend, fullgraph=True, dynamic=False)
    else:
        if use_op:
            for m in (net.a, net.b):
                m.use_custom_op = True
        f = net
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    torch.manual_seed(3)
    losses = []
    for s in range(steps):
        if lrs is not None:
            for m in (net.a, net.b):
                m.lr = lrs[s]
        t = torch.randint(0, 20, (4, 8), device=dev)
        loss = F.cross_entropy(f(t).float().reshape(-1, 20), t.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return net, losses


@pytest.fixture
def update_spy(monkeypatch):
    calls = []
    orig = LG._op_update_
    monkeypatch.setattr(LG, "_op_update_", lambda *a, **k: (calls.append(1), orig(*a, **k))[1])
    return calls


# ── (1) lr / weight_decay are data ────────────────────────────────────────────────────────────────────────────

def test_lr_setter_updates_the_device_buffer_in_place():
    lay = LatentGridLinear(64, 32, k=8, lr=0.05)
    buf = lay._lr_t
    lay.lr = 0.0123
    assert lay._lr_t is buf and float(buf) == pytest.approx(0.0123) and lay.lr == 0.0123
    lay.weight_decay = 0.2
    assert float(lay._wd_t) == pytest.approx(0.2) and lay._has_wd is True
    lay.weight_decay = 0.0
    assert lay._has_wd is False
    assert "_lr_t" not in lay.state_dict() and "_wd_t" not in lay.state_dict()


@needs_ops
def test_compiled_cosine_schedule_is_one_graph_and_matches_eager(update_spy):
    steps = 5
    lrs = [cosine(s, steps) for s in range(steps)]
    dynamo.reset()
    dynamo.utils.counters.clear()
    ne, le = train(steps, lrs)
    n_eager = len(update_spy)
    update_spy.clear()
    nc, lc = train(steps, lrs, compiled=True)
    c = dynamo.utils.counters
    assert c["stats"]["unique_graphs"] == 1, dict(c["stats"])                # no per-step recompile
    assert sum(c["graph_break"].values()) == 0
    assert n_eager == 2 * steps and len(update_spy) == 2 * steps             # layers x steps: exactly one update per step
    assert le == lc
    for l1, l2 in ((ne.a, nc.a), (ne.b, nc.b)):
        assert torch.equal(l1.idx, l2.idx) and torch.equal(l1.v_rms, l2.v_rms)
    assert float(nc.a._lr_t) == pytest.approx(lrs[-1])


@needs_ops
def test_lr_is_read_at_run_time_not_baked_at_trace_time():
    """Null twin of the recompile test: lr = 0 AFTER the graph was traced with lr > 0 must freeze the indices."""
    dynamo.reset()
    torch.manual_seed(0)
    net = Net()
    prepare_lg_for_compile(net, "op")
    f = torch.compile(net, backend="aot_eager", fullgraph=True, dynamic=False)
    t = torch.randint(0, 20, (4, 8))
    f(t).square().sum().backward()                                           # traced at lr = 0.05
    idx1 = net.a.idx.clone()
    for m in (net.a, net.b):
        m.lr = 0.0
    f(t).square().sum().backward()
    assert torch.equal(net.a.idx, idx1), "lr=0 step moved the index: lr was baked into the graph"
    for m in (net.a, net.b):
        m.lr = 0.05
    f(t).square().sum().backward()
    assert not torch.equal(net.a.idx, idx1)


@needs_ops
def test_weight_decay_buffer_matches_eager_function_path():
    ne, le = train(3, [0.05, 0.03, 0.02], use_op=False, wd=0.3)
    nc, lc = train(3, [0.05, 0.03, 0.02], compiled=True, wd=0.3)
    assert le == lc and torch.equal(ne.a.idx, nc.a.idx)
    n0, _ = train(3, [0.05, 0.03, 0.02], use_op=False, wd=0.0)
    assert not torch.equal(n0.a.idx, ne.a.idx)                              # null twin: wd really acts


@needs_ops
def test_every_step_draws_fresh_randomness_under_compile():
    """Compiled (op in the backward graph) must consume the global RNG once per update like eager: bit-equal indices over
    steps with a constant lr only if the stream advances identically (a baked seed / mask would diverge)."""
    ne, _ = train(4, [0.05] * 4)
    nc, _ = train(4, [0.05] * 4, compiled=True)
    assert torch.equal(ne.a.idx, nc.a.idx) and torch.equal(ne.b.idx, nc.b.idx)
    s0 = torch.get_rng_state()
    lay = LatentGridLinear(64, 32, k=8, lr=0.05, native=True)
    lay.use_custom_op = True
    lay(torch.randn(6, 64)).square().sum().backward()
    assert not torch.equal(torch.get_rng_state(), s0)                        # an update consumes the global stream


# ── (2) gradient checkpointing ────────────────────────────────────────────────────────────────────────────────

@needs_ops
def test_checkpoint_identical_updates_and_one_update_per_layer_per_step(update_spy):
    steps = 3
    lrs = [0.05, 0.04, 0.03]
    n_plain, l_plain = train(steps, lrs)
    assert len(update_spy) == 2 * steps
    update_spy.clear()
    n_ck, l_ck = train(steps, lrs, ckpt=True)
    assert len(update_spy) == 2 * steps, "recompute must not repeat (or skip) the update"
    assert l_plain == l_ck
    for l1, l2 in ((n_plain.a, n_ck.a), (n_plain.b, n_ck.b)):
        assert torch.equal(l1.idx, l2.idx) and torch.equal(l1.v_rms, l2.v_rms)
    assert not torch.equal(n_ck.a.idx, Net().a.idx)                          # null twin: it trained


@needs_ops
def test_checkpoint_forward_really_runs_twice(monkeypatch):
    """Null twin of the checkpoint tests: the recompute does happen (otherwise they prove nothing)."""
    seen = {"n": 0}
    real_forward = LatentGridLinear.forward

    def counting_forward(self, x):
        seen["n"] += 1
        return real_forward(self, x)

    monkeypatch.setattr(LatentGridLinear, "forward", counting_forward)
    train(1, ckpt=True)
    assert seen["n"] == 4, seen                                              # 2 layers x (forward + recompute)
    seen["n"] = 0
    train(1, ckpt=False)
    assert seen["n"] == 2


@needs_ops
def test_checkpoint_under_aot_eager_compile(update_spy):
    steps = 3
    lrs = [0.05, 0.04, 0.03]
    dynamo.reset()
    n_e, l_e = train(steps, lrs, ckpt=True)
    update_spy.clear()
    n_c, l_c = train(steps, lrs, ckpt=True, compiled=True)
    assert len(update_spy) == 2 * steps
    assert l_e == l_c and torch.equal(n_e.a.idx, n_c.a.idx)


def test_function_path_with_checkpoint_is_still_refused():
    with pytest.raises(RuntimeError, match="forwarded twice"):
        train(1, ckpt=True, use_op=False)


# ── GPU ───────────────────────────────────────────────────────────────────────────────────────────────────────

@GPU
@needs_ops
def test_gpu_tensor_lr_kernel_matches_float_lr_kernel():
    torch.manual_seed(0)
    lay = LatentGridLinear(256, 128, k=8, lr=0.03, backend="triton", weight_decay=0.1).cuda()
    g = torch.randn(128, 256, device="cuda").bfloat16()
    idx0, v0 = lay.idx.clone(), lay.v_rms.clone()
    K.lg_update_(g, lay.idx, lay._scale, lay._tscale, lay.v_rms, lr=0.03, beta=0.9, eps=1e-3, weight_decay=0.1,
                 m=lay.fmt.m, seed=5)
    idx_f, v_f = lay.idx.clone(), lay.v_rms.clone()
    lay.idx.copy_(idx0)
    lay.v_rms.copy_(v0)
    K.lg_update_(g, lay.idx, lay._scale, lay._tscale, lay.v_rms, lr=lay._lr_t, beta=0.9, eps=1e-3,
                 weight_decay=lay._wd_t, m=lay.fmt.m, seed=5, has_wd=True)
    assert torch.equal(lay.idx, idx_f) and torch.equal(lay.v_rms, v_f)
    assert not torch.equal(idx_f, idx0)


@GPU
@needs_ops
def test_gpu_inductor_cosine_one_graph_checkpoint_and_fresh_seeds(monkeypatch):
    seeds = []
    orig = LG._draw_seed
    monkeypatch.setattr(LG, "_draw_seed", lambda: (seeds.append(orig()), seeds[-1])[1])

    def run(compiled, ckpt):
        torch.manual_seed(0)
        net = Net("cuda", native="auto", ckpt=ckpt)
        prepare_lg_for_compile(net, "op")
        f = torch.compile(net, fullgraph=True, dynamic=False) if compiled else net
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        out = []
        for s in range(5):
            for m in (net.a, net.b):
                m.lr = cosine(s, 5)
            t = torch.randint(0, 20, (16, 32), device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = f(t)
            loss = F.cross_entropy(logits.float().reshape(-1, 20), t.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            out.append(float(loss.detach()))
        return out

    dynamo.reset()
    dynamo.utils.counters.clear()
    le = run(False, False)
    assert len(seeds) == 10
    seeds.clear()
    lc = run(True, False)
    assert dynamo.utils.counters["stats"]["unique_graphs"] == 1
    assert len(seeds) == 10 and len(set(seeds)) == 10, "every update must draw its own seed"
    seeds.clear()
    lk = run(True, True)                                                     # compiled + checkpointed
    assert len(seeds) == 10
    assert all(abs(a - b) < 0.3 for a, b in zip(le, lc)) and all(abs(a - b) < 0.3 for a, b in zip(lc, lk))
