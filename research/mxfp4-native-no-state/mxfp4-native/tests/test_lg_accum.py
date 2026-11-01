"""Opt-in gradient accumulation for LatentGridLinear (mx6/lg_accum.py): buffer mode (== one big batch), latent mode (lr/N
micro-updates), fp8 stochastic buffer, update counts, torch.compile (1 graph, no recompiles), checkpointing, FP-param accumulation,
DDP all-reduce hook (gloo, world_size 1). CPU: torch backend / emulation; GPU tests are skipped without CUDA + Triton."""
from __future__ import annotations

import socket
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
from mx6 import lg_accum as A                                          # noqa: E402
from mx6 import lg_update_kernel as K                                  # noqa: E402
from mx6.latent_grid import LatentGridLinear, prepare_lg_for_compile   # noqa: E402

GPU = pytest.mark.skipif(not (torch.cuda.is_available() and K.HAS_TRITON), reason="needs CUDA + Triton")
needs_ops = pytest.mark.skipif(not (LG.HAS_LG_OPS and LG.HAS_LG_ACC_OPS), reason="custom ops unavailable")
dynamo = pytest.importorskip("torch._dynamo")
PATHS = [pytest.param("function"), pytest.param("op", marks=needs_ops)]


def make(i=64, o=48, lr=0.02, path="function", seed=0, **kw):
    torch.manual_seed(seed)
    lay = LatentGridLinear(i, o, k=8, lr=lr, native=True, **kw)
    lay.use_custom_op = path == "op"
    return lay


def clone_layer(lay, path):
    c = make(lay.in_features, lay.out_features, lay.lr, path)
    c.load_state_dict(lay.state_dict())
    return c


def per_row_loss(y):
    return y.square().sum(-1)


def run_big(lay, x, seed=7, autocast=False):
    torch.manual_seed(seed)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        y = lay(x)
    per_row_loss(y.float()).mean().backward()


def run_accum(lay, x, n, seed=7, autocast=False, scaled=True):
    torch.manual_seed(seed)
    with A.lg_accumulate(lay, n) as acc:
        for i, xi in zip(acc.micro_batches(), x.chunk(n)):
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
                y = lay(xi)
            l = per_row_loss(y.float()).mean()
            (l / n if scaled else l).backward()


# ── buffer mode == one big batch ───────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("n", [2, 4])
def test_buffer_fp32_equals_single_big_batch(path, n):
    big, acc = make(path=path), make(path=path)
    A.set_accum(acc, n, "buffer", "fp32")
    x = torch.randn(8 * n, 64)
    idx0 = big.idx.clone()
    run_big(big, x)
    run_accum(acc, x, n)
    assert (big.idx != idx0).float().mean() > 0.2                          # null twin: the update really moved indices
    assert (big.idx != acc.idx).float().mean() < 2e-3                      # fp32 summation order only
    assert torch.allclose(big.v_rms, acc.v_rms, rtol=1e-5)
    assert float(acc._acc_buf.abs().max()) == 0.0                          # buffer zeroed after the update


@pytest.mark.parametrize("path", PATHS)
def test_buffer_bf16_equals_single_big_batch_within_bf16_tolerance(path):
    big, acc = make(path=path), make(path=path)
    A.set_accum(acc, 4, "buffer", "bf16")
    assert acc._acc_buf.dtype == torch.bfloat16
    x = torch.randn(32, 64)
    idx0 = big.idx.clone()
    run_big(big, x, autocast=True)
    run_accum(acc, x, 4, autocast=True)
    moved = (big.idx != idx0).float().mean()
    assert moved > 0.2
    d = (big.idx.float() - acc.idx.float()).abs()
    assert d.mean() < 0.15 * (big.idx.float() - idx0.float()).abs().mean()  # bf16 sums: small vs the step itself
    assert torch.allclose(big.v_rms, acc.v_rms, rtol=3e-2)


def test_buffer_loss_not_scaled_means_sum_over_n():
    big, acc = make(), make()
    A.set_accum(acc, 4, "buffer", "fp32", loss_scaled=False)
    x = torch.randn(32, 64)
    run_big(big, x)
    run_accum(acc, x, 4, scaled=False)
    assert (big.idx != acc.idx).float().mean() < 2e-3 and torch.allclose(big.v_rms, acc.v_rms, rtol=1e-5)


# ── update counts ───────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("mode,per_step", [("buffer", 1), ("latent", 4)])
def test_update_count_once_per_step_buffer_and_n_per_step_latent(path, mode, per_step, monkeypatch):
    calls = []
    o_fn, o_op = LatentGridLinear.apply_update_, LG._op_update_
    monkeypatch.setattr(LatentGridLinear, "apply_update_", lambda s, *a, **k: (calls.append("f"), o_fn(s, *a, **k))[1])
    monkeypatch.setattr(LG, "_op_update_", lambda *a, **k: (calls.append("o"), o_op(*a, **k))[1])
    lay = make(path=path)
    st = A.set_accum(lay, 4, mode, "bf16")
    for step in range(3):
        run_accum(lay, torch.randn(32, 64), 4, seed=step)
    assert len(calls) == 3 * per_step, calls
    assert set(calls) == {"f" if path == "function" else "o"}
    assert st.n_updates == 3 * per_step


# ── latent mode: unbiased (MC) and variance vs buffer ───────────────────────────────────────────────────────

def _mc_layer(path, mode, n, lr, seed):
    lay = make(64, 512, lr=lr, path=path, seed=seed)
    A.set_accum(lay, n, mode, "fp32")
    return lay


def _expected_change(lay, dW, mode, n, lr):
    """Expected change of the latent weight (weight units) = sum of the deterministic steps, from the module's own formulas."""
    beta, eps = lay.rms_beta, lay.rms_eps
    v = lay.v_rms.clone()
    tot = torch.zeros(lay.out_features, lay.in_features)
    if mode == "buffer":
        G = torch.stack(dW).mean(0)
        v = v * beta + G.square().mean(1, keepdim=True) * (1 - beta)
        return -lr * G / v.sqrt().clamp_min(eps)
    b = beta ** (1.0 / n)
    for g in dW:
        v = v * b + g.square().mean(1, keepdim=True) * (1 - b)
        tot = tot - (lr / n) * g / v.sqrt().clamp_min(eps)
    return tot


def _mc_run(mode, n=4, lr=0.004, seed=0, path="function"):
    lay = _mc_layer(path, mode, n, lr, seed)
    C = torch.randn(n, 8, 512)
    X = torch.randn(n, 8, 64)
    w0, i0 = lay.latent_weight().clone(), lay.idx.clone()
    dW = [(C[i].t() @ X[i]) for i in range(n)]                            # loss = (y * C).sum(): dW independent of W
    # steady-state RMS EMA (a fresh v = eps would make the first steps huge and hide the SR comparison): the micro-batch
    # second moment for latent, the second moment of the n-mean (1/n of it, pure noise) for buffer
    s2 = torch.stack(dW).square().mean(dim=(0, 2)).unsqueeze(1)
    lay.v_rms.copy_(s2 if mode == "latent" else s2 / n)
    v_init = lay.v_rms.clone()
    torch.manual_seed(100 + seed)
    with A.lg_accumulate(lay, n) as acc:
        for i in acc.micro_batches():
            ((lay(X[i]) * C[i]).sum() / n).backward()
    return lay, w0, i0, dW, v_init


class _Snap:
    """layer state before the run (v_rms = init), for the expectation formula."""
    def __init__(self, lay, v_init):
        self.rms_beta, self.rms_eps = lay.rms_beta, lay.rms_eps
        self.v_rms = v_init
        self.out_features, self.in_features = lay.out_features, lay.in_features


def _residual(mode, n, lr, seed, path="function"):
    lay, w0, i0, dW, v_init = _mc_run(mode, n, lr, seed, path)
    exp = _expected_change(_Snap(lay, v_init), dW, mode, n, lr)
    scale = lay._scale_full()
    # only elements far from the grid ends BEFORE the run (selecting on the outcome would bias the residual): no clamping
    ok = (i0.long() - lay.fmt.n_pts // 2).abs() <= (25 if lr <= 0.01 else 12)
    res = (lay.latent_weight() - w0 - exp) / scale                           # block-scale (E2M1) units
    return res[ok], (exp / scale)[ok]


@pytest.mark.parametrize("mode", ["buffer", "latent"])
@pytest.mark.parametrize("path", PATHS)
def test_mc_unbiased_both_modes(mode, path):
    res = torch.cat([_residual(mode, 4, 0.004, s, path)[0] for s in range(3)])
    z = float(res.mean() / (res.std() / res.numel() ** 0.5))
    assert abs(z) < 4.5, (mode, z, float(res.mean()), float(res.std()))


def test_mc_variance_latent_vs_buffer_report(capsys):
    out = {}
    for lr in (0.0002, 0.002, 0.01):
        rb = torch.cat([_residual("buffer", 4, lr, s)[0] for s in range(3)])
        rl_, exp = zip(*[_residual("latent", 4, lr, s) for s in range(3)])
        rl, exp = torch.cat(rl_), torch.cat(exp)
        out[lr] = (float(rb.var()), float(rl.var()), float(exp.abs().mean()))
        z = float(rl.mean() / (rl.std() / rl.numel() ** 0.5))
        assert abs(z) < 4.5, (lr, z)
    with capsys.disabled():
        for lr, (vb, vl, d) in out.items():
            print(f"\n[latent-vs-buffer SR variance] lr={lr}: var_buffer={vb:.3e} var_latent={vl:.3e} ratio={vl / vb:.2f} "
                  f"(mean |delta| = {d:.3f} E2M1 units)")
    # theory (module docstring): tiny micro-steps (|delta|/N << grid step d) -> the N SR draws add up to ~ one draw (ratio ~ 1);
    # micro-steps >= d -> every draw costs ~ d^2/6 while the single big draw is bounded -> ratio -> N
    lrs = sorted(out)
    assert 0.5 < out[lrs[0]][1] / out[lrs[0]][0] < 2.0
    assert 3.0 < out[lrs[-1]][1] / out[lrs[-1]][0] < 5.0


# ── fp8 stochastic buffer ─────────────────────────────────────────────────────────────────────────────────────

def test_sr_fp8_unbiased_and_on_grid():
    torch.manual_seed(0)
    v = torch.randn(64) * torch.logspace(-4, 0, 64)                          # dynamic range 1e4: subnormal path included
    v = v / v.abs().max() * 400.0
    T = 20000
    x = v.expand(T, 64).contiguous()
    q = A.sr_fp8(x)
    assert q.dtype == torch.float8_e4m3fn and torch.isfinite(q.float()).all()
    qf = q.float()
    err = qf.mean(0) - v
    se = qf.std(0) / T ** 0.5
    z = err / se.clamp_min(1e-12)
    exact = qf.std(0) == 0                                                  # x already on the grid -> exact
    assert z[~exact].abs().max() < 5.0 and abs(float(z[~exact].mean())) < 1.0
    assert (qf.max(0).values - qf.min(0).values).max() <= 2 * 32.0           # only the two neighbours are ever drawn


def test_fp8_buffer_accumulation_unbiased_and_error_report(capsys):
    torch.manual_seed(1)
    T, c = 20000, 64
    gs = [torch.randn(c) * torch.logspace(-2, 0, c) for _ in range(4)]
    total = sum(gs)
    buf = torch.zeros(T, c, dtype=torch.float8_e4m3fn)
    sc = torch.zeros(T, 1)
    for g in gs[:3]:
        A.fp8_add_(buf, sc, g.expand(T, c).contiguous())
    last = gs[3].expand(T, c).contiguous()
    out = A.acc_finish_(buf, sc, last, "fp8", 1.0)
    se = out.std(0) / T ** 0.5
    keep = se > 1e-6 * total.abs()                                         # a column that is exactly representable has no noise
    z = ((out.mean(0) - total) / se)[keep]
    assert keep.sum() > 50 and z.abs().max() < 5.0 and abs(float(z.mean())) < 1.0
    rel = float((out - total).square().mean().sqrt() / total.square().mean().sqrt())
    assert 0.0 < rel < 0.1
    assert float(buf.float().abs().max()) == 0.0 and float(sc.abs().max()) == 0.0
    with capsys.disabled():
        print(f"\n[fp8 buffer, N=4 -> 3 requantisations] relative rms error of the accumulated gradient: {rel * 100:.2f} %")


@pytest.mark.parametrize("path", PATHS)
def test_fp8_layer_runs_and_matches_bf16_buffer_loosely(path):
    a, b = make(path=path), make(path=path)
    A.set_accum(a, 4, "buffer", "fp8")
    A.set_accum(b, 4, "buffer", "fp32")
    assert a._acc_buf.dtype == torch.float8_e4m3fn and a._acc_scale.shape == (48, 1)
    x = torch.randn(32, 64)
    idx0 = a.idx.clone()
    run_accum(a, x, 4)
    run_accum(b, x, 4)
    assert (a.idx != idx0).float().mean() > 0.2
    assert (a.idx.float() - b.idx.float()).abs().mean() < 0.35 * (b.idx.float() - idx0.float()).abs().mean()


# ── torch.compile: 1 graph, no recompiles, same numbers as eager ───────────────────────────────────────────────

class Net(nn.Module):
    def __init__(self, path="op"):
        super().__init__()
        torch.manual_seed(0)
        self.emb = nn.Embedding(20, 64)
        self.ln = nn.LayerNorm(64)
        self.a = LatentGridLinear(64, 128, k=8, lr=0.05, native=True)
        self.b = LatentGridLinear(128, 64, k=8, lr=0.05, native=True)
        self.head = nn.Linear(64, 20, bias=False)
        for m in (self.a, self.b):
            m.use_custom_op = path == "op"

    def forward(self, t):
        h = self.emb(t)
        h = h + self.b(F.gelu(self.a(self.ln(h))))
        return self.head(h)


def train_net(n, mode, compiled, steps=3, ckpt=False, kind="bf16"):
    net = Net("op")
    if n > 1:
        A.set_accum(net, n, mode, kind)
    f = net
    cnt = None
    if compiled:
        from torch._dynamo.testing import CompileCounterWithBackend
        cnt = CompileCounterWithBackend("aot_eager")
        prepare_lg_for_compile(net, "op")
        f = torch.compile(net, backend=cnt, fullgraph=True, dynamic=False)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    torch.manual_seed(3)
    losses = []
    for _ in range(steps):
        T = torch.randint(0, 20, (4 * n, 8))
        opt.zero_grad()
        ctx = A.lg_accumulate(net, n) if n > 1 else None
        if ctx is None:
            loss = F.cross_entropy(f(T).float().reshape(-1, 20), T.reshape(-1))
            loss.backward()
            losses.append(float(loss))
        else:
            with ctx as acc:
                tot = 0.0
                for i in acc.micro_batches():
                    ti = T.chunk(n)[i]
                    loss = F.cross_entropy(f(ti).float().reshape(-1, 20), ti.reshape(-1))
                    (loss / n).backward()
                    tot += float(loss) / n
            losses.append(tot)
        opt.step()
    return net, losses, cnt


@needs_ops
@pytest.mark.parametrize("mode", ["buffer", "latent"])
def test_compiled_accumulation_one_graph_matches_eager(mode):
    dynamo.reset()
    ne, le, _ = train_net(4, mode, False)
    nc, lc, cnt = train_net(4, mode, True)
    assert cnt.frame_count == 1, f"{cnt.frame_count} frames: a recompile happened"
    assert le == lc
    for l1, l2 in ((ne.a, nc.a), (ne.b, nc.b)):
        assert torch.equal(l1.idx, l2.idx) and torch.equal(l1.v_rms, l2.v_rms)
    assert (nc.a.idx != Net().a.idx).any()
    dynamo.reset()


# ── checkpointing ──────────────────────────────────────────────────────────────────────────────────────────

class CkptNet(Net):
    def __init__(self, use_ckpt):
        super().__init__("op")
        self.use_ckpt = use_ckpt

    def forward(self, t):
        h = self.emb(t)
        blk = lambda z: self.b(F.gelu(self.a(self.ln(z))))              # noqa: E731
        h = h + (checkpoint(blk, h, use_reentrant=False) if self.use_ckpt else blk(h))
        return self.head(h)


@needs_ops
@pytest.mark.parametrize("mode", ["buffer", "latent"])
def test_gradient_checkpointing_with_accumulation(mode):
    res = []
    for use in (False, True):
        net = CkptNet(use)
        st = A.set_accum(net, 4, mode, "bf16")
        torch.manual_seed(3)
        for _ in range(2):
            T = torch.randint(0, 20, (16, 8))
            with A.lg_accumulate(net, 4) as acc:
                for i in acc.micro_batches():
                    ti = T.chunk(4)[i]
                    (F.cross_entropy(net(ti).reshape(-1, 20), ti.reshape(-1)) / 4).backward()
        res.append((net.a.idx.clone(), net.b.idx.clone(), st.n_updates))
    assert torch.equal(res[0][0], res[1][0]) and torch.equal(res[0][1], res[1][1])
    assert res[0][2] == res[1][2] == 2 * 2 * (1 if mode == "buffer" else 4)


# ── FP params keep the normal autograd accumulation ──────────────────────────────────────────────────────────

class Mixed(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.pre = nn.Linear(16, 64)
        self.lg = LatentGridLinear(64, 32, k=8, lr=0.02, native=True)
        self.post = nn.Linear(32, 5)

    def forward(self, x):
        return self.post(F.gelu(self.lg(F.gelu(self.pre(x)))))


def test_fp_params_accumulate_normally_and_optimizer_steps_once():
    n = 4
    x, y = torch.randn(32, 16), torch.randint(0, 5, (32,))
    big, acc = Mixed(), Mixed()
    A.set_accum(acc, n, "buffer", "fp32")
    torch.manual_seed(5)
    F.cross_entropy(big(x), y).backward()
    torch.manual_seed(5)
    with A.lg_accumulate(acc, n) as a:
        for i in a.micro_batches():
            (F.cross_entropy(acc(x.chunk(n)[i]), y.chunk(n)[i]) / n).backward()
    for (na, pb), (_, pa) in zip(big.named_parameters(), acc.named_parameters()):
        assert torch.allclose(pb.grad, pa.grad, rtol=1e-4, atol=1e-6), na
    assert (big.lg.idx != acc.lg.idx).float().mean() < 2e-3
    opt = torch.optim.AdamW(acc.parameters(), lr=1e-2)
    w0 = acc.pre.weight.detach().clone()
    opt.step()
    assert not torch.equal(w0, acc.pre.weight)


# ── DDP hook (gloo, world_size 1) ─────────────────────────────────────────────────────────────────────────

def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def gloo():
    dist = pytest.importorskip("torch.distributed")
    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("gloo unavailable")
    try:
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{_free_port()}", rank=0, world_size=1)
    except Exception as e:                                                 # noqa: BLE001
        pytest.skip(f"cannot init gloo: {e}")
    yield dist
    dist.destroy_process_group()


@pytest.mark.parametrize("path", PATHS)
def test_allreduce_hook_called_once_per_optimizer_step_per_layer(path, gloo, monkeypatch):
    real = gloo.all_reduce
    reduced = []
    monkeypatch.setattr(gloo, "all_reduce", lambda t, *a, **k: (reduced.append(tuple(t.shape)), real(t, *a, **k))[1])
    seen = []

    def hook(g):
        seen.append(tuple(g.shape))
        return A.dist_allreduce_mean(g)

    net = Net(path)
    st = A.set_accum(net, 4, "buffer", "bf16", grad_allreduce=hook)
    ref = Net(path)
    A.set_accum(ref, 4, "buffer", "bf16")
    for step in range(3):
        T = torch.randint(0, 20, (16, 8))
        for m in (net, ref):
            torch.manual_seed(step)
            with A.lg_accumulate(m, 4) as acc:
                for i in acc.micro_batches():
                    ti = T.chunk(4)[i]
                    (F.cross_entropy(m(ti).reshape(-1, 20), ti.reshape(-1)) / 4).backward()
    assert len(seen) == 3 * 2 and st.n_hook == 6                          # 2 layers x 3 optimizer steps, not x 4 micro-batches
    assert sorted(set(seen)) == [(64, 128), (128, 64)] and len(reduced) == 6
    assert torch.equal(net.a.idx, ref.a.idx)                              # world_size 1: the all-reduce is an identity


def test_allreduce_default_none_and_dist_string_and_uninitialised_is_noop():
    g = torch.randn(4, 4)
    assert torch.equal(A.dist_allreduce_mean(g.clone()), g)                # not initialised -> identity
    lay = make()
    st = A.set_accum(lay, 2, "buffer", "bf16")
    assert st.grad_allreduce is None
    assert A.set_accum(lay, 2, "buffer", "bf16", grad_allreduce="dist").grad_allreduce is A.dist_allreduce_mean
    with pytest.raises(ValueError):
        A.set_accum(lay, 2, "buffer", "bf16", grad_allreduce=3)


def test_allreduce_in_latent_mode_runs_every_micro_batch(gloo):
    seen = []
    lay = make()
    st = A.set_accum(lay, 4, "latent", grad_allreduce=lambda g: (seen.append(1), A.dist_allreduce_mean(g))[1])
    run_accum(lay, torch.randn(32, 64), 4)
    assert len(seen) == 4 == st.n_hook


# ── API contracts ──────────────────────────────────────────────────────────────────────────────────────────

def test_default_off_is_untouched_and_n1_disables():
    a, b = make(), make()
    assert a._acc is None and a._acc_id == -1 and a._acc_buf is None
    assert A.set_accum(b, 1, "buffer") is None and b._acc is None and b._acc_buf is None
    x = torch.randn(8, 64)
    run_big(a, x)
    run_big(b, x)
    assert torch.equal(a.idx, b.idx) and torch.equal(a.v_rms, b.v_rms)
    assert "_acc_buf" not in a.state_dict()


def test_backward_outside_context_and_incomplete_micro_batches_raise_and_reset():
    lay = make()
    A.set_accum(lay, 4, "buffer", "bf16")
    with pytest.raises(RuntimeError, match="outside"):
        lay(torch.randn(8, 64)).sum().backward()
    lay2 = make()
    A.set_accum(lay2, 4, "buffer", "bf16")
    with pytest.raises(RuntimeError, match="expected 4"):
        with A.lg_accumulate(lay2, 4) as acc:
            for i in acc.micro_batches():
                lay2(torch.randn(8, 64)).sum().backward()
                if i == 1:
                    break
    assert float(lay2._acc_buf.float().abs().max()) == 0.0
    with pytest.raises(ValueError):
        A.lg_accumulate(lay2, 3).__enter__()
    with pytest.raises(ValueError):
        A.set_accum(lay2, 4, "nope")
    with pytest.raises(ValueError):
        A.set_accum(lay2, 4, "buffer", "fp16")


def test_micro_context_and_memory_accounting():
    lay = make(64, 32)
    A.set_accum(lay, 2, "buffer", "bf16")
    assert A.accum_resident_bytes(lay) == 2 * 64 * 32
    A.set_accum(lay, 2, "buffer", "fp8")
    assert A.accum_resident_bytes(lay) == 64 * 32 + 4 * 32
    assert A.accum_bytes_per_param("fp8", 64) == pytest.approx(1 + 4 / 64)
    A.set_accum(lay, 2, "latent")
    assert A.accum_resident_bytes(lay) == 0
    st = A.get_accum(lay)
    with A.lg_accumulate(lay, 2) as acc:
        for i in (0, 1):
            with acc.micro(i):
                lay(torch.randn(8, 64)).square().mean().backward()
    assert st.n_updates == 2
    A.clear_accum(lay)
    assert lay._acc is None and A.get_accum(lay) is None


def test_latent_beta_and_lr_scaling_formulas():
    lay = make()
    st = A.set_accum(lay, 4, "latent")
    assert st.lr_mult == 0.25 and st.beta_eff(0.9) == pytest.approx(0.9 ** 0.25)
    assert st.beta_eff(0.9) ** 4 == pytest.approx(0.9)                     # EMA horizon in optimizer steps unchanged
    st2 = A.set_accum(lay, 4, "buffer")
    assert st2.lr_mult == 1.0 and st2.beta_eff(0.9) == 0.9


# ── GPU (skipped on CPU) ───────────────────────────────────────────────────────────────────────────────────

@GPU
@needs_ops
@pytest.mark.parametrize("kind", ["bf16", "fp8"])
def test_gpu_triton_buffer_matches_big_batch(kind):
    def mk():
        torch.manual_seed(0)
        l = LatentGridLinear(512, 256, k=8, lr=0.02, native="auto").cuda()
        l.use_custom_op = True
        return l
    big, acc = mk(), mk()
    assert big.native_enabled()
    A.set_accum(acc, 4, "buffer", kind)
    x = torch.randn(256, 512, device="cuda")
    idx0 = big.idx.clone()
    for lay, n in ((big, 1), (acc, 4)):
        torch.manual_seed(7)
        if n == 1:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = lay(x)
            y.float().square().sum(-1).mean().backward()
        else:
            with A.lg_accumulate(lay, n) as a:
                for i in a.micro_batches():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        y = lay(x.chunk(n)[i])
                    (y.float().square().sum(-1).mean() / n).backward()
    moved = (big.idx.float() - idx0.float()).abs().mean()
    assert moved > 0
    assert (big.idx.float() - acc.idx.float()).abs().mean() < (0.2 if kind == "bf16" else 0.4) * moved
    assert float(acc._acc_buf.float().abs().max()) == 0.0


@GPU
@needs_ops
def test_gpu_inductor_compile_accumulation_one_graph_and_update_count(monkeypatch):
    calls = []
    orig = LG.lg_update_
    monkeypatch.setattr(LG, "lg_update_", lambda *a, **k: (calls.append(1), orig(*a, **k))[1])
    from torch._dynamo.testing import CompileCounterWithBackend
    dynamo.reset()
    net = Net("op").cuda()
    A.set_accum(net, 4, "buffer", "bf16")
    prepare_lg_for_compile(net, "op")
    cnt = CompileCounterWithBackend("inductor")
    f = torch.compile(net, backend=cnt, fullgraph=True, dynamic=False)
    for step in range(3):
        T = torch.randint(0, 20, (32, 16), device="cuda")
        with A.lg_accumulate(net, 4) as acc:
            for i in acc.micro_batches():
                ti = T.chunk(4)[i]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = f(ti)
                (F.cross_entropy(out.float().reshape(-1, 20), ti.reshape(-1)) / 4).backward()
    assert cnt.frame_count == 1
    assert len(calls) == 2 * 3                                            # 2 layers x 3 optimizer steps
    dynamo.reset()



# ── probes/train_1b.py and probes/e1_quality.py wiring (CPU, tiny model) ───────────────────────────────────────

import importlib.util as _ilu                                            # noqa: E402

import numpy as np                                                       # noqa: E402

_spec = _ilu.spec_from_file_location("train_1b_acc", ROOT / "probes" / "train_1b.py")
sys.path.insert(0, str(ROOT / "probes"))
T1B = _ilu.module_from_spec(_spec)
sys.modules["train_1b_acc"] = T1B
_spec.loader.exec_module(T1B)
TINY = ["--d", "32", "--layers", "2", "--heads", "2", "--seq", "16", "--precision", "fp32", "--no-compile",
        "--eval-windows", "4", "--eval-batches", "2", "--log-every", "2", "--arm", "lg6_fp4_h2", "--adam-lr", "3e-3",
        "--lr", "0.02", "--eval-every", "100", "--ckpt-every", "4"]


@pytest.fixture
def corpus(tmp_path):
    rng = np.random.default_rng(0)
    base = np.tile(np.arange(97, 123, dtype=np.uint8), 400)
    arr = np.where(rng.random(base.size) < 0.9, base, rng.integers(0, 256, size=base.size).astype(np.uint8)).astype(np.uint8)
    tr, va = tmp_path / "train.txt", tmp_path / "valid.txt"
    tr.write_bytes(arr.tobytes())
    va.write_bytes(arr[:3000].tobytes())
    return tr, va


def targv(corpus, out, *extra):
    return TINY + ["--train-file", str(corpus[0]), "--val-file", str(corpus[1]), "--out", str(out)] + list(extra)


def run1b(corpus, out, *extra):
    rec = []
    r = T1B.run_train(T1B.build_parser().parse_args(targv(corpus, out, *extra)), record=rec)
    return rec, r


@needs_ops
@pytest.mark.parametrize("mode,dtype", [("buffer", "bf16"), ("buffer", "fp8"), ("latent", "bf16")])
def test_train_1b_accum_runs_with_grad_checkpoint_and_logs_effective_batch(corpus, tmp_path, mode, dtype):
    rec, r = run1b(corpus, tmp_path / "a", "--batch", "4", "--accum", "4", "--accum-mode", mode, "--accum-dtype", dtype,
                   "--steps", "30")
    assert len(rec) == 30 and np.isfinite(rec).all() and not r["diverged"]
    assert np.mean(rec[-5:]) < np.mean(rec[:5])
    log = (tmp_path / "a" / "train.log").read_text()
    assert "effective batch 16" in log and f"accum_mode={mode}" in log and "tokens/step=256" in log   # 4 x 4 x 16
    big, _ = run1b(corpus, tmp_path / "b", "--batch", "16", "--steps", "30")
    assert "no accumulation" in (tmp_path / "b" / "train.log").read_text()
    if mode == "buffer" and dtype == "bf16":
        assert rec[0] == pytest.approx(big[0], rel=1e-4)                  # same windows, same init weights -> same first loss
        assert np.allclose(rec[:6], big[:6], rtol=0.05)                   # == one big batch up to bf16 sums / SR draws


@needs_ops
def test_train_1b_accum_resume_bit_exact_and_rejects_changed_accum(corpus, tmp_path):
    common = ["--batch", "2", "--accum", "2", "--steps", "8", "--ckpt-every", "4"]
    full, _ = run1b(corpus, tmp_path / "A", *common)
    pre, _ = run1b(corpus, tmp_path / "B", *common, "--stop-after", "4")
    post, _ = run1b(corpus, tmp_path / "B", *common, "--resume")
    assert pre == full[:4] and post == full[4:]
    with pytest.raises(ValueError, match="accum"):
        run1b(corpus, tmp_path / "B", "--batch", "2", "--accum", "4", "--steps", "8", "--ckpt-every", "4", "--resume")


def test_train_1b_dense_accum_equals_big_batch(corpus, tmp_path):
    a, _ = run1b(corpus, tmp_path / "a", "--arm", "dense", "--batch", "4", "--accum", "4", "--steps", "6")
    b, _ = run1b(corpus, tmp_path / "b", "--arm", "dense", "--batch", "16", "--steps", "6")
    assert np.allclose(a, b, rtol=1e-3)


def test_train_1b_default_parser_has_accum_off():
    ns = T1B.build_parser().parse_args([])
    assert (ns.accum, ns.accum_mode, ns.accum_dtype) == (1, "buffer", "bf16")


@needs_ops
@pytest.mark.parametrize("accum,mode", [(1, "buffer"), (2, "buffer"), (2, "latent")])
def test_e1_quality_run_arm_accum(accum, mode):
    e1 = T1B.get_e1()
    e1.SDPA, e1.SCALE_FORMAT = False, "nvfp4"
    ids = torch.from_numpy((np.tile(np.arange(97, 123), 300)).astype(np.int64)) - 97
    r = e1.run_arm("lg6_fp4_h2", 0, 32, 6, 4, 16, ids, 26, 0.02, 3e-3, eval_every=3, val_ids=ids[:2000], blocks=2, heads=2,
                   accum=accum, accum_mode=mode)
    assert np.isfinite(r["val"])
