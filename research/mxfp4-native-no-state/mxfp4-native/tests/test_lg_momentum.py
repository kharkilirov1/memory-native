"""Per-weight momentum for LatentGridLinear (mx6/lg_accum.set_momentum, the "ema" accumulation mode)."""
import pytest
import torch

from mx6.latent_grid import LatentGridLinear
from mx6.lg_accum import (accum_bytes_per_param, set_momentum, AccumState, ema_step_, ema_accum_,
                          lg_accumulate)


@pytest.mark.parametrize("kind", ["fp32", "bf16", "fp8"])
@pytest.mark.parametrize("nesterov", [False, True])
def test_momentum_trains_and_buffer_is_ema(kind, nesterov):
    torch.manual_seed(0)
    lay = LatentGridLinear(64, 32, k=6, lr=0.01)
    st = set_momentum(lay, 0.9, kind, nesterov)
    assert st.mode == "ema" and st.n == 1
    x = torch.randn(16, 64)
    w0 = lay.weight().clone()
    for _ in range(5):
        lay(x).pow(2).mean().backward()
    assert st.n_updates == 5
    assert lay._acc_buf.float().abs().sum() > 0
    assert (lay.weight() != w0).any()


def test_ema_step_matches_formula():
    st = AccumState(1, "ema", "fp32", True, None)
    st.mu = 0.8
    buf = torch.zeros(4, 8)
    g1, g2 = torch.randn(4, 8), torch.randn(4, 8)
    out1 = ema_step_(st, g1.clone(), buf, None)
    assert torch.allclose(out1, 0.2 * g1)
    out2 = ema_step_(st, g2.clone(), buf, None)
    want = 0.8 * 0.2 * g1 + 0.2 * g2
    assert torch.allclose(out2, want, atol=1e-6) and torch.allclose(buf, want, atol=1e-6)
    st.nesterov = True
    buf2 = want.clone()
    g3 = torch.randn(4, 8)
    out3 = ema_step_(st, g3.clone(), buf2, None)
    b3 = 0.8 * want + 0.2 * g3
    assert torch.allclose(out3, 0.8 * b3 + 0.2 * g3, atol=1e-6)


def test_fp8_ema_unbiased():
    torch.manual_seed(1)
    st = AccumState(1, "ema", "fp8", True, None)
    st.mu = 0.9
    g = torch.randn(8, 256)
    acc = torch.zeros(8, 256)
    for _ in range(200):
        buf = torch.zeros(8, 256, dtype=torch.float8_e4m3fn)
        sc = torch.zeros(8, 1)
        ema_step_(st, g.clone(), buf, sc)
        acc += buf.float() * sc
    assert torch.allclose(acc / 200, 0.1 * g, atol=3e-3)


def test_momentum_off_and_validation():
    lay = LatentGridLinear(32, 16, k=6)
    assert set_momentum(lay, 0.0) is None and lay._acc is None
    with pytest.raises(ValueError):
        set_momentum(lay, 1.0)
    assert accum_bytes_per_param("bf16", 32) == 2.0


def _accum_run(st, buf, bscale, grads, n):
    st.active = True
    out = None
    for i, g in enumerate(grads):
        st.idx = i
        r = ema_accum_(st, g.clone(), buf, bscale)
        assert (r is None) == (i < n - 1)
        out = r if r is not None else out
    st.active, st.idx = False, 0
    return out


def test_ema_accum_equals_ema_of_big_batch_grad():
    torch.manual_seed(2)
    n, mu = 4, 0.9
    st = AccumState(n, "ema", "fp32", True, None)
    st.mu = mu
    buf = torch.randn(6, 16)
    old = buf.clone()
    gs = [torch.randn(6, 16) / n for _ in range(n)]       # loss_scaled: micro grads of loss/n
    out = _accum_run(st, buf, None, gs, n)
    want = mu * old + (1 - mu) * sum(gs)
    assert torch.allclose(out, want, atol=1e-6) and torch.allclose(buf, want, atol=1e-6)
    assert st.n_updates == 1


def test_ema_accum_fp8_unbiased():
    torch.manual_seed(3)
    n, mu = 4, 0.9
    gs = [torch.randn(4, 128) / n for _ in range(n)]
    acc = torch.zeros(4, 128)
    reps = 300
    for _ in range(reps):
        st = AccumState(n, "ema", "fp8", True, None)
        st.mu = mu
        buf = torch.zeros(4, 128, dtype=torch.float8_e4m3fn)
        sc = torch.zeros(4, 1)
        out = _accum_run(st, buf, sc, gs, n)
        acc += out.float()
    assert torch.allclose(acc / reps, (1 - mu) * sum(gs), atol=2e-3)


def test_layer_momentum_with_accum_runs_and_matches_one_update():
    torch.manual_seed(0)
    lay = LatentGridLinear(64, 32, k=6, lr=0.01)
    st = set_momentum(lay, 0.9, "bf16", n=4)
    x = torch.randn(32, 64)
    w0 = lay.weight().clone()
    with lg_accumulate(lay, 4) as acc:
        for i in acc.micro_batches():
            (lay(x.chunk(4)[i]).pow(2).mean() / 4).backward()
    assert st.n_updates == 1 and (lay.weight() != w0).any()
    with pytest.raises(RuntimeError):                       # outside the context: refused
        lay(x[:8]).pow(2).mean().backward()
    with pytest.raises(ValueError):
        set_momentum(lay, 0.9, nesterov=True, n=4)
