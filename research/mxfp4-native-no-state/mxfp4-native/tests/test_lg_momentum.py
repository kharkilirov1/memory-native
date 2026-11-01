"""Per-weight momentum for LatentGridLinear (mx6/lg_accum.set_momentum, the "ema" accumulation mode)."""
import pytest
import torch

from mx6.latent_grid import LatentGridLinear
from mx6.lg_accum import accum_bytes_per_param, set_momentum, AccumState, ema_step_


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
