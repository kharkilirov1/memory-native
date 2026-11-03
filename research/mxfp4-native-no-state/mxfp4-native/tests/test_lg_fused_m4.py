"""Fused int4 momentum inside the Triton lg update kernel (lg_update_(momentum=...)).

Runs on CUDA, or on CPU under the Triton interpreter (TRITON_INTERPRET=1). The reference is the SAME kernel without
momentum, fed the exact fp32 EMA t = mu * deq(m_old) + (1 - mu) * g, with the same seed: v_rms must agree and idx must
agree up to rare stochastic-rounding ties; the new buffer must be an unbiased int4 quantisation of t."""
import os

import pytest
import torch

from mx6.lg_update_kernel import HAS_TRITON

INTERP = os.environ.get("TRITON_INTERPRET") == "1"
pytestmark = pytest.mark.skipif(not HAS_TRITON or not (torch.cuda.is_available() or INTERP),
                                reason="needs Triton on CUDA or TRITON_INTERPRET=1")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _state(rows, cols, seed=0):
    from mx6.latent_grid import LatentGridLinear
    torch.manual_seed(seed)
    lay = LatentGridLinear(cols, rows, k=6, lr=0.03)
    idx = lay.idx.to(DEV).contiguous()
    return idx, lay._scale.to(DEV).contiguous(), lay._tscale.to(DEV), lay.v_rms.to(DEV).clone().contiguous(), lay.fmt.m


def _m4(rows, cols, seed=1):
    from mx6.lg_accum import q4_shapes, q4_store_
    torch.manual_seed(seed)
    bs, ss = q4_shapes(rows, cols)
    mb, ms = torch.empty(bs, dtype=torch.uint8), torch.empty(ss, dtype=torch.bfloat16)
    q4_store_(mb, ms, 0, rows, torch.randn(rows, cols) * 1e-3)
    return mb.to(DEV), ms.to(DEV)


@pytest.mark.parametrize("cols", [64, 96, 512])
def test_fused_matches_unfused_on_exact_ema(cols):
    from mx6.lg_accum import q4_dequant
    from mx6.lg_update_kernel import lg_update_
    rows, mu = 8, 0.9
    idx, sb, ts, v, m = _state(rows, cols)
    mb, ms = _m4(rows, cols)
    g = (torch.randn(rows, cols) * 1e-3).to(DEV)
    t = q4_dequant(mb.cpu(), ms.cpu(), 0, rows, cols).to(DEV) * mu + g.float() * (1 - mu)
    idx_r, v_r = idx.clone(), v.clone()
    lg_update_(t, idx_r, sb, ts, v_r, lr=0.03, beta=0.9, eps=1e-3, weight_decay=0.0, m=m, seed=123)
    lg_update_(g, idx, sb, ts, v, lr=0.03, beta=0.9, eps=1e-3, weight_decay=0.0, m=m, seed=123,
               momentum=(mb, ms, mu, 77))
    assert torch.allclose(v, v_r, rtol=1e-5, atol=1e-12)
    assert (idx != idx_r).float().mean() < 1e-3
    # new buffer: on the grid, finite scales
    d = q4_dequant(mb.cpu(), ms.cpu(), 0, rows, cols)
    k = d.view(rows, -1, 32) / ms.cpu().float().unsqueeze(-1)
    assert torch.allclose(k, k.round(), atol=1e-4) and k.abs().max() <= 7
    assert torch.isfinite(ms.float()).all()


def test_fused_buffer_is_unbiased():
    from mx6.lg_accum import q4_dequant
    from mx6.lg_update_kernel import lg_update_
    rows, cols, mu = 4, 64, 0.8
    idx0, sb, ts, v0, m = _state(rows, cols)
    mb0, ms0 = _m4(rows, cols)
    g = (torch.randn(rows, cols) * 1e-3).to(DEV)
    t = q4_dequant(mb0.cpu(), ms0.cpu(), 0, rows, cols) * mu + g.cpu().float() * (1 - mu)
    acc = torch.zeros(rows, cols)
    n = 200
    for s in range(n):
        mb, ms = mb0.clone(), ms0.clone()
        lg_update_(g, idx0.clone(), sb, ts, v0.clone(), lr=0.03, beta=0.9, eps=1e-3, weight_decay=0.0, m=m, seed=1,
                   momentum=(mb, ms, mu, 1000 + s))
        acc += q4_dequant(mb.cpu(), ms.cpu(), 0, rows, cols)
    err = (acc / n - t).abs().amax(1) / t.abs().amax(1)
    assert (err < 0.05).all(), err


def test_module_path_uses_fused_kernel(monkeypatch):
    """set_momentum(int4) on a Triton-backed layer goes through the fused kernel (the torch ema_step_ is NOT called)."""
    from mx6 import lg_accum
    from mx6.latent_grid import LatentGridLinear
    torch.manual_seed(0)
    lay = LatentGridLinear(64, 32, k=6, lr=0.03).to(DEV)
    if DEV == "cpu":                                  # interpreter: let the CPU layer take the Triton update path
        monkeypatch.setattr(lay, "_use_triton", lambda: True)
    st = lg_accum.set_momentum(lay, 0.9, "int4")
    called = []
    monkeypatch.setattr(lg_accum, "ema_step_", lambda *a, **k: called.append(1))
    x = torch.randn(16, 64, device=DEV)
    w0 = lay.weight().clone()
    for _ in range(3):
        lay(x).pow(2).mean().backward()
    assert not called and st.n_updates == 3
    assert (lay.weight() != w0).any()
    assert (lay._acc_buf != 0x88).any()
