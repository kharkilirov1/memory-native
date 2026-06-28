"""int4 weight-gradient correlation: a coarser operator that gives the SAME flips.

The flip needs only sign + in-row rank of G = Delta^T X, so the update correlation can run in int4
(INT4 IMMA on Turing+). These pin the witness: int4 is unbiased, recovers a teacher as well as fp32,
and is far better than 1-bit (sign-only), which is too coarse.
"""
import math

import torch

from memory_native import RMSCounterLinear
from memory_native.int8_compute import int4_correlation


def test_int4_correlation_unbiased():
    torch.manual_seed(0)
    M, N, K = 256, 8, 12
    D = torch.randn(M, N); X = torch.randn(M, K)
    exact = D.t() @ X
    est = torch.stack([int4_correlation(D, X) for _ in range(800)]).mean(0)
    rel = (est - exact).abs().mean() / exact.abs().mean()
    assert rel.item() < 0.05, f"int4 correlation biased: rel {rel.item():.4f}"


def _recover(update_compute):
    torch.manual_seed(0)
    n, N, C = 24, 256, 11
    ts = 0.25
    base = math.sqrt(3.0 / (2.0 * n))
    tw = torch.randint(-1, 2, (n, n)).float()
    x = torch.randn(N, n); y = x @ (ts * tw).t()
    lay = RMSCounterLinear(n, n, C=C, lr=0.02, lr_scale=2e-4, init_gain=ts / base,
                           update_compute=update_compute).train()
    for _ in range(600):
        (lay(x) - y).pow(2).mean().backward()
    with torch.no_grad():
        return (lay(x) - y).pow(2).mean().item()


def test_int4_update_recovers_teacher_like_fp():
    """The decisive witness: int4 correlation drives training to the same outcome as fp32."""
    assert _recover("int4") < 0.02
    assert _recover("int4") < max(3 * _recover("fp"), 0.01)


def test_int4_beats_one_bit_on_rank():
    torch.manual_seed(0)
    M, N, K = 2048, 64, 512
    D = torch.randn(M, N); X = torch.randn(M, K)
    exact = D.t() @ X

    def spearman(est):
        ra = est.argsort(1).argsort(1).float(); rb = exact.argsort(1).argsort(1).float()
        ra = ra - ra.mean(1, keepdim=True); rb = rb - rb.mean(1, keepdim=True)
        return ((ra * rb).sum(1) / (ra.pow(2).sum(1).sqrt() * rb.pow(2).sum(1).sqrt()).clamp_min(1e-12)).mean()
    int4_s = spearman(int4_correlation(D, X))
    onebit_s = spearman((D.sign().t() @ X.sign()).float())
    assert int4_s > onebit_s + 0.1, f"int4 {int4_s:.3f} not clearly > 1-bit {onebit_s:.3f}"
