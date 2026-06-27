"""int8 Tensor-Core compute path (acceleration memo M6).

The int8 GEMM estimate of the update correlation must be UNBIASED (E[Q(D)^T Q(X)] = D^T X) and a
counter layer using it must still recover a teacher. The Tensor-Core speedup needs a GPU; these
CPU tests pin the numerics so the CUDA path drops in unchanged.
"""
import math

import torch

from memory_native import RMSCounterLinear, int8_correlation, quantize_int8_cols


def test_quantizer_unbiased():
    torch.manual_seed(0)
    x = torch.randn(64, 32)
    q, s = quantize_int8_cols(x)
    assert q.dtype == torch.int8 and int(q.abs().max()) <= 127
    avg = torch.stack([(lambda qs: qs[1] * qs[0].float())(quantize_int8_cols(x))
                       for _ in range(400)]).mean(0)
    assert (avg - x).abs().mean().item() < 0.02      # E[scale*q] = x


def test_int8_correlation_unbiased():
    """E[ int8_correlation(D, X) ] == D^T X."""
    torch.manual_seed(0)
    M, N, K = 128, 8, 12
    D = torch.randn(M, N)
    X = torch.randn(M, K)
    exact = D.t() @ X
    est = torch.stack([int8_correlation(D, X) for _ in range(600)]).mean(0)
    rel = (est - exact).abs().mean() / exact.abs().mean()
    assert rel.item() < 0.02, f"int8 correlation biased: rel err {rel.item():.4f}"


def test_int8_update_recovers_teacher():
    torch.manual_seed(0)
    n, N, C = 24, 256, 11
    ts = 0.25
    base = math.sqrt(3.0 / (2.0 * n))
    tw = torch.randint(-1, 2, (n, n)).float()
    x = torch.randn(N, n)
    y = x @ (ts * tw).t()
    lay = RMSCounterLinear(n, n, C=C, lr=0.02, lr_scale=2e-4, init_gain=ts / base,
                           update_compute="int8").train()
    for _ in range(600):
        (lay(x) - y).pow(2).mean().backward()
    with torch.no_grad():
        assert (lay(x) - y).pow(2).mean().item() < 0.03
