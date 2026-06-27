"""int8 update that reuses the saved activation codes (no re-quantization of x).

The forward already saves X as int8 (act_save_bits=8, per-token row scale). The int8 update should
fold that row scale into grad_out and reuse the saved codes, quantizing only grad_out -- removing
the x re-quantization that made the plain int8 correlation lose to fp32 cuBLAS. Must stay unbiased
and still recover a teacher.
"""
import math

import torch

from memory_native import RMSCounterLinear, int8_correlation_presaved
from memory_native.actquant import quantize_codes


def test_presaved_correlation_unbiased():
    torch.manual_seed(0)
    M, N, K = 128, 8, 12
    D = torch.randn(M, N); X = torch.randn(M, K)
    exact = D.t() @ X

    def one_draw():
        qx, ax = quantize_codes(X, 8, dim=-1)          # the saved int8 activation (row scale)
        return int8_correlation_presaved(D, qx.to(torch.int8), ax)
    est = torch.stack([one_draw() for _ in range(600)]).mean(0)
    rel = (est - exact).abs().mean() / exact.abs().mean()
    assert rel.item() < 0.03, f"presaved correlation biased: rel {rel.item():.4f}"


def test_int8_update_with_saved_activation_recovers_teacher():
    torch.manual_seed(0)
    n, N, C = 24, 256, 11
    ts = 0.25
    base = math.sqrt(3.0 / (2.0 * n))
    tw = torch.randint(-1, 2, (n, n)).float()
    x = torch.randn(N, n); y = x @ (ts * tw).t()
    # act_save_bits=8 -> the forward saves int8 codes; update_compute=int8 -> reuse them
    lay = RMSCounterLinear(n, n, C=C, lr=0.02, lr_scale=2e-4, init_gain=ts / base,
                           act_save_bits=8, update_compute="int8").train()
    for _ in range(600):
        (lay(x) - y).pow(2).mean().backward()
    with torch.no_grad():
        assert (lay(x) - y).pow(2).mean().item() < 0.03
