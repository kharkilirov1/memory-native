"""int8 Tensor-Core forward as a training path (forward_compute="int8").

The forward Y = X T^T runs on int8 (per-token X scale + the int8 visible cache). It must be close
to the fp forward, DETERMINISTIC (round-to-nearest, so reversible/eager stay valid), and still
train a teacher. grad_x and the update stay fp (straight-through past the forward quant).
"""
import math

import torch

from memory_native import RMSCounterLinear
from memory_native.packed import PackedRMSCounterLinear


def test_int8_forward_close_to_fp_and_deterministic():
    torch.manual_seed(0)
    ref = PackedRMSCounterLinear(64, 48, C=11)
    lay = PackedRMSCounterLinear(64, 48, C=11, forward_compute="int8")
    lay.state.copy_(ref.state); lay.scale.copy_(ref.scale); lay._build_t_cache()
    x = torch.randn(8, 5, 64)
    with torch.no_grad():
        y_fp = ref(x)
        y_i8 = lay(x)
        y_i8b = lay(x)
    assert lay.forward_compute == "int8" and lay.cache_mode == "int8"
    assert torch.equal(y_i8, y_i8b)                                  # deterministic
    rel = (y_i8 - y_fp).abs().mean() / y_fp.abs().mean().clamp_min(1e-6)
    assert rel.item() < 0.05, f"int8 forward too far from fp: rel {rel.item():.4f}"


def test_int8_forward_recovers_teacher():
    torch.manual_seed(0)
    n, N, C = 24, 256, 11
    ts = 0.25
    base = math.sqrt(3.0 / (2.0 * n))
    tw = torch.randint(-1, 2, (n, n)).float()
    x = torch.randn(N, n); y = x @ (ts * tw).t()
    lay = RMSCounterLinear(n, n, C=C, lr=0.02, lr_scale=2e-4, init_gain=ts / base,
                           forward_compute="int8").train()
    for _ in range(600):
        (lay(x) - y).pow(2).mean().backward()
    with torch.no_grad():
        # int8-forward training is evaluated with its own (int8) forward
        assert (lay(x) - y).pow(2).mean().item() < 0.05
