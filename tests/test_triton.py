import math

import pytest
import torch

from memory_native.triton_counter import HAS_TRITON, TritonCounterLinear

CUDA = torch.cuda.is_available()


def test_cpu_fallback_trains():
    """Without CUDA/triton, TritonCounterLinear must transparently fall back to the packed
    PyTorch forward and still learn (so the class is safe to use anywhere)."""
    torch.manual_seed(0)
    n, N, C = 16, 256, 11
    ts = 0.25
    base = math.sqrt(3.0 / (2.0 * n))
    tw = torch.randint(-1, 2, (n, n)).float()
    x = torch.randn(N, n)
    y = x @ (ts * tw).t()
    layer = TritonCounterLinear(n, n, C=C, lr=0.005, lr_scale=0.0, init_gain=ts / base).train()
    for _ in range(300):
        ((layer(x) - y) ** 2).mean().backward()
    with torch.no_grad():
        mse = ((layer(x) - y) ** 2).mean().item()
    assert mse < 5e-3, mse
    # state stays packed 0.75 byte/weight
    assert layer.state.numel() == n * n * 3 // 4


@pytest.mark.skipif(not (CUDA and HAS_TRITON), reason="needs CUDA + triton")
def test_triton_forward_matches_reference():
    """The in-kernel decode+matmul must match the dense-decode reference within f32 tol."""
    torch.manual_seed(0)
    n_in, n_out, M, C = 64, 48, 32, 11
    layer = TritonCounterLinear(n_in, n_out, C=C).cuda()
    x = torch.randn(M, n_in, device="cuda")

    with torch.no_grad():
        ref = torch.nn.functional.linear(x, layer._dense_weight(torch.float32))
        from memory_native.triton_counter import triton_decode_matmul
        got = triton_decode_matmul(x, layer.state, layer.scale, C, n_in, n_out)
    assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4), (got - ref).abs().max()
