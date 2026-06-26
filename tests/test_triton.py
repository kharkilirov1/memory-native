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


@pytest.mark.skipif(not (CUDA and HAS_TRITON), reason="needs CUDA + triton")
def test_triton_grad_x_matches_reference():
    """grad_x straight from packed state must match grad_out @ dense_W within f32 tol."""
    torch.manual_seed(0)
    n_in, n_out, M, C = 128, 96, 64, 11
    layer = TritonCounterLinear(n_in, n_out, C=C).cuda()
    grad_out = torch.randn(M, n_out, device="cuda")
    with torch.no_grad():
        ref = grad_out @ layer._dense_weight(torch.float32)        # [M, n_in]
        from memory_native.triton_counter import triton_grad_x
        got = triton_grad_x(grad_out, layer.state, layer.scale, C, n_in, n_out)
    assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4), (got - ref).abs().max()


@pytest.mark.skipif(not (CUDA and HAS_TRITON), reason="needs CUDA + triton")
def test_triton_full_backward_trains():
    """End-to-end on CUDA: forward + grad_x in-kernel, layer recovers a ternary teacher."""
    import math
    torch.manual_seed(0)
    n, N, C = 64, 256, 11
    ts = 0.25
    base = math.sqrt(3.0 / (2.0 * n))
    tw = torch.randint(-1, 2, (n, n), device="cuda").float()
    x = torch.randn(N, n, device="cuda")
    y = x @ (ts * tw).t()
    lay = TritonCounterLinear(n, n, C=C, lr=0.005, lr_scale=0.0, init_gain=ts / base).cuda().train()
    for _ in range(400):
        ((lay(x) - y) ** 2).mean().backward()
    with torch.no_grad():
        assert ((lay(x) - y) ** 2).mean().item() < 5e-3
