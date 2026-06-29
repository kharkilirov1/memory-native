import math

import pytest
import torch

from memory_native.triton_counter import HAS_TRITON, TritonCounterLinear

CUDA = torch.cuda.is_available()


def _matmul_tol():
    """The decode-in-GEMM kernel uses tl.dot (Tensor Cores). On Ampere+ (sm>=80) that means TF32
    accumulation -> ~1e-3 vs the fp32 reference (NOT a kernel bug; the dynamics are unaffected).
    On T4 (sm_75, no TF32 for this path) it matches to ~1e-6. So gate fp32-tight on pre-Ampere and
    TF32-tight on Ampere+. (Verified: T4 max err 7e-7; Blackwell RTX PRO 6000 1.7e-3.)"""
    if CUDA and torch.cuda.get_device_capability()[0] >= 8:
        return dict(atol=3e-3, rtol=3e-3)
    return dict(atol=1e-4, rtol=1e-4)


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
    assert torch.allclose(got, ref, **_matmul_tol()), (got - ref).abs().max()


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
    assert torch.allclose(got, ref, **_matmul_tol()), (got - ref).abs().max()


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


@pytest.mark.skipif(not (CUDA and HAS_TRITON), reason="needs CUDA + triton")
def test_triton_forward_handles_3d_input():
    """Transformer activations are [B,T,d]; the kernel path must flatten leading dims."""
    lay = TritonCounterLinear(64, 48, C=11).cuda()
    x = torch.randn(4, 7, 64, device="cuda")  # [B, T, d]
    y = lay(x)
    assert y.shape == (4, 7, 48)


def test_counter_layer_accepts_3d_input_cpu():
    """Regression for the 3D bug the GPU run caught: a counter layer used inside a transformer
    receives [B,T,d]. The base (non-kernel) path must handle it."""
    from memory_native import RMSCounterLinear
    lay = RMSCounterLinear(32, 48, C=11).train()
    x = torch.randn(3, 5, 32)
    y = lay(x)
    assert y.shape == (3, 5, 48)
    y.sum().backward()  # drains the outstanding forward without error
