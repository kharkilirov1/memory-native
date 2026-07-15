"""CUDA contract gate for the review-discovered group-boundary bug."""
import pytest
import torch

from memory_native.group_scale_kernels import HAS_TRITON, triton_group_counter_update_from_io

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAS_TRITON), reason="needs CUDA + Triton"
)


def test_non_power_of_two_strict_group_is_rejected_instead_of_bleeding():
    torch.manual_seed(101)
    m, out, k, group = 37, 9, 48, 12
    state = torch.zeros(out, (k // 4) * 3, dtype=torch.uint8, device="cuda")
    scale = torch.ones(out, k // group, device="cuda") * 0.1
    v = torch.zeros(out, 1, device="cuda")
    x = torch.randn(m, k, device="cuda")
    go = torch.randn(m, out, device="cuda")
    perm = torch.randperm(k, device="cuda")
    with pytest.raises(ValueError, match="power-of-two"):
        triton_group_counter_update_from_io(
            state, scale, v, x, go, perm,
            group=group, C=11, lr=2e-3, lr_scale=2e-4,
            rms_beta=0.9, rms_eps=1e-3, seed=23,
            residual_alpha=0.4, clip=1.0,
        )
