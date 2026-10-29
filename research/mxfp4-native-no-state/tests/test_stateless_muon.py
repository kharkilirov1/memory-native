import pytest
import torch

from probes.stateless_muon import zeropower_ns


@pytest.mark.parametrize("shape", [(8, 16), (16, 8), (8, 8)])
def test_stateless_muon_direction_preserves_shape_and_is_finite(shape):
    grad = torch.randn(shape)
    out = zeropower_ns(grad, steps=5)
    assert out.shape == grad.shape
    assert torch.isfinite(out).all()


def test_rejects_non_matrix():
    with pytest.raises(ValueError):
        zeropower_ns(torch.randn(8))
