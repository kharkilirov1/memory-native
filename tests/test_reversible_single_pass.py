"""Single-pass inverse backward must preserve gradients and update each counter once."""
import copy

import pytest
import torch
from torch import nn

from memory_native import PackedRMSCounterLinear, ReversibleCouplingBlock, ReversibleSequence
from memory_native.reversible import _couple_fwd


class CountingMLP(nn.Sequential):
    def __init__(self, width):
        super().__init__(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, width))
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return super().forward(x)


@pytest.mark.parametrize("whole_chain", [False, True])
def test_constant_branches_preserve_identity_gradient(whole_chain):
    class Constant(nn.Module):
        def forward(self, x):
            return torch.full_like(x, 0.25)
    block = ReversibleCouplingBlock(8, Constant(), Constant())
    model = ReversibleSequence([block]) if whole_chain else block
    x = torch.randn(2, 8, requires_grad=True)
    upstream = torch.randn_like(x)
    model(x).backward(upstream)
    torch.testing.assert_close(x.grad, upstream, atol=0, rtol=0)


@pytest.mark.parametrize("depth", [1, 5])
@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("anchor_every", [0, 2])
def test_nonlinear_parameter_and_input_gradients(depth, frozen, anchor_every):
    torch.manual_seed(23)
    blocks = [ReversibleCouplingBlock(16, CountingMLP(8), CountingMLP(8)).double()
              for _ in range(depth)]
    if frozen:
        for block in blocks:
            block.F[0].requires_grad_(False)
    reference = copy.deepcopy(blocks)
    # Exercise the per-block Function too, rather than only the whole-chain Function.
    model = blocks[0] if depth == 1 and anchor_every == 0 else ReversibleSequence(blocks, anchor_every)
    x = torch.randn(3, 4, 16, dtype=torch.double, requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)
    upstream = torch.randn_like(x)
    out = model(x)
    out.backward(upstream)
    ref = xr
    for block in reference:
        ref = _couple_fwd(block, ref)
    ref.backward(upstream)
    torch.testing.assert_close(out, ref, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(x.grad, xr.grad, atol=1e-10, rtol=1e-10)
    for block, ref_block in zip(blocks, reference):
        assert block.F.calls == block.G.calls == 2  # one forward + one backward recompute
        for param, ref_param in zip(block.parameters(), ref_block.parameters()):
            if param.requires_grad:
                torch.testing.assert_close(param.grad, ref_param.grad, atol=1e-10, rtol=1e-10)
            else:
                assert param.grad is ref_param.grad is None


@pytest.mark.parametrize("whole_chain", [False, True])
def test_single_pass_gradcheck(whole_chain):
    torch.manual_seed(2)
    blocks = [ReversibleCouplingBlock(4, nn.Sequential(nn.Linear(2, 2), nn.Tanh()),
                                     nn.Sequential(nn.Linear(2, 2), nn.SiLU())).double()
              for _ in range(3 if whole_chain else 1)]
    model = ReversibleSequence(blocks) if whole_chain else blocks[0]
    x = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(model, (x,))


@pytest.mark.parametrize("whole_chain", [False, True])
def test_packed_counter_updates_match_stored_activation_reference(whole_chain):
    """Compare optimizer state as well as gradients, with no activation quantization noise."""
    torch.manual_seed(7)
    def branch():
        return nn.Sequential(PackedRMSCounterLinear(8, 8, C=11, lr=3e-3), nn.Tanh())
    blocks = [ReversibleCouplingBlock(16, branch(), branch())
              for _ in range(3 if whole_chain else 1)]
    reference = copy.deepcopy(blocks)
    model = ReversibleSequence(blocks) if whole_chain else blocks[0]
    x = torch.randn(4, 16)  # no input gradient: the update tap must still work
    upstream = torch.randn_like(x)
    torch.manual_seed(123)
    model(x).backward(upstream)
    torch.manual_seed(123)
    ref = x
    for block in reference:
        ref = _couple_fwd(block, ref)
    ref.backward(upstream)
    for block, ref_block in zip(blocks, reference):
        for branch_name in ("F", "G"):
            layer, ref_layer = getattr(block, branch_name)[0], getattr(ref_block, branch_name)[0]
            assert not layer._outstanding_forward
            torch.testing.assert_close(layer.state, ref_layer.state, atol=0, rtol=0)
            torch.testing.assert_close(layer.v, ref_layer.v, atol=1e-7, rtol=1e-5)
            torch.testing.assert_close(layer.scale, ref_layer.scale, atol=1e-7, rtol=1e-5)
            assert int(layer.update_events) == int(ref_layer.update_events)
