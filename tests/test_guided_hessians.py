"""GuidedQuant-style loss-weighted Hessian gates (collect_hessians_guided).

Claims:
  * with a loss whose grad wrt every layer output is CONSTANT (loss = sum(y)),
    the token weights are all equal, so H_guided == H_plain exactly (up to the
    constant factor -- here exactly 1);
  * with per-token loss weights, H_guided equals the hand-built sum g_n x_n x_n^T;
  * collection touches no weights and restores requires_grad flags;
  * ptq_warm_start(hessian_weighting='end_loss') runs end-to-end and rejects the
    asym combination.
"""
import copy

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from memory_native.donor.ptq import (
    collect_hessians,
    collect_hessians_guided,
    ptq_warm_start,
)


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = nn.Linear(16, 12, bias=False)
        self.out = nn.Linear(12, 8, bias=False)

    def forward(self, x):
        return self.out(torch.relu(self.inner(x)))


def test_constant_grad_reduces_to_plain_hessian():
    torch.manual_seed(0)
    model = _TinyNet()
    calib = [torch.randn(6, 16), torch.randn(4, 16)]
    # loss = sum(final y): dL/dy is constant 1 only at the FINAL layer 'out'
    plain = collect_hessians(model, ["out"], calib)["out"]
    guided = collect_hessians_guided(
        model, ["out"], calib, loss_fn=lambda m, b: m(b).sum())["out"]
    assert torch.allclose(guided, plain, atol=1e-4), (guided - plain).abs().max()


def test_token_weights_match_hand_built_sum():
    torch.manual_seed(1)
    model = _TinyNet()
    x = torch.randn(5, 16)
    w_tok = torch.tensor([1.0, 2.0, 0.5, 3.0, 0.1])
    # loss = sum_n w_n * sum_o y_{n,o}  ->  dL/dy_out[n, :] = w_n
    guided = collect_hessians_guided(
        model, ["out"], [x],
        loss_fn=lambda m, b: (m(b).sum(dim=1) * w_tok).sum())["out"]
    with torch.no_grad():
        x_out = torch.relu(model.inner(x))               # the input 'out' sees
    ref = (x_out * (w_tok ** 2).unsqueeze(1)).t() @ x_out    # g_n = w_n^2
    assert torch.allclose(guided, ref, atol=1e-4), (guided - ref).abs().max()


def test_collection_touches_no_weights_and_restores_flags():
    torch.manual_seed(2)
    model = _TinyNet()
    model.inner.weight.requires_grad_(False)             # pre-existing mixed flags
    before = copy.deepcopy(model.state_dict())
    flags = {n: p.requires_grad for n, p in model.named_parameters()}
    collect_hessians_guided(model, ["inner", "out"], [torch.randn(4, 16)],
                            loss_fn=lambda m, b: m(b).pow(2).mean())
    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert {n: p.requires_grad for n, p in model.named_parameters()} == flags
    assert all(p.grad is None for p in model.parameters())


def test_warm_start_end_loss_end_to_end_and_asym_rejected():
    torch.manual_seed(3)
    model = _TinyNet()
    calib = [torch.randn(6, 16), torch.randn(6, 16)]
    report = ptq_warm_start(
        model, calib, mode="gptq_group", kind="counter_packed", group=8, C=11,
        progress=False, kernel_mode="torch", grid="itf", scale_refit="align",
        hessian_weighting="end_loss", loss_fn=lambda m, b: m(b).pow(2).mean())
    assert len(report.swapped) == 2
    with pytest.raises(ValueError, match="hessian_weighting"):
        ptq_warm_start(_TinyNet(), calib, mode="gptq_group", kind="counter_packed",
                       group=8, progress=False, calibration="asym",
                       hessian_weighting="end_loss")
