"""End-to-end PTQ import gate for solver-v3 packed group recovery."""
import copy

import torch
import torch.nn as nn

from memory_native.convert import CounterLinearWithBias
from memory_native.donor.ptq import gptq_group_ternary, ptq_warm_start
from memory_native.group_scale_packed import PackedGroupScaleCounterLinear


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(16, 12, bias=True)

    def forward(self, x):
        return self.proj(x)


def test_ptq_group_uses_packed_and_preserves_solver_reconstruction():
    torch.manual_seed(0)
    model = Tiny()
    original = copy.deepcopy(model)
    calib = [torch.randn(4, 5, 16), torch.randn(3, 5, 16)]
    xcal = torch.cat([x.reshape(-1, 16) for x in calib], dim=0)
    q, _, _ = gptq_group_ternary(
        original.proj.weight, xcal.t() @ xcal, group=8, refine_iters=2
    )
    report = ptq_warm_start(
        model, calib, mode="gptq_group", kind="counter_packed",
        group=8, C=11, progress=False, kernel_mode="torch",
    )
    assert report.coeffs == 16 * 12
    assert isinstance(model.proj, CounterLinearWithBias)
    assert isinstance(model.proj.counter, PackedGroupScaleCounterLinear)
    assert torch.allclose(model.proj.counter.visible_weight(), q, atol=1e-5)
    x = torch.randn(2, 16)
    assert torch.allclose(model(x), x @ q.t() + original.proj.bias, atol=1e-5)
