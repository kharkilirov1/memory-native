import math

import torch

from memory_native import RMSCounterLinear


def test_teacher_recovery_raw_input():
    """A counter layer must recover a ternary teacher, AND it must self-update even when the
    input does not require grad (no requires_grad on x)."""
    torch.manual_seed(0)
    n, N, C = 16, 256, 11
    teacher_scale = 0.25
    tw = torch.randint(-1, 2, (n, n)).float()
    x = torch.randn(N, n)  # NOTE: no requires_grad_
    y = x @ (teacher_scale * tw).t()

    base = math.sqrt(3.0 / (2.0 * n))
    layer = RMSCounterLinear(n, n, C=C, lr=0.005, lr_scale=0.0, init_gain=teacher_scale / base)
    layer.train()

    for _ in range(400):
        pred = layer(x)
        loss = ((pred - y) ** 2).mean()
        loss.backward()

    with torch.no_grad():
        mse = ((layer(x) - y) ** 2).mean().item()
    assert mse < 5e-3, f"counter did not recover teacher: mse={mse}"
    assert int(layer.weight_flips) > 0, "layer never updated on raw (no-grad) input"


def test_loss_decreases():
    torch.manual_seed(1)
    layer = RMSCounterLinear(32, 32, C=11, lr=4e-3, lr_scale=2e-4)
    layer.train()
    x = torch.randn(64, 32)
    target = torch.randn(64, 32)
    with torch.no_grad():  # measurement: no update, no graph
        first = ((layer(x) - target) ** 2).mean().item()
    for _ in range(300):
        loss = ((layer(x) - target) ** 2).mean()
        loss.backward()
    with torch.no_grad():
        last = ((layer(x) - target) ** 2).mean().item()
    assert last < first
