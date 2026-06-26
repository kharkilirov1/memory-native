import torch
import torch.nn as nn

from memory_native import GaLoreAdamW, LoMo, build_optimizer


def _fit(opt_factory, steps=300):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 64))
    x = torch.randn(128, 64)
    target = torch.randn(128, 64)
    opt = opt_factory(model.parameters())
    first = None
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = ((model(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    with torch.no_grad():
        last = ((model(x) - target) ** 2).mean().item()
    return first, last


def test_galore_reduces_loss():
    first, last = _fit(lambda p: GaLoreAdamW(p, lr=1e-2, rank=16, update_proj_gap=50))
    assert last < first * 0.9, (first, last)


def test_lomo_reduces_loss():
    first, last = _fit(lambda p: LoMo(p, lr=5e-2))
    assert last < first * 0.95, (first, last)


def test_lomo_holds_no_optimizer_state():
    model = nn.Linear(32, 32)
    opt = LoMo(model.parameters(), lr=1e-2)
    x = torch.randn(8, 32)
    (model(x) ** 2).mean().backward()  # hook updates params, frees grads
    opt.step()
    # LoMo keeps no per-parameter moment buffers
    assert all(len(s) == 0 for s in opt.state.values())


def test_build_optimizer_names():
    model = nn.Linear(8, 8)
    for name in ("adamw", "galore", "lomo"):
        assert build_optimizer(name, model.parameters(), lr=1e-3) is not None
