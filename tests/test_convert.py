"""Model-level warm-start: swap_linears_to_counter over an arbitrary nn.Module (Phase 2 core)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from memory_native.convert import swap_linears_to_counter, CounterLinearWithBias
from memory_native.counter import CompactCounterLinear, weight_to_counter_state


class ToyNet(nn.Module):
    def __init__(self, vocab=40, d=32):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d, bias=False)
        self.fc2 = nn.Linear(d, d, bias=False)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        h = self.norm(self.emb(idx))
        h = F.relu(self.fc1(h))
        h = self.fc2(h)
        return self.head(h)


def test_swaps_linears_leaves_embed_norm_and_skipped_head():
    torch.manual_seed(0)
    m = ToyNet()
    report = swap_linears_to_counter(m, kind="counter_rms", skip=["head"], C=11)
    assert set(report.swapped) == {"fc1", "fc2"}
    assert report.skipped == ["head"]
    assert report.coeffs == 32 * 32 * 2
    # embedding + norm untouched, head left as a plain Linear (skipped), fc1/fc2 now counter layers
    assert isinstance(m.emb, nn.Embedding) and isinstance(m.norm, nn.LayerNorm)
    assert isinstance(m.head, nn.Linear)
    assert isinstance(m.fc1, CompactCounterLinear) and isinstance(m.fc2, CompactCounterLinear)
    # forward still runs end-to-end
    out = m(torch.randint(0, 40, (2, 8)))
    assert out.shape == (2, 8, 40) and torch.isfinite(out).all()


def test_swapped_linear_is_faithful_to_ternary_reconstruction():
    torch.manual_seed(1)
    m = ToyNet()
    w1 = m.fc1.weight.detach().clone()
    swap_linears_to_counter(m, kind="counter_rms", skip=["head"], C=11)
    x = torch.randn(4, 32)
    with torch.no_grad():
        y = m.fc1(x)
        s, t, _ = weight_to_counter_state(w1, C=11)
        y_ref = F.linear(x, s * t.float())
    assert torch.allclose(y, y_ref, atol=1e-5)


def test_bias_is_preserved():
    torch.manual_seed(2)
    lin = nn.Linear(16, 16, bias=True)
    m = nn.Sequential(lin)
    swap_linears_to_counter(m, kind="counter_rms", C=11)
    assert isinstance(m[0], CounterLinearWithBias)
    x = torch.randn(3, 16)
    with torch.no_grad():
        y = m(x)
        y_ref = m[0].counter(x) + m[0].bias
    assert torch.allclose(y, y_ref, atol=1e-6)
    assert torch.allclose(m[0].bias, lin.bias, atol=1e-6)


def test_packed_kind_swaps_and_matches_unpacked():
    torch.manual_seed(3)
    a = ToyNet(); b = ToyNet(); b.load_state_dict(a.state_dict())
    swap_linears_to_counter(a, kind="counter_rms", skip=["head"], C=11)
    swap_linears_to_counter(b, kind="counter_packed", skip=["head"], C=11)
    idx = torch.randint(0, 40, (2, 8))
    with torch.no_grad():
        assert torch.allclose(a(idx), b(idx), atol=1e-5)


def test_swapped_model_is_trainable():
    # the swapped counter layers self-update in backward: state evolves and grads flow.
    torch.manual_seed(4)
    m = ToyNet()
    swap_linears_to_counter(m, kind="counter_rms", skip=["head"], C=11, lr=0.02, lr_scale=1e-4)
    idx = torch.randint(0, 40, (4, 8))
    tgt = torch.randint(0, 40, (4, 8))
    for _ in range(30):
        F.cross_entropy(m(idx).reshape(-1, 40), tgt.reshape(-1)).backward()
    with torch.no_grad():                                  # eval forward must not self-update
        l1 = F.cross_entropy(m(idx).reshape(-1, 40), tgt.reshape(-1)).item()
    assert torch.isfinite(torch.tensor(l1))
    # the counter state actually moved (updates fired)
    flips = sum(int(c.weight_flips) for c in (m.fc1, m.fc2))
    assert flips > 0
