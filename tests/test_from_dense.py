"""Warm-start a counter layer from pretrained dense weights (Phase 1 of the finetune-pretrained
pipeline). Verifies the conversion primitive `weight_to_counter_state` and the `from_dense` /
`from_linear` constructors on RMSCounterLinear and the packed 6-bit subclass."""
import torch
import torch.nn.functional as F

from memory_native.counter import (
    weight_to_counter_state, decode_state, encode_state,
    RMSCounterLinear,
)
from memory_native.packed import PackedRMSCounterLinear


def test_state_ranges_and_codec_roundtrip():
    torch.manual_seed(0)
    w = torch.randn(32, 64)
    for C in (8, 11):
        s, t, c = weight_to_counter_state(w, C=C)
        assert s.shape == (32, 1)
        assert t.shape == w.shape and c.shape == w.shape
        assert set(t.unique().tolist()).issubset({-1, 0, 1})
        assert int(c.abs().max()) <= C - 1
        # the (t, c) pair survives the uint8 state codec exactly
        code = encode_state(t, c, C)
        t2, c2 = decode_state(code, C)
        assert torch.equal(t2, t) and torch.equal(c2, c)


def test_near_ternary_donor_is_lossless():
    # a donor that is ALREADY scale*ternary imports with ~0 forward error and ~0 residual counter.
    torch.manual_seed(1)
    s0 = torch.rand(16, 1) * 0.5 + 0.1
    tern = torch.randint(-1, 2, (16, 64)).float()
    w = s0 * tern
    s, t, c = weight_to_counter_state(w, C=11)
    recon = s * t.float()                                   # what the layer forward uses
    err = (recon - w).norm() / w.norm().clamp_min(1e-9)
    assert err < 1e-3, err
    assert c.abs().float().mean() < 0.05                    # essentially no residual to seed


def test_from_linear_forward_matches_reconstruction():
    torch.manual_seed(2)
    lin = torch.nn.Linear(64, 32, bias=False)
    layer = RMSCounterLinear.from_linear(lin, C=11)
    x = torch.randn(8, 64)
    with torch.no_grad():                                   # no_grad -> pure forward, no self-update
        y = layer(x)
        s, t, c = weight_to_counter_state(lin.weight, C=11)
        y_ref = F.linear(x, s * t.float())
    assert torch.allclose(y, y_ref, atol=1e-5)


def test_packed_and_unpacked_import_identically():
    torch.manual_seed(3)
    w = torch.randn(32, 64)                                 # in_features divisible by 4 for packing
    a = RMSCounterLinear.from_dense(w, C=11)
    b = PackedRMSCounterLinear.from_dense(w, C=11)
    ta, ca = decode_state(a.state, a.C)
    tb, cb = decode_state(b._all_codes(), b.C)
    assert torch.equal(ta, tb) and torch.equal(ca, cb)
    assert torch.allclose(a.scale, b.scale)
    x = torch.randn(4, 64)
    with torch.no_grad():
        assert torch.allclose(a(x), b(x), atol=1e-5)


def test_warmstart_beats_random_init_on_donor_outputs():
    # the whole point: reproducing a pretrained layer's outputs is far better from a warm-start
    # ternary import than from the random ternary init used when training from scratch.
    torch.manual_seed(4)
    donor = torch.nn.Linear(64, 64, bias=False)
    x = torch.randn(128, 64)
    with torch.no_grad():
        y = donor(x)
        warm = RMSCounterLinear.from_dense(donor.weight, C=11)
        rand = RMSCounterLinear(64, 64, C=11)
        err_warm = (warm(x) - y).pow(2).mean().item()
        err_rand = (rand(x) - y).pow(2).mean().item()
    assert err_warm < err_rand


def test_recovery_finetune_reduces_gap_to_representable_target():
    # a warm-started counter layer keeps training. NOTE: recovering a *full-precision* donor's
    # own outputs is not a single-layer win -- the TWN warm-start is already near the ternary
    # optimum in weight space, so per-layer self-update only adds stochastic-rounding noise (real
    # recovery is a network-level effect: composed layers + task loss + distillation, Phase 4).
    # Here we isolate that the self-update itself descends when there IS a reachable gap: import a
    # NOISY copy of a counter-representable (scale*ternary) target and finetune back toward it.
    torch.manual_seed(7)
    tern = torch.randint(-1, 2, (48, 48)).float()
    W = 0.2 * tern                                         # reachable by a counter layer
    x = torch.randn(128, 48)
    with torch.no_grad():
        y = x @ W.t()
    layer = RMSCounterLinear.from_dense(W + 0.06 * torch.randn_like(W), C=11, lr=0.01, lr_scale=1e-4)
    with torch.no_grad():
        l0 = (layer(x) - y).pow(2).mean().item()
    for _ in range(400):
        (layer(x) - y).pow(2).mean().backward()
    with torch.no_grad():
        l1 = (layer(x) - y).pow(2).mean().item()
    assert l1 < l0, (l0, l1)
