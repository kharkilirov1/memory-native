"""MLX port validation — cross-checked against the PyTorch reference implementation.

Runs on any MLX backend: on Linux CI this is the mlx[cpu] wheel (how the port was
developed); on an Apple-silicon Mac the same tests run on Metal, plus the fused Metal
kernel parity gate engages. Skips cleanly when mlx (or torch, for the cross-checks) is
not installed.
"""
from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="mlx not installed (pip install mlx[cpu] on Linux)")

import mlx.nn as mnn  # noqa: E402

from memory_native_mlx import (  # noqa: E402
    PackedRMSCounterLinear,
    ReversibleCouplingBlock,
    ReversibleSequence,
    ReversibleSequential,
    RMSCounterLinear,
    counter_update_hashsr,
    decode_state,
    encode_state,
    pack_codes,
    unpack_codes,
)

torch = pytest.importorskip("torch", reason="torch needed for cross-implementation parity")

import memory_native as tref  # noqa: E402
import memory_native.fused_update as tfu  # noqa: E402
import memory_native.packed as tpk  # noqa: E402


# --------------------------------------------------------------------------------------
# bit-level parity with the torch reference
# --------------------------------------------------------------------------------------

def test_encode_decode_matches_torch():
    rng = np.random.default_rng(0)
    C = 11
    t = rng.integers(-1, 2, size=(32, 64)).astype(np.int16)
    c = rng.integers(-(C - 1), C, size=(32, 64)).astype(np.int16)

    mlx_codes = encode_state(mx.array(t.astype(np.int32)), mx.array(c.astype(np.int32)), C)
    torch_codes = tref.encode_state(torch.from_numpy(t), torch.from_numpy(c), C)
    assert np.array_equal(np.array(mlx_codes), torch_codes.numpy())

    mt, mc = decode_state(mlx_codes, C)
    assert np.array_equal(np.array(mt), t.astype(np.int32))
    assert np.array_equal(np.array(mc), c.astype(np.int32))


def test_pack_unpack_matches_torch():
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 63, size=(16, 128)).astype(np.uint8)

    mlx_packed = pack_codes(mx.array(codes))
    torch_packed = tpk.pack_codes(torch.from_numpy(codes))
    assert np.array_equal(np.array(mlx_packed), torch_packed.numpy())

    roundtrip = unpack_codes(mlx_packed, 128)
    assert np.array_equal(np.array(roundtrip), codes)


def test_hash_u32_matches_torch():
    from memory_native_mlx import hash_u32, uniform01

    x = np.arange(0, 100000, 7, dtype=np.uint32)
    got = np.array(hash_u32(mx.array(x)))
    exp = tfu.hash_u32(torch.from_numpy(x.astype(np.int64))).numpy().astype(np.uint32)
    assert np.array_equal(got, exp)
    ru = np.array(uniform01(mx.array(x)))
    re = tfu.uniform01(torch.from_numpy(x.astype(np.int64))).numpy()
    assert np.allclose(ru, re)


def test_hashsr_update_matches_torch_reference():
    """The full RMS+SR update vs memory_native.fused_update.counter_update_hashsr.

    Same deterministic hash-SR stream on both sides; the only permitted difference is fp
    reduction order in the row stats, which can tip an SR boundary on a vanishing fraction
    of weights (the same caveat the Triton kernel documents vs this very reference)."""
    rng = np.random.default_rng(2)
    out, in_, C = 32, 64, 8
    t = rng.integers(-1, 2, size=(out, in_)).astype(np.int16)
    c = rng.integers(-(C - 1), C, size=(out, in_)).astype(np.int16)
    codes = tref.encode_state(torch.from_numpy(t), torch.from_numpy(c), C)
    scale = np.full((out, 1), 0.11, dtype=np.float32)
    v = (rng.random((out, 1)) * 1e-4).astype(np.float32)
    gw = (rng.standard_normal((out, in_)) * 0.01).astype(np.float32)
    kw = dict(C=C, lr=0.04, lr_scale=2e-4, rms_beta=0.9, rms_eps=1e-3, seed=12345)

    for lagged in (False, True):
        t_scale = torch.from_numpy(scale.copy())
        t_v = torch.from_numpy(v.copy())
        exp_codes = tfu.counter_update_hashsr(
            codes.clone(), t_scale, t_v, torch.from_numpy(gw), lagged=lagged, **kw)

        got_codes, got_scale, got_v = counter_update_hashsr(
            mx.array(codes.numpy()), mx.array(scale), mx.array(v), mx.array(gw),
            lagged=lagged, **kw)
        mx.eval(got_codes, got_scale, got_v)

        same = np.array(got_codes) == exp_codes.numpy()
        assert same.mean() > 0.99, f"codes disagree beyond SR-boundary noise: {1 - same.mean():.4f}"
        assert np.allclose(np.array(got_scale), t_scale.numpy(), atol=1e-6)
        assert np.allclose(np.array(got_v), t_v.numpy(), atol=1e-9)


# --------------------------------------------------------------------------------------
# layer behavior (self-update through the custom VJP)
# --------------------------------------------------------------------------------------

def _sgd_free_step(layer, x, y):
    """One training step: value_and_grad drives the VJP; no optimizer exists at all."""

    def loss_fn(model, xb, yb):
        return mx.mean((model(xb) - yb) ** 2)

    loss, _ = mnn.value_and_grad(layer, loss_fn)(layer, x, y)
    mx.eval(loss, layer.parameters())
    return loss.item()


def test_layer_updates_on_raw_input_and_no_trainable_weights():
    """The MLX analogue of the torch 'raw input' contract: the layer must self-update even
    when nothing upstream needs a gradient — the tap makes the VJP fire."""
    layer = RMSCounterLinear(32, 32, C=11, lr=4e-3)
    assert list(layer.trainable_parameters().keys()) == ["tap"]

    x = mx.random.normal((64, 32), key=mx.random.key(1))
    y = mx.random.normal((64, 32), key=mx.random.key(2))
    before = np.array(layer.codes)
    _sgd_free_step(layer, x, y)
    after = np.array(layer.codes)
    assert (before != after).any(), "layer never updated on raw input"


def test_eval_mode_and_plain_call_do_not_update():
    layer = RMSCounterLinear(16, 16, C=11)
    x = mx.random.normal((8, 16), key=mx.random.key(3))
    before = np.array(layer.codes)
    mx.eval(layer(x))  # plain call: no VJP -> no update
    layer.eval()
    _sgd_free_step(layer, x, mx.zeros((8, 16)))  # eval mode: VJP fires but update is gated
    assert np.array_equal(before, np.array(layer.codes))


def test_teacher_recovery():
    """Port of test_learning.test_teacher_recovery_raw_input (same arch/lr/thresholds)."""
    import math

    n, N, C = 16, 256, 11
    teacher_scale = 0.25
    tw = mx.random.randint(-1, 2, (n, n), key=mx.random.key(7)).astype(mx.float32)
    x = mx.random.normal((N, n), key=mx.random.key(8))
    y = x @ (teacher_scale * tw).T
    mx.eval(tw, x, y)

    base = math.sqrt(3.0 / (2.0 * n))
    layer = RMSCounterLinear(n, n, C=C, lr=0.005, lr_scale=0.0,
                             init_gain=teacher_scale / base, key=mx.random.key(9))
    for _ in range(400):
        _sgd_free_step(layer, x, y)
    layer.eval()
    mse = mx.mean((layer(x) - y) ** 2).item()
    assert mse < 5e-3, f"counter did not recover teacher: mse={mse}"


def test_loss_decreases():
    layer = RMSCounterLinear(32, 32, C=11, lr=4e-3, lr_scale=2e-4, key=mx.random.key(4))
    x = mx.random.normal((64, 32), key=mx.random.key(5))
    target = mx.random.normal((64, 32), key=mx.random.key(6))
    layer.eval()
    first = mx.mean((layer(x) - target) ** 2).item()
    layer.train()
    for _ in range(300):
        _sgd_free_step(layer, x, target)
    layer.eval()
    last = mx.mean((layer(x) - target) ** 2).item()
    assert last < first


def test_packed_matches_unpacked_bit_for_bit():
    """hash-SR is deterministic, so the packed layer must track the unpacked one exactly."""
    kw = dict(C=11, lr=6e-3, lr_scale=2e-4, key=mx.random.key(10))
    a = RMSCounterLinear(24, 20, **kw)
    b = PackedRMSCounterLinear(24, 20, **kw)
    assert np.array_equal(np.array(a.codes), np.array(unpack_codes(b.codes, 24)))

    x = mx.random.normal((32, 24), key=mx.random.key(11))
    y = mx.random.normal((32, 20), key=mx.random.key(12))
    for _ in range(25):
        _sgd_free_step(a, x, y)
        _sgd_free_step(b, x, y)
    assert np.array_equal(np.array(a.codes), np.array(unpack_codes(b.codes, 24)))
    assert np.allclose(np.array(a.scale), np.array(b.scale))
    assert np.allclose(np.array(a.v), np.array(b.v))


def test_mixed_model_with_adamw_head():
    """Counter layer + ordinary trainable modules: mlx AdamW trains the head, the counter
    trains itself, in one value_and_grad step (the standard MLX loop)."""
    import mlx.optimizers as optim

    class Model(mnn.Module):
        def __init__(self):
            super().__init__()
            self.counter = RMSCounterLinear(16, 32, C=11, lr=6e-3, key=mx.random.key(13))
            self.head = mnn.Linear(32, 4)

        def __call__(self, x):
            return self.head(mnn.relu(self.counter(x)))

    model = Model()
    opt = optim.AdamW(learning_rate=1e-2)
    x = mx.random.normal((64, 16), key=mx.random.key(14))
    y = mx.random.normal((64, 4), key=mx.random.key(15))

    def loss_fn(m, xb, yb):
        return mx.mean((m(xb) - yb) ** 2)

    vg = mnn.value_and_grad(model, loss_fn)
    codes_before = np.array(model.counter.codes)
    head_before = np.array(model.head.weight)
    first = None
    for _ in range(60):
        loss, grads = vg(model, x, y)
        opt.update(model, grads)
        mx.eval(loss, model.parameters(), opt.state)
        first = loss.item() if first is None else first
    assert loss.item() < first
    assert (np.array(model.counter.codes) != codes_before).any(), "counter never self-updated"
    assert (np.array(model.head.weight) != head_before).any(), "AdamW never touched the head"
    # tap must stay (numerically) zero: it gets zero grads and 0 is a weight-decay fixpoint.
    assert abs(model.counter.tap.item()) < 1e-6


# --------------------------------------------------------------------------------------
# reversible
# --------------------------------------------------------------------------------------

def _make_blocks(dim, n, key0):
    blocks = []
    for i in range(n):
        d = dim // 2
        F = mnn.Linear(d, d, bias=False)
        G = mnn.Linear(d, d, bias=False)
        F.weight = mx.random.normal((d, d), key=mx.random.key(key0 + 2 * i)) * 0.2
        G.weight = mx.random.normal((d, d), key=mx.random.key(key0 + 2 * i + 1)) * 0.2
        blocks.append(ReversibleCouplingBlock(dim, F, G))
    return blocks


def test_reversible_sequence_matches_sequential_grads():
    """The O(1)-memory chain must produce the same loss and the same parameter/input grads
    as the plain differentiated stack (the torch port's grad-check, on MLX)."""
    dim, n = 16, 4
    blocks = _make_blocks(dim, n, 100)
    seq = ReversibleSequential(blocks)
    rev = ReversibleSequence(blocks)  # same underlying modules -> same params

    x = mx.random.normal((8, dim), key=mx.random.key(200))
    y = mx.random.normal((8, dim), key=mx.random.key(201))

    def loss_ref(m, xb):
        return mx.mean((m(xb) - y) ** 2)

    l_ref, g_ref = mnn.value_and_grad(seq, loss_ref)(seq, x)
    l_rev, g_rev = mnn.value_and_grad(rev, loss_ref)(rev, x)
    mx.eval(l_ref, g_ref, l_rev, g_rev)
    assert abs(l_ref.item() - l_rev.item()) < 1e-5

    from mlx.utils import tree_flatten
    fr = dict(tree_flatten(g_ref))
    fv = dict(tree_flatten(g_rev))
    assert set(fr) == set(fv)
    for k in fr:
        assert np.allclose(np.array(fr[k]), np.array(fv[k]), atol=1e-4), f"grad mismatch at {k}"


def test_reversible_anchored_matches_pure():
    dim, n = 12, 6
    blocks = _make_blocks(dim, n, 300)
    pure = ReversibleSequence(blocks)
    anch = ReversibleSequence(blocks, anchor_every=2)
    x = mx.random.normal((4, dim), key=mx.random.key(400))
    y = mx.random.normal((4, dim), key=mx.random.key(401))

    def loss_fn(m, xb):
        return mx.mean((m(xb) - y) ** 2)

    l1, g1 = mnn.value_and_grad(pure, loss_fn)(pure, x)
    l2, g2 = mnn.value_and_grad(anch, loss_fn)(anch, x)
    mx.eval(l1, g1, l2, g2)
    assert abs(l1.item() - l2.item()) < 1e-5
    from mlx.utils import tree_flatten
    f1, f2 = dict(tree_flatten(g1)), dict(tree_flatten(g2))
    for k in f1:
        assert np.allclose(np.array(f1[k]), np.array(f2[k]), atol=1e-4)


def test_reversible_with_counter_layers_updates_once_per_step():
    """Counter layers as F/G inside a reversible chain: the whole method on MLX. Each
    counter must self-update exactly once per step (during the backward walk)."""
    dim = 16
    d = dim // 2
    blocks = [
        ReversibleCouplingBlock(
            dim,
            RMSCounterLinear(d, d, C=11, lr=6e-3, key=mx.random.key(500 + i)),
            RMSCounterLinear(d, d, C=11, lr=6e-3, key=mx.random.key(600 + i)),
        )
        for i in range(3)
    ]
    rev = ReversibleSequence(blocks)
    x = mx.random.normal((16, dim), key=mx.random.key(700))
    y = mx.random.normal((16, dim), key=mx.random.key(701))

    def loss_fn(m, xb):
        return mx.mean((m(xb) - y) ** 2)

    counters = [blocks[i].F for i in range(3)] + [blocks[i].G for i in range(3)]
    steps_before = [c._sr_step for c in counters]
    first = None
    for _ in range(40):
        loss, _ = mnn.value_and_grad(rev, loss_fn)(rev, x)
        mx.eval(loss, rev.parameters())
        first = loss.item() if first is None else first
    assert loss.item() < first, "reversible counter stack failed to learn"
    for c, s0 in zip(counters, steps_before):
        assert c._sr_step - s0 == 40, "counter did not update exactly once per step"


# --------------------------------------------------------------------------------------
# torch <-> mlx interop (the CUDA -> MacBook handoff)
# --------------------------------------------------------------------------------------

def test_torch_to_mlx_handoff_and_back():
    from memory_native_mlx.interop import export_counter_to_torch, mlx_counter_from_torch

    torch.manual_seed(0)
    tl = tref.RMSCounterLinear(16, 12, C=11, lr=6e-3, lr_scale=2e-4)
    tl.train()
    xt = torch.randn(32, 16)
    yt = torch.randn(32, 12)
    for _ in range(20):  # some torch training first
        loss = ((tl(xt) - yt) ** 2).mean()
        loss.backward()

    ml = mlx_counter_from_torch(tl)
    # forward parity on the handoff point
    with torch.no_grad():
        exp = tl(xt).numpy()
    ml.eval()
    got = np.array(ml(mx.array(xt.numpy())))
    assert np.allclose(got, exp, atol=1e-5)

    # continue training on MLX: it keeps learning
    ml.train()
    x = mx.array(xt.numpy())
    y = mx.array(yt.numpy())
    first = mx.mean((ml(x) - y) ** 2).item()
    for _ in range(100):
        _sgd_free_step(ml, x, y)
    ml.eval()
    assert mx.mean((ml(x) - y) ** 2).item() < first

    # and export back into torch, bit-faithfully
    tl2 = tref.RMSCounterLinear(16, 12, C=11, lr=6e-3, lr_scale=2e-4)
    export_counter_to_torch(ml, tl2)
    tt, tc = tl2._decode_rows(0, 12)
    mt, mc = decode_state(ml._codes(), ml.C)
    assert np.array_equal(tt.numpy().astype(np.int32), np.array(mt))
    assert np.array_equal(tc.numpy().astype(np.int32), np.array(mc))
    with torch.no_grad():
        exp2 = tl2(xt).numpy()
    assert np.allclose(np.array(ml(mx.array(xt.numpy()))), exp2, atol=1e-5)


def test_packed_torch_to_mlx_packed():
    from memory_native_mlx.interop import mlx_counter_from_torch

    torch.manual_seed(1)
    tl = tref.__dict__.get("PackedRMSCounterLinear") or tpk.PackedRMSCounterLinear
    tl = tl(16, 8, C=11)
    ml = mlx_counter_from_torch(tl)
    assert isinstance(ml, PackedRMSCounterLinear)
    # identical packed bytes (both pack 4 codes / 3 bytes, engine layout)
    assert np.array_equal(np.array(ml.codes), tl.state.numpy())


# --------------------------------------------------------------------------------------
# fused Metal kernel (engages on Apple silicon only)
# --------------------------------------------------------------------------------------

def test_metal_fused_update_matches_reference():
    from memory_native_mlx.metal_update import fused_counter_update_metal, metal_available

    if not metal_available():
        pytest.skip("Metal GPU not available (run on an Apple-silicon Mac)")
    rng = np.random.default_rng(3)
    out, in_, C = 16, 32, 8
    codes = rng.integers(0, 3 * (2 * C - 1), size=(out, in_)).astype(np.uint8)
    packed = pack_codes(mx.array(codes))
    scale = mx.full((out, 1), 0.11)
    v = mx.array((rng.random((out, 1)) * 1e-4).astype(np.float32))
    gw = mx.array((rng.standard_normal((out, in_)) * 0.01).astype(np.float32))
    kw = dict(C=C, lr=0.04, lr_scale=2e-4, rms_beta=0.9, rms_eps=1e-3, seed=99)

    ns, nsc, nv = fused_counter_update_metal(packed, scale, v, gw, **kw)
    ref_codes, ref_scale, ref_v = counter_update_hashsr(
        mx.array(codes), scale, v, gw, **kw)
    mx.eval(ns, nsc, nv, ref_codes, ref_scale, ref_v)
    got = np.array(unpack_codes(ns, in_))
    same = got == np.array(ref_codes)
    assert same.mean() > 0.99
    assert np.allclose(np.array(nsc), np.array(ref_scale), atol=1e-6)
    assert np.allclose(np.array(nv), np.array(ref_v), atol=1e-9)
