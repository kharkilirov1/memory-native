"""Group-scale counter layer + Bonsai-format import/export on MLX.

Runs on any MLX backend (Linux mlx[cpu] in CI). No network: Bonsai-format inputs are
synthesized in the exact layout of the released checkpoints (ternary {-1,0,+1} weights,
one fp16 scale per group of 128 / MLX affine 2-bit quant tensors)."""
from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="mlx not installed (pip install mlx[cpu] on Linux)")

import mlx.nn as mnn  # noqa: E402

from memory_native_mlx import (  # noqa: E402
    GroupScaleCounterLinear,
    RMSCounterLinear,
    group_counter_from_dense,
    group_counter_from_quantized,
    to_mlx_quantized,
)


def _step(layer, x, y):
    def loss_fn(model, xb, yb):
        return mx.mean((model(xb) - yb) ** 2)

    loss, _ = mnn.value_and_grad(layer, loss_fn)(layer, x, y)
    mx.eval(loss, layer.parameters())
    return loss.item()


# --------------------------------------------------------------------------------------
# the layer itself
# --------------------------------------------------------------------------------------

def test_single_group_degenerates_to_row_scale_layer():
    """With one group spanning the row (group=in) and alpha=0, the group layer's update is
    exactly RMSCounterLinear's — same hash-SR stream, bit-identical state trajectory."""
    n_in, n_out = 24, 20
    row = RMSCounterLinear(n_in, n_out, C=11, lr=6e-3, lr_scale=2e-4, key=mx.random.key(1))
    grp = GroupScaleCounterLinear(n_in, n_out, group=n_in, C=11, lr=6e-3, lr_scale=2e-4,
                                  key=mx.random.key(1))
    # align initial scales (row layer scale is [out,1]; group layer [out,1] too here)
    grp.scale = mx.array(np.array(row.scale))
    assert np.array_equal(np.array(row.codes), np.array(grp.codes))

    x = mx.random.normal((32, n_in), key=mx.random.key(2))
    y = mx.random.normal((32, n_out), key=mx.random.key(3))
    for _ in range(20):
        _step(row, x, y)
        _step(grp, x, y)
    assert np.array_equal(np.array(row.codes), np.array(grp.codes)), "state trajectories diverged"
    assert np.allclose(np.array(row.scale), np.array(grp.scale))
    assert np.allclose(np.array(row.v), np.array(grp.v))


def test_group_layer_learns_group_scaled_teacher():
    """A teacher whose scale varies per group-of-8 (x7.4 range): the group layer must both
    learn it (>5x loss drop) and beat the row-scale layer, which cannot represent the
    within-row scale variation — the reason group scales exist."""
    n_in, n_out, g = 32, 16, 8
    t_true = mx.random.randint(-1, 2, (n_out, n_in), key=mx.random.key(4)).astype(mx.float32)
    s_true = mx.exp(mx.random.uniform(low=-2.0, high=0.0, shape=(n_out, n_in // g),
                                      key=mx.random.key(5)))
    w_true = mx.repeat(s_true, g, axis=1) * t_true
    x = mx.random.normal((256, n_in), key=mx.random.key(6))
    y = x @ w_true.T
    mx.eval(w_true, x, y)

    grp = GroupScaleCounterLinear(n_in, n_out, group=g, C=11, lr=4e-3, lr_scale=1e-2,
                                  key=mx.random.key(7))
    row = RMSCounterLinear(n_in, n_out, C=11, lr=4e-3, lr_scale=1e-2, key=mx.random.key(7))
    first = grp_loss = row_loss = None
    for _ in range(600):
        grp_loss = _step(grp, x, y)
        row_loss = _step(row, x, y)
        first = grp_loss if first is None else first
    assert grp_loss < first * 0.2, f"group layer failed to learn: {first} -> {grp_loss}"
    assert grp_loss < row_loss * 0.8, (
        f"group scales gave no advantage over a row scale: {grp_loss} vs {row_loss}")


def test_residual_alpha_homotopy():
    """alpha>0 exposes t + alpha*c/C; alpha=0 must return the pure ternary weight."""
    layer = GroupScaleCounterLinear(16, 8, group=8, C=11, key=mx.random.key(8))
    t = mx.random.randint(-1, 2, (8, 16), key=mx.random.key(9)).astype(mx.int32)
    c = mx.random.randint(-5, 6, (8, 16), key=mx.random.key(10)).astype(mx.int32)
    s = mx.abs(mx.random.normal((8, 2), key=mx.random.key(11))) + 0.1
    layer.load_group_state(s, t, c)

    layer.set_residual_alpha(0.0)
    w0 = np.array(layer.visible_weight())
    col = np.repeat(np.array(s), 8, axis=1)
    assert np.allclose(w0, col * np.array(t), atol=1e-6)

    layer.set_residual_alpha(0.5)
    w5 = np.array(layer.visible_weight())
    expected = col * (np.array(t) + 0.5 * np.array(c) / 11)
    assert np.allclose(w5, expected, atol=1e-5)


def test_act_order_permutation_groups_columns_correctly():
    """With an act-order perm, each scale group must cover the PERMUTED columns."""
    n_in, n_out, g = 12, 4, 4
    perm = mx.array(np.random.default_rng(0).permutation(n_in).astype(np.int32))
    layer = GroupScaleCounterLinear(n_in, n_out, group=g, key=mx.random.key(12), perm=perm)
    t = mx.ones((n_out, n_in), dtype=mx.int32)
    s = mx.array(np.arange(1, 1 + n_out * (n_in // g), dtype=np.float32).reshape(n_out, n_in // g))
    layer.load_group_state(s, t)
    w = np.array(layer.visible_weight())
    perm_np = np.array(perm)
    for j in range(n_in):
        gid = int(np.where(perm_np == j)[0][0]) // g
        assert np.allclose(w[:, j], np.array(s)[:, gid]), f"column {j} took the wrong group scale"


# --------------------------------------------------------------------------------------
# Bonsai-format import / export
# --------------------------------------------------------------------------------------

def _synthetic_bonsai_dense(out=16, in_=256, group=128, seed=0):
    """Dense fp16 weight in the exact Bonsai layout: per-group scale times ternary."""
    rng = np.random.default_rng(seed)
    t = rng.integers(-1, 2, size=(out, in_)).astype(np.float32)
    s = (rng.random((out, in_ // group)).astype(np.float32) * 0.05 + 0.01)
    w = np.repeat(s, group, axis=1) * t
    return mx.array(w.astype(np.float16)), t, s


def test_import_dense_bonsai_layout():
    w16, t, s = _synthetic_bonsai_dense()
    layer, err = group_counter_from_dense(w16, group=128)
    assert err < 1e-3
    # visible weight reproduces the checkpoint (fp16-cast tolerance)
    assert np.allclose(np.array(layer.visible_weight()), np.array(w16.astype(mx.float32)),
                       atol=1e-3)
    # and the recovered ternary is the true one wherever the scale is nonzero
    tt, _ = layer._decode()
    assert np.array_equal(np.array(tt), t)


def test_import_rejects_non_ternary():
    w = mx.random.normal((8, 256), key=mx.random.key(13))  # gaussian: not ternary
    with pytest.raises(ValueError):
        group_counter_from_dense(w, group=128)


def test_import_mlx_2bit_quant_roundtrip():
    """MLX affine 2-bit tensors (the -mlx-2bit release layout: q=t+1, scale=s, bias=-s)
    -> counter layer. Note mx.quantize itself cannot produce this layout losslessly (its
    fitted 2-bit grid has no zero), hence the manual construction — same as the releases."""
    from memory_native_mlx import ternary_to_mlx_quant

    w16, t, s = _synthetic_bonsai_dense(out=8, in_=256, seed=1)
    q, qs, qb = ternary_to_mlx_quant(mx.array(t.astype(np.int32)), mx.array(s), group=128)
    # sanity: these tensors ARE a valid mlx quant triple for the native kernels
    back = mx.dequantize(q, qs, qb, group_size=128, bits=2)
    assert np.allclose(np.array(back), np.array(w16.astype(mx.float32)), atol=1e-3)
    layer, err = group_counter_from_quantized(q, qs, qb, group=128, bits=2)
    assert err < 1e-3
    tt, _ = layer._decode()
    assert np.array_equal(np.array(tt), t)


def test_imported_layer_finetunes_and_exports_back():
    """The full loop: Bonsai checkpoint -> trainable counter -> fine-tune -> export to
    MLX native quant -> quantized_matmul output parity with the layer's own forward."""
    w16, _, _ = _synthetic_bonsai_dense(out=16, in_=256, seed=2)
    layer, _ = group_counter_from_dense(w16, group=128, lr=4e-3, lr_scale=1e-3)

    x = mx.random.normal((64, 256), key=mx.random.key(14))
    y = mx.random.normal((64, 16), key=mx.random.key(15))
    before = np.array(layer.codes)
    first = None
    for _ in range(60):
        loss = _step(layer, x, y)
        first = loss if first is None else first
    assert loss < first, "imported layer failed to fine-tune"
    assert (np.array(layer.codes) != before).any(), "fine-tune never touched the state"

    q, qs, qb = to_mlx_quantized(layer, bits=2)
    layer.eval()
    y_layer = np.array(layer(x))
    y_quant = np.array(mx.quantized_matmul(x, q, qs, qb, group_size=128, bits=2))
    assert np.allclose(y_layer, y_quant, atol=1e-3), "native-kernel inference diverges"
