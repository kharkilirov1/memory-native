"""Unit gate for salient_refit='align' (the L1 lever, results/SALIENT_REFIT_WITNESS.md).

The refit touches ONLY the salient channel: ternary codes, scales, perm and the residual
counter are bit-identical to the salient_refit='none' solve; the H-error never increases
(the refit is the exact minimizer over the salient coordinates with everything else
fixed)."""
import torch

from memory_native.donor.ptq import solve_group_state


def _toy_layer(out=24, cols=64, seed=0):
    torch.manual_seed(seed)
    w = torch.randn(out, cols) * 0.1
    X = torch.randn(256, cols)
    H = X.t() @ X / X.shape[0]
    return w, H


def _h_err(w, Q, H):
    E = (w - Q).double()
    return float(torch.einsum("oi,ij,oj->", E, H.double(), E))


def test_refit_only_moves_salient_and_never_hurts():
    w, H = _toy_layer()
    kw = dict(group=16, C=11, grid="itf", scale_refit="align", refine_iters=1,
              salient_first=0.05, salient_scope="layer")
    (S0, t0, c0, p0, idx0, val0), Q0 = solve_group_state(w, H, salient_refit="none", **kw)
    (S1, t1, c1, p1, idx1, val1), Q1 = solve_group_state(w, H, salient_refit="align", **kw)
    assert torch.equal(t0, t1) and torch.equal(S0, S1)
    assert torch.equal(c0, c1) and torch.equal(p0, p1)
    assert torch.equal(idx0, idx1)
    assert not torch.equal(val0, val1), "refit must move the salient values"
    cols = w.shape[1]
    mask = torch.zeros(w.numel(), dtype=torch.bool)
    mask[idx0.long()] = True
    assert torch.equal(Q0.reshape(-1)[~mask], Q1.reshape(-1)[~mask])
    e0, e1 = _h_err(w, Q0, H), _h_err(w, Q1, H)
    assert e1 <= e0 * (1.0 + 1e-9), (e0, e1)
    assert e1 < e0, "refit should strictly reduce the training-H error here"


def test_refit_noop_without_salient():
    w, H = _toy_layer(seed=1)
    kw = dict(group=16, C=11, grid="sym", salient_first=0.0)
    (_, t0, *_), Q0 = solve_group_state(w, H, salient_refit="none", **kw)
    (_, t1, *_), Q1 = solve_group_state(w, H, salient_refit="align", **kw)
    assert torch.equal(t0, t1) and torch.equal(Q0, Q1)
