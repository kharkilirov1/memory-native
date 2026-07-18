"""Carry/saturation semantics of the counter transition.

The fused kernel and its CPU reference (fused_update.counter_update_hashsr) pin a BLOCKED
flip's residual to the counter edge +-(C-1): a weight already at t=+-1 whose counter
overflows further out keeps its accumulated pressure at the wall. The torch path's
`blocked` branch was dead code (in-place clamp_ aliased `proposed_t` and `new_t`, so
`proposed_t != new_t` was all-False) and silently RESET that pressure instead -- a real
torch-vs-kernel divergence, not a design choice. These tests hold the torch path to the
kernel-reference semantics."""
import pytest

torch = pytest.importorskip("torch")

from memory_native.counter import _carry_resolve


def _kernel_reference(cc, t_f, C):
    """Verbatim carry block of fused_update.counter_update_hashsr (the Triton kernel mirrors it)."""
    carry = torch.trunc(cc / C)
    rem = cc - carry * C
    nt = t_f + carry
    ct = nt.clamp(-1, 1)
    rem = torch.where(ct != nt, torch.sign(cc) * (C - 1), rem).clamp_(-(C - 1), C - 1)
    return ct, rem


def test_saturated_counter_pins_to_edge():
    # t=+1 (at the wall), counter overflows upward: cc=+9 at C=8. The flip is blocked, so
    # the residual must PIN to +7 (keep the pressure), not reset to cc - C = +1.
    new_t, rem = _carry_resolve(torch.tensor([9.0]), torch.tensor([1.0]), 8)
    assert float(new_t) == 1.0
    assert float(rem) == 7.0
    # mirror case downward
    new_t, rem = _carry_resolve(torch.tensor([-9.0]), torch.tensor([-1.0]), 8)
    assert float(new_t) == -1.0
    assert float(rem) == -7.0


def test_carry_resolve_matches_kernel_reference_exhaustively():
    C = 8
    ts, ccs = [], []
    for t in (-1.0, 0.0, 1.0):
        for cc in range(-3 * C, 3 * C + 1):
            ts.append(t)
            ccs.append(float(cc))
    t_f = torch.tensor(ts)
    cc = torch.tensor(ccs)
    ref_t, ref_rem = _kernel_reference(cc.clone(), t_f.clone(), C)
    new_t, rem = _carry_resolve(cc.clone(), t_f.clone(), C)
    assert torch.equal(new_t, ref_t)
    assert torch.equal(rem, ref_rem)


def test_unblocked_transitions_unchanged():
    # plain sub-threshold tick (no carry) and a clean single flip keep their old semantics
    C = 8
    new_t, rem = _carry_resolve(torch.tensor([3.0]), torch.tensor([0.0]), C)
    assert float(new_t) == 0.0 and float(rem) == 3.0
    new_t, rem = _carry_resolve(torch.tensor([8.0]), torch.tensor([0.0]), C)
    assert float(new_t) == 1.0 and float(rem) == 0.0
