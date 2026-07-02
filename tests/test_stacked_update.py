"""Fused batched counter update for stacked experts (stacked_update.py).

CPU: the hash-SR stacked reference equals counter_update_hashsr on the flattened rows (global
element indexing) and honors the active mask. CUDA: the Triton kernel matches the reference up to
one SR quantum on an O(1) fraction (chunked reduction), including a bf16 grad_w input.
"""
import pytest
import torch

from memory_native.counter import decode_state, encode_state
from memory_native.fused_update import HAS_TRITON, counter_update_hashsr
from memory_native.stacked_update import stacked_update_hashsr, triton_stacked_update

CUDA = torch.cuda.is_available()
KW = dict(C=11, lr=0.02, lr_scale=2e-4, rms_beta=0.9, rms_eps=1e-3, seed=7)


def _stack(E=3, out=8, in_=16, device="cpu"):
    torch.manual_seed(0)
    C = KW["C"]
    st = encode_state(torch.randint(-1, 2, (E, out, in_)).short(),
                      torch.randint(-(C - 1), C, (E, out, in_)).short(), C).to(device)
    sc = torch.full((E, out, 1), 0.3, device=device)
    v = (torch.rand(E, out, 1, device=device) * 1e-2)
    gw = torch.randn(E, out, in_, device=device)
    return st, sc, v, gw


def test_stacked_reference_equals_flat_rows():
    """Full-active stack == counter_update_hashsr on the [E*out, in] flat view (same global elem
    indices, same seed) -- the reference is DEFINED by that equality."""
    E, out, in_ = 3, 8, 16
    st, sc, v, gw = _stack(E, out, in_)
    ref_c = st.reshape(E * out, in_).clone()
    ref_s = sc.reshape(E * out, 1).clone()
    ref_v = v.reshape(E * out, 1).clone()
    ref_new = counter_update_hashsr(ref_c, ref_s, ref_v, gw.reshape(E * out, in_), **KW)
    stacked_update_hashsr(st, sc, v, gw, torch.ones(E, dtype=torch.bool), **KW)
    assert torch.equal(st.reshape(E * out, in_), ref_new)
    assert torch.equal(sc.reshape(E * out, 1), ref_s)
    assert torch.equal(v.reshape(E * out, 1), ref_v)


def test_stacked_reference_active_mask():
    """Inactive experts keep state/scale/v bit-identical."""
    st, sc, v, gw = _stack()
    active = torch.tensor([True, False, True])
    s0, c0, v0 = st[1].clone(), sc[1].clone(), v[1].clone()
    stacked_update_hashsr(st, sc, v, gw, active, **KW)
    assert torch.equal(st[1], s0) and torch.equal(sc[1], c0) and torch.equal(v[1], v0)
    assert not torch.equal(st[0], _stack()[0][0])              # active experts did move


@pytest.mark.skipif(not (CUDA and HAS_TRITON), reason="needs CUDA + triton")
@pytest.mark.parametrize("grad_dtype", [torch.float32, torch.bfloat16])
def test_triton_stacked_matches_reference(grad_dtype):
    """Kernel vs reference on GPU: same hash-SR, chunked reduction -> at most one counter quantum
    on an O(1) fraction of weights (same tolerance as the packed fused kernel). bf16 grad_w is
    cast in-register -- the bf16 GEMM path feeds the update without an fp32 copy."""
    E, out, in_ = 4, 64, 512
    st, sc, v, gw = _stack(E, out, in_, device="cuda")
    active = torch.tensor([True, True, False, True], device="cuda")
    C = KW["C"]

    ref_st, ref_sc, ref_v = st.clone(), sc.clone(), v.clone()
    stacked_update_hashsr(ref_st, ref_sc, ref_v, gw.to(grad_dtype).float(), active, **KW)
    triton_stacked_update(st, sc, v, gw.to(grad_dtype), active, **KW)

    assert torch.allclose(sc, ref_sc, atol=1e-5)
    assert torch.allclose(v, ref_v, atol=1e-6)
    gt, gc = decode_state(st, C)
    rt, rc = decode_state(ref_st, C)
    pos_got = gt.float() + gc.float() / C
    pos_ref = rt.float() + rc.float() / C
    frac = (st != ref_st).float().mean().item()
    assert frac < 2e-3, f"too many SR-boundary flips: {frac:.2e}"
    assert (pos_got - pos_ref).abs().max().item() < 1.0 / C + 1e-6
    assert torch.equal(st[2], ref_st[2])                        # inactive expert untouched
