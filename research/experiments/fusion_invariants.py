"""Confirm the architectural INVARIANTS the kernel-fusion plan (PERF_ANATOMY levers) relies on, on
CPU, with no kernel needed. These turn 'hypothesis' into 'invariant confirmed -> only the kernel
remains'. What needs a GPU/Nsight (the actual speedups, overlap) is NOT claimed here.

  #2 prologue-fusion enabler: the FORWARD weight depends only on the visible ternary t, not on the
     counter c -> forward mixed-input is ternary x activation (BitNet/Marlin-ternary class), the
     6-bit packing only matters for the update path.
  #1 epilogue-fusion enabler: in rms_mode='lagged' the state-write tick is PER-ELEMENT (new
     state[o,i] depends on grad_w[o,i] only, given last step's v/scale) -> it can run in a tiled
     GEMM epilogue from the accumulator, no full-row reduction. In rms_mode='exact' the denom needs
     the whole row's g_sq -> the state-write is NOT per-element -> exact blocks epilogue fusion.
     (So #1 is unblocked SPECIFICALLY by the lagged one-pass mode we already ship.)
"""
import torch

from memory_native import RMSCounterLinear
from memory_native.counter import decode_state, encode_state


def confirm_forward_is_ternary_only():
    """#2: zeroing the counter c (keeping ternary t) must not change the forward."""
    torch.manual_seed(0)
    lay = RMSCounterLinear(16, 8, C=11).eval()
    x = torch.randn(4, 16)
    with torch.no_grad():
        y_before = lay(x).clone()
        t, c = decode_state(lay.state, lay.C)   # set every counter c to 0, keep ternary t
        lay.state.copy_(encode_state(t, torch.zeros_like(c), lay.C))
        y_after = lay(x)
    ok = torch.equal(y_before, y_after)
    print(f"  #2 forward independent of counter c (ternary-only):  {'CONFIRMED' if ok else 'FAIL'}"
          f"   max|Δ|={float((y_before-y_after).abs().max()):.2e}")
    return ok


def _row_dependency(mode):
    """Return the set of state positions that change when ONE grad_w element [0,j] is perturbed,
    with the SR RNG held fixed and the layer reset to the same start. Per-element <=> only {[0,j]}."""
    torch.manual_seed(0)
    lay = RMSCounterLinear(8, 4, C=11, lr=0.1, lr_scale=0.0, rms_mode=mode)
    lo, hi = 0, lay.out_features
    gw = torch.randn(hi, lay.in_features)
    # warm up v so the lagged denom is a real (nonzero) previous-step stat, not the init eps
    t, c = lay._decode_rows(lo, hi); torch.manual_seed(1)
    lay._update_tile(lo, hi, gw.clone(), t, c, lay.scale[lo:hi].clone())
    snap_state, snap_v, snap_scale = lay.state.clone(), lay.v.clone(), lay.scale.clone()

    def run(g):
        lay.state.copy_(snap_state); lay.v.copy_(snap_v); lay.scale.copy_(snap_scale)
        t, c = lay._decode_rows(lo, hi)
        torch.manual_seed(7)                    # identical SR draws for both runs
        lay._update_tile(lo, hi, g, t, c, lay.scale[lo:hi].clone())
        return lay.state.clone()

    base = run(gw.clone())
    g2 = gw.clone(); g2[0, 5] += 3.0            # perturb a single element of row 0
    pert = run(g2)
    changed = torch.nonzero(base != pert)       # which state bytes changed
    return changed, base.shape


def confirm_lagged_is_per_element():
    print("  #1 per-element dependency of the state-write (perturb grad_w[0,5]):")
    ok = True
    for mode in ("lagged", "exact"):
        changed, shape = _row_dependency(mode)
        rows = sorted(set(int(r) for r, _ in changed.tolist()))
        cols = sorted(set(int(cc) for _, cc in changed.tolist()))
        per_element = (changed.shape[0] <= 1 and (changed.tolist() in ([], [[0, 5]])))
        # 'lagged' must be per-element (only [0,5]); 'exact' must spill across the whole row
        verdict = ("per-element (epilogue-fusable)" if per_element
                   else f"row-coupled (changed cols={cols}) -> needs full-row reduction")
        print(f"     rms_mode={mode:7s}: {changed.shape[0]} byte(s) changed, rows={rows} -> {verdict}")
        if mode == "lagged":
            ok = ok and per_element
        else:
            ok = ok and (not per_element)
    print(f"  #1 lagged=per-element AND exact=row-coupled (confirms lagged unblocks epilogue fusion):"
          f"  {'CONFIRMED' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("=== Fusion-plan invariants (CPU, no kernel) ===")
    a = confirm_forward_is_ternary_only()
    b = confirm_lagged_is_per_element()
    print(f"\nOVERALL: {'ALL CONFIRMED' if (a and b) else 'SOME FAILED'}")
