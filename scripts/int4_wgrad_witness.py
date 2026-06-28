"""int4 weight-gradient witness: does a low-bit correlation give the SAME flips as fp32?

The counter flip needs only the SIGN and in-row RANK of G = Delta^T X, not its fp32 value. So the
update correlation should run in int4 (INT4 IMMA on Turing+, ~4x fp16 / ~2x int8), not fp32. This
witness, before any CUDA kernel, checks two things on correlated (activation-like) data:

  1. fidelity of int4 / int8 / 1-bit(sign) vs the exact fp32 G: sign agreement, Spearman rank, and
     the overlap of the top-0.5% by |G| (the weights that actually flip);
  2. teacher recovery: does training the counter with the int4 correlation reach the same MSE as fp32.

    python scripts/int4_wgrad_witness.py
"""
from __future__ import annotations

import math

import torch

from memory_native import RMSCounterLinear
from memory_native.counter import decode_state, encode_state
from memory_native.int8_compute import int4_correlation, int8_correlation


def _correlated(M, D, rank, noise, g):
    """Activation-like: low-rank structure (correlated columns) + noise."""
    z = torch.randn(M, rank, generator=g)
    w = torch.randn(rank, D, generator=g) / math.sqrt(rank)
    return z @ w + noise * torch.randn(M, D, generator=g)


def _spearman_rows(a, b):
    ra = a.argsort(1).argsort(1).float(); rb = b.argsort(1).argsort(1).float()
    ra = ra - ra.mean(1, keepdim=True); rb = rb - rb.mean(1, keepdim=True)
    num = (ra * rb).sum(1)
    den = (ra.pow(2).sum(1).sqrt() * rb.pow(2).sum(1).sqrt()).clamp_min(1e-12)
    return (num / den).mean().item()


def _top_overlap(g_est, g_exact, frac=0.005):
    k = max(1, int(frac * g_exact.shape[1]))
    te = g_exact.abs().topk(k, dim=1).indices
    ts = g_est.abs().topk(k, dim=1).indices
    hits = sum(len(set(te[o].tolist()) & set(ts[o].tolist())) for o in range(te.shape[0]))
    return hits / (te.shape[0] * k)


def _onebit(delta, x):
    return delta.sign().t() @ x.sign()         # XNOR/popcount correlation (sign only)


def fidelity():
    g = torch.Generator().manual_seed(0)
    M, N, K = 4096, 256, 2048
    X = _correlated(M, K, rank=64, noise=0.3, g=g)
    D = _correlated(M, N, rank=64, noise=0.3, g=g)
    exact = D.t() @ X
    print("operator        sign-agree   Spearman-rank   top-0.5% overlap")
    print("-" * 64)
    for name, est in [("int4", int4_correlation(D, X)),
                      ("int8", int8_correlation(D, X)),
                      ("1-bit (XNOR)", _onebit(D, X))]:
        sign = (est.sign() == exact.sign()).float().mean().item()
        print(f"  {name:13s} {sign:8.2f}      {_spearman_rows(est, exact):8.3f}       {_top_overlap(est, exact):8.2f}")


def teacher_recovery(update_compute, steps=600):
    torch.manual_seed(0)
    n, N, C = 24, 256, 11
    ts = 0.25
    base = math.sqrt(3.0 / (2.0 * n))
    tw = torch.randint(-1, 2, (n, n)).float()
    x = torch.randn(N, n); y = x @ (ts * tw).t()
    lay = RMSCounterLinear(n, n, C=C, lr=0.02, lr_scale=2e-4, init_gain=ts / base,
                           update_compute=update_compute).train()
    for _ in range(steps):
        (lay(x) - y).pow(2).mean().backward()
    with torch.no_grad():
        return (lay(x) - y).pow(2).mean().item()


def speedup_ceiling():
    # measured isolated GEMM speedups vs fp on a T4 (results/ACCELERATION.md); int4 ~2x int8 IMMA
    print("\nstep-speedup framework (the 3 GEMMs per counter layer vs fp):")
    print("  forward  X T^T     : int8  ~x2.05   (measured)")
    print("  grad_x   Delta T    : int8  ~x2      (int8 GEMM)")
    print("  wgrad    Delta^T X  : int4  ~x3-4    (int4 IMMA ~2x the int8 presaved x1.45-2.16)")
    print("  -> if the 3 GEMMs are a fraction f of the step and go ~k x faster, the step speedup is")
    print("     1 / (1 - f + f/k). On d=2048 the GEMMs dominate the non-reversible cost, so with")
    print("     int8 forward + int4 wgrad + int8 grad_x the GEMM block is ~2.5-3x; the full-step")
    print("     ceiling depends on the reversible-recompute share (cut it with anchors).")


if __name__ == "__main__":
    print("=== fidelity: low-bit correlation vs exact fp32 G (correlated data) ===")
    fidelity()
    print("\n=== teacher recovery: final MSE (lower = same learning) ===")
    for uc in ("fp", "int8", "int4"):
        print(f"  update_compute={uc:4s} -> MSE {teacher_recovery(uc):.5f}")
    speedup_ceiling()
