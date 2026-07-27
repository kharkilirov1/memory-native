"""Calibrated ternary PTQ and trainable group-counter warm starts."""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..convert import CounterLinearWithBias, SwapReport
from ..counter import C_DEFAULT
from ..group_scale_counter import GroupScaleCounterLinear
from ..group_scale_packed import PackedGroupScaleCounterLinear

__all__ = [
    "optimal_ternary", "gptq_ternary", "gptq_group_ternary", "residual_counter",
    "group_residual_counter", "collect_hessians", "collect_hessians_guided",
    "quantize_dense_group_ternary",
    "ptq_warm_start", "itf_grid", "align_scales_output", "solve_group_state",
]


@torch.no_grad()
def optimal_ternary(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact per-row L2 minimizer of ||w-s*t|| over t in {-1,0,1}."""
    w = w.to(torch.float32)
    absw = w.abs()
    vals, _ = absw.sort(dim=1, descending=True)
    csum = vals.cumsum(dim=1)
    k = torch.arange(1, w.shape[1] + 1, device=w.device, dtype=w.dtype)
    kstar = (csum.pow(2) / k).argmax(dim=1, keepdim=True)
    s = (csum.gather(1, kstar) / (kstar + 1).to(w.dtype)).clamp_min(1e-8)
    thr = vals.gather(1, kstar)
    t = torch.sign(w) * (absw >= thr).to(w.dtype)
    return s, t


@torch.no_grad()
def residual_counter(w: torch.Tensor, s: torch.Tensor, t: torch.Tensor,
                     C: int = C_DEFAULT) -> torch.Tensor:
    c = ((w.to(torch.float32) / s - t) * C).round().clamp_(-(C - 1), C - 1)
    return c.to(torch.int16)


def _prep_hinv(H: torch.Tensor, W: torch.Tensor, percdamp: float):
    H = H.detach().to(torch.float32).clone()
    diag = torch.diagonal(H)
    dead = diag == 0
    if dead.any():
        H[dead, dead] = 1.0
        W[:, dead] = 0.0
    damp = percdamp * torch.mean(torch.diagonal(H))
    H += torch.eye(H.shape[0], device=H.device, dtype=H.dtype) * damp
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    return torch.linalg.cholesky(Hinv, upper=True), H


@torch.no_grad()
def gptq_ternary(w: torch.Tensor, H: torch.Tensor, *, C: int = C_DEFAULT,
                  blocksize: int = 128, percdamp: float = 0.01, act_order: bool = True,
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    W = w.detach().to(torch.float32).clone()
    cols = W.shape[1]
    H = H.detach().to(torch.float32).clone()
    diag = torch.diagonal(H)
    dead = diag == 0
    if dead.any():
        H[dead, dead] = 1.0
        W[:, dead] = 0.0
    if act_order:
        perm = torch.argsort(torch.diagonal(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
        invperm = torch.argsort(perm)
    s, _ = optimal_ternary(W)
    sq = s.squeeze(1)
    Hinv, _ = _prep_hinv(H, W, percdamp)
    Q = torch.zeros_like(W)
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        Wb = W[:, i1:i2].clone()
        Qb = torch.zeros_like(Wb)
        Eb = torch.zeros_like(Wb)
        Hb = Hinv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            wcol = Wb[:, j]
            q = (wcol / sq).round_().clamp_(-1, 1) * sq
            Qb[:, j] = q
            e = (wcol - q) / Hb[j, j]
            Eb[:, j] = e
            if j + 1 < i2 - i1:
                Wb[:, j + 1:] -= e.unsqueeze(1) * Hb[j, j + 1:].unsqueeze(0)
        Q[:, i1:i2] = Qb
        W[:, i1:i2] = Wb
        if i2 < cols:
            W[:, i2:] -= Eb @ Hinv[i1:i2, i2:]
    if act_order:
        Q = Q[:, invperm]
        W = W[:, invperm]
    t = (Q / s).round()
    c = residual_counter(W, s, t, C)
    return s, t.to(torch.int16), c


@torch.no_grad()
def itf_grid(Wg: torch.Tensor, *, iters: int = 3,
             s_init: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A5 — full asymmetric ternary grid fit for one (row, group-block): the ITF step of
    PT2-LLM adapted to the g128 layout.

    Coordinate descent between (i) assigning each weight to the NEAREST grid point of
    {-s_neg, 0, +s_pos} and (ii) the L2-optimal scale on each achieved support
    (s_pos = mean w on {t=+1}, s_neg = mean |w| on {t=-1}). Both steps are exact given
    the other, so the block MSE is non-increasing; 2-3 iterations are enough in practice.
    Asymmetry matters on skewed blocks (post-SwiGLU / outlier rows): the symmetric grid
    forces one scale onto two differently-shaped lobes.

    Wg: [out, g] fp32-ish. Returns (s_pos [out], s_neg [out], t [out, g] in {-1,0,+1}).

    Init note (measured on the 0.5B donor): seeding BOTH scales from one shared symmetric
    fit traps the descent on skewed blocks — the large shared scale keeps the
    opposite-sign lobe at 0, its support stays empty and its scale never updates; while a
    naive per-lobe MEAN init loses to optimal_ternary on plain gaussian blocks (it puts no
    mass on 0). The init below runs the exact per-row optimal ternary SEPARATELY on each
    lobe — each lobe gets its own optimal scale AND its own zeros."""
    w = Wg.to(torch.float32)
    if s_init is None:
        s_pos = optimal_ternary(w.clamp_min(0))[0].squeeze(1).clamp_min(1e-8)
        s_neg = optimal_ternary((-w).clamp_min(0))[0].squeeze(1).clamp_min(1e-8)
    else:
        s_pos = s_neg = s_init.clamp_min(1e-8)
    t = torch.zeros_like(w)
    for _ in range(iters):
        d0 = w.abs()
        dp = (w - s_pos.unsqueeze(1)).abs()
        dn = (w + s_neg.unsqueeze(1)).abs()
        t = torch.where((dp < d0) & (dp <= dn), torch.ones_like(w), torch.zeros_like(w))
        t = torch.where((dn < d0) & (dn < dp), -torch.ones_like(w), t)
        pos, neg = t > 0, t < 0
        sp = torch.where(pos.any(dim=1), (w * pos).sum(dim=1) / pos.sum(dim=1).clamp_min(1), s_pos)
        sn = torch.where(neg.any(dim=1), -(w * neg).sum(dim=1) / neg.sum(dim=1).clamp_min(1), s_neg)
        s_pos, s_neg = sp.clamp_min(1e-8), sn.clamp_min(1e-8)
    return s_pos, s_neg, t


@torch.no_grad()
def align_scales_output(w: torch.Tensor, T: torch.Tensor, H: torch.Tensor, *,
                        group: int = 128, grid: str = "sym",
                        ridge: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """A7 — exact activation-aware scale alignment: with the ternary SUPPORT fixed, solve
    for per-row group scales minimizing the OUTPUT error

        ||X(W - Q)||^2 = (w - q)^T H (w - q),   q = sum_k s_k * T_k,

    where T_k are the (group, sign)-block code matrices (K = G for sym, 2G for itf).
    Everything is formed from H = X^T X — no activations are stored:
        A[o,k,l] = <T_k H, T_l>_row-o,  b[o,k] = <T_k H, w>_row-o,  solve A s = b.
    Each T_k lives on its group block, so T_k @ H costs [out, bs] @ [bs, in].
    The positive clamp on s is the only approximation (unconstrained solve, then clamp).

    w, T, H must share the SAME column order (call in the permuted/group-aligned space).
    Returns (s [out, K] fp32 — for itf interleaved (g0_pos, g0_neg, g1_pos, ...) —,
    Q [out, in] fp32 reconstruction)."""
    w = w.to(torch.float32)
    T = T.to(torch.float32)
    out, cols = T.shape
    G = (cols + group - 1) // group
    codes, blocks = [], []
    for g in range(G):
        i1, i2 = g * group, min((g + 1) * group, cols)
        blk = slice(i1, i2)
        cg = T[:, blk]
        if grid == "itf":
            codes.append(cg.clamp(min=0)); blocks.append(blk)    # +1 on positives (s_pos)
            codes.append(cg.clamp(max=0)); blocks.append(blk)    # -1 on negatives (s_neg)
        else:
            codes.append(cg); blocks.append(blk)
    K = len(codes)
    A = w.new_zeros(out, K, K)
    b = w.new_zeros(out, K)
    for k in range(K):
        U = codes[k] @ H[blocks[k], :]                     # [out, in]
        b[:, k] = (U * w).sum(dim=1)
        for l in range(K):
            A[:, k, l] = (U[:, blocks[l]] * codes[l]).sum(dim=1)
    diag = A.diagonal(dim1=1, dim2=2)
    diag.add_(ridge * diag.mean(dim=1, keepdim=True).clamp_min(1e-12))
    s = torch.linalg.solve(A, b.unsqueeze(2)).squeeze(2).clamp_min(1e-8)
    Q = torch.zeros_like(w)
    for k in range(K):
        Q[:, blocks[k]] += s[:, k].unsqueeze(1) * codes[k]
    return s, Q


def _initial_group_scales(W: torch.Tensor, group: int, grid: str = "sym",
                          itf_iters: int = 3, smask: torch.Tensor | None = None
                          ) -> torch.Tensor:
    """Per-(row, group) start scales as [out, G, 2] = (s_pos, s_neg); sym keeps both equal.

    With a salient mask the grid is fit on the NON-salient remainder (BiLLM-style split:
    salient weights leave the ternary grid and must not pull its scale up)."""
    out, cols = W.shape
    n_groups = (cols + group - 1) // group
    S = torch.empty((out, n_groups, 2), device=W.device, dtype=torch.float32)
    for g in range(n_groups):
        i1, i2 = g * group, min((g + 1) * group, cols)
        Wb = W[:, i1:i2]
        Wfit = Wb.masked_fill(smask[:, i1:i2], 0.0) if smask is not None else Wb
        if grid == "itf":
            sp, sn, _ = itf_grid(Wfit, iters=itf_iters)
        elif grid == "sym":
            sp = sn = optimal_ternary(Wfit)[0].squeeze(1)
        else:
            raise ValueError(f"unknown grid {grid!r}")
        S[:, g, 0], S[:, g, 1] = sp.clamp_min(1e-8), sn.clamp_min(1e-8)
    return S


def _nearest_ternary(wcol: torch.Tensor, sp: torch.Tensor, sn: torch.Tensor):
    """Assign each entry to the nearest grid point of {-sn, 0, +sp}.
    For sp == sn this coincides with round-then-clamp (ties included)."""
    d0 = wcol.abs()
    dp = (wcol - sp).abs()
    dn = (wcol + sn).abs()
    tcol = torch.where((dp < d0) & (dp <= dn), torch.ones_like(wcol), torch.zeros_like(wcol))
    tcol = torch.where((dn < d0) & (dn < dp), -torch.ones_like(wcol), tcol)
    q = torch.where(tcol > 0, sp, torch.zeros_like(wcol))
    q = torch.where(tcol < 0, -sn, q)
    return tcol, q


def _group_sweep(W0: torch.Tensor, Hinv: torch.Tensor, S: torch.Tensor, group: int,
                 smask: torch.Tensor | None = None, *, in_sweep_refit: bool = False,
                 grid: str = "sym", itf_iters: int = 3):
    """One GPTQ sweep; all tensors are in act-ordered layout.

    S is [out, G, 2] = (s_pos, s_neg). With ``in_sweep_refit`` each group's scales are
    re-solved from the CURRENT feedback-adjusted block right before its columns are swept
    (the v2-sweep behaviour whose absence explained the start-quality gap); otherwise the
    incoming S is used as-is (fixed scales — what the alternation loop needs).

    With a salient mask the salient entries leave the ternary grid: q = s2*sign(w) on
    their own sign-magnitude component (s2 = per-(row, group) mean |w| over the salient
    set, t = 0 there) — still INSIDE the error feedback, so later columns compensate the
    total (ternary + salient) error. Returns (Q, T, W_adjusted, Q_salient, S_used)."""
    W = W0.clone()
    cols = W.shape[1]
    Q = torch.zeros_like(W)
    T = torch.zeros_like(W)
    Qsal = torch.zeros_like(W)
    S_used = S.clone()
    for g in range(S.shape[1]):
        i1, i2 = g * group, min((g + 1) * group, cols)
        Wb = W[:, i1:i2].clone()
        Qb = torch.zeros_like(Wb)
        Eb = torch.zeros_like(Wb)
        Hb = Hinv[i1:i2, i1:i2]
        Mb = smask[:, i1:i2] if smask is not None else None
        if in_sweep_refit:
            Wfit = Wb.masked_fill(Mb, 0.0) if Mb is not None else Wb
            if grid == "itf":
                sp, sn, _ = itf_grid(Wfit, iters=itf_iters)
            else:
                sp = sn = optimal_ternary(Wfit)[0].squeeze(1).clamp_min(1e-8)
            S_used[:, g, 0], S_used[:, g, 1] = sp, sn
        else:
            sp, sn = S_used[:, g, 0], S_used[:, g, 1]
        for j in range(i2 - i1):
            wcol = Wb[:, j]
            tcol, q = _nearest_ternary(wcol, sp, sn)
            if Mb is not None:
                # Salient override = the EXACT ORIGINAL weight (the fp16 channel stores a
                # value per position anyway). Deriving it from the feedback-ADJUSTED block
                # (the old s2*sign(w_adj) form) amplified salient values catastrophically on
                # real layers: |w|*sqrt(diagH) saliency puts large low-energy weights into
                # the act-order TAIL, exactly where GPTQ error feedback inflates W.
                mc = Mb[:, j]
                qsal = W0[:, i1 + j]
                q = torch.where(mc, qsal, q)
                tcol = torch.where(mc, torch.zeros_like(tcol), tcol)
                Qsal[:, i1 + j] = torch.where(mc, qsal, torch.zeros_like(qsal))
            Qb[:, j] = q
            T[:, i1 + j] = tcol
            e = (wcol - q) / Hb[j, j]
            Eb[:, j] = e
            if j + 1 < i2 - i1:
                Wb[:, j + 1:] -= e.unsqueeze(1) * Hb[j, j + 1:].unsqueeze(0)
        Q[:, i1:i2] = Qb
        W[:, i1:i2] = Wb
        if i2 < cols:
            W[:, i2:] -= Eb @ Hinv[i1:i2, i2:]
    return Q, T, W, Qsal, S_used


def _refit_scales(W: torch.Tensor, T: torch.Tensor, H: torch.Tensor, group: int,
                  mode: str, previous: torch.Tensor, *, grid: str = "sym",
                  w_target: torch.Tensor | None = None) -> torch.Tensor:
    """Refit scales after the achieved ternary support is known.

    previous is [out, G, 2] = (s_pos, s_neg); sym grids keep both lobes equal.
    w_target defaults to W; with a salient-first split the caller passes W - Q_salient so
    every mode fits the scales on the remainder. Modes:
      l2 / hdiag   per-lobe least squares (diag(H)-weighted for hdiag); salient entries
                   carry t = 0 and drop out of both numerator and denominator.
      hessian_cd   greedy per-group (per-lobe for itf) coordinate descent in the full
                   H-metric against w_target.
      align        A7 — EXACT joint per-row solve of all group scales in the H-metric on
                   the same support; supersedes the greedy pass (never worse in the
                   unconstrained solve, positivity clamp is the shared approximation)."""
    if mode not in {"l2", "hdiag", "hessian_cd", "align"}:
        raise ValueError("scale_refit must be 'l2', 'hdiag', 'hessian_cd' or 'align'")
    if w_target is None:
        w_target = W
    _, cols = W.shape
    n_groups = previous.shape[1]
    S = previous.clone()

    if mode == "align":
        s_al, _ = align_scales_output(w_target, T, H, group=group, grid=grid)
        if grid == "itf":
            S[:, :, 0] = s_al[:, 0::2]
            S[:, :, 1] = s_al[:, 1::2]
        else:
            S[:, :, 0] = S[:, :, 1] = s_al
        return S

    if mode in {"l2", "hdiag"}:
        d = torch.ones(cols, device=W.device, dtype=W.dtype)
        if mode == "hdiag":
            d = torch.diagonal(H).clamp_min(1e-12)
        for g in range(n_groups):
            i1, i2 = g * group, min((g + 1) * group, cols)
            tg = T[:, i1:i2]
            dg = d[i1:i2].unsqueeze(0)
            wg = w_target[:, i1:i2]
            if grid == "itf":
                for lobe, sign in ((0, 1.0), (1, -1.0)):
                    tm = (tg == sign).to(W.dtype)              # 1 on this lobe
                    num = (sign * wg * tm * dg).sum(dim=1)
                    den = (tm * dg).sum(dim=1)
                    valid = den > 1e-12
                    cand = (num / den.clamp_min(1e-12)).clamp_min(1e-8)
                    S[:, g, lobe] = torch.where(valid & (num > 0), cand, S[:, g, lobe])
            else:
                num = (wg * tg * dg).sum(dim=1)
                den = (tg.square() * dg).sum(dim=1)
                valid = den > 1e-12
                cand = (num / den.clamp_min(1e-12)).clamp_min(1e-8)
                s_new = torch.where(valid & (num > 0), cand, S[:, g, 0])
                S[:, g, 0] = S[:, g, 1] = s_new
        return S

    gidx = torch.div(torch.arange(cols, device=W.device), group, rounding_mode="floor")
    if grid == "itf":
        recon = S[:, gidx, 0] * T.clamp(min=0) + S[:, gidx, 1] * T.clamp(max=0)
    else:
        recon = T * S[:, gidx, 0]
    for g in range(n_groups):
        i1, i2 = g * group, min((g + 1) * group, cols)
        tg = T[:, i1:i2]
        if not tg.count_nonzero():
            continue
        Hgg = H[i1:i2, i1:i2]
        if grid == "itf":
            for lobe in (0, 1):
                basis = tg.clamp(min=0) if lobe == 0 else tg.clamp(max=0)
                if not basis.count_nonzero():
                    continue
                residual = w_target - recon
                residual[:, i1:i2] += S[:, g:g + 1, lobe] * basis
                Hr = residual @ H
                num = (basis * Hr[:, i1:i2]).sum(dim=1)
                den = ((basis @ Hgg) * basis).sum(dim=1).clamp_min(1e-12)
                cand = (num / den).clamp_min(1e-8)
                valid = (num > 0) & torch.isfinite(cand)
                new_s = torch.where(valid, cand, S[:, g, lobe])
                recon[:, i1:i2] = new_s.unsqueeze(1) * basis
                S[:, g, lobe] = new_s
        else:
            residual = w_target - recon
            residual[:, i1:i2] += S[:, g:g + 1, 0] * tg
            Hr = residual @ H
            num = (tg * Hr[:, i1:i2]).sum(dim=1)
            den = ((tg @ Hgg) * tg).sum(dim=1).clamp_min(1e-12)
            cand = (num / den).clamp_min(1e-8)
            valid = (num > 0) & torch.isfinite(cand)
            new_s = torch.where(valid, cand, S[:, g, 0])
            recon[:, i1:i2] = new_s.unsqueeze(1) * tg
            S[:, g, 0] = S[:, g, 1] = new_s
    return S


def _hessian_error(W: torch.Tensor, Q: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    E = W - Q
    return ((E @ H) * E).sum()


@torch.no_grad()
def gptq_group_ternary(
    w: torch.Tensor,
    H: torch.Tensor,
    *,
    group: int = 128,
    percdamp: float = 0.01,
    act_order: bool = True,
    refine_scale: bool = True,
    refine_iters: int = 2,
    scale_refit: str = "hdiag",
    grid: str = "sym",
    itf_iters: int = 3,
    salient_first: float = 0.0,
    salient_scope: str = "row",
    in_sweep_refit: bool = False,
    return_perm: bool = False,
    return_salient: bool = False,
):
    """Group-scale GPTQ v3, consolidated: the agent v3 refine cycle plus the measured
    Stage-A solver ingredients (Stage-A pass), defaults unchanged.

    Base cycle: act-order, one sweep at fixed per-(row, group) scales, then refine_iters
    rounds of scale refit -> full re-sweep, keeping the best by measured Hessian error.

    Consolidated ingredients (each gated separately on the 0.5B donor, relative
    H-weighted layer output error vs the v2 start):
      * grid="itf"            A5 asymmetric {-s_neg, 0, +s_pos} grid per group (-2.0%
                              alone, best on skewed blocks). NOTE: the packed counter
                              format is sym-scale — ptq_warm_start finishes an itf solve
                              with an exact sym re-solve on the achieved support.
      * scale_refit="align"   A7 exact joint per-row scale solve in the H-metric
                              (supersedes the greedy hessian_cd on the same support).
      * salient_first > 0     A4.1 BiLLM-style pre-sweep split: the top fraction by
                              |w|*sqrt(diag H) leaves the ternary grid for its own
                              s2*sign(w) component that participates in the error
                              feedback (-5.8% alone at 0.01; -10.1% in the full chain).
      * salient_scope         "row" (default): an equal per-row budget, the original
                              behaviour. "layer": one global top-K over the whole
                              layer -- rows compete for the same fp16 slots, so hard
                              rows take more and easy rows give theirs up. Total
                              budget (and bpw) is unchanged; the packed salient
                              channel stores flat indices, so any per-row split loads.
      A6 (SSR reordering) is deliberately NOT ported: measured +94% error — diag(H)
      order is the compensation order, not a grouping artifact.

    Returns (Q, S, t): Q [out,in] fp32 reconstruction (ternary + salient components),
    S [out, n_groups] for sym / [out, n_groups, 2] for itf indexed by PERMUTED groups,
    t [out,in] int16 in ORIGINAL column order (0 at salient entries).
    return_perm adds (perm, W_adjusted); return_salient (requires return_perm) further
    adds (salient_idx, salient_val): flat ORIGINAL-order indices (int32) of the salient
    set and their exact fp32 values s2*sign(w), ready for the packed salient channel."""
    if return_salient and not return_perm:
        raise ValueError("return_salient requires return_perm=True")
    W = w.detach().to(torch.float32).clone()
    Hwork = H.detach().to(torch.float32).clone()
    cols = W.shape[1]
    if act_order:
        perm = torch.argsort(torch.diagonal(Hwork), descending=True)
        W = W[:, perm]
        Hwork = Hwork[perm][:, perm]
        invperm = torch.argsort(perm)
    else:
        perm = torch.arange(cols, device=W.device)
        invperm = perm
    Hinv, Hdamped = _prep_hinv(Hwork, W, percdamp)

    smask = None
    if salient_first > 0.0:
        # BiLLM-style activation-aware saliency, static across refine iterations.
        sal = W.abs() * torch.diagonal(Hwork).sqrt().clamp_min(1e-12).unsqueeze(0)
        if salient_scope == "layer":
            k = max(1, int(round(salient_first * sal.numel())))
            thr = sal.reshape(-1).kthvalue(sal.numel() - k + 1).values
        elif salient_scope == "row":
            k = max(1, int(round(salient_first * cols)))
            thr = sal.kthvalue(cols - k + 1, dim=1, keepdim=True).values
        else:
            raise ValueError(f"salient_scope must be 'row' or 'layer', got {salient_scope!r}")
        smask = sal >= thr

    S = _initial_group_scales(W, group, grid, itf_iters, smask)
    # First sweep: with in_sweep_refit each group's scales are re-solved from the
    # feedback-adjusted block right before its columns (the v2 start-quality behaviour).
    # Alternation sweeps below run at FIXED candidate scales -- that is what makes the
    # post-sweep refit meaningful; the monotone Hessian gate keeps every step safe.
    Q, T, W_adjusted, Qsal, S = _group_sweep(
        W, Hinv, S, group, smask,
        in_sweep_refit=in_sweep_refit, grid=grid, itf_iters=itf_iters)
    best_err = _hessian_error(W, Q, Hdamped)

    if refine_scale:
        for _ in range(max(0, int(refine_iters))):
            candidate_S = _refit_scales(W, T, Hdamped, group, scale_refit, S,
                                        grid=grid, w_target=W - Qsal)
            candidate_Q, candidate_T, candidate_W, candidate_Qsal, candidate_S = _group_sweep(
                W, Hinv, candidate_S, group, smask)
            candidate_err = _hessian_error(W, candidate_Q, Hdamped)
            if not torch.isfinite(candidate_err) or candidate_err > best_err * (1.0 + 1e-7):
                break
            S, Q, T, W_adjusted, Qsal = (candidate_S, candidate_Q, candidate_T,
                                         candidate_W, candidate_Qsal)
            best_err = candidate_err

    Q_orig = Q[:, invperm]
    T_orig = T[:, invperm]
    W_adjusted_orig = W_adjusted[:, invperm]
    S_out = S[:, :, 0] if grid == "sym" else S
    if return_perm:
        if return_salient:
            if smask is not None:
                mask_orig = smask[:, invperm]
                Qsal_orig = Qsal[:, invperm]
                salient_idx = mask_orig.reshape(-1).nonzero().squeeze(1).to(torch.int32)
                salient_val = Qsal_orig.reshape(-1)[salient_idx.long()].to(torch.float32)
            else:
                salient_idx = torch.zeros(0, dtype=torch.int32)
                salient_val = torch.zeros(0, dtype=torch.float32)
            return (Q_orig, S_out, T_orig.to(torch.int16), perm, W_adjusted_orig,
                    (salient_idx, salient_val))
        return Q_orig, S_out, T_orig.to(torch.int16), perm, W_adjusted_orig
    return Q_orig, S_out, T_orig.to(torch.int16)


@torch.no_grad()
def solve_group_state(
    w: torch.Tensor,
    H: torch.Tensor,
    *,
    group: int = 128,
    C: int = 11,   # group layers default to C=11 (deployed config); C_DEFAULT=8 is legacy
    percdamp: float = 0.01,
    act_order: bool = True,
    refine_iters: int = 2,
    scale_refit: str = "hdiag",
    grid: str = "sym",
    itf_iters: int = 3,
    salient_first: float = 0.0,
    salient_scope: str = "row",
    in_sweep_refit: bool = False,
):
    """One layer's full DEPLOY solve: the v3 solver, then (for itf grids) the exact sym
    re-solve on the achieved support (the packed format is sym-scale), the residual
    counter from the feedback-adjusted weights, and the salient channel.

    Returns ``(state, Q)``: ``state = (S, t, c, perm, salient_idx, salient_val)`` on the
    input device (sym scales [out, n_groups]) and ``Q`` the deployable dense
    reconstruction in ORIGINAL column order (exact fp32 salient values; the packed layer
    stores them as fp16). Shared by the fp and asymmetric calibration paths of
    ``ptq_warm_start``."""
    _, S, t, perm, Wadj, (salient_idx, salient_val) = gptq_group_ternary(
        w, H, group=group, percdamp=percdamp,
        act_order=act_order, refine_iters=refine_iters, scale_refit=scale_refit,
        grid=grid, itf_iters=itf_iters, salient_first=salient_first,
        salient_scope=salient_scope, in_sweep_refit=in_sweep_refit,
        return_perm=True, return_salient=True,
    )
    cols = w.shape[1]
    invperm = torch.argsort(perm)
    if grid == "itf":
        # The packed format is sym-scale: exact joint sym re-solve (A7) on the
        # achieved itf support, against the non-salient remainder.
        Hp = H.detach().to(torch.float32)[perm][:, perm]
        w_perm = w.detach().to(torch.float32)[:, perm]
        t_perm = t[:, perm].to(torch.float32)
        w_target = w_perm
        if salient_idx.numel():
            o = salient_idx.long() // cols
            j = salient_idx.long() % cols
            qsal = torch.zeros_like(w_perm).reshape(-1)
            qsal[o * cols + invperm[j]] = salient_val.float()
            w_target = w_perm - qsal.view_as(w_perm)
        S, _ = align_scales_output(w_target, t_perm, Hp, group=group, grid="sym")
    c = group_residual_counter(Wadj, S, t, perm, group, C)
    if salient_idx.numel():
        c = c.clone()
        c.reshape(-1)[salient_idx.long()] = 0
    group_perm = torch.div(torch.arange(cols, device=perm.device), group,
                           rounding_mode="floor")
    gidx = torch.empty_like(group_perm)
    gidx[perm] = group_perm
    Q = S[:, gidx] * t.to(torch.float32)
    if salient_idx.numel():
        Q = Q.clone()
        Q.reshape(-1)[salient_idx.long()] = salient_val.float()
    return (S, t, c, perm, salient_idx, salient_val), Q


@torch.no_grad()
def group_residual_counter(w_adjusted: torch.Tensor, scales: torch.Tensor, t: torch.Tensor,
                           perm: torch.Tensor, group: int, C: int = 11) -> torch.Tensor:
    cols = t.shape[1]
    group_perm = torch.div(torch.arange(cols, device=perm.device), group, rounding_mode="floor")
    group_index = torch.empty_like(group_perm)
    group_index[perm] = group_perm
    s_col = scales[:, group_index]
    c = ((w_adjusted.to(torch.float32) / s_col - t.to(torch.float32)) * C)
    return c.round().clamp_(-(C - 1), C - 1).to(torch.int16)


@dataclass(frozen=True)
class _StackedMoETarget:
    path: str
    module: nn.Module
    num_experts: int
    hidden_dim: int
    intermediate_dim: int

    def gate_up_path(self, expert: int) -> str:
        return f"{self.path}.gate_up_proj[{expert}]"

    def down_path(self, expert: int) -> str:
        return f"{self.path}.down_proj[{expert}]"

    def expert_path(self, expert: int) -> str:
        return f"{self.path}[{expert}]"


_LEGACY_EXPERT_LINEAR_NAMES = {
    "w1", "w2", "w3", "gate_proj", "up_proj", "down_proj",
}


def _moe_router_paths(model: nn.Module) -> set[str]:
    paths = set()
    for parent_path, parent in model.named_modules():
        if "experts" not in parent._modules:
            continue
        for name in ("gate", "router"):
            if isinstance(parent._modules.get(name), nn.Module):
                paths.add(f"{parent_path}.{name}" if parent_path else name)
    return paths


def _target_paths(model: nn.Module, skip) -> list[str]:
    out = []
    routers = _moe_router_paths(model)
    for parent_path, parent in model.named_modules():
        for child_name, child in parent.named_children():
            if isinstance(child, nn.Linear):
                path = f"{parent_path}.{child_name}" if parent_path else child_name
                if path not in routers and not any(sub in path for sub in skip):
                    out.append(path)
    return out


def _stacked_moe_targets(model: nn.Module) -> list[_StackedMoETarget]:
    targets = []
    for path, module in model.named_modules():
        gate_up = getattr(module, "gate_up_proj", None)
        down = getattr(module, "down_proj", None)
        if not isinstance(gate_up, nn.Parameter) or not isinstance(down, nn.Parameter):
            continue
        if gate_up.ndim != 3 or down.ndim != 3:
            continue
        experts, twice_hidden, hidden_dim = gate_up.shape
        down_experts, down_hidden, intermediate = down.shape
        act_name = type(getattr(module, "act_fn", None)).__name__.lower()
        if (
            experts != down_experts
            or hidden_dim != down_hidden
            or twice_hidden != 2 * intermediate
            or "silu" not in act_name
        ):
            continue
        targets.append(
            _StackedMoETarget(
                path, module, int(experts), int(hidden_dim), int(intermediate)
            )
        )
    return targets


def _legacy_expert_targets(model: nn.Module) -> dict[str, str]:
    """Map legacy expert-linear path -> per-expert path."""
    mapping = {}
    for block_path, block in model.named_modules():
        experts = block._modules.get("experts")
        if not isinstance(experts, nn.ModuleList):
            continue
        experts_path = f"{block_path}.experts" if block_path else "experts"
        for expert_idx, expert in enumerate(experts):
            expert_path = f"{experts_path}.{expert_idx}"
            for relative, module in expert.named_modules():
                if not isinstance(module, nn.Linear) or not relative:
                    continue
                if relative.rsplit(".", 1)[-1] not in _LEGACY_EXPERT_LINEAR_NAMES:
                    continue
                mapping[f"{expert_path}.{relative}"] = expert_path
    return mapping


def _unhandled_moe_paths(
    model: nn.Module,
    stacked_targets: list[_StackedMoETarget],
    legacy_targets: dict[str, str],
) -> list[str]:
    """Find expert containers that no supported target representation covers."""
    stacked_parameters = {
        id(target.module): {
            id(target.module.gate_up_proj), id(target.module.down_proj),
        }
        for target in stacked_targets
    }
    legacy_parameters = set()
    for path in legacy_targets:
        legacy_parameters.update(
            id(parameter)
            for parameter in model.get_submodule(path).parameters(recurse=False)
        )
    out = []
    for block_path, block in model.named_modules():
        experts = block._modules.get("experts")
        if experts is None:
            continue
        experts_path = f"{block_path}.experts" if block_path else "experts"
        parameters = list(experts.parameters(recurse=True))
        if id(experts) in stacked_parameters:
            if all(id(parameter) in stacked_parameters[id(experts)] for parameter in parameters):
                continue
        if isinstance(experts, nn.ModuleList):
            if not parameters:
                continue
            if all(id(parameter) in legacy_parameters for parameter in parameters):
                continue
        elif not parameters:
            continue
        out.append(experts_path)
    return out


def _assert_no_unhandled_moe(
    model: nn.Module,
    stacked_targets: list[_StackedMoETarget],
    legacy_targets: dict[str, str],
) -> None:
    paths = _unhandled_moe_paths(model, stacked_targets, legacy_targets)
    if paths:
        raise RuntimeError(
            "unsupported MoE expert layout at "
            f"{paths}; refusing partial PTQ conversion because attention-only conversion "
            "would leave expert weights unconverted"
        )


def _hessian_chunks(model: nn.Module, targets: list[str], budget_bytes: int) -> list[list[str]]:
    """Greedy-split targets so each chunk's fp32 Hessians fit the GPU budget.

    Large donors overflow VRAM if every H (in_features^2 fp32) is resident at once --
    e.g. gemma-4-12B needs ~64 GiB of Hessians alone (48 down_proj at 15360^2). Chunked
    collection re-runs the calibration forward once per chunk and offloads each chunk to
    CPU; the solve loop moves one layer's H back to the weight device at a time."""
    chunks: list[list[str]] = []
    cur: list[str] = []
    size = 0
    for path in targets:
        need = model.get_submodule(path).in_features ** 2 * 4
        if cur and size + need > budget_bytes:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(path)
        size += need
    if cur:
        chunks.append(cur)
    return chunks


def _parent_and_name(model: nn.Module, path: str):
    if "." not in path:
        return model, path
    parent_path, name = path.rsplit(".", 1)
    return model.get_submodule(parent_path), name


@torch.no_grad()
def collect_hessians(
    model: nn.Module,
    targets: list[str],
    calib_batches,
    *,
    moe_targets: list[_StackedMoETarget] | None = None,
    return_counts: bool = False,
) -> dict | tuple[dict, dict[str, int]]:
    hessians: dict[str, torch.Tensor] = {}
    sample_counts: dict[str, int] = {}
    hooks = []
    was_training = model.training
    model.eval()

    def accumulate(path, x):
        x = x.detach().reshape(-1, x.shape[-1]).to(torch.float32)
        h = hessians.get(path)
        if h is None:
            h = torch.zeros(
                x.shape[1], x.shape[1], dtype=torch.float32, device=x.device
            )
            hessians[path] = h
        h.addmm_(x.t(), x)
        sample_counts[path] = sample_counts.get(path, 0) + x.shape[0]

    def make_hook(path, in_features):
        def hook(_mod, inputs):
            x = inputs[0].detach().reshape(-1, in_features).to(torch.float32)
            accumulate(path, x)
        return hook

    def make_moe_hook(target):
        def hook(module, inputs):
            hidden_states, top_k_index = inputs[:2]
            for expert in range(target.num_experts):
                token_idx = torch.where(top_k_index == expert)[0]
                if token_idx.numel() == 0:
                    continue
                current = hidden_states[token_idx]
                gate_up_path = target.gate_up_path(expert)
                down_path = target.down_path(expert)
                accumulate(gate_up_path, current)
                gate, up = F.linear(
                    current, module.gate_up_proj[expert]
                ).chunk(2, dim=-1)
                accumulate(down_path, module.act_fn(gate) * up)
        return hook

    for path in targets:
        lin = model.get_submodule(path)
        hooks.append(lin.register_forward_pre_hook(make_hook(path, lin.in_features)))
    for target in moe_targets or []:
        hooks.append(target.module.register_forward_pre_hook(make_moe_hook(target)))
    try:
        for ids in calib_batches:
            model(ids)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)
    if return_counts:
        return hessians, sample_counts
    return hessians


def _default_lm_loss(model: nn.Module, ids: torch.Tensor) -> torch.Tensor:
    """HF causal-LM next-token CE (labels = inputs). Override via loss_fn for
    non-HF models."""
    return model(ids, labels=ids).loss


def collect_hessians_guided(model: nn.Module, targets: list[str], calib_batches,
                            *, loss_fn=None) -> dict:
    """End-loss-weighted Hessians (GuidedQuant with one output group, arXiv:2505.07004):

        H = sum_n g_n x_n x_n^T,    g_n = mean_o (dL/dy_{n,o})^2,

    token importance from ONE backward per batch. This is statistics collection, NOT
    training: every parameter's requires_grad is forced off except the input embedding
    (kept on so autograd builds the graph and grad_output reaches each layer), nothing
    is stepped, and the embedding grad is dropped after each batch. The absolute scale
    of H is irrelevant to the solver (percdamp, the grids and align are all
    scale-invariant), only the RELATIVE token weighting matters.

    Memory note: the backward graph of a full LM is the dominant cost -- on CPU boxes
    feed SMALL batches (e.g. [1, seq]); a 1.5B model with [2, 128] batches peaked over
    15 GiB and got OOM-killed, while the same token budget in [1, 128] batches halves
    the graph."""
    if loss_fn is None:
        loss_fn = _default_lm_loss
    hessians: dict[str, torch.Tensor] = {}
    hooks = []
    was_training = model.training
    model.eval()
    req = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in req:
        p.requires_grad_(False)
    get_emb = getattr(model, "get_input_embeddings", None)
    if callable(get_emb) and get_emb() is not None:
        grad_anchor = get_emb().weight
    else:
        grad_anchor = next(model.parameters())
    grad_anchor.requires_grad_(True)

    def make_fwd(path, in_features):
        def hook(_mod, inputs, output):
            x = inputs[0].detach().reshape(-1, in_features).to(torch.float32)

            def grab(grad):
                g = grad.detach().reshape(x.shape[0], -1).to(torch.float32)
                g = g.pow(2).mean(dim=1)
                h = hessians.get(path)
                if h is None:
                    h = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32,
                                    device=x.device)
                    hessians[path] = h
                h.addmm_((x * g.unsqueeze(1)).t(), x)

            # tensor hook on the layer OUTPUT: raises immediately if the graph does
            # not reach this layer (output.requires_grad False), instead of silently
            # skipping it the way a module backward hook would.
            output.register_hook(grab)
        return hook

    for path in targets:
        lin = model.get_submodule(path)
        hooks.append(lin.register_forward_hook(make_fwd(path, lin.in_features)))
    try:
        for ids in calib_batches:
            with torch.enable_grad():
                loss = loss_fn(model, ids)
                loss.backward()
            grad_anchor.grad = None
    finally:
        for hook in hooks:
            hook.remove()
        for p, r in req:
            p.requires_grad_(r)
        model.train(was_training)
    missing = [p for p in targets if p not in hessians]
    if missing:
        raise RuntimeError(f"no gradient reached: {missing} -- did the loss depend "
                           "on these layers?")
    return hessians


@torch.no_grad()
def quantize_dense_group_ternary(model: nn.Module, calib_batches, *, group: int = 128,
                                  percdamp: float = 0.01, extra_skip=None,
                                  refine_iters: int = 2, scale_refit: str = "hdiag",
                                  grid: str = "sym", itf_iters: int = 3,
                                  salient_first: float = 0.0,
                                  salient_scope: str = "row",
                                  in_sweep_refit: bool = False,
                                  progress: bool = True) -> None:
    skip = ["lm_head"] + (list(extra_skip) if extra_skip is not None else [])
    targets = _target_paths(model, skip)
    hessians = collect_hessians(model, targets, calib_batches)
    for i, path in enumerate(targets):
        lin = model.get_submodule(path)
        w_hat, _, _ = gptq_group_ternary(
            lin.weight, hessians.pop(path), group=group, percdamp=percdamp,
            refine_iters=refine_iters, scale_refit=scale_refit, grid=grid,
            itf_iters=itf_iters, salient_first=salient_first,
            salient_scope=salient_scope,
            in_sweep_refit=in_sweep_refit,
        )
        lin.weight.copy_(w_hat.to(lin.weight.dtype))
        if progress and (i + 1) % 25 == 0:
            print(f"[group{group}-v3] {i+1}/{len(targets)} layers quantized", flush=True)


def _optimal_group_state(w: torch.Tensor, *, group: int, C: int):
    """Represent the exact data-free rowwise ternary optimum in a group counter."""
    s, t = optimal_ternary(w)
    groups = (w.shape[1] + group - 1) // group
    scales = s.repeat(1, groups)
    perm = torch.arange(w.shape[1], device=w.device)
    c = group_residual_counter(w, scales, t, perm, group, C)
    return (
        scales, t.to(torch.int16), c, perm,
        torch.zeros(0, dtype=torch.int32, device=w.device),
        torch.zeros(0, dtype=torch.float32, device=w.device),
    )


def _slice_group_state(state, start: int, end: int):
    S, t, c, perm, salient_idx, salient_val = state
    cols = t.shape[1]
    if salient_idx.numel():
        rows = salient_idx.long() // cols
        keep = (rows >= start) & (rows < end)
        sliced_idx = salient_idx[keep].long() - start * cols
        sliced_idx = sliced_idx.to(torch.int32)
        sliced_val = salient_val[keep]
    else:
        sliced_idx = salient_idx
        sliced_val = salient_val
    return S[start:end], t[start:end], c[start:end], perm, sliced_idx, sliced_val


def _group_counter_from_state(
    state, *, in_features: int, out_features: int, group: int, C: int,
    kind: str, counter_kw: dict,
) -> nn.Module:
    packed_kinds = {
        "counter_packed", "counter_triton", "group_packed", "group_scale_packed",
    }
    reference_supported = {
        "lr", "lr_scale", "rms_beta", "rms_eps", "local_grad_clip", "residual_alpha",
    }
    packed_supported = reference_supported | {
        "kernel_mode", "strict_update", "flip_sample_size",
    }
    S, t, c, perm, salient_idx, salient_val = state
    packed = (
        kind in packed_kinds and in_features % 4 == 0 and group % 4 == 0
    )
    if packed:
        kw = {key: value for key, value in counter_kw.items() if key in packed_supported}
        counter: nn.Module = PackedGroupScaleCounterLinear(
            in_features, out_features, group=group, C=C, perm=perm, **kw
        )
    else:
        kw = {
            key: value for key, value in counter_kw.items()
            if key in reference_supported
        }
        counter = GroupScaleCounterLinear(
            in_features, out_features, group=group, C=C, perm=perm, **kw
        )
    counter.load_group_state(
        S, t, c, perm, salient_idx=salient_idx, salient_val=salient_val
    )
    return counter


def _swap_stacked_moe(
    model: nn.Module,
    targets: list[_StackedMoETarget],
    states: dict[str, list[tuple]],
    report: SwapReport,
    *,
    is_group: bool,
    kind: str,
    group: int,
    C: int,
    counter_kw: dict,
) -> None:
    from ..moe_ffn import (
        HFModuleListSwiGLUExperts,
        HFStackedSwiGLUExperts,
        SwiGLUCounterExpert,
    )

    for target in targets:
        if is_group:
            experts = []
            for expert, (gate_up_state, down_state) in enumerate(states[target.path]):
                split = target.intermediate_dim
                gate_state = _slice_group_state(gate_up_state, 0, split)
                up_state = _slice_group_state(gate_up_state, split, 2 * split)
                gate = _group_counter_from_state(
                    gate_state, in_features=target.hidden_dim,
                    out_features=split, group=group, C=C, kind=kind,
                    counter_kw=counter_kw,
                )
                up = _group_counter_from_state(
                    up_state, in_features=target.hidden_dim,
                    out_features=split, group=group, C=C, kind=kind,
                    counter_kw=counter_kw,
                )
                down = _group_counter_from_state(
                    down_state, in_features=split,
                    out_features=target.hidden_dim, group=group, C=C, kind=kind,
                    counter_kw=counter_kw,
                )
                experts.append(SwiGLUCounterExpert(gate, up, down))
            replacement: nn.Module = HFModuleListSwiGLUExperts(
                experts, hidden_dim=target.hidden_dim,
                intermediate_dim=target.intermediate_dim,
            )
        else:
            allowed = {"lr", "lr_scale", "rms_beta", "rms_eps", "compute_dtype"}
            kw = {key: value for key, value in counter_kw.items() if key in allowed}
            replacement = HFStackedSwiGLUExperts(
                target.num_experts, target.hidden_dim, target.intermediate_dim,
                C=C, **kw,
            )
            for expert, (gate_up_state, down_state) in enumerate(states[target.path]):
                replacement.load_counter_state(expert, gate_up_state, down_state)

        parent, name = _parent_and_name(model, target.path)
        setattr(parent, name, replacement)
        for expert in range(target.num_experts):
            report.swapped.extend(
                [target.gate_up_path(expert), target.down_path(expert)]
            )
        report.coeffs += target.num_experts * (
            3 * target.hidden_dim * target.intermediate_dim
        )


def _moe_calibration_diagnostics(
    model: nn.Module,
    stacked_targets: list[_StackedMoETarget],
    legacy_targets: dict[str, str],
    sample_counts: dict[str, int],
):
    counts: dict[str, int] = {}
    required: dict[str, int] = {}
    for target in stacked_targets:
        for expert in range(target.num_experts):
            expert_path = target.expert_path(expert)
            counts[expert_path] = sample_counts.get(target.gate_up_path(expert), 0)
            required[expert_path] = max(target.hidden_dim, target.intermediate_dim)
    for path, expert_path in legacy_targets.items():
        counts[expert_path] = max(
            counts.get(expert_path, 0), sample_counts.get(path, 0)
        )
        required[expert_path] = max(
            required.get(expert_path, 0), model.get_submodule(path).in_features
        )

    dead, rank_deficient = [], []
    for expert_path, count in counts.items():
        if count == 0:
            dead.append(expert_path)
            warnings.warn(
                f"dead MoE expert {expert_path}: zero routed calibration tokens; "
                "using the data-free optimal ternary solve",
                RuntimeWarning,
            )
        elif count < required[expert_path]:
            rank_deficient.append(expert_path)
            warnings.warn(
                f"rank-deficient MoE Hessian for {expert_path}: {count} routed "
                f"tokens < {required[expert_path]} input features; damping will "
                "regularize the solve, increase the calibration budget if quality suffers",
                RuntimeWarning,
            )
    return counts, dead, rank_deficient


@torch.no_grad()
def ptq_warm_start(
    model: nn.Module,
    calib_batches,
    *,
    mode: str = "gptq",
    kind: str = "counter_rms",
    C: int = C_DEFAULT,
    keep_bias: bool = True,
    extra_skip=None,
    blocksize: int = 128,
    group: int = 128,
    percdamp: float = 0.01,
    act_order: bool = True,
    refine_iters: int = 2,
    scale_refit: str = "hdiag",
    grid: str = "sym",
    itf_iters: int = 3,
    salient_first: float = 0.0,
    salient_scope: str = "row",
    in_sweep_refit: bool = False,
    calibration: str = "fp",
    asym_chunk_layers: int = 7,
    asym_strength: float = 1.0,
    asym_passes: int = 1,
    asym_fp_device: str | None = None,
    hessian_weighting: str = "none",
    loss_fn=None,
    hessian_gpu_budget_gib: float = 24.0,
    progress: bool = True,
    **counter_kw,
) -> SwapReport:
    """Swap body linears to a calibrated counter format.

    For ``mode='gptq_group'``, packed kinds preserve `(S,t,c,perm)` in
    ``PackedGroupScaleCounterLinear``. On CUDA+Triton this gives group-aware decode-in-GEMM,
    group-aware grad_x, and strict update-from-IO with no dense W/grad_w. Non-packed kinds keep the
    pure-PyTorch ``GroupScaleCounterLinear`` reference.

    Consolidated solver ingredients: ``grid='itf'`` runs the asymmetric-grid sweep; since
    the packed format is sym-scale, the achieved support then gets an EXACT sym re-solve
    (align) before packing. ``salient_first > 0`` splits the top-|w|*sqrt(diag H) fraction
    out before the sweep (A4.1) and ships it as the packed salient channel
    (salient_idx/salient_val, exact fp16 overrides) instead of forcing it onto the grid.

    ``calibration='asym'`` (group modes only) switches to the GPTAQ-style cascade-aware
    objective ||X_q Q - X_fp W||^2: layers are solved sequentially against the inputs of
    the PARTIALLY QUANTIZED model with the damped target w~ = H_q^{-1} (X_q^T X_fp) w
    (see donor/asym.py). Costs 2*ceil(len(targets)/asym_chunk_layers) calibration passes
    and one resident fp copy of the model; ``asym_strength`` interpolates w -> w~.

    ``hessian_weighting='end_loss'`` collects GuidedQuant-style loss-weighted Hessians
    (one backward per calibration batch, no weight updates; see
    ``collect_hessians_guided``). Not combinable with ``calibration='asym'`` yet.
    """
    stacked_targets = _stacked_moe_targets(model)
    all_legacy_targets = _legacy_expert_targets(model)
    _assert_no_unhandled_moe(model, stacked_targets, all_legacy_targets)
    router_paths = sorted(_moe_router_paths(model))
    skip = ["lm_head"]
    skip += list(extra_skip) if extra_skip is not None else []
    targets = _target_paths(model, skip)
    legacy_targets = {
        path: expert for path, expert in all_legacy_targets.items() if path in targets
    }
    skipped_legacy = sorted(set(all_legacy_targets) - set(legacy_targets))
    if skipped_legacy:
        raise RuntimeError(
            "refusing partial PTQ conversion: expert weights were excluded by skip rules: "
            f"{skipped_legacy}"
        )
    skipped_stacked = [
        target.path for target in stacked_targets
        if any(token in target.path for token in skip)
    ]
    if skipped_stacked:
        raise RuntimeError(
            "refusing partial PTQ conversion: stacked expert weights were excluded by "
            f"skip rules: {skipped_stacked}"
        )
    has_moe = bool(stacked_targets or legacy_targets)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = None

    is_group = mode in {"gptq_group", "group128v3", "group"}
    if mode not in {"optimal", "gptq", "gptq_group", "group128v3", "group"}:
        raise ValueError("mode must be 'optimal', 'gptq' or 'gptq_group'")
    if not is_group:
        # Group-only controls must not leak into the legacy counter path.
        for key in ("residual_alpha", "kernel_mode", "strict_update", "flip_sample_size"):
            counter_kw.pop(key, None)
    if calibration not in {"fp", "asym"}:
        raise ValueError("calibration must be 'fp' or 'asym'")
    if hessian_weighting not in {"none", "end_loss"}:
        raise ValueError("hessian_weighting must be 'none' or 'end_loss'")
    if has_moe and hessian_weighting == "end_loss":
        raise RuntimeError(
            "hessian_weighting='end_loss' does not support MoE routing yet; refusing "
            "partial expert conversion"
        )
    sample_counts: dict[str, int] = {}
    moe_states: dict[str, list[tuple]] = {}
    if calibration == "asym":
        if not is_group:
            raise ValueError("calibration='asym' requires a group mode")
        if hessian_weighting != "none":
            raise ValueError("hessian_weighting is not supported with calibration='asym'")
        import copy as _copy

        from .asym import asym_solve_states
        passes = max(1, int(asym_passes))
        # Multi-pass: the fp tower and the ORIGINAL weights must be captured once —
        # after pass 1 the model's dense weights are already the quantized recon.
        model_fp = _copy.deepcopy(model) if passes > 1 else None
        if model_fp is not None and asym_fp_device is not None:
            model_fp = model_fp.to(asym_fp_device)
        w0 = ({p: model.get_submodule(p).weight.detach().clone() for p in targets}
              if passes > 1 else None)
        states: dict[str, tuple] = {}
        for pass_i in range(passes):
            if progress and passes > 1:
                print(f"[ptq:asym] pass {pass_i + 1}/{passes}", flush=True)
            states = asym_solve_states(
                model, calib_batches, targets, group=group, C=C, percdamp=percdamp,
                moe_targets=stacked_targets,
                moe_targets_fp=(_stacked_moe_targets(model_fp)
                                if model_fp is not None else None),
                routers=sorted(router_paths) if stacked_targets else None,
                moe_states=moe_states,
                act_order=act_order, refine_iters=refine_iters, scale_refit=scale_refit,
                grid=grid, itf_iters=itf_iters, salient_first=salient_first,
                salient_scope=salient_scope, in_sweep_refit=in_sweep_refit,
                chunk_layers=asym_chunk_layers, strength=asym_strength,
                model_fp=model_fp, w0=w0, fp_device=asym_fp_device, progress=progress,
            )
    else:
        if hessian_weighting == "end_loss":
            def _collect(m, t, c):
                return collect_hessians_guided(m, t, c, loss_fn=loss_fn)
        else:
            _collect = collect_hessians
        hessians: dict[str, torch.Tensor] = {}
        if mode.startswith("gptq") or is_group:
            chunks = _hessian_chunks(model, targets, int(hessian_gpu_budget_gib * 2**30))
            if hessian_weighting == "end_loss":
                if len(chunks) == 1:
                    hessians = _collect(model, targets, calib_batches)
                else:
                    if progress:
                        print(f"[ptq:{mode}] hessians in {len(chunks)} chunks "
                              f"(budget {hessian_gpu_budget_gib:g} GiB)", flush=True)
                    for chunk in chunks:
                        part = _collect(model, chunk, calib_batches)
                        hessians.update({k: v.cpu() for k, v in part.items()})
                        del part
                        if device is not None and device.type == "cuda":
                            torch.cuda.empty_cache()
            else:
                collection_chunks = chunks or ([[]] if stacked_targets else [])
                if progress and len(collection_chunks) > 1:
                    print(f"[ptq:{mode}] hessians in {len(chunks)} chunks "
                          f"(budget {hessian_gpu_budget_gib:g} GiB)", flush=True)
                for chunk_idx, chunk in enumerate(collection_chunks):
                    part, counts = collect_hessians(
                        model, chunk, calib_batches,
                        moe_targets=stacked_targets if chunk_idx == 0 else None,
                        return_counts=True,
                    )
                    park_cpu = len(collection_chunks) > 1
                    hessians.update({
                        key: value.cpu() if park_cpu else value
                        for key, value in part.items()
                    })
                    sample_counts.update(counts)
                    del part
                    if device is not None and device.type == "cuda":
                        torch.cuda.empty_cache()
        elif has_moe:
            # Optimal mode is data-free, but one routing pass is still needed for
            # operator-visible dead/rank-deficient expert diagnostics.
            _, sample_counts = collect_hessians(
                model, list(legacy_targets), calib_batches,
                moe_targets=stacked_targets, return_counts=True,
            )
        states = {}
        for i, path in enumerate(targets):
            w = model.get_submodule(path).weight
            dead_legacy = (
                path in legacy_targets and sample_counts.get(path, 0) == 0
            )
            if is_group:
                if dead_legacy:
                    S, t, c, perm, salient_idx, salient_val = _optimal_group_state(
                        w, group=group, C=C
                    )
                else:
                    H_layer = hessians.pop(path).to(w.device)
                    (S, t, c, perm, salient_idx, salient_val), _ = solve_group_state(
                        w, H_layer, group=group, C=C, percdamp=percdamp,
                        act_order=act_order, refine_iters=refine_iters,
                        scale_refit=scale_refit, grid=grid, itf_iters=itf_iters,
                        salient_first=salient_first, salient_scope=salient_scope,
                        in_sweep_refit=in_sweep_refit,
                    )
                states[path] = (S.cpu(), t.cpu(), c.cpu(), perm.cpu(),
                                salient_idx.cpu(), salient_val.cpu())
            elif mode == "gptq":
                if dead_legacy:
                    s, t = optimal_ternary(w)
                    c = residual_counter(w, s, t, C)
                else:
                    s, t, c = gptq_ternary(
                        w, hessians.pop(path).to(w.device), C=C, blocksize=blocksize,
                        percdamp=percdamp, act_order=act_order,
                    )
                states[path] = (s.cpu(), t.cpu(), c.cpu())
            elif mode == "optimal":
                s, t = optimal_ternary(w)
                c = residual_counter(w, s, t, C)
                states[path] = (s.cpu(), t.to(torch.int16).cpu(), c.cpu())
            if progress and (i + 1) % 25 == 0:
                print(f"[ptq:{mode}] {i+1}/{len(targets)} layers solved", flush=True)

        for target in stacked_targets:
            target_states = []
            for expert in range(target.num_experts):
                gate_up = target.module.gate_up_proj[expert]
                down = target.module.down_proj[expert]
                count = sample_counts.get(target.gate_up_path(expert), 0)
                if is_group:
                    if count == 0:
                        gate_up_state = _optimal_group_state(
                            gate_up, group=group, C=C
                        )
                        down_state = _optimal_group_state(down, group=group, C=C)
                    else:
                        gate_up_state, _ = solve_group_state(
                            gate_up,
                            hessians.pop(target.gate_up_path(expert)).to(gate_up.device),
                            group=group, C=C, percdamp=percdamp,
                            act_order=act_order, refine_iters=refine_iters,
                            scale_refit=scale_refit, grid=grid, itf_iters=itf_iters,
                            salient_first=salient_first, salient_scope=salient_scope,
                            in_sweep_refit=in_sweep_refit,
                        )
                        down_state, _ = solve_group_state(
                            down,
                            hessians.pop(target.down_path(expert)).to(down.device),
                            group=group, C=C, percdamp=percdamp,
                            act_order=act_order, refine_iters=refine_iters,
                            scale_refit=scale_refit, grid=grid, itf_iters=itf_iters,
                            salient_first=salient_first, salient_scope=salient_scope,
                            in_sweep_refit=in_sweep_refit,
                        )
                    target_states.append(
                        (
                            tuple(value.cpu() for value in gate_up_state),
                            tuple(value.cpu() for value in down_state),
                        )
                    )
                elif mode == "gptq" and count > 0:
                    gate_up_state = gptq_ternary(
                        gate_up,
                        hessians.pop(target.gate_up_path(expert)).to(gate_up.device),
                        C=C, blocksize=blocksize, percdamp=percdamp,
                        act_order=act_order,
                    )
                    down_state = gptq_ternary(
                        down,
                        hessians.pop(target.down_path(expert)).to(down.device),
                        C=C, blocksize=blocksize, percdamp=percdamp,
                        act_order=act_order,
                    )
                    target_states.append(
                        (
                            tuple(value.cpu() for value in gate_up_state),
                            tuple(value.cpu() for value in down_state),
                        )
                    )
                else:
                    s_gu, t_gu = optimal_ternary(gate_up)
                    s_d, t_d = optimal_ternary(down)
                    target_states.append(
                        (
                            (
                                s_gu.cpu(), t_gu.to(torch.int16).cpu(),
                                residual_counter(gate_up, s_gu, t_gu, C).cpu(),
                            ),
                            (
                                s_d.cpu(), t_d.to(torch.int16).cpu(),
                                residual_counter(down, s_d, t_d, C).cpu(),
                            ),
                        )
                    )
            moe_states[target.path] = target_states

    expert_counts, dead_experts, rank_deficient = _moe_calibration_diagnostics(
        model, stacked_targets, legacy_targets, sample_counts
    )
    if is_group:
        report = SwapReport()
        packed_kinds = {
            "counter_packed", "counter_triton", "group_packed", "group_scale_packed",
        }
        want_packed = kind in packed_kinds
        reference_supported = {
            "lr", "lr_scale", "rms_beta", "rms_eps", "local_grad_clip", "residual_alpha",
        }
        packed_supported = reference_supported | {
            "kernel_mode", "strict_update", "flip_sample_size",
        }
        warned_fallback = False
        for path in targets:
            parent, name = _parent_and_name(model, path)
            lin = getattr(parent, name)
            S, t, c, perm, salient_idx, salient_val = states[path]
            packed_ok = want_packed and lin.in_features % 4 == 0 and group % 4 == 0
            if packed_ok:
                kw = {k: v for k, v in counter_kw.items() if k in packed_supported}
                counter: nn.Module = PackedGroupScaleCounterLinear(
                    lin.in_features, lin.out_features, group=group, C=C, perm=perm, **kw
                )
            else:
                if want_packed and progress and not warned_fallback:
                    print(
                        "[ptq:gptq_group] packed kernel requires in_features%4==0 and group%4==0; "
                        "falling back to the torch group layer for unsupported shapes",
                        flush=True,
                    )
                    warned_fallback = True
                kw = {k: v for k, v in counter_kw.items() if k in reference_supported}
                counter = GroupScaleCounterLinear(
                    lin.in_features, lin.out_features, group=group, C=C, perm=perm, **kw
                )
            counter.load_group_state(S, t, c, perm,
                                     salient_idx=salient_idx, salient_val=salient_val)
            replacement: nn.Module = counter
            if lin.bias is not None and keep_bias:
                replacement = CounterLinearWithBias(counter, lin.bias)
            setattr(parent, name, replacement)
            report.swapped.append(path)
            report.coeffs += lin.in_features * lin.out_features
    else:
        from ..convert import swap_linears_to_counter
        def swap_skip(path):
            return path in router_paths or any(token in path for token in skip)

        report = swap_linears_to_counter(
            model, kind=kind, skip=swap_skip, C=C, keep_bias=keep_bias, **counter_kw
        )
        for path, (s, t, c) in states.items():
            mod = model.get_submodule(path)
            if isinstance(mod, CounterLinearWithBias):
                mod = mod.counter
            mod.load_counter_state(s, t, c)

    _swap_stacked_moe(
        model, stacked_targets, moe_states, report, is_group=is_group, kind=kind,
        group=group, C=C, counter_kw=counter_kw,
    )
    report.expert_token_counts.update(expert_counts)
    report.dead_experts.extend(dead_experts)
    report.rank_deficient_experts.extend(rank_deficient)
    if device is not None:
        model.to(device)
    return report
