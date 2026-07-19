"""Calibrated ternary PTQ and trainable group-counter warm starts."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..convert import CounterLinearWithBias, SwapReport
from ..counter import C_DEFAULT
from ..group_scale_counter import GroupScaleCounterLinear
from ..group_scale_packed import PackedGroupScaleCounterLinear

__all__ = [
    "optimal_ternary", "gptq_ternary", "gptq_group_ternary", "residual_counter",
    "group_residual_counter", "collect_hessians", "quantize_dense_group_ternary",
    "ptq_warm_start", "itf_grid", "align_scales_output",
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
        k = max(1, int(round(salient_first * cols)))
        thr = sal.kthvalue(cols - k + 1, dim=1, keepdim=True).values
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
def group_residual_counter(w_adjusted: torch.Tensor, scales: torch.Tensor, t: torch.Tensor,
                           perm: torch.Tensor, group: int, C: int = 11) -> torch.Tensor:
    cols = t.shape[1]
    group_perm = torch.div(torch.arange(cols, device=perm.device), group, rounding_mode="floor")
    group_index = torch.empty_like(group_perm)
    group_index[perm] = group_perm
    s_col = scales[:, group_index]
    c = ((w_adjusted.to(torch.float32) / s_col - t.to(torch.float32)) * C)
    return c.round().clamp_(-(C - 1), C - 1).to(torch.int16)


def _target_paths(model: nn.Module, skip) -> list[str]:
    out = []
    for parent_path, parent in model.named_modules():
        for child_name, child in parent.named_children():
            if isinstance(child, nn.Linear):
                path = f"{parent_path}.{child_name}" if parent_path else child_name
                if not any(sub in path for sub in skip):
                    out.append(path)
    return out


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
def collect_hessians(model: nn.Module, targets: list[str], calib_batches) -> dict:
    hessians: dict[str, torch.Tensor] = {}
    hooks = []
    was_training = model.training
    model.eval()

    def make_hook(path, in_features):
        def hook(_mod, inputs):
            x = inputs[0].detach().reshape(-1, in_features).to(torch.float32)
            h = hessians.get(path)
            if h is None:
                h = torch.zeros(in_features, in_features, dtype=torch.float32, device=x.device)
                hessians[path] = h
            h.addmm_(x.t(), x)
        return hook

    for path in targets:
        lin = model.get_submodule(path)
        hooks.append(lin.register_forward_pre_hook(make_hook(path, lin.in_features)))
    for ids in calib_batches:
        model(ids)
    for hook in hooks:
        hook.remove()
    model.train(was_training)
    return hessians


@torch.no_grad()
def quantize_dense_group_ternary(model: nn.Module, calib_batches, *, group: int = 128,
                                  percdamp: float = 0.01, extra_skip=None,
                                  refine_iters: int = 2, scale_refit: str = "hdiag",
                                  grid: str = "sym", itf_iters: int = 3,
                                  salient_first: float = 0.0,
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
            in_sweep_refit=in_sweep_refit,
        )
        lin.weight.copy_(w_hat.to(lin.weight.dtype))
        if progress and (i + 1) % 25 == 0:
            print(f"[group{group}-v3] {i+1}/{len(targets)} layers quantized", flush=True)


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
    in_sweep_refit: bool = False,
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
    """
    skip = ["lm_head"] + (list(extra_skip) if extra_skip is not None else [])
    targets = _target_paths(model, skip)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = None

    is_group = mode in {"gptq_group", "group128v3", "group"}
    if not is_group:
        # Group-only controls must not leak into the legacy counter path.
        for key in ("residual_alpha", "kernel_mode", "strict_update", "flip_sample_size"):
            counter_kw.pop(key, None)
    hessians: dict[str, torch.Tensor] = {}
    if mode.startswith("gptq") or is_group:
        chunks = _hessian_chunks(model, targets, int(hessian_gpu_budget_gib * 2**30))
        if len(chunks) == 1:
            hessians = collect_hessians(model, targets, calib_batches)
        else:
            # Each chunk re-runs the calibration forwards; Hessians park on CPU and the
            # solve loop below moves them back one layer at a time.
            if progress:
                print(f"[ptq:{mode}] hessians in {len(chunks)} chunks "
                      f"(budget {hessian_gpu_budget_gib:g} GiB)", flush=True)
            for chunk in chunks:
                part = collect_hessians(model, chunk, calib_batches)
                hessians.update({k: v.cpu() for k, v in part.items()})
                del part
                if device is not None and device.type == "cuda":
                    torch.cuda.empty_cache()
    states: dict[str, tuple] = {}
    for i, path in enumerate(targets):
        w = model.get_submodule(path).weight
        if is_group:
            H_layer = hessians.pop(path).to(w.device)
            _, S, t, perm, Wadj, (salient_idx, salient_val) = gptq_group_ternary(
                w, H_layer, group=group, percdamp=percdamp,
                act_order=act_order, refine_iters=refine_iters, scale_refit=scale_refit,
                grid=grid, itf_iters=itf_iters, salient_first=salient_first,
                in_sweep_refit=in_sweep_refit,
                return_perm=True, return_salient=True,
            )
            if grid == "itf":
                # The packed format is sym-scale: exact joint sym re-solve (A7) on the
                # achieved itf support, against the non-salient remainder.
                Hp = H_layer.detach().to(torch.float32)[perm][:, perm]
                w_perm = w.detach().to(torch.float32)[:, perm]
                t_perm = t[:, perm].to(torch.float32)
                w_target = w_perm
                if salient_idx.numel():
                    cols = w.shape[1]
                    invperm = torch.argsort(perm)
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
            states[path] = (S.cpu(), t.cpu(), c.cpu(), perm.cpu(),
                            salient_idx.cpu(), salient_val.cpu())
        elif mode == "gptq":
            s, t, c = gptq_ternary(
                w, hessians.pop(path).to(w.device), C=C, blocksize=blocksize,
                percdamp=percdamp, act_order=act_order,
            )
            states[path] = (s.cpu(), t.cpu(), c.cpu())
        elif mode == "optimal":
            s, t = optimal_ternary(w)
            c = residual_counter(w, s, t, C)
            states[path] = (s.cpu(), t.to(torch.int16).cpu(), c.cpu())
        else:
            raise ValueError("mode must be 'optimal', 'gptq' or 'gptq_group'")
        if progress and (i + 1) % 25 == 0:
            print(f"[ptq:{mode}] {i+1}/{len(targets)} layers solved", flush=True)

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
        report = swap_linears_to_counter(
            model, kind=kind, skip=skip, C=C, keep_bias=keep_bias, **counter_kw
        )
        for path, (s, t, c) in states.items():
            mod = model.get_submodule(path)
            if isinstance(mod, CounterLinearWithBias):
                mod = mod.counter
            mod.load_counter_state(s, t, c)

    if device is not None:
        model.to(device)
    return report
