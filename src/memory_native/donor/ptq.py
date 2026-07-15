"""Calibrated ternary PTQ and trainable group-counter warm starts."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..convert import CounterLinearWithBias, SwapReport
from ..counter import C_DEFAULT
from ..group_scale_counter import GroupScaleCounterLinear

__all__ = [
    "optimal_ternary", "gptq_ternary", "gptq_group_ternary", "residual_counter",
    "group_residual_counter", "collect_hessians", "quantize_dense_group_ternary",
    "ptq_warm_start",
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


def _initial_group_scales(W: torch.Tensor, group: int) -> torch.Tensor:
    out, cols = W.shape
    n_groups = (cols + group - 1) // group
    S = torch.empty((out, n_groups), device=W.device, dtype=torch.float32)
    for g in range(n_groups):
        i1, i2 = g * group, min((g + 1) * group, cols)
        S[:, g] = optimal_ternary(W[:, i1:i2])[0].squeeze(1)
    return S


def _group_sweep(W0: torch.Tensor, Hinv: torch.Tensor, S: torch.Tensor, group: int):
    """One GPTQ sweep at fixed group scales; all tensors are in act-ordered layout."""
    W = W0.clone()
    cols = W.shape[1]
    Q = torch.zeros_like(W)
    T = torch.zeros_like(W)
    for g in range(S.shape[1]):
        i1, i2 = g * group, min((g + 1) * group, cols)
        Wb = W[:, i1:i2].clone()
        Qb = torch.zeros_like(Wb)
        Eb = torch.zeros_like(Wb)
        Hb = Hinv[i1:i2, i1:i2]
        sg = S[:, g]
        for j in range(i2 - i1):
            wcol = Wb[:, j]
            tcol = (wcol / sg).round_().clamp_(-1, 1)
            q = tcol * sg
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
    return Q, T, W


def _refit_scales(W: torch.Tensor, T: torch.Tensor, H: torch.Tensor, group: int,
                  mode: str, previous: torch.Tensor) -> torch.Tensor:
    """Refit scales after the achieved ternary support is known.

    ``hdiag`` is the stable default: exact minimizer under diag(H), so calibration salience
    participates without an O(groups^2) state. ``hessian_cd`` performs exact-H coordinate
    updates and is intended for research/small group counts because it is much more expensive.
    """
    _, cols = W.shape
    n_groups = previous.shape[1]
    S = previous.clone()
    if mode not in {"l2", "hdiag", "hessian_cd"}:
        raise ValueError("scale_refit must be 'l2', 'hdiag' or 'hessian_cd'")
    if mode in {"l2", "hdiag"}:
        d = torch.ones(cols, device=W.device, dtype=W.dtype)
        if mode == "hdiag":
            d = torch.diagonal(H).clamp_min(1e-12)
        for g in range(n_groups):
            i1, i2 = g * group, min((g + 1) * group, cols)
            tg = T[:, i1:i2]
            dg = d[i1:i2].unsqueeze(0)
            num = (W[:, i1:i2] * tg * dg).sum(dim=1)
            den = (tg.square() * dg).sum(dim=1)
            valid = den > 1e-12
            cand = (num / den.clamp_min(1e-12)).clamp_min(1e-8)
            S[:, g] = torch.where(valid & (num > 0), cand, S[:, g])
        return S

    gidx = torch.div(torch.arange(cols, device=W.device), group, rounding_mode="floor")
    recon = T * S[:, gidx]
    for g in range(n_groups):
        i1, i2 = g * group, min((g + 1) * group, cols)
        tg = T[:, i1:i2]
        if not tg.count_nonzero():
            continue
        residual = W - recon
        residual[:, i1:i2] += S[:, g:g + 1] * tg
        Hr = residual @ H
        num = (tg * Hr[:, i1:i2]).sum(dim=1)
        Hgg = H[i1:i2, i1:i2]
        den = ((tg @ Hgg) * tg).sum(dim=1).clamp_min(1e-12)
        cand = (num / den).clamp_min(1e-8)
        valid = (num > 0) & torch.isfinite(cand)
        new_s = torch.where(valid, cand, S[:, g])
        recon[:, i1:i2] = new_s.unsqueeze(1) * tg
        S[:, g] = new_s
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
    return_perm: bool = False,
):
    """Group-scale GPTQ v3 with true post-sweep s<->t alternation.

    v2 estimated a scale before the sequential GPTQ sweep and never refit against the support
    actually produced by error feedback. v3 alternates: fixed-S GPTQ sweep -> Hessian-weighted
    scale refit on achieved T -> fresh GPTQ sweep. Candidate iterations are accepted only when
    the calibration Hessian objective does not increase.

    Returns ``(Q, S, T)`` for compatibility, or ``(Q, S, T, perm, W_adjusted)`` with
    ``return_perm=True``. Q/T are restored to original input order; S indexes groups in
    permuted act-order.
    """
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
    S = _initial_group_scales(W, group)
    Q, T, W_adjusted = _group_sweep(W, Hinv, S, group)
    best_err = _hessian_error(W, Q, Hdamped)

    if refine_scale:
        for _ in range(max(0, int(refine_iters))):
            candidate_S = _refit_scales(W, T, Hdamped, group, scale_refit, S)
            candidate_Q, candidate_T, candidate_W = _group_sweep(W, Hinv, candidate_S, group)
            candidate_err = _hessian_error(W, candidate_Q, Hdamped)
            if not torch.isfinite(candidate_err) or candidate_err > best_err * (1.0 + 1e-7):
                break
            S, Q, T, W_adjusted = candidate_S, candidate_Q, candidate_T, candidate_W
            best_err = candidate_err

    Q_orig = Q[:, invperm]
    T_orig = T[:, invperm]
    W_adjusted_orig = W_adjusted[:, invperm]
    if return_perm:
        return Q_orig, S, T_orig.to(torch.int16), perm, W_adjusted_orig
    return Q_orig, S, T_orig.to(torch.int16)


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
                                  progress: bool = True) -> None:
    skip = ["lm_head"] + (list(extra_skip) if extra_skip is not None else [])
    targets = _target_paths(model, skip)
    hessians = collect_hessians(model, targets, calib_batches)
    for i, path in enumerate(targets):
        lin = model.get_submodule(path)
        w_hat, _, _ = gptq_group_ternary(
            lin.weight, hessians.pop(path), group=group, percdamp=percdamp,
            refine_iters=refine_iters, scale_refit=scale_refit,
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
    progress: bool = True,
    **counter_kw,
) -> SwapReport:
    """Swap body linears to a calibrated counter format.

    ``mode='gptq_group'`` is solver-v3's end-to-end bridge: it preserves group scales and
    act-order metadata in a trainable ``GroupScaleCounterLinear`` instead of collapsing the
    strong group PTQ solution back to a single row scale.
    """
    skip = ["lm_head"] + (list(extra_skip) if extra_skip is not None else [])
    targets = _target_paths(model, skip)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = None

    is_group = mode in {"gptq_group", "group128v3", "group"}
    hessians = (
        collect_hessians(model, targets, calib_batches)
        if mode.startswith("gptq") or is_group else {}
    )
    states: dict[str, tuple] = {}
    for i, path in enumerate(targets):
        w = model.get_submodule(path).weight
        if is_group:
            _, S, t, perm, Wadj = gptq_group_ternary(
                w, hessians.pop(path), group=group, percdamp=percdamp,
                act_order=act_order, refine_iters=refine_iters, scale_refit=scale_refit,
                return_perm=True,
            )
            c = group_residual_counter(Wadj, S, t, perm, group, C)
            states[path] = (S.cpu(), t.cpu(), c.cpu(), perm.cpu())
        elif mode == "gptq":
            s, t, c = gptq_ternary(
                w, hessians.pop(path), C=C, blocksize=blocksize,
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
        supported = {"lr", "lr_scale", "rms_beta", "rms_eps", "local_grad_clip", "residual_alpha"}
        group_kw = {k: v for k, v in counter_kw.items() if k in supported}
        for path in targets:
            parent, name = _parent_and_name(model, path)
            lin = getattr(parent, name)
            S, t, c, perm = states[path]
            counter = GroupScaleCounterLinear(
                lin.in_features, lin.out_features, group=group, C=C, perm=perm, **group_kw
            )
            counter.load_group_state(S, t, c, perm)
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
