"""Calibrated PTQ warm-start: put the counter format's best foot forward BEFORE recovery.

Motivation (measured, 2026-07): the naive TWN threshold warm-start collapses a 1.5B donor to
PPL ~5e5 and recovery has to resurrect it. Bonsai-27B (PrismML) showed post-training
quantization alone -- group scales + calibrated reconstruction, NO retraining -- retains ~90%
of an fp donor at ~1.1-1.7 bpw. Our format is per-ROW scale + ternary (w ~= s_row * t), so we
adapt the two proven PTQ ingredients to it:

  1. optimal_ternary  -- the EXACT per-row L2 minimizer of ||w - s*t|| (search over support
     size k; TWN's fixed threshold_ratio is a heuristic approximation of this).
  2. gptq_ternary     -- GPTQ-style column-sequential rounding to the {-s,0,+s} grid with
     Hessian error feedback (H = X^T X from calibration data, Cholesky inverse, optional
     activation ordering). Weight-only errors stop mattering; what the LAYER OUTPUT sees does.

`ptq_warm_start(model, batches, ...)` = collect Hessians on the fp model -> compute (s,t,c)
per target linear -> swap_linears_to_counter -> load_counter_state. Drop-in replacement for
qwen_to_counter when calibration batches are available.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..convert import CounterLinearWithBias, SwapReport, swap_linears_to_counter
from ..counter import C_DEFAULT

__all__ = ["optimal_ternary", "gptq_ternary", "gptq_group_ternary", "residual_counter",
           "collect_hessians", "quantize_dense_group_ternary", "ptq_warm_start"]


@torch.no_grad()
def optimal_ternary(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact per-row L2-optimal (s, t): for support size k the best choice is the top-k |w|
    with s = mean of those, and the objective (sum top-k)^2 / k is maximized over k."""
    w = w.to(torch.float32)
    absw = w.abs()
    vals, _ = absw.sort(dim=1, descending=True)
    csum = vals.cumsum(dim=1)
    k = torch.arange(1, w.shape[1] + 1, device=w.device, dtype=w.dtype)
    kstar = (csum.pow(2) / k).argmax(dim=1, keepdim=True)             # [out,1]
    s = (csum.gather(1, kstar) / (kstar + 1).to(w.dtype)).clamp_min(1e-8)
    thr = vals.gather(1, kstar)                                        # k-th largest |w|
    t = torch.sign(w) * (absw >= thr).to(w.dtype)
    return s, t


@torch.no_grad()
def residual_counter(w: torch.Tensor, s: torch.Tensor, t: torch.Tensor,
                     C: int = C_DEFAULT) -> torch.Tensor:
    """Seed the sub-threshold counter with the quantization residual (same rule as the TWN path)."""
    c = ((w.to(torch.float32) / s - t) * C).round().clamp_(-(C - 1), C - 1)
    return c.to(torch.int16)


@torch.no_grad()
def gptq_ternary(w: torch.Tensor, H: torch.Tensor, *, C: int = C_DEFAULT,
                 blocksize: int = 128, percdamp: float = 0.01, act_order: bool = True,
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPTQ-adapted ternary: per-row scale from optimal_ternary, then column-sequential
    nearest-grid rounding to {-s, 0, +s} with Hessian error propagation.

    Returns (s [out,1] fp32, t [out,in] int16, c [out,in] int16); c is seeded from the
    error-adjusted weight, i.e. the residual the counter 'was carrying' after reconstruction."""
    W = w.detach().to(torch.float32).clone()
    out, cols = W.shape
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

    s, _ = optimal_ternary(W)                     # row stats are permutation-invariant
    sq = s.squeeze(1)

    damp = percdamp * torch.mean(torch.diagonal(H))
    H += torch.eye(cols, device=H.device) * damp
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    Q = torch.zeros_like(W)
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        Wb = W[:, i1:i2].clone()
        Qb = torch.zeros_like(Wb)
        Eb = torch.zeros_like(Wb)
        Hb = Hinv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            wcol = Wb[:, j]
            q = (wcol / sq).round_().clamp_(-1, 1) * sq       # nearest of {-s,0,+s}
            Qb[:, j] = q
            e = (wcol - q) / Hb[j, j]
            Eb[:, j] = e
            if j + 1 < i2 - i1:
                Wb[:, j + 1:] -= e.unsqueeze(1) * Hb[j, j + 1:].unsqueeze(0)
        Q[:, i1:i2] = Qb
        W[:, i1:i2] = Wb                                       # adjusted (for the residual seed)
        if i2 < cols:
            W[:, i2:] -= Eb @ Hinv[i1:i2, i2:]

    if act_order:
        Q = Q[:, invperm]
        W = W[:, invperm]

    t = (Q / s).round()
    c = residual_counter(W, s, t, C)
    return s, t.to(torch.int16), c


def _prep_hinv(H: torch.Tensor, W: torch.Tensor, percdamp: float):
    """Shared GPTQ Hessian prep: dead-column handling, damping, Cholesky-inverse-Cholesky."""
    H = H.detach().to(torch.float32).clone()
    diag = torch.diagonal(H)
    dead = diag == 0
    if dead.any():
        H[dead, dead] = 1.0
        W[:, dead] = 0.0
    damp = percdamp * torch.mean(torch.diagonal(H))
    H += torch.eye(H.shape[0], device=H.device) * damp
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    return torch.linalg.cholesky(Hinv, upper=True)


@torch.no_grad()
def gptq_group_ternary(w: torch.Tensor, H: torch.Tensor, *, group: int = 128,
                       percdamp: float = 0.01,
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bonsai-granularity probe: GPTQ ternary with a scale PER (row, group-of-`group`-inputs)
    instead of per row -- ~in/group times finer, i.e. the ~1.71 bpw ternary layout
    (log2(3) + 16/group). Group scales are recomputed from the error-ADJUSTED weights on
    entry to each group (AutoGPTQ-style dynamic groups); no act-order (groups stay aligned
    to the input layout). This is an INFERENCE-quantization experiment -- it returns the
    reconstructed dense weight, not a counter-format state.

    Returns (w_hat [out,in] fp32, s [out, ceil(in/group)] fp32, t [out,in] int16)."""
    W = w.detach().to(torch.float32).clone()
    out, cols = W.shape
    Hinv = _prep_hinv(H, W, percdamp)

    n_groups = (cols + group - 1) // group
    S = torch.zeros(out, n_groups, dtype=torch.float32, device=W.device)
    Q = torch.zeros_like(W)
    for g in range(n_groups):
        i1, i2 = g * group, min((g + 1) * group, cols)
        Wb = W[:, i1:i2].clone()
        sg = optimal_ternary(Wb)[0].squeeze(1)            # per-row scale for THIS group
        S[:, g] = sg
        Qb = torch.zeros_like(Wb)
        Eb = torch.zeros_like(Wb)
        Hb = Hinv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            wcol = Wb[:, j]
            q = (wcol / sg).round_().clamp_(-1, 1) * sg
            Qb[:, j] = q
            e = (wcol - q) / Hb[j, j]
            Eb[:, j] = e
            if j + 1 < i2 - i1:
                Wb[:, j + 1:] -= e.unsqueeze(1) * Hb[j, j + 1:].unsqueeze(0)
        Q[:, i1:i2] = Qb
        W[:, i1:i2] = Wb
        if i2 < cols:
            W[:, i2:] -= Eb @ Hinv[i1:i2, i2:]

    t = torch.zeros_like(Q, dtype=torch.int16)
    for g in range(n_groups):                             # decode t group-wise (sign of Q/s)
        i1, i2 = g * group, min((g + 1) * group, cols)
        t[:, i1:i2] = (Q[:, i1:i2] / S[:, g:g + 1].clamp_min(1e-12)).round().to(torch.int16)
    return Q, S, t


def _target_paths(model: nn.Module, skip) -> list[str]:
    out = []
    for parent_path, parent in model.named_modules():
        for child_name, child in parent.named_children():
            if isinstance(child, nn.Linear):
                path = f"{parent_path}.{child_name}" if parent_path else child_name
                if not any(sub in path for sub in skip):
                    out.append(path)
    return out


@torch.no_grad()
def collect_hessians(model: nn.Module, targets: list[str], calib_batches) -> dict:
    """Accumulate H = X^T X (fp32, on-device) per target linear over the calibration batches."""
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
    for h in hooks:
        h.remove()
    model.train(was_training)
    return hessians


@torch.no_grad()
def quantize_dense_group_ternary(model: nn.Module, calib_batches, *, group: int = 128,
                                 percdamp: float = 0.01, extra_skip=None,
                                 progress: bool = True) -> None:
    """INFERENCE-quantization probe (Bonsai layout, our solver): overwrite every body linear's
    dense weight with its group-`group` GPTQ-ternary reconstruction, in place. No counter
    format, no training semantics -- this measures what scale GRANULARITY alone buys."""
    skip = ["lm_head"] + (list(extra_skip) if extra_skip is not None else [])
    targets = _target_paths(model, skip)
    hessians = collect_hessians(model, targets, calib_batches)
    for i, path in enumerate(targets):
        lin = model.get_submodule(path)
        w_hat, _, _ = gptq_group_ternary(lin.weight, hessians.pop(path),
                                         group=group, percdamp=percdamp)
        lin.weight.copy_(w_hat.to(lin.weight.dtype))
        if progress and (i + 1) % 25 == 0:
            print(f"[group{group}] {i+1}/{len(targets)} layers quantized", flush=True)


@torch.no_grad()
def ptq_warm_start(model: nn.Module, calib_batches, *, mode: str = "gptq",
                   kind: str = "counter_rms", C: int = C_DEFAULT, keep_bias: bool = True,
                   extra_skip=None, blocksize: int = 128, percdamp: float = 0.01,
                   act_order: bool = True, progress: bool = True,
                   **counter_kw) -> SwapReport:
    """Swap ``model``'s body linears to counter layers warm-started by CALIBRATED PTQ.

    mode="optimal": exact per-row optimal ternary (no calibration data used).
    mode="gptq":    optimal per-row scale + GPTQ error-feedback rounding against H = X^T X
                    accumulated on ``calib_batches`` (input_ids tensors, run through the fp
                    model with hooks; batches are only needed in this mode).

    Everything else mirrors qwen_to_counter (skip tied lm_head, keep q/k/v bias, re-.to(device)
    after the swap)."""
    skip = ["lm_head"] + (list(extra_skip) if extra_skip is not None else [])
    targets = _target_paths(model, skip)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = None

    hessians = collect_hessians(model, targets, calib_batches) if mode == "gptq" else {}

    # per-layer PTQ states BEFORE the swap frees the fp weights
    states: dict[str, tuple] = {}
    for i, path in enumerate(targets):
        w = model.get_submodule(path).weight
        if mode == "gptq":
            s, t, c = gptq_ternary(w, hessians.pop(path), C=C, blocksize=blocksize,
                                   percdamp=percdamp, act_order=act_order)
        elif mode == "optimal":
            s, t = optimal_ternary(w)
            c = residual_counter(w, s, t, C)
            t = t.to(torch.int16)
        else:
            raise ValueError("mode must be 'gptq' or 'optimal'")
        states[path] = (s.cpu(), t.cpu(), c.cpu())
        if progress and (i + 1) % 25 == 0:
            print(f"[ptq] {i+1}/{len(targets)} layers quantized", flush=True)

    report = swap_linears_to_counter(model, kind=kind, skip=skip, C=C,
                                     keep_bias=keep_bias, **counter_kw)
    for path, (s, t, c) in states.items():
        mod = model.get_submodule(path)
        if isinstance(mod, CounterLinearWithBias):
            mod = mod.counter
        mod.load_counter_state(s, t, c)
    if device is not None:
        model.to(device)
    return report
