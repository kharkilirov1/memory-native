"""GPTAQ-style asymmetric (cascade-aware) calibration for the group-ternary solver.

The plain pipeline collects H = X_fp^T X_fp from the UNQUANTIZED model and solves every
layer against its full-precision input. At inference the layer actually sees X_q -- the
input distorted by every quantized layer before it -- so the deployed per-layer
objective is really

    min_Q ||X_q Q^T - X_fp W^T||_F^2
        = sum_rows (q - w~)^T H_q (q - w~) + const,
    H_q = X_q^T X_q,   G = X_q^T X_fp,   w~ = H_q^{-1} G w   (damped solve).

So the EXISTING solver runs unchanged on (w~, H_q); only the statistics change. Later
layers then actively cancel the error accumulated by earlier ones instead of ignoring
it. Measured on a 2-layer SiLU cascade: -47% network error vs the classic objective,
while merely re-collecting H on quantized inputs at the ORIGINAL target (H_q, w) buys
nothing -- the target shift is the whole effect. External evidence at W2 (Llama-2-7B
wiki2): GPTQ 6875 -> GPTAQ 1269 (arXiv:2504.02692) -> LoaQ 214 (arXiv:2509.06297).

Mechanics: two towers. ``model`` (the one being quantized) advances chunk by chunk --
each solved layer's dense reconstruction is written back into its weight so later
chunks see quantized inputs; ``model_fp`` is an untouched deepcopy providing the fp
stream. Per chunk, one pass over the calibration runs BOTH towers batch-synchronized
and accumulates (H_q, G) for the chunk's layers. Cost: 2 x len(chunks) calibration
passes and a second resident model; ``chunk_layers=7`` is one Qwen2 decoder block
(q,k,v,o,gate,up,down -- named_modules order matches execution order, which is what
makes the sequential scheme valid).

``strength`` interpolates the target w -> w~ (QEP-style, arXiv:2504.09629): the exact
asymmetric target can overcorrect on thin calibrations; 1.0 = full asymmetric.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .ptq import solve_group_state

__all__ = ["asym_target_weights", "collect_asym_stats", "asym_solve_states"]


@torch.no_grad()
def asym_target_weights(w: torch.Tensor, H_q: torch.Tensor, G: torch.Tensor, *,
                        percdamp: float = 0.01, strength: float = 1.0) -> torch.Tensor:
    """Damped asymmetric target in RESIDUAL form (the GPTAQ parameterization):

        w~ = w + (H_q + lambda I)^{-1} (G - H_q) w,

    i.e. Tikhonov-regularized toward w, NOT toward zero. The naive form
    (H_q + lambda I)^{-1} G w solves the same undamped problem but its damping shrinks
    weights toward 0 in weakly excited directions -- measured on real attn layers with
    near-identical streams it LOST 4x network-KL vs classic for exactly that reason.
    In residual form the correction vanishes identically when the streams coincide
    (G == H_q), for ANY damping; lambda only tempers the correction itself.

    Dead quantized channels (zero H_q diagonal: the q-stream never excites them) keep
    the original weight -- their contribution to the H_q metric is zero either way."""
    W = w.detach().to(torch.float32)
    H = H_q.detach().to(torch.float32).clone()
    diag = torch.diagonal(H)
    dead = diag == 0
    if dead.any():
        H[dead, dead] = 1.0
    damp = percdamp * torch.diagonal(H).mean()
    H += torch.eye(H.shape[0], device=H.device, dtype=H.dtype) * damp
    L = torch.linalg.cholesky(H)
    R = (G.to(torch.float32) - H_q.detach().to(torch.float32)) @ W.t()
    Wt = W + torch.cholesky_solve(R, L).t()
    if dead.any():
        Wt[:, dead] = W[:, dead]
    if strength != 1.0:
        Wt = W + float(strength) * (Wt - W)
    return Wt


@torch.no_grad()
def collect_asym_stats(model_q: nn.Module, model_fp: nn.Module, targets: list[str],
                       calib_batches) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """One pass over the calibration, both towers batch-synchronized: returns
    {path: (H_q, G)} for the given targets. The fp tower runs FIRST each batch and its
    per-layer inputs are held until the q tower consumes them (memory: one batch of
    activations for every target in the chunk -- size the chunk accordingly)."""
    stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    cap_fp: dict[str, torch.Tensor] = {}
    hooks = []
    fp_dev = next(model_fp.parameters()).device
    q_dev = next(model_q.parameters()).device

    def fp_hook(path, fin):
        def hook(_m, inputs):
            cap_fp[path] = inputs[0].detach().reshape(-1, fin).to(torch.float32)
        return hook

    def q_hook(path, fin):
        def hook(_m, inputs):
            xq = inputs[0].detach().reshape(-1, fin).to(torch.float32)
            # The fp tower may live on another device (2-GPU solve): hop its capture over.
            xf = cap_fp.pop(path).to(xq.device, non_blocking=True)
            if xf.shape != xq.shape:
                raise RuntimeError(f"fp/q stream shape mismatch at {path}: "
                                   f"{tuple(xf.shape)} vs {tuple(xq.shape)}")
            pair = stats.get(path)
            if pair is None:
                pair = (torch.zeros(fin, fin, dtype=torch.float32, device=xq.device),
                        torch.zeros(fin, fin, dtype=torch.float32, device=xq.device))
                stats[path] = pair
            pair[0].addmm_(xq.t(), xq)
            pair[1].addmm_(xq.t(), xf)
        return hook

    for path in targets:
        lin_fp = model_fp.get_submodule(path)
        lin_q = model_q.get_submodule(path)
        hooks.append(lin_fp.register_forward_pre_hook(fp_hook(path, lin_fp.in_features)))
        hooks.append(lin_q.register_forward_pre_hook(q_hook(path, lin_q.in_features)))
    was_fp, was_q = model_fp.training, model_q.training
    model_fp.eval()
    model_q.eval()
    try:
        for ids in calib_batches:
            model_fp(ids.to(fp_dev) if hasattr(ids, "to") else ids)
            model_q(ids.to(q_dev) if hasattr(ids, "to") else ids)
            if cap_fp:
                raise RuntimeError(f"unconsumed fp captures: {sorted(cap_fp)} -- "
                                   "fp/q towers disagree on executed targets")
    finally:
        for h in hooks:
            h.remove()
        model_fp.train(was_fp)
        model_q.train(was_q)
    return stats


@torch.no_grad()
def asym_solve_states(model: nn.Module, calib_batches, targets: list[str], *,
                      group: int, C: int, percdamp: float = 0.01,
                      act_order: bool = True, refine_iters: int = 2,
                      scale_refit: str = "hdiag", grid: str = "sym",
                      itf_iters: int = 3, salient_first: float = 0.0,
                      salient_scope: str = "row", in_sweep_refit: bool = False,
                      chunk_layers: int = 7, strength: float = 1.0,
                      model_fp: nn.Module | None = None,
                      w0: dict[str, torch.Tensor] | None = None,
                      fp_device: str | None = None,
                      progress: bool = True) -> dict[str, tuple]:
    """Sequential asymmetric solve over ``targets`` (named_modules order == execution
    order). Returns {path: (S, t, c, perm, salient_idx, salient_val)} on CPU, mutating
    MODEL's dense weights to the deployable reconstruction as it goes -- the caller
    (``ptq_warm_start``) swaps the counter layers in afterwards.

    For MULTI-PASS solving (iterated asym: re-collect stats on the fully quantized net,
    re-solve from the ORIGINAL weights against the updated streams) pass ``model_fp``
    (the untouched fp tower) and ``w0`` (the original dense weights per path) captured
    BEFORE the first pass -- on a second call ``model`` is already mutated, so both
    must come from outside. Single-pass callers can omit them.

    Memory: one full deepcopy of the model (the fp tower) for the duration."""
    if model_fp is None:
        model_fp = copy.deepcopy(model)
        if fp_device is not None:
            model_fp = model_fp.to(fp_device)
    model_fp.eval()
    states: dict[str, tuple] = {}
    chunks = [targets[i: i + max(1, int(chunk_layers))]
              for i in range(0, len(targets), max(1, int(chunk_layers)))]
    done = 0
    try:
        for ci, chunk in enumerate(chunks):
            stats = collect_asym_stats(model, model_fp, chunk, calib_batches)
            for path in chunk:
                H_q, G = stats.pop(path)
                lin = model.get_submodule(path)
                w = w0[path] if w0 is not None else lin.weight
                wt = asym_target_weights(w, H_q, G, percdamp=percdamp,
                                         strength=strength)
                state, Q = solve_group_state(
                    wt, H_q, group=group, C=C, percdamp=percdamp,
                    act_order=act_order, refine_iters=refine_iters,
                    scale_refit=scale_refit, grid=grid, itf_iters=itf_iters,
                    salient_first=salient_first, salient_scope=salient_scope,
                    in_sweep_refit=in_sweep_refit,
                )
                S, t, c, perm, salient_idx, salient_val = state
                states[path] = (S.cpu(), t.cpu(), c.cpu(), perm.cpu(),
                                salient_idx.cpu(), salient_val.cpu())
                lin.weight.copy_(Q.to(lin.weight.dtype))
                del H_q, G
                done += 1
                if progress and done % 25 == 0:
                    print(f"[ptq:asym] {done}/{len(targets)} layers solved "
                          f"(chunk {ci + 1}/{len(chunks)})", flush=True)
    finally:
        del model_fp
    return states
