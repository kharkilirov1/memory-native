"""Research-only input-metric preconditioning of small counter BLAST factors.

For each factor, G <- beta G + (1-beta) X.T X / n on a training forward,
starting from identity. Its correlation H receives the right preconditioner
H (G + ridge * max(trace(G)/d, eps) I + eps I)^-1 before the existing
RMS/counter rule. Both counter ticks and scale learning use this new signal.
The input derivative continues to use the forward-time visible weights.

This is an uncentered second moment, not a centered covariance or a full
natural gradient. Input-metric preconditioners are familiar from K-FAC and
Shampoo; compatibility with discrete counter factors is an experimental
hypothesis. Persistent fp32 d-by-d Grams cost memory and the Gram/Cholesky
arithmetic costs compute. No packed-state or speed benefit is asserted.
The reference is limited to single-process exact-correlation eager training.
"""
from __future__ import annotations

import math

import torch

from ..counter import RMSCounterLinear
from .structured import CounterBLASTLinear

__all__ = ["CovarianceRMSCounterLinear", "precondition_counter_blast"]


class CovarianceRMSCounterLinear(RMSCounterLinear):
    """RMS counter layer with a persistent, locally sized input Gram EMA."""

    def __init__(self, *args, cov_beta=.95, cov_ridge=.1, cov_eps=1e-6, **kwargs):
        beta, ridge, eps = float(cov_beta), float(cov_ridge), float(cov_eps)
        if not math.isfinite(beta) or not 0 <= beta < 1:
            raise ValueError("cov_beta must be finite in [0, 1)")
        if not math.isfinite(ridge) or ridge < 0:
            raise ValueError("cov_ridge must be finite and nonnegative")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("cov_eps must be positive and finite")
        # Exclude paths that would change the meaning of the measured signal.
        for key, expected in (("rms_mode", "exact"), ("scale_rebase", "eager"),
                              ("pulse_mode", "direct"), ("update_compute", "fp"),
                              ("forward_compute", "fp"), ("act_save_bits", 0),
                              ("decimate_updates", False), ("compile_update", False)):
            if key in kwargs and kwargs[key] != expected:
                raise ValueError(f"covariance reference requires {key}={expected!r}")
        super().__init__(*args, **kwargs)
        self.cov_beta, self.cov_ridge, self.cov_eps = beta, ridge, eps
        self.register_buffer("input_gram", torch.eye(self.in_features, dtype=torch.float32))
        self.register_buffer("gram_updates", torch.zeros((), dtype=torch.int64))
        self.register_buffer("_cov_config", torch.tensor([
            self.in_features, self.out_features, self.C, beta, ridge, eps,
        ], dtype=torch.float64))

    def forward(self, x):
        observe = torch.is_grad_enabled() and self.training and self.update_enabled
        if observe:
            if torch.distributed.is_available() and torch.distributed.is_initialized() \
                    and torch.distributed.get_world_size() > 1:
                raise RuntimeError("covariance reference supports single-process training only")
            if x.numel() == 0 or not bool(torch.isfinite(x).all()):
                raise ValueError("Gram observation requires nonempty finite inputs")
        # Keep the inherited reuse guard: a refused second forward must not
        # change the statistic used by the outstanding backward.
        result = super().forward(x)
        if observe:
            with torch.no_grad():
                rows = x.detach().reshape(-1, self.in_features).to(self.input_gram.dtype)
                gram = rows.T @ rows / rows.shape[0]
                self.input_gram.mul_(self.cov_beta).add_(gram, alpha=1-self.cov_beta)
                self.gram_updates.add_(1)
        return result

    @torch.no_grad()
    def precondition(self, grad_w):
        """Return H M^-1 by an SPD Cholesky solve; never form the inverse."""
        gram = self.input_gram.to(device=grad_w.device, dtype=grad_w.dtype)
        gram = (gram + gram.T) * .5
        mean_diagonal = gram.diagonal().mean().clamp_min(self.cov_eps)
        damping = self.cov_ridge * mean_diagonal + self.cov_eps
        metric = gram + damping * torch.eye(self.in_features, device=gram.device, dtype=gram.dtype)
        factor = torch.linalg.cholesky(metric)
        return torch.cholesky_solve(grad_w.T.contiguous(), factor).T

    @torch.no_grad()
    def _update_tile(self, lo, hi, grad_w, t_i, c_i, s_i, proxy_gsq=None):
        super()._update_tile(lo, hi, self.precondition(grad_w), t_i, c_i, s_i)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        saved = state_dict.get(prefix + "_cov_config")
        if not isinstance(saved, torch.Tensor) or saved.shape != self._cov_config.shape \
                or not torch.equal(saved.detach().cpu(), self._cov_config.detach().cpu()):
            raise RuntimeError("covariance checkpoint dimensions, C or metric configuration mismatch")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)


def precondition_counter_blast(model, *, cov_beta=.95, cov_ridge=.1, cov_eps=1e-6):
    """Replace U/V/S in-place, preserving original factor state and the RNG.

    Construct the ordinary CounterBLASTLinear first for matched initialization.
    This factory adds only local Grams: V sees block_size inputs, while U and S
    see rank and number-of-input-blocks inputs. No full operator Gram is used.
    """
    if not isinstance(model, CounterBLASTLinear):
        raise TypeError("expected CounterBLASTLinear")
    groups = (model.V, model.S, model.U)
    if any(type(old) is not RMSCounterLinear for group in groups for old in group):
        raise TypeError("expected ordinary, unmodified RMSCounterLinear factors")
    if any(old._outstanding_forward for group in groups for old in group):
        raise RuntimeError("cannot convert factors with an outstanding forward")
    if any(old.scale.dtype != torch.float32 for group in groups for old in group):
        raise TypeError("covariance conversion requires float32 counter buffers")
    replacements = []
    # Constructors sample disposable counter codes. Restoring RNG makes
    # conversion itself neutral to the subsequent stochastic-rounding stream.
    with torch.random.fork_rng(devices=[]):
        for group in groups:
            new_group = []
            for old in group:
                options = {name: getattr(old, name) for name in (
                    "C", "lr", "lr_scale", "tile_rows", "local_grad_clip", "pulse_mode",
                    "act_save_bits", "decimate_updates", "cache_mode", "cache_patch",
                    "update_compute", "forward_compute", "compile_update", "rms_beta",
                    "rms_eps", "use_rms", "rms_mode", "scale_rebase")}
                new = CovarianceRMSCounterLinear(old.in_features, old.out_features,
                    cov_beta=cov_beta, cov_ridge=cov_ridge, cov_eps=cov_eps, **options)
                new.to(device=old.scale.device)
                state = new.state_dict()
                state.update(old.state_dict())
                new.load_state_dict(state)
                new.train(old.training)
                new.update_enabled = old.update_enabled
                new_group.append(new)
            replacements.append(new_group)
    for group, replacement in zip(groups, replacements):
        for index, new in enumerate(replacement):
            group[index] = new
    return model
