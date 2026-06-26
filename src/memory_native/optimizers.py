"""Optimizer factory for the baselines the counter synapse should be compared against.

The point of the method is memory, so the honest comparison is not just FP32 AdamW but the
real memory-efficient-training optimizers. Supported names:

  adamw   : torch.optim.AdamW (FP32 moments) -- the heavy reference.
  bnb8    : bitsandbytes 8-bit AdamW (optional dep, CUDA only) -- 4x smaller optimizer state.
  galore  : GaLore-style low-rank projected AdamW (moments live in a low-rank subspace).
  lomo    : LoMo-style fused-backward SGD (update during backward, no grad/moment storage).

GaLore and LoMo are implemented here in plain PyTorch so they run on CPU and need no extra
deps. bnb8 is imported lazily and only works where bitsandbytes + CUDA are available.
"""
from __future__ import annotations

import torch

__all__ = ["build_optimizer", "GaLoreAdamW", "LoMo", "available_optimizers"]


def available_optimizers() -> list[str]:
    return ["adamw", "bnb8", "galore", "lomo"]


def build_optimizer(name: str, params, lr: float, **kw):
    params = list(params)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    if name == "bnb8":
        try:
            import bitsandbytes as bnb
        except Exception as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "optimizer 'bnb8' needs bitsandbytes (CUDA): pip install bitsandbytes"
            ) from exc
        return bnb.optim.AdamW8bit(params, lr=lr)
    if name == "galore":
        return GaLoreAdamW(params, lr=lr, **kw)
    if name == "lomo":
        return LoMo(params, lr=lr, **kw)
    raise ValueError(f"unknown optimizer {name!r}; choices: {available_optimizers()}")


class GaLoreAdamW(torch.optim.Optimizer):
    """Gradient Low-Rank Projection AdamW (Zhao et al. 2024), minimal pure-PyTorch version.

    For each 2-D parameter, the gradient G [m,n] is projected to a rank-r subspace P^T G (P
    from a periodic SVD of G), Adam runs on the small [r,n] moments, and the update is
    projected back. Optimizer-state memory for that parameter drops from 2*m*n to ~2*r*n.
    1-D parameters fall back to plain AdamW. This is a faithful-enough reimplementation for
    a memory baseline, not the reference package.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2,
                 rank=128, update_proj_gap=200, scale=0.25):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        rank=rank, update_proj_gap=update_proj_gap, scale=scale)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                low_rank = g.ndim == 2 and min(g.shape) > group["rank"]

                if "step" not in st:
                    st["step"] = 0
                    if low_rank:
                        st["proj"] = None
                    else:
                        st["exp_avg"] = torch.zeros_like(p)
                        st["exp_avg_sq"] = torch.zeros_like(p)
                st["step"] += 1
                t = st["step"]

                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])

                if not low_rank:
                    m, v = st["exp_avg"], st["exp_avg_sq"]
                    m.mul_(b1).add_(g, alpha=1 - b1)
                    v.mul_(b2).addcmul_(g, g, value=1 - b2)
                    mhat = m / (1 - b1 ** t)
                    vhat = v / (1 - b2 ** t)
                    p.addcdiv_(mhat, vhat.sqrt().add_(group["eps"]), value=-group["lr"])
                    continue

                # tall vs wide: project the long dimension down to rank r.
                m_, n_ = g.shape
                r = group["rank"]
                if st["proj"] is None or t % group["update_proj_gap"] == 1:
                    # periodic projection basis from the current gradient (left singular vecs)
                    U, _, Vh = torch.linalg.svd(g.float(), full_matrices=False)
                    if m_ >= n_:
                        st["side"] = "left"
                        st["proj"] = U[:, :r].contiguous()          # [m, r]
                    else:
                        st["side"] = "right"
                        st["proj"] = Vh[:r, :].contiguous()         # [r, n]
                    st["exp_avg"] = torch.zeros(
                        (r, n_) if st["side"] == "left" else (m_, r), device=p.device, dtype=p.dtype)
                    st["exp_avg_sq"] = torch.zeros_like(st["exp_avg"])

                P = st["proj"]
                if st["side"] == "left":
                    gr = P.t().to(g.dtype) @ g          # [r, n]
                else:
                    gr = g @ P.t().to(g.dtype)          # [m, r]
                m, v = st["exp_avg"], st["exp_avg_sq"]
                m.mul_(b1).add_(gr, alpha=1 - b1)
                v.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                mhat = m / (1 - b1 ** t)
                vhat = v / (1 - b2 ** t)
                upd = mhat / vhat.sqrt().add_(group["eps"])
                full = (P.to(g.dtype) @ upd) if st["side"] == "left" else (upd @ P.to(g.dtype))
                p.add_(full, alpha=-group["lr"] * group["scale"])
        return loss


class LoMo(torch.optim.Optimizer):
    """LoMo-style fused SGD (Lv et al. 2023): apply the update during backward via a grad
    hook and immediately free the gradient, so the full gradient tensor and any optimizer
    moments are never simultaneously resident. Pure SGD (no momentum) -> zero optimizer state.

    Usage differs from a normal optimizer: register hooks once, then DON'T call .backward via
    an outer optimizer.step that needs grads -- the hook does the update. We expose .step() as
    a no-op so the training loop stays uniform.
    """

    def __init__(self, params, lr=1e-3, weight_decay=0.0):
        params = list(params)
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._handles = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    self._handles.append(p.register_post_accumulate_grad_hook(self._make_hook(group)))

    def _make_hook(self, group):
        lr, wd = group["lr"], group["weight_decay"]

        @torch.no_grad()
        def hook(p):
            if p.grad is None:
                return
            if wd:
                p.mul_(1 - lr * wd)
            p.add_(p.grad, alpha=-lr)
            p.grad = None  # free immediately: no grad lingers, no moments exist
        return hook

    @torch.no_grad()
    def step(self, closure=None):
        return closure() if closure is not None else None

    def zero_grad(self, set_to_none: bool = True):
        # grads are already freed in the hook; nothing to clear.
        return
