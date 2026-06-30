"""CounterMoEFFN -- the architecture lever (verification-plan M4): a counter-state Mixture-of-Experts.

Replace a dense FFN (two d x 4d GEMMs touched by every token) with a *sparse* MoE: a small fp
router picks top_k of E experts per token; each expert is a counter-MLP (RMSCounterLinear d->h,
gelu, RMSCounterLinear h->d). Only the router is an fp Parameter (AdamW owns it); the experts are
counter-state (0.75 byte/weight visible) and self-update in their own backward.

Why this fits the counter method exactly (no approximation, no bias):
    A token NOT routed to an expert contributes nothing to that expert's output, so its gradient
    w.r.t. that expert is EXACTLY zero. We gather each expert's routed tokens into one contiguous
    batch and run that expert's two counter layers on exactly those tokens, so each counter layer
    sees only -- and all of -- its tokens. The fused counter update over that batch is therefore the
    exact per-expert gradient, not a sparse approximation. (Gather-per-expert also satisfies the
    RMSCounterLinear "one forward per backward" contract: every expert layer is called at most once.)

EQUAL ACTIVE COMPUTE (the gate's pivot):
    A dense FFN does 2*d*(4d) active MACs/token. We size each expert to hidden h so that the top_k
    experts a token actually visits cost ~ the dense active MACs: top_k * (2*d*h) ~ 2*d*4d, i.e.
    h ~ 4d / top_k. E (the expert count) grows TOTAL capacity / persistent bytes without touching the
    per-token active compute (a token still only visits top_k experts).

Load balancing:
    Top-k routing collapses (a few experts hog every token) unless penalized. We add the standard
    switch-transformer auxiliary loss  aux = E * sum_e (frac_tokens_e * mean_router_prob_e), exposed
    via aux_loss_weight; the last forward's value is stashed on .last_aux_loss so the training loop
    can add aux_loss_weight * ffn.last_aux_loss to the task loss.

Pure PyTorch, CPU/CUDA.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .counter import RMSCounterLinear
from .packed import PackedRMSCounterLinear

__all__ = ["CounterMoEFFN"]


def _gelu_grad(x: torch.Tensor) -> torch.Tensor:
    """d/dx of the exact (erf) GELU = Phi(x) + x*phi(x)."""
    cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    pdf = torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    return cdf + x * pdf


class _GroupedExperts(torch.autograd.Function):
    """Run ALL experts' two counter MLPs as grouped GEMMs (one launch each via torch._grouped_mm)
    instead of a python loop over E experts, then self-update each expert's counter state from its
    grouped weight-gradient. Tokens arrive already SORTED by expert; `offs` are the cumulative
    per-expert row ends. Routing, the convex-combination weighting, and the scatter stay in ordinary
    autograd OUTSIDE this Function (so the router gets its gradient for free); the Function owns only
    x->grad_x and the counter update -- the counter-specific part autograd can't express. `tap`
    forces backward to run even when x needs no gradient (so the experts always self-update)."""

    @staticmethod
    def forward(ctx, x_sorted, offs, tap, moe):
        E, d, h = moe.E, moe.d, moe.h
        W1 = torch.empty(E, h, d, device=x_sorted.device, dtype=torch.float32)   # scale1*T1 [out,in]
        W2 = torch.empty(E, d, h, device=x_sorted.device, dtype=torch.float32)
        for e, ex in enumerate(moe.experts):
            t1, _ = ex.fc1._decode_rows(0, h); W1[e] = ex.fc1.scale * t1.float()
            t2, _ = ex.fc2._decode_rows(0, d); W2[e] = ex.fc2.scale * t2.float()
        xs = x_sorted.float()
        y1 = torch._grouped_mm(xs, W1.transpose(1, 2).contiguous(), offs=offs)   # [M,h]
        a = F.gelu(y1)
        y2 = torch._grouped_mm(a, W2.transpose(1, 2).contiguous(), offs=offs)    # [M,d]
        ctx.moe = moe; ctx.offs = offs
        ctx.save_for_backward(xs, y1, a, W1, W2)
        return y2.to(x_sorted.dtype)

    @staticmethod
    def backward(ctx, grad_y2):
        xs, y1, a, W1, W2 = ctx.saved_tensors
        offs = ctx.offs; moe = ctx.moe
        gy2 = grad_y2.float()
        grad_a = torch._grouped_mm(gy2, W2.contiguous(), offs=offs)              # [M,h]
        grad_y1 = grad_a * _gelu_grad(y1)
        grad_x = torch._grouped_mm(grad_y1, W1.contiguous(), offs=offs)          # [M,d]
        # per-expert counter update from the grouped weight-gradients (the cheap ~9% part).
        bounds = [0] + offs.tolist()
        for e, ex in enumerate(moe.experts):
            s, t = bounds[e], bounds[e + 1]
            if t <= s:
                continue
            ex.fc1.apply_update_from_grad_w(grad_y1[s:t].t() @ xs[s:t])          # [h,d]
            ex.fc2.apply_update_from_grad_w(gy2[s:t].t() @ a[s:t])               # [d,h]
        return grad_x.to(grad_y2.dtype), None, None, None


def _expert_linear(fin: int, fout: int, *, C: int, lr: float, lr_scale: float, packed: bool):
    """Pick the expert's counter linear. packed=True -> PackedRMSCounterLinear, which on CUDA fires
    the ONE-launch fused Triton update (vs ~15 torch ops) and stores 0.75 B/weight; it needs
    in_features % 4 == 0, so fall back to RMSCounterLinear when the width isn't divisible by 4. The
    learning dynamics are identical on CPU (packed only changes storage + which update path runs)."""
    if packed and fin % 4 == 0:
        return PackedRMSCounterLinear(fin, fout, C=C, lr=lr, lr_scale=lr_scale)
    return RMSCounterLinear(fin, fout, C=C, lr=lr, lr_scale=lr_scale)


class _CounterExpert(nn.Module):
    """A single counter-MLP expert: counter-linear(d->h) -> gelu -> counter-linear(h->d). Holds no
    fp Parameters; both linears are counter-state and self-update in backward. With packed=True the
    two linears are PackedRMSCounterLinear so the per-expert update runs as one fused kernel."""

    def __init__(self, dim: int, hidden: int, *, C: int, lr: float, lr_scale: float,
                 packed: bool = True) -> None:
        super().__init__()
        self.fc1 = _expert_linear(dim, hidden, C=C, lr=lr, lr_scale=lr_scale, packed=packed)
        self.fc2 = _expert_linear(hidden, dim, C=C, lr=lr, lr_scale=lr_scale, packed=packed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class CounterMoEFFN(nn.Module):
    """Sparse Mixture-of-counter-Experts FFN drop-in.

    dim          : model width d.
    n_experts    : E -- total expert count (grows capacity / persistent bytes, not active compute).
    top_k        : experts visited per token (active compute scales with top_k, not E).
    expert_hidden: per-expert hidden width h. Default 4*dim // top_k -> top_k experts ~ dense active
                   MACs (equal-active-compute gate). Pass an explicit value to deviate.
    aux_loss_weight: caller-side weight for the load-balance aux loss (see module docstring). The
                   FFN itself only COMPUTES the aux loss (.last_aux_loss); the training loop scales
                   and adds it. Stored here so the arm carries its own recommended weight.
    """

    def __init__(self, dim: int, n_experts: int = 8, top_k: int = 2,
                 expert_hidden: int | None = None, *, C: int = 11, lr: float = 0.04,
                 lr_scale: float = 2e-4, aux_loss_weight: float = 1e-2,
                 packed_experts: bool = True, grouped: bool = False) -> None:
        super().__init__()
        self.d = int(dim)
        self.E = int(n_experts)
        self.k = int(top_k)
        if self.k > self.E:
            raise ValueError(f"top_k ({self.k}) cannot exceed n_experts ({self.E})")
        # h = 4d / top_k -> top_k experts cost ~ the dense FFN's 2*d*4d active MACs/token.
        self.h = int(expert_hidden) if expert_hidden is not None else max(1, (4 * self.d) // self.k)
        self.aux_loss_weight = float(aux_loss_weight)
        # grouped=True replaces the python per-expert loop with grouped GEMMs (torch._grouped_mm):
        # all experts' fc1/fc2 run in one launch each, the per-expert update reads its grouped
        # weight-gradient. Same math as the loop (fp32, up to reduction order). Needs torch._grouped_mm.
        self.grouped = bool(grouped) and hasattr(torch, "_grouped_mm")

        # The ONLY fp Parameter: the router (a tiny d->E linear). AdamW owns it.
        self.router = nn.Linear(self.d, self.E, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)

        # packed_experts=True (default): experts are PackedRMSCounterLinear -> fused Triton update
        # on CUDA + 0.75 B/weight. Identical dynamics on CPU.
        self.experts = nn.ModuleList(
            _CounterExpert(self.d, self.h, C=C, lr=lr, lr_scale=lr_scale, packed=packed_experts)
            for _ in range(self.E)
        )

        # Diagnostics (not optimizer state): O(E) scalars.
        self.register_buffer("token_count", torch.zeros(self.E, dtype=torch.float64),
                             persistent=False)
        # The last forward's load-balance aux loss, for the training loop to add (scaled).
        self.last_aux_loss: torch.Tensor = torch.zeros(())
        # The last forward's per-expert token fraction, for the routing-collapse check.
        self.last_token_fraction: torch.Tensor = torch.zeros(self.E)

    def _aux_loss(self, probs: torch.Tensor, top_idx: torch.Tensor) -> torch.Tensor:
        """Switch-transformer load-balance loss: E * sum_e f_e * P_e, where
        f_e = fraction of tokens routed to e (hard, from the top-k assignment) and
        P_e = mean router probability mass on e (soft). Minimized when both are uniform (1/E)."""
        N = probs.shape[0]
        # P_e: soft mean probability per expert.
        mean_prob = probs.mean(dim=0)                                      # [E]
        # f_e: hard fraction of (token, slot) assignments that picked e.
        one_hot = F.one_hot(top_idx.reshape(-1), num_classes=self.E).to(probs.dtype)  # [N*k, E]
        frac = one_hot.sum(dim=0) / max(N * self.k, 1)                     # [E]
        return self.E * (frac * mean_prob).sum()

    def _route(self, h: torch.Tensor):
        """Shared routing: router -> softmax -> top_k -> renormalized weights, aux loss, diagnostics.
        Returns (flat_tok, flat_exp, flat_w) over the N*k (token, slot) pairs."""
        N = h.shape[0]
        logits = self.router(h)                                           # [N, E]
        probs = torch.softmax(logits, dim=-1)                             # [N, E]
        top_w, top_idx = probs.topk(self.k, dim=-1)                       # [N, k], [N, k]
        top_w = top_w / top_w.sum(dim=-1, keepdim=True).clamp_min(1e-9)   # convex combination
        self.last_aux_loss = self._aux_loss(probs, top_idx)
        with torch.no_grad():
            counts = F.one_hot(top_idx.reshape(-1), num_classes=self.E).sum(dim=0).double()
            self.token_count += counts
            self.last_token_fraction = (counts / max(N * self.k, 1)).float()
        flat_tok = torch.arange(N, device=h.device).repeat_interleave(self.k)  # [N*k] token id
        return flat_tok, top_idx.reshape(-1), top_w.reshape(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.grouped:
            return self._forward_grouped(x)
        sh = x.shape
        h = x.reshape(-1, self.d)                                          # [N, d]
        flat_tok, flat_exp, flat_w = self._route(h)

        # Gather each expert's routed tokens into ONE batch so its two counter layers each see
        # exactly -- and only -- their tokens (exact update, one forward per counter layer per bwd).
        y = torch.zeros_like(h)                                                 # [N, d]
        for e, expert in enumerate(self.experts):
            sel = (flat_exp == e).nonzero(as_tuple=True)[0]                     # slots routed to e
            if sel.numel() == 0:
                continue
            tok_ids = flat_tok[sel]
            out_e = expert(h[tok_ids])                                          # [n_e, d]
            y.index_add_(0, tok_ids, flat_w[sel].unsqueeze(-1) * out_e)
        return y.reshape(sh)

    def _forward_grouped(self, x: torch.Tensor) -> torch.Tensor:
        """Loop-free path: sort the (token, slot) pairs by expert, run all experts as grouped GEMMs
        (one launch each), then the weighted scatter. Routing + weighting + scatter are ordinary
        autograd (router gets its grad); _GroupedExperts owns the grouped matmuls + counter update."""
        sh = x.shape
        h = x.reshape(-1, self.d)                                          # [N, d]
        flat_tok, flat_exp, flat_w = self._route(h)
        order = torch.argsort(flat_exp)                                    # sort pairs by expert
        sorted_tok = flat_tok[order]
        sorted_w = flat_w[order]
        offs = torch.bincount(flat_exp, minlength=self.E).cumsum(0).to(torch.int32)  # per-expert ends
        x_sorted = h[sorted_tok]                                          # [M, d] (autograd-tracked)
        tap = (torch.zeros((), device=h.device, dtype=h.dtype, requires_grad=True)
               if torch.is_grad_enabled() else h.new_zeros(()))           # forces experts to update
        y2 = _GroupedExperts.apply(x_sorted, offs, tap, self)             # [M, d] unweighted outputs
        weighted = y2 * sorted_w.unsqueeze(-1)                            # router grad flows here
        y = torch.zeros_like(h).index_add(0, sorted_tok, weighted)
        return y.reshape(sh)

    # --- accounting (for the equal-active-compute comparison) ----------------------
    def active_macs_per_token(self) -> int:
        """MACs a single token touches: router (d*E) + its top_k experts (each 2*d*h)."""
        return self.d * self.E + self.k * (2 * self.d * self.h)

    def persistent_bytes(self) -> int:
        """Counter experts (the bulk: ~0.75 B/weight visible + per-row scale/v) + fp router."""
        b = 0
        for expert in self.experts:
            for m in (expert.fc1, expert.fc2):
                b += m.state.numel() + m.scale.numel() * 4 + m.v.numel() * 4
        b += self.router.weight.numel() * 4                                # fp router
        return b

    @torch.no_grad()
    def routing_report(self) -> dict[str, float]:
        """Cumulative per-expert token fractions + collapse flags (for the witness)."""
        total = float(self.token_count.sum().clamp_min(1))
        frac = (self.token_count / total).tolist()
        return {
            "fractions": frac,
            "min_frac": min(frac),
            "max_frac": max(frac),
            "starved": any(f < 0.01 for f in frac),    # an expert getting < 1% of tokens
            "dominant": any(f > 0.90 for f in frac),   # one expert hogging > 90% of tokens
        }
