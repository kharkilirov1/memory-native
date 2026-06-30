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

import torch
import torch.nn as nn
import torch.nn.functional as F

from .counter import RMSCounterLinear
from .packed import PackedRMSCounterLinear

__all__ = ["CounterMoEFFN"]


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
                 packed_experts: bool = True) -> None:
        super().__init__()
        self.d = int(dim)
        self.E = int(n_experts)
        self.k = int(top_k)
        if self.k > self.E:
            raise ValueError(f"top_k ({self.k}) cannot exceed n_experts ({self.E})")
        # h = 4d / top_k -> top_k experts cost ~ the dense FFN's 2*d*4d active MACs/token.
        self.h = int(expert_hidden) if expert_hidden is not None else max(1, (4 * self.d) // self.k)
        self.aux_loss_weight = float(aux_loss_weight)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sh = x.shape
        h = x.reshape(-1, self.d)                                          # [N, d]
        N = h.shape[0]

        logits = self.router(h)                                           # [N, E]
        probs = torch.softmax(logits, dim=-1)                             # [N, E]
        top_w, top_idx = probs.topk(self.k, dim=-1)                       # [N, k], [N, k]
        # Renormalize the kept weights so the readout is a convex combination of the chosen experts.
        top_w = top_w / top_w.sum(dim=-1, keepdim=True).clamp_min(1e-9)   # [N, k]

        # Load-balance aux loss + diagnostics (computed every forward; cheap, O(N*E)).
        self.last_aux_loss = self._aux_loss(probs, top_idx)
        with torch.no_grad():
            counts = F.one_hot(top_idx.reshape(-1), num_classes=self.E).sum(dim=0).double()
            self.token_count += counts
            self.last_token_fraction = (counts / max(N * self.k, 1)).float()

        # Flatten (token, slot) pairs, then gather each expert's routed tokens into ONE batch so the
        # expert's two counter layers each see exactly -- and only -- their tokens (exact update,
        # and at most one forward per counter layer per backward).
        flat_tok = torch.arange(N, device=h.device).repeat_interleave(self.k)  # [N*k] token id
        flat_exp = top_idx.reshape(-1)                                          # [N*k] expert id
        flat_w = top_w.reshape(-1)                                              # [N*k] weight

        y = torch.zeros_like(h)                                                 # [N, d]
        for e, expert in enumerate(self.experts):
            sel = (flat_exp == e).nonzero(as_tuple=True)[0]                     # slots routed to e
            if sel.numel() == 0:
                continue
            tok_ids = flat_tok[sel]
            out_e = expert(h[tok_ids])                                          # [n_e, d]
            y.index_add_(0, tok_ids, flat_w[sel].unsqueeze(-1) * out_e)
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
