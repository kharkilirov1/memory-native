"""M9 — Multi-Token Prediction (MTP).

GOAL: more tokens-to-loss. From ONE trunk forward we predict k future tokens via k output heads,
so each forward yields more learning signal. This does NOT reduce per-layer FLOPs — the trunk is
unchanged; we only bolt extra (cheap) output heads onto the final hidden state.

External basis: Meta 2024 "Better & Faster Large Language Models via Multi-token Prediction"
(gains at no training-time overhead; up to ~3x inference via self-speculative decoding).

Design:
  * head j (j = 0 .. n_pred-1) predicts the token at position t+1+j from the final trunk hidden
    state at position t. j=0 is the ORDINARY next-token head.
  * loss = mean over heads of cross-entropy at the correspondingly-shifted targets.
    Given the standard next-token targets `y` (y[i] = idx[i+1]), head j's target at position i is
    idx[i+1+j] = y[i+j]. Positions whose target falls past the end of the sequence are MASKED
    (ignore_index = -100), never wrapped.
  * eval reports the j=0 (primary) head's val-loss, which is exactly the usual next-token loss.

Heads are kept cheap: the primary head is the model's existing tied head (weight == token
embedding); the j>=1 auxiliary heads are small Linears (optionally tied to the same embedding).

Pure PyTorch, CPU/CUDA.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MultiTokenHead", "shift_targets", "mtp_loss"]

IGNORE_INDEX = -100


def shift_targets(targets: torch.Tensor, j: int, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """Targets for head j, given the standard next-token targets `targets` (shape (b, t),
    targets[:, i] == idx[:, i+1]).

    Head j predicts idx[i+1+j] from position i, i.e. targets[:, i+j]. We left-shift `targets`
    by j and pad the j positions that run off the end with `ignore_index` (masked, NOT wrapped).
    j == 0 returns `targets` unchanged.
    """
    if j < 0:
        raise ValueError(f"head index j must be >= 0, got {j}")
    b, t = targets.shape
    if j == 0:
        return targets
    if j >= t:
        # every position's target is past the end -> all masked
        return torch.full_like(targets, ignore_index)
    shifted = targets[:, j:]                                   # (b, t-j) valid targets
    pad = torch.full((b, j), ignore_index, dtype=targets.dtype, device=targets.device)
    return torch.cat([shifted, pad], dim=1)                    # (b, t), last j masked


def mtp_loss(logits_list, targets: torch.Tensor, ignore_index: int = IGNORE_INDEX):
    """Mean over heads of the per-head cross-entropy at shifted, boundary-masked targets.

    logits_list[j] has shape (b, t, vocab). Returns (mean_loss, per_head_losses) where
    per_head_losses[0] is the primary next-token loss.

    With a single head this reduces EXACTLY to F.cross_entropy(logits, targets) (no position is
    masked when n_pred == 1, since head 0 uses targets verbatim).
    """
    per_head = []
    for j, logits in enumerate(logits_list):
        tgt = shift_targets(targets, j, ignore_index)
        loss_j = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=ignore_index,
        )
        per_head.append(loss_j)
    mean_loss = torch.stack(per_head).mean()
    return mean_loss, per_head


class MultiTokenHead(nn.Module):
    """n_pred output heads predicting tokens t+1 .. t+n_pred from one final hidden state.

    head 0 is the primary next-token head; if `primary_head` is supplied (e.g. the GPT's existing
    tied head) it is reused so the n_pred==1 case is byte-for-byte the ordinary model. The j>=1
    heads are cheap Linears; with `tie_embedding` they share `embedding.weight` and add no params.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_pred: int = 1,
        primary_head: nn.Module | None = None,
        embedding: nn.Embedding | None = None,
        tie_embedding: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if n_pred < 1:
            raise ValueError(f"n_pred must be >= 1, got {n_pred}")
        self.n_pred = n_pred
        self.d_model = d_model
        self.vocab_size = vocab_size

        # primary (j=0) head
        if primary_head is not None:
            self.primary = primary_head
        else:
            self.primary = nn.Linear(d_model, vocab_size, bias=bias)
            if tie_embedding and embedding is not None:
                self.primary.weight = embedding.weight

        # auxiliary (j>=1) heads
        self.aux = nn.ModuleList()
        for _ in range(n_pred - 1):
            lin = nn.Linear(d_model, vocab_size, bias=bias)
            if tie_embedding and embedding is not None:
                lin.weight = embedding.weight
            self.aux.append(lin)

    def heads(self):
        return [self.primary, *self.aux]

    def forward(self, h: torch.Tensor):
        """h: (b, t, d_model). Returns a list of n_pred logits tensors, each (b, t, vocab)."""
        return [head(h) for head in self.heads()]

    def loss(self, h: torch.Tensor, targets: torch.Tensor, ignore_index: int = IGNORE_INDEX):
        """Returns (mean_loss, per_head_losses). per_head_losses[0] is the primary loss."""
        return mtp_loss(self.forward(h), targets, ignore_index)


class MTPGPT(nn.Module):
    """Thin wrapper that runs a GPT-like trunk once and applies a MultiTokenHead.

    The wrapped `trunk` must expose a `.trunk_hidden(idx) -> (b, t, d)` method returning the final
    (post-lnf) hidden state, OR a `.forward_features(idx)`. This keeps the trunk forward at full
    FLOPs (no reduction) and adds only the cheap heads.
    """

    def __init__(self, trunk: nn.Module, d_model: int, vocab_size: int, n_pred: int = 1,
                 primary_head: nn.Module | None = None, embedding: nn.Embedding | None = None,
                 tie_embedding: bool = False) -> None:
        super().__init__()
        self.trunk = trunk
        self.mtp = MultiTokenHead(d_model, vocab_size, n_pred, primary_head,
                                  embedding, tie_embedding)

    def _features(self, idx: torch.Tensor) -> torch.Tensor:
        if hasattr(self.trunk, "trunk_hidden"):
            return self.trunk.trunk_hidden(idx)
        if hasattr(self.trunk, "forward_features"):
            return self.trunk.forward_features(idx)
        raise AttributeError("trunk must expose trunk_hidden() or forward_features()")

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        h = self._features(idx)
        logits_list = self.mtp(h)
        loss = None
        per_head = None
        if targets is not None:
            loss, per_head = mtp_loss(logits_list, targets)
        # primary logits first for downstream next-token use
        return logits_list, loss, per_head
