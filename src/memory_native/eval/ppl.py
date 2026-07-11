"""Perplexity witness for donor recovery.

Counter layers are eager-only, but under ``torch.no_grad()`` their forward is a pure inference
matmul (no update, no graph), so PPL measurement is safe and side-effect-free."""
from __future__ import annotations

import math

import torch

__all__ = ["perplexity"]


@torch.no_grad()
def perplexity(model, batches, *, device=None) -> float:
    """Token-averaged perplexity of a causal LM over an iterable of ``input_ids`` batches [B, T].

    Uses the model's own shifted cross-entropy (HF ``labels=input_ids``), weighting each batch by
    its shifted-token count so ragged batch sizes/lengths average correctly. Restores the model's
    train/eval mode on exit."""
    was_training = model.training
    model.eval()
    total_nll = None                                      # on-device accumulator: one host
    total_tok = 0                                         # sync at the end, not per batch
    try:
        for ids in batches:
            if device is not None:
                ids = ids.to(device)
            out = model(ids, labels=ids)
            n_shift = ids.numel() - ids.shape[0]          # (T-1) * B tokens actually scored
            if n_shift <= 0:
                continue
            nll = out.loss.detach() * n_shift
            total_nll = nll if total_nll is None else total_nll + nll
            total_tok += n_shift
    finally:
        model.train(was_training)
    if total_tok == 0:
        return float("nan")
    return math.exp(float(total_nll) / total_tok)
