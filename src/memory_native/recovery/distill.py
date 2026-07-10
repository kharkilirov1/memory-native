"""Recovery finetune by knowledge distillation from the fp donor (teacher).

The warm-started counter student starts degraded (ternarization is lossy for a full-precision
donor). Recovery is a *network-level* effect -- composed layers driven by a task/distill loss --
NOT a per-layer self-improvement (see CLAUDE.md). Here the loss is the KD divergence to the
resident fp teacher (optionally plus the true-token CE). The counter body updates itself in the
backward pass; the remaining fp params (embeddings, norms, tied lm_head, preserved biases) are
trained by AdamW.

The teacher is behind a ``TeacherSource`` seam so a later scale-up can swap the resident fp model
for an offline top-k logit cache without touching the training loop.
"""
from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ResidentTeacher", "TeacherSource", "distill_finetune", "kd_divergence"]


@runtime_checkable
class TeacherSource(Protocol):
    """Anything that can return teacher logits [B, T, V] for a batch of ``input_ids`` [B, T]."""

    def logits(self, input_ids: torch.Tensor) -> torch.Tensor: ...


class ResidentTeacher:
    """Hold the fp donor resident and return its logits (no grad). Fine for small donors where the
    teacher fits beside the student; swap for an offline cache when it does not."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model.eval()

    @torch.no_grad()
    def logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids).logits


def kd_divergence(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """Temperature-scaled KL(teacher || student), averaged per token (the classic KD term x T^2)."""
    T = float(temperature)
    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits / T, dim=-1)
    # sum KL over vocab, mean over the B*T tokens -> stable across batch/seq shapes.
    per_tok = F.kl_div(s, t, reduction="none").sum(dim=-1)
    return per_tok.mean() * (T * T)


def _as_teacher(teacher) -> TeacherSource:
    if isinstance(teacher, TeacherSource):
        return teacher
    if isinstance(teacher, nn.Module):
        return ResidentTeacher(teacher)
    raise TypeError("teacher must be a TeacherSource or an nn.Module")


def distill_finetune(
    student: nn.Module,
    teacher,
    batches: Iterable[torch.Tensor],
    *,
    steps: int,
    kd_alpha: float = 1.0,
    ce_alpha: float = 0.0,
    temperature: float = 2.0,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    grad_clip: float = 0.0,
    device=None,
    log_every: int = 0,
) -> list[float]:
    """Distill ``student`` toward ``teacher`` on ``batches`` for ``steps`` optimizer steps.

    Loss = ``kd_alpha`` * KD(student, teacher, T) + ``ce_alpha`` * CE(student, next-token). The
    counter body self-updates inside ``loss.backward()``; the fp params go through AdamW. Batches
    are cycled if ``steps`` exceeds their count. Returns the per-step loss history.

    ``grad_clip`` (>0) clips the fp-param grad-norm before AdamW steps -- important when recovering
    from a heavily degraded warm-start, where the first-step grads are huge and the run diverges
    without it. The counter body has its own per-layer ``local_grad_clip`` (set at swap time).
    """
    teacher = _as_teacher(teacher)
    batches = list(batches)
    if not batches:
        raise ValueError("distill_finetune needs at least one batch")

    student.train()
    fp_params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(fp_params, lr=lr, weight_decay=weight_decay) if fp_params else None

    history: list[float] = []
    for step in range(steps):
        ids = batches[step % len(batches)]
        if device is not None:
            ids = ids.to(device)

        teacher_logits = teacher.logits(ids)
        out = student(ids, labels=ids if ce_alpha else None)
        loss = kd_alpha * kd_divergence(out.logits, teacher_logits, temperature)
        if ce_alpha:
            loss = loss + ce_alpha * out.loss

        if opt is not None:
            opt.zero_grad(set_to_none=True)
        loss.backward()          # counter body self-updates here; fp grads accumulate for AdamW
        if opt is not None:
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(fp_params, grad_clip)
            opt.step()

        history.append(loss.item())
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"[distill] step {step:4d}  loss {history[-1]:.4f}")

    return history
