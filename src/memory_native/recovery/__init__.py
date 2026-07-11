"""Recovery finetune (Phase 4): distill a counter-warm-started student back toward its fp donor."""
from .distill import (
    ResidentTeacher,
    TeacherSource,
    TopKLogitCache,
    distill_finetune,
    kd_divergence,
)

__all__ = [
    "ResidentTeacher", "TeacherSource", "TopKLogitCache", "distill_finetune", "kd_divergence",
]
