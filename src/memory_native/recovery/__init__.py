"""Recovery finetune (Phase 4): distill a counter-warm-started student back toward its fp donor."""
from .distill import ResidentTeacher, TeacherSource, distill_finetune, kd_divergence

__all__ = ["ResidentTeacher", "TeacherSource", "distill_finetune", "kd_divergence"]
