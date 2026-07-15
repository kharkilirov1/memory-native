"""Recovery finetune and reproducible solver-v3 runtime helpers."""
from .distill import (
    ResidentTeacher,
    TeacherSource,
    TopKLogitCache,
    distill_finetune,
    kd_divergence,
)
from .runtime import (
    atomic_torch_save,
    build_ptq_counter_kwargs,
    capture_rng_state,
    evaluate_at_alpha,
    is_group_mode,
    metric_from_ppl,
    observe_counter_telemetry,
    prefix_metrics,
    restore_counter_structure,
    restore_rng_state,
    temporary_residual_alpha,
)

__all__ = [
    "ResidentTeacher", "TeacherSource", "TopKLogitCache", "distill_finetune", "kd_divergence",
    "atomic_torch_save", "build_ptq_counter_kwargs", "capture_rng_state", "evaluate_at_alpha",
    "is_group_mode", "metric_from_ppl", "observe_counter_telemetry", "prefix_metrics",
    "restore_counter_structure", "restore_rng_state", "temporary_residual_alpha",
]
