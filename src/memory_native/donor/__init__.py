"""Pretrained-donor loaders + in-place counter warm-start (Phase 2-remaining).

Donor-specific glue that turns an open-weights model into the counter format by swapping its
transformer-body linears. Kept out of the core package so `transformers` stays an optional
`[donor]` extra; import from here only when you actually have a donor to convert.
"""
from .qwen import DEFAULT_QWEN, load_qwen_donor, qwen_to_counter

__all__ = ["DEFAULT_QWEN", "load_qwen_donor", "qwen_to_counter"]
