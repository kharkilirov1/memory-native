"""Qwen2.5 (dense) donor: load + in-place counter warm-start.

A counter layer is a drop-in replacement for ``nn.Linear`` (same ``x -> y``), so a dense donor
needs almost no architecture adapter: we replace the transformer-body projections
(``q/k/v/o_proj``, ``gate/up/down_proj``) *inside* the loaded HF ``Qwen2ForCausalLM`` and leave
the donor's own embeddings, norms, RoPE, attention and LM head intact and fp. All architecture
numbers come from the donor's own config -- nothing here is hardcoded.
"""
from __future__ import annotations

from ..convert import SwapReport, swap_linears_to_counter
from ..counter import C_DEFAULT

__all__ = ["DEFAULT_QWEN", "load_qwen_donor", "qwen_to_counter"]

DEFAULT_QWEN = "Qwen/Qwen2.5-0.5B"


def qwen_to_counter(
    model,
    *,
    kind: str = "counter_rms",
    C: int = C_DEFAULT,
    threshold_ratio: float = 0.7,
    keep_bias: bool = True,
    extra_skip=None,
    **counter_kw,
) -> SwapReport:
    """Warm-start ``model`` in place: swap every transformer-body ``nn.Linear`` for a counter layer.

    ``lm_head`` is skipped -- in Qwen2.5 it is tied to ``embed_tokens``; it stays fp and is trained
    by AdamW alongside the embeddings and norms (exactly the fp slice the method leaves alone).
    ``keep_bias`` preserves Qwen2's q/k/v projection bias via ``CounterLinearWithBias``. Gradient
    checkpointing is disabled: counter layers are eager-only (one forward per backward), and
    checkpointing re-runs the forward, which would trip the reuse guard.

    ``cache_mode`` defaults to "fp16" here (override via ``counter_kw``): without the derived
    T-cache every forward re-decodes the whole state, which profiled at ~2x the student forward
    cost. The cache is a non-persistent derived view (+2 bytes/weight resident) -- pass
    ``cache_mode="none"`` to trade that memory back.

    ``extra_skip`` adds more substrings/predicate targets to leave fp (e.g. a specific layer).
    Extra ``counter_kw`` flow through to each counter layer's constructor. Returns a ``SwapReport``.
    """
    if getattr(model, "is_gradient_checkpointing", False):
        model.gradient_checkpointing_disable()
    counter_kw.setdefault("cache_mode", "fp16")

    skip = ["lm_head"]
    if extra_skip is not None:
        skip += list(extra_skip)

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = None

    report = swap_linears_to_counter(
        model,
        kind=kind,
        skip=skip,
        C=C,
        threshold_ratio=threshold_ratio,
        keep_bias=keep_bias,
        **counter_kw,
    )
    # counter layers build their state/scale buffers on CPU; move them to the model's device so a
    # GPU donor (swapped after .to("cuda")) does not hit a cpu/cuda mismatch in the first matmul.
    if device is not None:
        model.to(device)
    return report


def load_qwen_donor(
    name: str = DEFAULT_QWEN,
    *,
    dtype=None,
    attn_implementation: str = "sdpa",
    **from_pretrained_kw,
):
    """Load a Qwen2.5 donor (weights + tokenizer) from HuggingFace.

    ``attn_implementation="sdpa"`` is the default: SDPA fuses only the attention *math* --
    each q/k/v/o projection is still called exactly once per forward, so the eager-only counter
    guard holds (verified: a full swap + forward + backward step raises no reuse guard, and it
    profiled ~2.8x faster than "eager" on CPU). The thing that DOES break the guard is gradient
    checkpointing (re-runs forwards), which ``qwen_to_counter`` disables. Real weights download
    on first use; do the download + finetune on the GPU box (see the T4 script/notebook).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, attn_implementation=attn_implementation, **from_pretrained_kw
    )
    return model, tokenizer
