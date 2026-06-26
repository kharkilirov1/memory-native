"""Memory accounting + a peak-memory gate that anyone can run on stock PyTorch.

Two things live here:
  * memory_report(model): static byte accounting of persistent state (params + buffers),
    plus what the counter weights would cost packed to 6 bits.
  * peak_training_memory(step_fn): the *dynamic* peak during a real train step, measured
    with torch.cuda.max_memory_allocated on CUDA. This is the portable analogue of the
    engine's memory truth gate: it lets a user verify the counter optimizer actually uses
    less training memory than a dense AdamW baseline, on their own GPU, with no custom build.

Honesty: the pure-PyTorch counter layer decodes states to dense tensors around the GEMM, so
its *measured* peak is not yet the 0.75 byte/weight figure -- it still beats dense+Adam on
the optimizer-state pool (no FP master, no Adam moments), which is what these helpers show.
The sub-byte training peak needs the Triton/CUDA packed kernel (see README "Roadmap").
"""
from __future__ import annotations

from typing import Callable, Iterable

import torch
import torch.nn as nn

from .counter import CompactCounterLinear

__all__ = ["memory_report", "fmt_bytes", "peak_training_memory", "compare_training_peak"]


def fmt_bytes(n: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024
        i += 1
    return f"{n:.2f} {units[i]}"


def _unique(items: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    seen: set[int] = set()
    out: list[torch.Tensor] = []
    for t in items:
        if t is None:
            continue
        ptr = int(t.untyped_storage().data_ptr()) if t.numel() else id(t)
        if ptr not in seen:
            seen.add(ptr)
            out.append(t)
    return out


def memory_report(model: nn.Module) -> dict[str, float]:
    params = _unique(model.parameters())
    buffers = _unique(model.buffers())
    counter = [m for m in model.modules() if isinstance(m, CompactCounterLinear)]
    counter_weights = sum(int(m.state.numel()) for m in counter)
    fp_params = sum(int(p.numel()) for p in params)
    parameter_bytes = sum(p.numel() * p.element_size() for p in params)
    buffer_bytes = sum(b.numel() * b.element_size() for b in buffers)
    return {
        "counter_weights": float(counter_weights),
        "fp_parameters": float(fp_params),
        "parameter_bytes": float(parameter_bytes),
        "buffer_bytes": float(buffer_bytes),
        "persistent_bytes": float(parameter_bytes + buffer_bytes),
        "counter_packed_6bit_bytes": float(counter_weights * 6 / 8),
        "dense_bf16_adam_rule_of_thumb_bytes": float(16 * (counter_weights + fp_params)),
    }


def peak_training_memory(step_fn: Callable[[], None], device: torch.device) -> int:
    """Run one training step and return the peak allocated bytes during it.

    On CUDA this uses torch.cuda.max_memory_allocated (a true device peak). On CPU torch has
    no allocator peak API, so this returns 0 and callers should fall back to memory_report.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        step_fn()
        torch.cuda.synchronize(device)
        return int(torch.cuda.max_memory_allocated(device))
    step_fn()
    return 0


def compare_training_peak(build_counter: Callable[[], nn.Module],
                          build_dense: Callable[[], "tuple[nn.Module, torch.optim.Optimizer]"],
                          step_inputs: Callable[[], tuple],
                          device: torch.device) -> dict[str, float]:
    """Measure training-step peak memory for a counter model vs a dense+AdamW baseline.

    build_counter() -> counter model (self-updating in backward, no optimizer).
    build_dense()    -> (dense model, its AdamW optimizer).
    step_inputs()    -> (x, y) batch on `device`.
    Returns a dict with both peaks (CUDA) or both persistent-byte reports (CPU fallback).
    """
    import torch.nn.functional as F

    # counter
    cm = build_counter().to(device).train()
    x, y = step_inputs()

    def counter_step():
        logits = cm(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()  # counter layers self-update here; no optimizer.step()

    # dense + AdamW
    dm, opt = build_dense()
    dm = dm.to(device).train()

    def dense_step():
        opt.zero_grad(set_to_none=True)
        logits = dm(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        opt.step()

    if device.type == "cuda":
        counter_peak = peak_training_memory(counter_step, device)
        dense_peak = peak_training_memory(dense_step, device)
        return {
            "counter_peak_bytes": float(counter_peak),
            "dense_adam_peak_bytes": float(dense_peak),
            "ratio_dense_over_counter": float(dense_peak / max(counter_peak, 1)),
        }
    # CPU: no allocator peak; report persistent state instead.
    counter_step(); dense_step()
    cr, dr = memory_report(cm), memory_report(dm)
    return {
        "counter_persistent_bytes": cr["persistent_bytes"],
        "dense_persistent_bytes": dr["persistent_bytes"],
        "note": float("nan"),  # CPU: see memory_report; peak needs CUDA
    }
