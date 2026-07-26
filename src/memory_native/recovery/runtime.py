"""Reproducible runtime helpers for solver-v3 recovery."""
from __future__ import annotations

from contextlib import contextmanager
import os
import random
from typing import Callable, Iterable

import torch
import torch.nn as nn

from ..convert import CounterLinearWithBias, SwapReport
from ..group_scale_counter import GroupScaleCounterLinear
from ..group_scale_packed import PackedGroupScaleCounterLinear

__all__ = [
    "atomic_torch_save", "build_ptq_counter_kwargs", "capture_rng_state",
    "evaluate_at_alpha", "is_group_mode", "metric_from_ppl", "observe_counter_telemetry",
    "prefix_metrics", "restore_counter_structure", "restore_rng_state",
    "temporary_residual_alpha",
]


def is_group_mode(mode: str) -> bool:
    return mode in {"gptq_group", "group128v3", "group"}


def build_ptq_counter_kwargs(
    mode: str, *, lr: float, lr_scale: float, local_grad_clip: float,
    residual_alpha: float, cache_mode: str, kernel_mode: str,
    strict_update: bool, flip_sample_size: int,
) -> dict:
    common = {
        "lr": float(lr), "lr_scale": float(lr_scale),
        "local_grad_clip": float(local_grad_clip),
    }
    if is_group_mode(mode):
        common.update(
            residual_alpha=float(residual_alpha), kernel_mode=kernel_mode,
            strict_update=bool(strict_update), flip_sample_size=int(flip_sample_size),
        )
    else:
        common["cache_mode"] = cache_mode
    return common


def _alpha_modules(model: nn.Module) -> list[nn.Module]:
    return [m for m in model.modules() if hasattr(m, "set_residual_alpha")]


@contextmanager
def temporary_residual_alpha(model: nn.Module, alpha: float):
    modules = _alpha_modules(model)
    old = [float(getattr(m, "residual_alpha")) for m in modules]
    for module in modules:
        module.set_residual_alpha(alpha)
    try:
        yield
    finally:
        for module, value in zip(modules, old):
            module.set_residual_alpha(value)


def evaluate_at_alpha(model: nn.Module, alpha: float, evaluator: Callable[[], dict]) -> dict:
    was_training = model.training
    model.eval()
    try:
        with temporary_residual_alpha(model, alpha):
            return evaluator()
    finally:
        model.train(was_training)


def prefix_metrics(prefix: str, result: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in result.items()}


def metric_from_ppl(result: dict) -> float:
    ppls = [float(v) for k, v in result.items() if k.startswith("ppl") and float(v) > 0]
    if not ppls:
        raise ValueError("evaluation result contains no positive ppl* metrics")
    return sum(torch.log(torch.tensor(v, dtype=torch.float64)).item() for v in ppls) / len(ppls)


@torch.no_grad()
def observe_counter_telemetry(model_or_layers: nn.Module | Iterable[nn.Module]) -> dict[str, float]:
    layers = list(model_or_layers.modules()) if isinstance(model_or_layers, nn.Module) else list(model_or_layers)
    observations = []
    for layer in layers:
        observe = getattr(layer, "observe_flip_sample", None)
        if observe is not None:
            observations.append(observe())
    total = sum(item["sample_size"] for item in observations)
    if total <= 0:
        return {"flip_rate_alt": 0.0, "counter_edge_sample": 0.0, "flip_sample_size": 0.0}
    return {
        "flip_rate_alt": sum(item["flip_rate_alt"] * item["sample_size"] for item in observations) / total,
        "counter_edge_sample": sum(
            item["counter_edge_sample"] * item["sample_size"] for item in observations
        ) / total,
        "flip_sample_size": float(total),
    }


def _target_linears(model: nn.Module, skip: Iterable[str]):
    from ..donor.ptq import _moe_router_paths

    targets = []
    routers = _moe_router_paths(model)
    for parent_path, parent in model.named_modules():
        for child_name, child in parent.named_children():
            if isinstance(child, nn.Linear):
                path = f"{parent_path}.{child_name}" if parent_path else child_name
                if path not in routers and not any(token in path for token in skip):
                    targets.append((parent, child_name, child, path))
    return targets


def _state_prefix(state_dict: dict[str, torch.Tensor], path: str) -> tuple[str, bool]:
    wrapped, direct = f"{path}.counter.", f"{path}."
    if wrapped + "state" in state_dict:
        return wrapped, True
    if direct + "state" in state_dict:
        return direct, False
    raise KeyError(f"checkpoint has no counter state for {path}")


@torch.no_grad()
def restore_counter_structure(
    model: nn.Module, state_dict: dict[str, torch.Tensor], *, kind: str,
    group: int, C: int, keep_bias: bool = True, extra_skip=None, **counter_kw,
) -> SwapReport:
    """Recreate saved counter modules without collecting Hessians or rerunning PTQ."""
    from ..donor.ptq import (
        _assert_no_unhandled_moe,
        _legacy_expert_targets,
        _parent_and_name,
        _stacked_moe_targets,
    )
    from ..moe_ffn import (
        HFModuleListSwiGLUExperts,
        HFStackedSwiGLUExperts,
        SwiGLUCounterExpert,
    )

    skip = ["lm_head"] + (list(extra_skip) if extra_skip is not None else [])
    report = SwapReport()
    stacked_targets = _stacked_moe_targets(model)
    legacy_targets = _legacy_expert_targets(model)
    _assert_no_unhandled_moe(model, stacked_targets, legacy_targets)
    dense_targets = _target_linears(model, skip)
    dense_paths = {path for _, _, _, path in dense_targets}
    skipped_legacy = sorted(set(legacy_targets) - dense_paths)
    if skipped_legacy:
        raise RuntimeError(
            "refusing partial counter restore: expert weights were excluded by skip "
            f"rules: {skipped_legacy}"
        )
    for parent, child_name, linear, path in dense_targets:
        prefix, was_wrapped = _state_prefix(state_dict, path)
        saved_state = state_dict[prefix + "state"]
        saved_perm = state_dict.get(prefix + "perm")
        if saved_perm is not None:
            packed = saved_state.shape != (linear.out_features, linear.in_features)
            allowed = {
                "lr", "lr_scale", "rms_beta", "rms_eps", "local_grad_clip",
                "residual_alpha", "kernel_mode", "strict_update", "flip_sample_size",
            }
            kw = {key: value for key, value in counter_kw.items() if key in allowed}
            if packed:
                counter: nn.Module = PackedGroupScaleCounterLinear(
                    linear.in_features, linear.out_features, group=group, C=C,
                    perm=saved_perm, **kw,
                )
            else:
                for key in ("kernel_mode", "strict_update", "flip_sample_size"):
                    kw.pop(key, None)
                counter = GroupScaleCounterLinear(
                    linear.in_features, linear.out_features, group=group, C=C,
                    perm=saved_perm, **kw,
                )
        else:
            from ..baselines import make_linear
            group_only = {"residual_alpha", "kernel_mode", "strict_update", "flip_sample_size"}
            kw = {key: value for key, value in counter_kw.items() if key not in group_only}
            counter = make_linear(kind, linear.in_features, linear.out_features, 1.0, C=C, **kw)

        replacement: nn.Module = counter
        if was_wrapped or (linear.bias is not None and keep_bias):
            bias = linear.bias if linear.bias is not None else state_dict[f"{path}.bias"]
            replacement = CounterLinearWithBias(counter, bias)
        setattr(parent, child_name, replacement)
        report.swapped.append(path)
        report.coeffs += linear.in_features * linear.out_features

    def restore_group_counter(path: str, in_features: int, out_features: int):
        prefix, _ = _state_prefix(state_dict, path)
        saved_state = state_dict[prefix + "state"]
        saved_perm = state_dict.get(prefix + "perm")
        if saved_perm is None:
            raise KeyError(f"checkpoint has no group permutation for MoE matrix {path}")
        packed = saved_state.shape != (out_features, in_features)
        allowed = {
            "lr", "lr_scale", "rms_beta", "rms_eps", "local_grad_clip",
            "residual_alpha", "kernel_mode", "strict_update", "flip_sample_size",
        }
        kw = {key: value for key, value in counter_kw.items() if key in allowed}
        if packed:
            return PackedGroupScaleCounterLinear(
                in_features, out_features, group=group, C=C, perm=saved_perm, **kw
            )
        for key in ("kernel_mode", "strict_update", "flip_sample_size"):
            kw.pop(key, None)
        return GroupScaleCounterLinear(
            in_features, out_features, group=group, C=C, perm=saved_perm, **kw
        )

    for target in stacked_targets:
        stacked_key = f"{target.path}.stacked.sg"
        group_key = f"{target.path}.experts.0.gate.state"
        group_wrapped_key = f"{target.path}.experts.0.gate.counter.state"
        if stacked_key in state_dict:
            allowed = {"lr", "lr_scale", "rms_beta", "rms_eps", "compute_dtype"}
            kw = {key: value for key, value in counter_kw.items() if key in allowed}
            replacement: nn.Module = HFStackedSwiGLUExperts(
                target.num_experts, target.hidden_dim, target.intermediate_dim,
                C=C, **kw,
            )
        elif group_key in state_dict or group_wrapped_key in state_dict:
            experts = []
            for expert in range(target.num_experts):
                base = f"{target.path}.experts.{expert}"
                gate = restore_group_counter(
                    f"{base}.gate", target.hidden_dim, target.intermediate_dim
                )
                up = restore_group_counter(
                    f"{base}.up", target.hidden_dim, target.intermediate_dim
                )
                down = restore_group_counter(
                    f"{base}.down", target.intermediate_dim, target.hidden_dim
                )
                experts.append(SwiGLUCounterExpert(gate, up, down))
            replacement = HFModuleListSwiGLUExperts(
                experts, hidden_dim=target.hidden_dim,
                intermediate_dim=target.intermediate_dim,
            )
        else:
            raise KeyError(
                f"checkpoint has no supported counter MoE state for {target.path}"
            )
        parent, name = _parent_and_name(model, target.path)
        setattr(parent, name, replacement)
        for expert in range(target.num_experts):
            report.swapped.extend([
                target.gate_up_path(expert), target.down_path(expert),
            ])
        report.coeffs += target.num_experts * (
            3 * target.hidden_dim * target.intermediate_dim
        )
    return report


def capture_rng_state() -> dict:
    payload = {"torch_rng_state": torch.get_rng_state(), "python_rng_state": random.getstate()}
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np
        payload["numpy_rng_state"] = np.random.get_state()
    except Exception:
        pass
    return payload


def restore_rng_state(payload: dict) -> None:
    if "torch_rng_state" in payload:
        torch.set_rng_state(payload["torch_rng_state"])
    if "python_rng_state" in payload:
        random.setstate(payload["python_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in payload:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
    if "numpy_rng_state" in payload:
        try:
            import numpy as np
            np.random.set_state(payload["numpy_rng_state"])
        except Exception:
            pass


def atomic_torch_save(payload: dict, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
