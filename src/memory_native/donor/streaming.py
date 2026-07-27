"""Block-sequential streaming conversion: convert donors far larger than memory.

``ptq_warm_start`` needs the whole donor resident -- it collects every Hessian,
then solves. That caps conversion at what fits in RAM/VRAM. This driver instead
walks the transformer one block at a time:

    embeddings -> activation buffer
    for each block i:
        materialize block i's weights from the safetensors shards (lazily)
        run the buffer through it, accumulating H for the block's targets
        solve + swap the block to counter layers
        re-run the CONVERTED block to produce the buffer for block i+1
        write block i's counter state to disk, free the block

Only one block plus the activation buffer is ever resident, so peak memory is
set by the block size, not the model size.

The re-run in step 4 is deliberate: block i+1 calibrates against the outputs of
the ALREADY-CONVERTED block i, so quantization error compounds into the
statistics rather than being hidden -- the same error-cascade signal the asym
calibration exploits, obtained here for free. Pass ``cascade=False`` for classic
fp-tower semantics (each block calibrated on unquantized inputs).

Output is incremental: one file per block plus a manifest, so an interrupted run
resumes at the first unfinished block instead of starting over.

MoE donors are NOT supported here yet and are rejected loudly: transformers
stores Mixtral-style experts on disk in the legacy per-expert layout and
converts them to stacked tensors at load time without exposing the mapping, so
resolving expert weights by name from the shards needs per-family conversion
code. The in-memory path (which lets transformers do the loading) handles MoE.
"""
from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass, field

import torch
from torch import nn

from ..convert import CounterLinearWithBias
from .ptq import (
    _assert_no_unhandled_moe,
    _group_counter_from_state,
    _moe_router_paths,
    _parent_and_name,
    _target_paths,
    solve_group_state,
)

__all__ = ["StreamingReport", "convert_streaming", "load_streamed_state"]

MANIFEST = "manifest.json"


@dataclass
class StreamingReport:
    """What a streaming conversion actually did (and cost)."""

    blocks_total: int = 0
    blocks_converted: int = 0
    blocks_resumed: int = 0
    targets: list[str] = field(default_factory=list)
    coeffs: int = 0
    peak_bytes: int = 0
    out_dir: str = ""

    @property
    def peak_gib(self) -> float:
        return self.peak_bytes / 2**30


def _tensor_bytes(*tensors) -> int:
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


class _WeightSource:
    """Lazy per-tensor access to a HF checkpoint (single-file or sharded)."""

    def __init__(self, path: str):
        from safetensors import safe_open

        self._safe_open = safe_open
        self.path = path
        index = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index, encoding="utf-8") as handle:
                weight_map = json.load(handle)["weight_map"]
            self._file_of = {k: os.path.join(path, v) for k, v in weight_map.items()}
        else:
            single = os.path.join(path, "model.safetensors")
            if not os.path.exists(single):
                raise FileNotFoundError(f"no safetensors checkpoint under {path}")
            with safe_open(single, framework="pt") as handle:
                self._file_of = {k: single for k in handle.keys()}
        self._handles: dict[str, object] = {}

    def keys(self):
        return self._file_of.keys()

    def get(self, name: str, dtype=None) -> torch.Tensor:
        file = self._file_of[name]
        handle = self._handles.get(file)
        if handle is None:
            handle = self._safe_open(file, framework="pt")
            self._handles[file] = handle
        t = handle.get_tensor(name)
        return t.to(dtype) if dtype is not None else t

    def close(self):
        self._handles.clear()


def _materialize(module: nn.Module, prefix: str, src: _WeightSource, device, dtype) -> None:
    """Give a meta-device module real weights, one tensor at a time.

    ``to_empty`` allocates UNINITIALIZED memory, so every tensor it creates has to
    be written before use. Anything the checkpoint does not carry is an error
    here, not a silent pass: a computed buffer left as garbage (rotary
    ``inv_freq`` is the classic one) still produces finite numbers and corrupts
    the calibration invisibly. Modules that own computed buffers are rebuilt from
    config instead -- see ``_build_rotary``.
    """
    module.to_empty(device=device)
    missing: list[str] = []
    with torch.no_grad():
        for name, param in list(module.named_parameters(recurse=True)):
            key = f"{prefix}.{name}"
            if key not in src.keys():
                missing.append(key)
                continue
            param.copy_(src.get(key, dtype).to(device))
        for name, buf in list(module.named_buffers(recurse=True)):
            key = f"{prefix}.{name}"
            if key in src.keys():
                buf.copy_(src.get(key, dtype).to(device))
            else:
                missing.append(key)
    if missing:
        # Distinguish the two ways a tensor can be absent, because the fix differs.
        expert_miss = [k for k in missing if "expert" in k or "mlp.gate." in k]
        if expert_miss and any("experts." in k for k in src.keys()):
            raise NotImplementedError(
                "this checkpoint stores MoE experts in the legacy per-expert layout "
                "(block_sparse_moe.experts.N.w1/w2/w3) while the module expects stacked "
                f"tensors ({expert_miss[:2]}); transformers converts between the two at "
                "load time and does not expose the mapping, so streaming cannot resolve "
                "expert weights by name yet. Use the in-memory path for MoE donors until "
                "the per-family checkpoint mapping lands."
            )
        raise RuntimeError(
            f"checkpoint has no tensor for {missing[:4]}"
            f"{' ...' if len(missing) > 4 else ''}; refusing to run on "
            "uninitialized memory (rebuild such modules from config instead)"
        )


def _build_rotary(config, device):
    """Rotary embeddings own COMPUTED buffers (inv_freq) that no checkpoint
    stores, so they are constructed from config on a real device rather than
    materialized from the shards."""
    from transformers import AutoModelForCausalLM

    with torch.device(device):
        probe = AutoModelForCausalLM.from_config(config)
    inner = probe.model if hasattr(probe, "model") else probe
    rotary = inner.rotary_emb
    del probe
    gc.collect()
    return rotary


def _block_targets(block: nn.Module, skip) -> list[str]:
    """Target paths inside one block, expressed relative to the block."""
    routers = _moe_router_paths(block)
    skip = list(skip) + list(routers)
    return _target_paths(block, skip)


@torch.no_grad()
def convert_streaming(
    model_path: str,
    calib_batches,
    out_dir: str,
    *,
    kind: str = "counter_packed",
    C: int = 11,
    group: int = 128,
    dtype=torch.float32,
    device: str | torch.device = "cpu",
    micro_batch: int = 4,
    cascade: bool = True,
    extra_skip=None,
    resume: bool = True,
    progress: bool = True,
    **solve_kw,
) -> StreamingReport:
    """Convert ``model_path`` block by block, writing counter state into ``out_dir``.

    ``calib_batches`` is an iterable of token-id tensors [B, T] (the same input the
    in-memory path takes). ``micro_batch`` bounds how many sequences are pushed
    through a block at once, which is the knob that bounds peak memory together
    with the per-block Hessians.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    device = torch.device(device)
    os.makedirs(out_dir, exist_ok=True)
    report = StreamingReport(out_dir=out_dir)

    manifest_path = os.path.join(out_dir, MANIFEST)
    done: set[int] = set()
    if resume and os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            done = set(json.load(handle).get("blocks_done", []))
        report.blocks_resumed = len(done)

    src = _WeightSource(model_path)
    config = AutoConfig.from_pretrained(model_path)
    with torch.device("meta"):
        skeleton = AutoModelForCausalLM.from_config(config)
    inner = skeleton.model if hasattr(skeleton, "model") else skeleton
    blocks = inner.layers
    report.blocks_total = len(blocks)

    # --- embeddings once: the activation buffer lives on CPU, blocks pull micro-batches ---
    _materialize(inner.embed_tokens, "model.embed_tokens", src, device, dtype)
    buffer: list[torch.Tensor] = []
    id_batches = [b for b in calib_batches]
    for ids in id_batches:
        buffer.append(inner.embed_tokens(ids.to(device)).to("cpu"))
    inner.embed_tokens.to("meta")

    rotary = _build_rotary(config, device)

    peak = _tensor_bytes(*buffer)
    counter_kw = {k: solve_kw.pop(k) for k in list(solve_kw) if k in {
        "lr", "lr_scale", "rms_beta", "rms_eps", "local_grad_clip", "residual_alpha",
        "kernel_mode", "strict_update", "flip_sample_size",
    }}

    for index, block in enumerate(blocks):
        if index in done:
            # A finished block still has to RUN: block i+1 calibrates on its output,
            # so skipping it outright would feed the next block unconverted
            # activations and silently change the result of a resumed run.
            _materialize(block, f"model.layers.{index}", src, device, dtype)
            _reload_block_counters(block, index, out_dir, kind=kind, group=group, C=C,
                                   counter_kw=counter_kw)
            buffer = _run_block(block, buffer, rotary, device, micro_batch, collect=True)
            block.to("meta")
            gc.collect()
            if progress:
                print(f"[stream] block {index}: already converted, replayed for cascade",
                      flush=True)
            continue

        _materialize(block, f"model.layers.{index}", src, device, dtype)
        # keep the fail-loudly property of the in-memory path: a MoE layout this
        # driver cannot convert must raise, never silently convert attention only.
        _assert_no_unhandled_moe(block, [], {})
        targets = _block_targets(block, extra_skip or [])

        # --- pass 1: Hessians for this block's targets ---
        hessians: dict[str, torch.Tensor] = {}
        hooks = []

        def make_hook(path, in_features):
            def hook(_mod, inputs):
                x = inputs[0].detach().reshape(-1, in_features).to(torch.float32)
                h = hessians.get(path)
                if h is None:
                    h = torch.zeros(in_features, in_features, dtype=torch.float32,
                                    device=x.device)
                    hessians[path] = h
                h.addmm_(x.t(), x)
            return hook

        for path in targets:
            lin = block.get_submodule(path)
            hooks.append(lin.register_forward_pre_hook(make_hook(path, lin.in_features)))
        fp_out = _run_block(block, buffer, rotary, device, micro_batch,
                            collect=not cascade)
        for hook in hooks:
            hook.remove()

        peak = max(peak, _tensor_bytes(*buffer, *hessians.values(),
                                       *[p for p in block.parameters()]))

        # --- solve + swap, one target at a time ---
        state_out: dict[str, torch.Tensor] = {}
        for path in targets:
            lin = block.get_submodule(path)
            H = hessians.get(path)
            if H is None:                      # never called (dead branch): data-free solve
                H = torch.eye(lin.in_features, dtype=torch.float32, device=lin.weight.device)
            state, _ = solve_group_state(lin.weight.data.float(), H, group=group, C=C,
                                         **solve_kw)
            counter = _group_counter_from_state(
                state, in_features=lin.in_features, out_features=lin.out_features,
                group=group, C=C, kind=kind, counter_kw=counter_kw,
            )
            parent, child = _parent_and_name(block, path)
            bias = getattr(lin, "bias", None)
            if bias is not None:
                counter = CounterLinearWithBias(counter, bias.detach().clone())
            setattr(parent, child, counter.to(device))
            report.coeffs += lin.in_features * lin.out_features
            report.targets.append(f"model.layers.{index}.{path}")
            for key, value in counter.state_dict().items():
                state_out[f"model.layers.{index}.{path}.{key}"] = value.cpu()
            hessians.pop(path, None)

        # --- next block's activations ---
        # cascade: re-run the CONVERTED block so block i+1 calibrates on quantized
        # outputs (error compounds into the statistics, like the asym cascade).
        # classic: reuse the fp outputs captured during the Hessian pass -- same
        # semantics as the in-memory path, and one pass cheaper.
        buffer = (_run_block(block, buffer, rotary, device, micro_batch, collect=True)
                  if cascade else fp_out)

        torch.save(state_out, os.path.join(out_dir, f"block_{index:04d}.pt"))
        done.add(index)
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump({"blocks_done": sorted(done), "blocks_total": report.blocks_total,
                       "model_path": model_path, "group": group, "C": C, "kind": kind},
                      handle, indent=2)
        report.blocks_converted += 1

        block.to("meta")
        gc.collect()
        if progress:
            print(f"[stream] block {index}: {len(targets)} targets, "
                  f"peak {peak / 2**30:.2f} GiB", flush=True)

    src.close()
    report.peak_bytes = peak
    return report


def _reload_block_counters(block, index: int, out_dir: str, *, kind, group, C,
                           counter_kw) -> None:
    """Rebuild an already-converted block's counter layers from its saved file."""
    from ..recovery.runtime import restore_counter_structure

    prefix = f"model.layers.{index}."
    saved = torch.load(os.path.join(out_dir, f"block_{index:04d}.pt"),
                       map_location="cpu", weights_only=True)
    stripped = {k[len(prefix):]: v for k, v in saved.items() if k.startswith(prefix)}
    restore_counter_structure(block, stripped, kind=kind, group=group, C=C, **counter_kw)
    missing, unexpected = block.load_state_dict(stripped, strict=False)
    if unexpected:
        raise RuntimeError(f"block {index}: unexpected keys on reload: {unexpected[:4]}")


def _run_block(block, buffer, rotary, device, micro_batch, collect):
    """Push the activation buffer through one block; optionally keep the output.

    ``micro_batch`` caps how many sequences are on the compute device at once,
    which together with the per-block Hessians is what bounds peak memory. Note
    it also sets the Hessian accumulation order, so changing it perturbs the last
    bits of H (and very occasionally a ternary code at a rounding boundary).
    """
    out: list[torch.Tensor] = []
    for chunk in buffer:
        pieces = (chunk.split(micro_batch, dim=0) if micro_batch and micro_batch > 0
                  else (chunk,))
        produced = []
        for piece in pieces:
            hidden = piece.to(device)
            positions = torch.arange(hidden.shape[1], device=device).unsqueeze(0)
            pos_emb = rotary(hidden, positions)
            result = block(hidden, position_embeddings=pos_emb)
            result = result[0] if isinstance(result, tuple) else result
            if collect:
                produced.append(result.to("cpu"))
            del hidden, result
        if collect:
            out.append(torch.cat(produced, dim=0) if len(produced) > 1 else produced[0])
    return out if collect else buffer


def load_streamed_state(out_dir: str) -> dict[str, torch.Tensor]:
    """Merge per-block files into one state dict for ``restore_counter_structure``."""
    manifest_path = os.path.join(out_dir, MANIFEST)
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    merged: dict[str, torch.Tensor] = {}
    for index in manifest["blocks_done"]:
        merged.update(torch.load(os.path.join(out_dir, f"block_{index:04d}.pt"),
                                 map_location="cpu", weights_only=True))
    return merged
