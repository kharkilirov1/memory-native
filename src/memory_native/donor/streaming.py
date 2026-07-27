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

MoE donors work: transformers stores Mixtral-style experts on disk one module
per expert and stacks them at load time without exposing the mapping, so the
weight reader reconstructs it (gate_up_proj[e] = cat([w1, w3]), down_proj[e] =
w2 -- verified bit-exact against from_pretrained). Layouts the reader does not
recognise still fail loudly rather than converting a subset.
"""
from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass, field

import torch
from torch import nn

from ..convert import CounterLinearWithBias
from ..convert import SwapReport
from .ptq import (
    _assert_no_unhandled_moe,
    _group_counter_from_state,
    _legacy_expert_targets,
    _moe_router_paths,
    _parent_and_name,
    _stacked_moe_targets,
    _swap_stacked_moe,
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

    def has(self, name: str) -> bool:
        """Present verbatim, or assemblable from a legacy expert layout."""
        return name in self._file_of or self._assemble_plan(name) is not None

    def get(self, name: str, dtype=None) -> torch.Tensor:
        if name not in self._file_of:
            assembled = self._assemble(name, dtype)
            if assembled is None:
                raise KeyError(name)
            return assembled
        file = self._file_of[name]
        handle = self._handles.get(file)
        if handle is None:
            handle = self._safe_open(file, framework="pt")
            self._handles[file] = handle
        t = handle.get_tensor(name)
        return t.to(dtype) if dtype is not None else t

    # --- legacy expert layout -------------------------------------------------
    # transformers stores Mixtral-style experts one module per expert on disk and
    # stacks them at load time; the mapping is not exposed, so the reader
    # reconstructs it. Verified bit-exact against from_pretrained:
    #   gate_up_proj[e] = cat([w1, w3], dim=0)      down_proj[e] = w2
    _LEGACY_LAYOUTS = (
        ("block_sparse_moe", ("w1", "w3", "w2")),        # Mixtral
        ("mlp", ("gate_proj", "up_proj", "down_proj")),  # Qwen3-MoE style
    )

    def _assemble_plan(self, name: str):
        """Return (layer_prefix, container, parts, what) if `name` can be built."""
        for suffix, what in ((".mlp.experts.gate_up_proj", "gate_up"),
                             (".mlp.experts.down_proj", "down"),
                             (".mlp.gate.weight", "router")):
            if not name.endswith(suffix):
                continue
            layer_prefix = name[: -len(suffix)]
            for container, parts in self._LEGACY_LAYOUTS:
                probe = (f"{layer_prefix}.{container}.gate.weight" if what == "router"
                         else f"{layer_prefix}.{container}.experts.0.{parts[0]}.weight")
                if probe in self._file_of and probe != name:
                    return layer_prefix, container, parts, what
        return None

    def _assemble(self, name: str, dtype):
        plan = self._assemble_plan(name)
        if plan is None:
            return None
        layer_prefix, container, parts, what = plan
        if what == "router":
            return self.get(f"{layer_prefix}.{container}.gate.weight", dtype)
        gate, up, down = parts
        slices = []
        expert = 0
        while True:
            base = f"{layer_prefix}.{container}.experts.{expert}"
            if f"{base}.{gate}.weight" not in self._file_of:
                break
            if what == "gate_up":
                slices.append(torch.cat([self.get(f"{base}.{gate}.weight", dtype),
                                         self.get(f"{base}.{up}.weight", dtype)], dim=0))
            else:
                slices.append(self.get(f"{base}.{down}.weight", dtype))
            expert += 1
        return torch.stack(slices) if slices else None

    def close(self):
        self._handles.clear()


def _set_by_path(module: nn.Module, dotted: str) -> torch.Tensor:
    """Fetch the buffer at ``dotted`` so it can be written in place."""
    owner, _, leaf = dotted.rpartition(".")
    return getattr(module.get_submodule(owner) if owner else module, leaf)


def _non_persistent_names(module: nn.Module) -> set[str]:
    """Dotted names of ``module``'s non-persistent buffers.

    PyTorch's own definition of "computed": a buffer registered with
    ``persistent=False`` is deliberately kept out of the state dict, so no
    checkpoint can ever carry it. That makes the flag an exact criterion rather
    than a name blacklist -- gemma-4 alone has five (``embed_scale`` plus four
    rotary ``inv_freq`` variants for its full/sliding attention split).
    """
    out: set[str] = set()
    for path, sub in module.named_modules():
        for name in getattr(sub, "_non_persistent_buffers_set", ()):
            out.add(f"{path}.{name}" if path else name)
    return out


def _materialize(module: nn.Module, prefix: str, src: _WeightSource, device, dtype,
                 computed: dict[str, torch.Tensor] | None = None) -> None:
    """Give a meta-device module real weights, one tensor at a time.

    ``to_empty`` allocates UNINITIALIZED memory, so every tensor it creates has to
    be written before use. Anything the checkpoint does not carry is an error
    here, not a silent pass: a computed buffer left as garbage (rotary
    ``inv_freq`` is the classic one) still produces finite numbers and corrupts
    the calibration invisibly.

    Non-persistent buffers are the computed ones, and they are supplied through
    ``computed`` (harvested from a real one-layer probe) rather than looked up in
    the checkpoint, where they do not and cannot exist. Missing from BOTH sources
    is still a hard error.
    """
    non_persistent = _non_persistent_names(module)
    module.to_empty(device=device)
    missing: list[str] = []
    with torch.no_grad():
        for name in sorted(non_persistent):
            value = (computed or {}).get(name)
            if value is None:
                missing.append(f"{prefix}.{name} (computed buffer, not in any checkpoint)")
                continue
            _set_by_path(module, name).copy_(value.to(device))
        for name, param in list(module.named_parameters(recurse=True)):
            key = f"{prefix}.{name}"
            if not src.has(key):
                missing.append(key)
                continue
            param.copy_(src.get(key, dtype).to(device))
        for name, buf in list(module.named_buffers(recurse=True)):
            if name in non_persistent:
                continue                       # already filled from the probe above
            key = f"{prefix}.{name}"
            if src.has(key):
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


def _shallow_config(config):
    """A copy of ``config`` with a single decoder layer.

    The probe below has to be built on a REAL device to get real computed
    buffers, and building the donor's full depth there would allocate the whole
    model -- ~24 GiB for gemma-4-12B, which is the very thing streaming exists to
    avoid (it also does not fit this 32 GiB box). No computed buffer depends on
    depth: rotary comes from head_dim/rope_theta/max_position, gemma's
    embed_scale from hidden_size. Depth is the one dimension that is safe to cut.
    """
    import copy

    probe = copy.deepcopy(config)
    for holder in (getattr(probe, "text_config", None), probe):
        if holder is not None and getattr(holder, "num_hidden_layers", None):
            holder.num_hidden_layers = 1
    return probe


def _build_probe(config, device):
    """One-layer real instance, the single source of every computed value."""
    from transformers import AutoModelForCausalLM

    with torch.device(device):
        return AutoModelForCausalLM.from_config(_shallow_config(config))


def _build_rotary(config, device):
    """Rotary embeddings own COMPUTED buffers (inv_freq) that no checkpoint
    stores, so they are constructed from config on a real device rather than
    materialized from the shards."""
    probe = _build_probe(config, device)
    inner, _ = _resolve_decoder(probe)
    rotary = inner.rotary_emb
    del probe
    gc.collect()
    return rotary


def _computed_buffers(module: nn.Module) -> dict[str, torch.Tensor]:
    """Every non-persistent buffer of ``module``, keyed by its dotted name."""
    return {name: _set_by_path(module, name).detach().clone()
            for name in _non_persistent_names(module)}


def _resolve_decoder(model: nn.Module) -> tuple[nn.Module, str]:
    """Return the module owning the decoder block list, and its dotted path.

    A text-only donor puts it at ``model`` (``model.layers``, ``model.embed_tokens``).
    A multimodal donor nests it: gemma-4 keeps the text tower at
    ``model.language_model``, so the old hardcoded ``model.layers`` raised
    AttributeError before a single block was read.

    The path is also the CHECKPOINT prefix -- HF writes tensors under the same
    dotted name as the module -- so returning it here is what lets ``_materialize``
    ask for ``model.language_model.layers.0.*`` instead of a wrong ``model.layers.0.*``.
    Verified against the real gemma-4-12B header: module keys and checkpoint keys
    match exactly, both sides empty on the difference.
    """
    found = [
        (path, module)
        for path, module in model.named_modules()
        if isinstance(getattr(module, "layers", None), nn.ModuleList)
        and len(module.layers) > 0
        and hasattr(module, "embed_tokens")
    ]
    if not found:
        raise NotImplementedError(
            f"{type(model).__name__}: found no decoder module carrying both a "
            "non-empty .layers ModuleList and .embed_tokens; streaming conversion "
            "cannot locate the transformer stack"
        )
    # The shallowest match is the text decoder itself; anything deeper would be a
    # nested sub-stack. Ties cannot happen -- two decoders at the same depth would
    # be an architecture this driver has no defined behaviour for.
    found.sort(key=lambda item: (item[0].count("."), item[0]))
    if len(found) > 1 and found[0][0].count(".") == found[1][0].count("."):
        raise NotImplementedError(
            f"{type(model).__name__}: ambiguous decoder stacks at "
            f"{[p for p, _ in found[:2]]}; refusing to guess which one to convert"
        )
    return found[0][1], found[0][0]


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
    inner, stack = _resolve_decoder(skeleton)
    blocks = inner.layers
    report.blocks_total = len(blocks)

    # --- computed buffers come from one real one-layer probe, never from the
    # checkpoint (they are non-persistent by construction and are not in it) ---
    probe = _build_probe(config, device)
    probe_inner, _ = _resolve_decoder(probe)
    computed_embed = _computed_buffers(probe_inner.embed_tokens)
    computed_block = _computed_buffers(probe_inner.layers[0])
    rotary = probe_inner.rotary_emb
    del probe
    gc.collect()

    # --- embeddings once: the activation buffer lives on CPU, blocks pull micro-batches ---
    _materialize(inner.embed_tokens, f"{stack}.embed_tokens", src, device, dtype,
                 computed=computed_embed)
    buffer: list[torch.Tensor] = []
    id_batches = [b for b in calib_batches]
    _assert_window_covers(config, id_batches)
    for ids in id_batches:
        buffer.append(inner.embed_tokens(ids.to(device)).to("cpu"))
    inner.embed_tokens.to("meta")

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
            _materialize(block, f"{stack}.layers.{index}", src, device, dtype,
                         computed=computed_block)
            _reload_block_counters(block, index, out_dir, kind=kind, group=group, C=C,
                                   counter_kw=counter_kw, stack=stack)
            buffer = _run_block(block, buffer, rotary, device, micro_batch, collect=True)
            block.to("meta")
            gc.collect()
            if progress:
                print(f"[stream] block {index}: already converted, replayed for cascade",
                      flush=True)
            continue

        _materialize(block, f"{stack}.layers.{index}", src, device, dtype,
                     computed=computed_block)
        stacked = _stacked_moe_targets(block)
        legacy = _legacy_expert_targets(block)
        # keep the fail-loudly property of the in-memory path: a MoE layout this
        # driver cannot convert must raise, never silently convert attention only.
        _assert_no_unhandled_moe(block, stacked, legacy)
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
        for target in stacked:
            hooks.append(target.module.register_forward_pre_hook(
                _make_stacked_moe_hook(target, hessians)))
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
            report.targets.append(f"{stack}.layers.{index}.{path}")
            for key, value in counter.state_dict().items():
                state_out[f"{stack}.layers.{index}.{path}.{key}"] = value.cpu()
            hessians.pop(path, None)

        # --- stacked MoE experts: solve each slice, then swap through ptq's builder ---
        if stacked:
            moe_states: dict[str, list[tuple]] = {}
            moe_report = SwapReport()
            for target in stacked:
                per_expert = []
                for expert in range(target.num_experts):
                    slices = []
                    for path, weight in (
                        (target.gate_up_path(expert), target.module.gate_up_proj[expert]),
                        (target.down_path(expert), target.module.down_proj[expert]),
                    ):
                        H = hessians.get(path)
                        if H is None:      # expert saw no tokens: data-free solve
                            H = torch.eye(weight.shape[1], dtype=torch.float32,
                                          device=weight.device)
                        state, _ = solve_group_state(weight.data.float(), H,
                                                     group=group, C=C, **solve_kw)
                        slices.append(state)
                        hessians.pop(path, None)
                    per_expert.append(tuple(slices))
                    report.coeffs += (target.module.gate_up_proj[expert].numel()
                                      + target.module.down_proj[expert].numel())
                    report.targets.append(
                        f"{stack}.layers.{index}.{target.expert_path(expert)}")
                moe_states[target.path] = per_expert
            _swap_stacked_moe(block, stacked, moe_states, moe_report, is_group=True,
                              kind=kind, group=group, C=C, counter_kw=counter_kw)
            for key, value in block.state_dict().items():
                if "experts" in key:
                    state_out[f"{stack}.layers.{index}.{key}"] = value.cpu()

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
                       "model_path": model_path, "group": group, "C": C, "kind": kind,
                       "stack": stack},
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


def _make_stacked_moe_hook(target, hessians: dict):
    """Accumulate H per expert over exactly the tokens routed to it, mirroring
    ptq.collect_hessians' MoE branch (the expert's own inputs are the routed
    slice, and down_proj sees the SwiGLU activation, not the block input)."""
    import torch.nn.functional as F

    def accumulate(path, x):
        x = x.detach().reshape(-1, x.shape[-1]).to(torch.float32)
        h = hessians.get(path)
        if h is None:
            h = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=x.device)
            hessians[path] = h
        h.addmm_(x.t(), x)

    def hook(module, inputs):
        hidden_states, top_k_index = inputs[:2]
        for expert in range(target.num_experts):
            token_idx = torch.where(top_k_index == expert)[0]
            if token_idx.numel() == 0:
                continue
            current = hidden_states[token_idx]
            accumulate(target.gate_up_path(expert), current)
            gate, up = F.linear(current, module.gate_up_proj[expert]).chunk(2, dim=-1)
            accumulate(target.down_path(expert), module.act_fn(gate) * up)
    return hook


def _reload_block_counters(block, index: int, out_dir: str, *, kind, group, C,
                           counter_kw, stack: str = "model") -> None:
    """Rebuild an already-converted block's counter layers from its saved file."""
    from ..recovery.runtime import restore_counter_structure

    prefix = f"{stack}.layers.{index}."
    saved = torch.load(os.path.join(out_dir, f"block_{index:04d}.pt"),
                       map_location="cpu", weights_only=True)
    stripped = {k[len(prefix):]: v for k, v in saved.items() if k.startswith(prefix)}
    restore_counter_structure(block, stripped, kind=kind, group=group, C=C, **counter_kw)
    missing, unexpected = block.load_state_dict(stripped, strict=False)
    if unexpected:
        raise RuntimeError(f"block {index}: unexpected keys on reload: {unexpected[:4]}")


def _assert_window_covers(config, id_batches) -> None:
    """Refuse sequences longer than a sliding-attention donor's window.

    Blocks are run bare, and a bare HF attention call with no mask is plain
    causal -- correct for a full-attention layer, WRONG for a sliding one as
    soon as the sequence outgrows the window: those layers would attend to the
    whole prefix and the Hessians would be silently collected off the model the
    donor actually is. Under the window the two masks coincide exactly, so short
    calibration sequences are safe; longer ones need real sliding masks, which
    this driver does not build yet.
    """
    text = getattr(config, "text_config", config)
    window = getattr(text, "sliding_window", None)
    types = set(getattr(text, "layer_types", None) or ())
    if not window or "sliding_attention" not in types:
        return
    longest = max((int(b.shape[-1]) for b in id_batches), default=0)
    if longest > window:
        raise NotImplementedError(
            f"calibration sequence length {longest} exceeds the donor's sliding "
            f"window {window}; the driver runs blocks without a sliding mask, so "
            "sliding-attention layers would see the full prefix and their Hessians "
            "would not match the real model. Use sequences <= the window."
        )


def _rotary_kwargs(rotary, block) -> dict:
    """Extra kwargs this donor's rotary needs, derived from its own signature.

    A single-rope donor takes (x, position_ids). Gemma-4 interleaves sliding and
    full attention and keeps one inv_freq per kind, so its rotary picks the set
    by ``layer_type`` -- calling it without one raises AttributeError on
    ``None_inv_freq``. The block carries its own ``layer_type``, so the value is
    read from the block rather than guessed.
    """
    import inspect

    try:
        params = inspect.signature(rotary.forward).parameters
    except (TypeError, ValueError):
        return {}
    if "layer_type" not in params:
        return {}
    # gemma-4 hangs it off the attention submodule, not the decoder layer.
    layer_type = next(
        (value for value in (getattr(sub, "layer_type", None)
                             for _, sub in block.named_modules())
         if value is not None),
        None,
    )
    if layer_type is None:
        raise NotImplementedError(
            f"{type(rotary).__name__} selects its frequencies by layer_type, but "
            f"nothing under {type(block).__name__} exposes one; streaming cannot "
            "pick the right rope without guessing"
        )
    return {"layer_type": layer_type}


def _run_block(block, buffer, rotary, device, micro_batch, collect):
    """Push the activation buffer through one block; optionally keep the output.

    ``micro_batch`` caps how many sequences are on the compute device at once,
    which together with the per-block Hessians is what bounds peak memory. Note
    it also sets the Hessian accumulation order, so changing it perturbs the last
    bits of H (and very occasionally a ternary code at a rounding boundary).

    No attention mask is passed: with ``attention_mask=None`` the HF attention
    path is already causal (checked -- editing the last token leaves the first
    token's output bit-identical). Sliding-window donors are handled by the
    guard in ``convert_streaming``, not here.
    """
    rotary_kw = _rotary_kwargs(rotary, block)
    out: list[torch.Tensor] = []
    for chunk in buffer:
        pieces = (chunk.split(micro_batch, dim=0) if micro_batch and micro_batch > 0
                  else (chunk,))
        produced = []
        for piece in pieces:
            hidden = piece.to(device)
            positions = torch.arange(hidden.shape[1], device=device).unsqueeze(0)
            pos_emb = rotary(hidden, positions, **rotary_kw)
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
