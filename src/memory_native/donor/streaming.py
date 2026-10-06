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

import copy
import gc
import hashlib
import json
import os
import struct
import warnings
from dataclasses import dataclass, field

import torch
from torch import nn

from ..convert import CounterLinearWithBias
from ..convert import SwapReport
from .provenance import (
    atomic_torch_save,
    atomic_write_json,
    canonical_json,
    checkpoint_fingerprint,
    file_fingerprint,
)
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
MANIFEST_VERSION = 2
_COUNTER_KWARGS = {
    "lr", "lr_scale", "rms_beta", "rms_eps", "local_grad_clip", "residual_alpha",
    "kernel_mode", "strict_update", "flip_sample_size", "stats_scope", "decimation",
}

# Vocabulary the computed-buffer probe is built with (see _shallow_config).
PROBE_VOCAB = 2048


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


def _calibration_fingerprint(batches) -> dict:
    """Bind batch boundaries/order and the exact token stream used by the solve."""
    digest = hashlib.sha256()
    shapes = []
    tokens = 0
    if not batches:
        raise ValueError("calibration must contain at least one nonempty [B, T] batch")
    for batch in batches:
        if (not isinstance(batch, torch.Tensor) or batch.ndim != 2
                or batch.numel() == 0 or batch.dtype not in (torch.int32, torch.int64)):
            raise ValueError("calibration batches must be nonempty int32/int64 [B, T] tensors")
        descriptor = {"shape": list(batch.shape), "dtype": str(batch.dtype)}
        shapes.append(descriptor)
        digest.update(canonical_json(descriptor))
        digest.update(batch.detach().to(device="cpu", dtype=torch.int64)
                      .contiguous().numpy().astype("<i8", copy=False).tobytes())
        tokens += batch.numel()
    return {"sha256": digest.hexdigest(), "batches": shapes, "tokens": tokens}


def _validated_blocks(manifest, out_dir, *, require_complete=False,
                      require_verified=False) -> list[int]:
    """Validate committed progress and file integrity before loading any tensors."""
    version = manifest.get("schema_version")
    if require_verified and version != MANIFEST_VERSION:
        raise ValueError("legacy streamed manifest has no verified provenance; "
                         "resume requires a fresh conversion directory")
    if version not in (None, MANIFEST_VERSION):
        raise ValueError(f"unsupported streamed manifest schema_version={version!r}")
    total = manifest.get("blocks_total")
    done = manifest.get("blocks_done")
    if (type(total) is not int or total <= 0 or not isinstance(done, list)
            or any(type(i) is not int for i in done)
            or done != list(range(len(done))) or len(done) > total):
        raise ValueError("streamed manifest blocks_done must be a contiguous prefix "
                         "within blocks_total")
    if require_complete and len(done) != total:
        raise ValueError(f"incomplete streamed conversion: {len(done)}/{total} blocks")
    metadata = manifest.get("block_files", {})
    if version == MANIFEST_VERSION and not isinstance(metadata, dict):
        raise ValueError("streamed manifest block_files must be an object")
    for index in done:
        filename = f"block_{index:04d}.pt"
        path = os.path.join(out_dir, filename)
        if not os.path.isfile(path):
            raise ValueError(f"streamed block {index}: missing checkpoint {filename}")
        if version == MANIFEST_VERSION:
            record = metadata.get(str(index))
            if not isinstance(record, dict) or record.get("file") != filename:
                raise ValueError(f"streamed block {index}: missing integrity metadata")
            if file_fingerprint(path) != {k: record.get(k) for k in ("bytes", "sha256")}:
                raise ValueError(f"streamed block {index}: checkpoint integrity mismatch")
    return done


def _load_block_state(path, index, stack) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    prefix = f"{stack}.layers.{index}."
    if (not isinstance(state, dict) or not state
            or any(not isinstance(k, str) or not k.startswith(prefix)
                   or not isinstance(v, torch.Tensor) for k, v in state.items())):
        raise ValueError(f"streamed block {index}: invalid tensor state or key prefix")
    if any(v.is_floating_point() and not torch.isfinite(v).all().item()
           for v in state.values()):
        raise ValueError(f"streamed block {index}: nonfinite tensor state")
    return state


class _WeightSource:
    """Lazy per-tensor access to a HF checkpoint (single-file or sharded)."""

    # safetensors hands back tensors that VIEW its memory map. On this box
    # (safetensors 0.8.0, torch 2.12.1, Windows) reading a large one out of a
    # 22 GiB file access-violates inside UntypedStorage.__getitem__ — not an
    # exception, the process just dies — and it is INTERMITTENT: the identical
    # call succeeded minutes earlier. Plain file I/O plus torch.frombuffer
    # reproduces the same bytes without ever mapping the file, so it is used for
    # every tensor. A 256 MiB threshold was tried first and was not enough: the
    # fault also hit ordinary block weights (down_proj, 118 MiB bf16). The mmap
    # path stays only as a fallback for tensors the header cannot describe.
    _MMAP_BYTES_LIMIT = 0

    _DTYPES = {
        "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
        "F64": torch.float64, "I64": torch.int64, "I32": torch.int32,
        "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
        "BOOL": torch.bool,
    }

    def __init__(self, path: str):
        from safetensors import safe_open

        self._safe_open = safe_open
        self.path = path
        self._headers: dict[str, tuple[int, dict]] = {}
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

    def get_rows(self, name: str, indices: torch.Tensor, chunk: int = 8192
                 ) -> torch.Tensor:
        """Gather rows of a 2-D tensor without materializing the whole thing.

        INDEXING a safetensors tensor faults on this platform. ``get_tensor``
        returns a view over the memory map, and indexing it goes through
        ``UntypedStorage.__getitem__``, which dies with a Windows access
        violation — not an exception — on the 262144x3840 embedding inside
        gemma-4's 22 GiB file. Bulk reads of the same tensor are fine, which is
        why this only ever showed up as a mysterious process death. ``.clone()``
        forces the bytes out of the mapping first; the gather then runs on
        ordinary memory. ``chunk`` bounds how much is cloned at once.
        """
        full = self.get(name)
        if full.ndim != 2 or full.shape[0] <= chunk:
            return full.clone()[indices]
        order = torch.argsort(indices)
        wanted = indices[order]
        parts = []
        for start in range(0, int(full.shape[0]), chunk):
            stop = min(start + chunk, int(full.shape[0]))
            hit = wanted[(wanted >= start) & (wanted < stop)]
            if hit.numel() == 0:
                continue
            parts.append(full[start:stop].clone()[hit - start])
        gathered = torch.cat(parts, dim=0)
        out = torch.empty_like(gathered)
        out[order] = gathered                     # restore the caller's order
        return out

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
        t = self._read_direct(file, name)
        if t is None:
            handle = self._handles.get(file)
            if handle is None:
                handle = self._safe_open(file, framework="pt")
                self._handles[file] = handle
            t = handle.get_tensor(name)
        return t.to(dtype) if dtype is not None else t

    def _file_header(self, file: str) -> tuple[int, dict]:
        cached = self._headers.get(file)
        if cached is None:
            with open(file, "rb") as handle:
                size = struct.unpack("<Q", handle.read(8))[0]
                cached = (size, json.loads(handle.read(size).decode("utf-8")))
            self._headers[file] = cached
        return cached

    def _read_direct(self, file: str, name: str):
        """Read one tensor with ordinary file I/O, bypassing the memory map.

        Returns None for small tensors (the mmap path is fine and cheaper there)
        or if the header does not describe this tensor the way we expect, so the
        caller falls back rather than guessing.
        """
        header_size, header = self._file_header(file)
        info = header.get(name)
        if not isinstance(info, dict) or "data_offsets" not in info:
            return None
        start, stop = info["data_offsets"]
        if stop - start <= self._MMAP_BYTES_LIMIT:
            return None
        dtype = self._DTYPES.get(info.get("dtype"))
        if dtype is None:
            return None
        if os.environ.get("MN_MMAP_BIG") == "1":
            # commit-starved host (winerror 1455: the pagefile cannot grow):
            # copy-on-write file mapping -- reads are served by the shard file
            # (zero commit charge), any write lands on a private page instead of
            # faulting on a read-only mapping, and the file itself is never
            # touched. Reinterpreted to the real dtype without any copy.
            import numpy as np

            flat = np.memmap(file, dtype=np.uint8, mode="c",
                             offset=8 + header_size + start, shape=(stop - start,))
            return (torch.from_numpy(flat).view(dtype).reshape(info["shape"]))
        with open(file, "rb") as handle:
            handle.seek(8 + header_size + start)
            # single presized buffer + readinto: handle.read() would allocate the
            # bytes AND then copy into a bytearray (2x transient on 2.5 GB tensors)
            raw = bytearray(stop - start)
            view = memoryview(raw)
            pos = 0
            while pos < stop - start:
                got = handle.readinto(view[pos:pos + (1 << 29)])
                if not got:
                    break
                pos += got
            del view
        if pos != stop - start:
            raise RuntimeError(
                f"{name}: short read, {pos} of {stop - start} bytes"
            )
        return torch.frombuffer(raw, dtype=dtype).reshape(info["shape"])

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
            # Read in the checkpoint's own dtype and let copy_ do the conversion.
            # Asking the source to cast first materializes a SECOND full-size fp32
            # tensor on top of the destination to_empty already allocated -- for
            # gemma-4's 262144x3840 embeddings that is 3.75 GiB of avoidable peak
            # on top of an open 22 GiB mmap, and it crashed the run with a Windows
            # access violation (the OS fails a mapped page rather than raising
            # MemoryError). copy_ converts elementwise, no temporary.
            param.copy_(src.get(key).to(device))
        for name, buf in list(module.named_buffers(recurse=True)):
            if name in non_persistent:
                continue                       # already filled from the probe above
            key = f"{prefix}.{name}"
            if src.has(key):
                buf.copy_(src.get(key).to(device))
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
        if holder is None:
            continue
        if getattr(holder, "num_hidden_layers", None):
            holder.num_hidden_layers = 1
        # The vocabulary only sizes the embedding table, which the probe never
        # reads -- it exists to expose computed buffers. Keeping the donor's real
        # 262144 rows costs 2.23 GiB against 0.55 GiB here, and all five of
        # gemma-4's computed buffers were measured bit-identical either way.
        # _build_probe re-checks that nothing actually depends on it.
        if getattr(holder, "vocab_size", 0) > PROBE_VOCAB:
            holder.vocab_size = PROBE_VOCAB
    # A multimodal config's vision tower is dead weight on a REAL-device probe
    # (qwen3_5: 27 layers x 1152 ≈ 1.6 GB for buffers nobody reads), and no
    # computed value the probe exists for depends on tower depth -- same
    # argument as the text depth cut above.
    vision = getattr(probe, "vision_config", None)
    if vision is not None and getattr(vision, "depth", None):
        vision.depth = 2
    return probe


def _build_skeleton(config):
    """Meta-device skeleton, preferring the checkpoint's own declared class.

    ``AutoModelForCausalLM`` maps some multimodal configs to the TEXT-ONLY class
    (qwen3_5: ``Qwen3_5Config`` -> ``Qwen3_5ForCausalLM``), which moves the
    decoder to ``model`` while the checkpoint still keys it as
    ``model.language_model.*`` -- ``_materialize`` would then fail on the very
    first tensor. Building the class named in ``config.architectures`` keeps the
    module tree and the checkpoint prefix identical (the same invariant the
    gemma-4 VLM already satisfied through its own auto-mapping). The text-only
    class stays the fallback: it is what every text-only donor builds anyway.
    """
    from transformers import AutoModelForCausalLM, PreTrainedModel

    arch = (getattr(config, "architectures", None) or [None])[0]
    if arch:
        import transformers

        cls = getattr(transformers, arch, None)
        if isinstance(cls, type) and issubclass(cls, PreTrainedModel):
            try:
                return cls(config)
            except TypeError:
                pass                   # class wants a different constructor
    return AutoModelForCausalLM.from_config(config)


def _skeleton_for_checkpoint(src, config):
    """Meta skeleton whose decoder prefix matches the checkpoint -- or raise.

    ``AutoModelForCausalLM`` maps some multimodal configs to the text-only
    class (qwen3_5: decoder at ``model`` while the checkpoint keys it
    ``model.language_model.*``), which would make ``_materialize`` fail on the
    first tensor. The checkpoint's declared architectures are tried first, the
    multimodal auto-class second; the embed-key probe against the real
    checkpoint decides, and a persistent mismatch is a loud error instead of a
    KeyError from deep inside the reader.
    """
    with torch.device("meta"):
        skeleton = _build_skeleton(config)
        inner, stack = _resolve_decoder(skeleton)
        if not src.has(f"{stack}.embed_tokens.weight"):
            from transformers import AutoModelForImageTextToText

            skeleton = AutoModelForImageTextToText.from_config(config)
            inner, stack = _resolve_decoder(skeleton)
    if not src.has(f"{stack}.embed_tokens.weight"):
        raise RuntimeError(
            f"decoder resolved to {stack!r} but the checkpoint carries no "
            f"{stack}.embed_tokens.weight; the skeleton class "
            f"{type(skeleton).__name__} does not match how this checkpoint "
            "was saved"
        )
    return skeleton, inner, stack


def _build_probe(config, device, cls=None):
    """One-layer, small-vocab real instance: the single source of every computed value.

    ``cls`` builds the probe with the SAME top-level class as the conversion
    skeleton (a VLM config probed as the text-only class names modules
    ``model.*`` while the checkpoint and skeleton say ``model.language_model.*``
    -- computed-buffer lookups by full path would miss).
    """
    from transformers import AutoModelForCausalLM

    shallow = _shallow_config(config)
    with torch.device(device):
        probe = (cls(shallow) if cls is not None
                 else AutoModelForCausalLM.from_config(shallow))
    # The shrink is only safe while no computed buffer is sized by the vocabulary.
    # A buffer carrying the probe's vocab in its shape would be exactly that, and
    # would be silently wrong for the real donor -- refuse instead.
    for path, module in probe.named_modules():
        for name in getattr(module, "_non_persistent_buffers_set", ()):
            buf = getattr(module, name, None)
            if buf is not None and PROBE_VOCAB in tuple(buf.shape):
                raise NotImplementedError(
                    f"computed buffer {path}.{name} has the probe's vocabulary in its "
                    f"shape {tuple(buf.shape)}; it depends on vocab_size and the "
                    "shrunken probe would give the wrong value"
                )
    return probe


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
    asym_strength: float = 0.0,
    asym_passes: int = 1,
    **solve_kw,
) -> StreamingReport:
    """Convert ``model_path`` block by block, writing counter state into ``out_dir``.

    ``calib_batches`` is an iterable of token-id tensors [B, T] (the same input the
    in-memory path takes). ``micro_batch`` bounds how many sequences are pushed
    through a block at once, which is the knob that bounds peak memory together
    with the per-block Hessians.

    ``asym_strength > 0`` enables the GPTAQ-style asymmetric objective per block
    (H_q from the converted-prefix stream, G paired against the untouched-donor
    stream, residual-form target w~ tempered by strength) -- the streaming mirror
    of ``donor/asym.py``. ``asym_passes`` re-collects on the pass-1 quantized net
    and re-solves from the original weights (pass-N states land in
    ``out_dir_pN``, then replace ``out_dir``); later blocks' q stream always
    replays the FROZEN pass-1 states, matching the in-memory iterated asym.
    Dense donors only (MoE + asym raises).
    """
    from transformers import AutoConfig, AutoModelForCausalLM
    import transformers

    device = torch.device(device)
    model_path = os.fspath(model_path)
    out_dir = os.fspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    report = StreamingReport(out_dir=out_dir)

    manifest_path = os.path.join(out_dir, MANIFEST)
    previous = None
    if resume and os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            previous = json.load(handle)
        if not isinstance(previous, dict):
            raise ValueError("streamed manifest must be a JSON object")
        for index in _validated_blocks(previous, out_dir, require_verified=True):
            # Validate structure even in classic mode, where replay uses the
            # donor block and would otherwise never deserialize this file.
            _load_block_state(os.path.join(out_dir, f"block_{index:04d}.pt"),
                              index, previous.get("stack", "model"))

    counter_kw = {k: solve_kw.pop(k) for k in list(solve_kw) if k in _COUNTER_KWARGS}
    extra_skip = list(extra_skip or [])
    # Bind defaults as well as supplied options so omitting a solver option is
    # equivalent to spelling out its current default.
    import inspect as _inspect
    solver_call = _inspect.signature(solve_group_state).bind(
        None, None, group=group, C=C, **solve_kw)
    solver_call.apply_defaults()
    solver_options = {k: v for k, v in solver_call.arguments.items() if k not in {"w", "H"}}
    conversion_options = {
        "kind": kind, "C": C, "group": group, "dtype": str(dtype),
        "device": str(device), "micro_batch": micro_batch, "cascade": cascade,
        "extra_skip": sorted(extra_skip), "solver": solver_options,
        "counter": counter_kw, "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
        "asym_strength": float(asym_strength), "asym_passes": int(asym_passes),
    }
    # Reject non-JSON / non-finite knobs before producing any checkpoint files.
    canonical_json(conversion_options)
    id_batches = [b for b in calib_batches]
    calibration = _calibration_fingerprint(id_batches)
    source_fingerprint = checkpoint_fingerprint(model_path)
    for key, expected in (("source_fingerprint", source_fingerprint),
                          ("calibration_fingerprint", calibration),
                          ("conversion_options", conversion_options)):
        if previous is not None and previous.get(key) != expected:
            raise ValueError(f"streamed resume {key} mismatch; use a fresh output directory")

    src = _WeightSource(model_path)
    config = AutoConfig.from_pretrained(model_path)
    text_config = getattr(config, "text_config", config)
    vocab_size = getattr(text_config, "vocab_size", None)
    if vocab_size is not None and any(
            b.min().item() < 0 or b.max().item() >= vocab_size for b in id_batches):
        raise ValueError(f"calibration token ids must be within [0, {vocab_size})")
    skeleton, inner, stack = _skeleton_for_checkpoint(src, config)
    blocks = inner.layers
    report.blocks_total = len(blocks)
    if previous is not None and (
            previous["blocks_total"] != report.blocks_total or previous.get("stack") != stack):
        raise ValueError("streamed resume decoder layout mismatch")
    done: set[int] = set(previous["blocks_done"] if previous is not None else [])
    report.blocks_resumed = len(done)
    manifest = {
        "schema_version": MANIFEST_VERSION, "blocks_done": sorted(done),
        "blocks_total": report.blocks_total, "model_path": model_path,
        "group": group, "C": C, "kind": kind, "stack": stack,
        "asym_strength": float(asym_strength), "asym_passes": int(asym_passes),
        "source_fingerprint": source_fingerprint,
        "calibration_fingerprint": calibration, "conversion_options": conversion_options,
        "block_files": dict(previous.get("block_files", {})) if previous else {},
    }
    _assert_window_covers(config, id_batches)
    atomic_write_json(manifest_path, manifest)

    # --- computed buffers come from one real one-layer probe, never from the
    # checkpoint (they are non-persistent by construction and are not in it) ---
    probe = _build_probe(config, device, cls=type(skeleton))
    probe_inner, _ = _resolve_decoder(probe)
    computed_embed = _computed_buffers(probe_inner.embed_tokens)
    computed_block = _computed_buffers(probe_inner.layers[0])
    rotary = probe_inner.rotary_emb
    del probe
    gc.collect()

    # --- embeddings: only the rows the calibration actually touches ---
    # Materializing the whole table costs vocab*dim*4 B — 3.75 GiB for gemma-4's
    # 262144x3840 — and every byte of it is dead once the activation buffer
    # exists. A lookup is a row gather, so the module runs against a compact
    # table holding just the unique ids, with the ids remapped onto it. The
    # module's own forward is kept (gemma scales by embed_scale, and rolling the
    # lookup by hand would silently drop that: measured 6.22 max error).
    buffer: list[torch.Tensor] = []
    flat = torch.cat([b.reshape(-1) for b in id_batches])
    unique, inverse = torch.unique(flat, return_inverse=True)
    compact_weight = src.get_rows(f"{stack}.embed_tokens.weight", unique)
    inner.embed_tokens.to_empty(device=device)
    # Match the dtype to_empty produced rather than forcing `dtype`: to_empty
    # keeps the skeleton's own dtype (bf16 for this donor's config), and the
    # blocks are materialized the same way, so forcing fp32 here alone makes the
    # first matmul fail on mismatched operands.
    embed_dtype = inner.embed_tokens.weight.dtype
    inner.embed_tokens.weight = nn.Parameter(
        compact_weight.to(embed_dtype).to(device), requires_grad=False)
    with torch.no_grad():
        for name, value in computed_embed.items():
            _set_by_path(inner.embed_tokens, name).copy_(value.to(device))
        cursor = 0
        for ids in id_batches:
            n = ids.numel()
            compact = inverse[cursor:cursor + n].reshape(ids.shape)
            cursor += n
            buffer.append(inner.embed_tokens(compact.to(device)).to("cpu"))
    del compact_weight
    inner.embed_tokens.to("meta")

    # The fp stream starts IDENTICAL to the q stream (same embeddings); it only
    # diverges once converted blocks advance the q side. Aliasing the tensors is
    # safe: buffers are replaced, never mutated in place.
    buffer_fp = list(buffer)
    # Iterated-asym passes restart both streams from the embeddings; keep the
    # initial activation tensors alive for that (references only -- the buffers
    # themselves are rebound, never mutated).
    buffer0 = list(buffer) if int(asym_passes) > 1 else None

    peak = _tensor_bytes(*buffer)

    if asym_strength > 0 and not cascade:
        raise ValueError("asym requires cascade=True (the q stream is the cascade)")

    # Iterated passes re-materialize blocks that still carry the counter modules
    # swapped in by the previous pass (their non-persistent buffers are not in any
    # checkpoint and _materialize must fail loudly on those) -- keep a pristine
    # meta snapshot of the layer list to reset the tree at each pass start.
    pristine_layers = (copy.deepcopy(inner.layers) if int(asym_passes) > 1 else None)

    for pass_no in range(1, max(1, int(asym_passes)) + 1):
        if pass_no > 1:
            inner.layers = copy.deepcopy(pristine_layers)
            blocks = inner.layers
        solve_dir = out_dir if pass_no == 1 else f"{out_dir}_p{pass_no}"
        if pass_no > 1:
            # The q stream of every later pass replays the FROZEN pass-1 states --
            # the streaming equivalent of in-memory iterated asym re-collecting on
            # the fully quantized net (never on the in-progress pass-N prefix).
            frozen = json.load(open(manifest_path, encoding="utf-8"))
            if set(frozen.get("blocks_done", [])) != set(range(len(blocks))):
                raise RuntimeError(
                    f"asym pass {pass_no} requires pass 1 complete in {out_dir}")
            os.makedirs(solve_dir, exist_ok=True)
            solve_manifest = os.path.join(solve_dir, MANIFEST)
            done = set()
            if resume and os.path.exists(solve_manifest):
                with open(solve_manifest, encoding="utf-8") as handle:
                    pass_previous = json.load(handle)
                done = set(_validated_blocks(pass_previous, solve_dir,
                                             require_verified=True))
            buffer = list(buffer0)
            buffer_fp = list(buffer0)
        save_manifest_path = os.path.join(solve_dir, MANIFEST)
        # Pass-local manifest: same verified provenance as pass 1, but its own
        # committed-block set and integrity records (each pass writes its own dir).
        pass_manifest = dict(manifest)
        pass_manifest["asym_pass"] = pass_no
        pass_manifest["blocks_done"] = sorted(done)
        pass_manifest["block_files"] = {
            str(i): rec for i, rec in manifest.get("block_files", {}).items()
        } if pass_no == 1 else {}

        for index, block in enumerate(blocks):
            if index in done:
                # A finished block still has to RUN: block i+1 calibrates on its output,
                # so skipping it outright would feed the next block unconverted
                # activations and silently change the result of a resumed run.
                _materialize(block, f"{stack}.layers.{index}", src, device, dtype,
                             computed=computed_block)
                if asym_strength > 0:
                    buffer_fp = _run_block(block, buffer_fp, rotary, device, micro_batch,
                                           collect=True)
                # Which stream the next block must calibrate on:
                #   cascade=True  -> the CONVERTED block (error compounds into the stats)
                #   cascade=False -> the original fp block, exactly as the in-memory path
                #   pass_no > 1   -> always the FROZEN pass-1 converted counters
                # Reloading unconditionally would silently turn a cascade=False resume
                # into a cascaded run and change the converted state.
                if cascade or pass_no > 1:
                    _reload_block_counters(block, index, out_dir,
                                           kind=kind, group=group, C=C,
                                           counter_kw=counter_kw, stack=stack)
                buffer = _run_block(block, buffer, rotary, device, micro_batch,
                                    collect=True)
                _release_block(block)
                gc.collect()
                if progress:
                    print(f"[stream] block {index}: already converted, replayed "
                          f"{'converted' if (cascade or pass_no > 1) else 'fp'} block",
                          flush=True)
                continue

            _materialize(block, f"{stack}.layers.{index}", src, device, dtype,
                         computed=computed_block)
            stacked = _stacked_moe_targets(block)
            legacy = _legacy_expert_targets(block)
            # keep the fail-loudly property of the in-memory path: a MoE layout this
            # driver cannot convert must raise, never silently convert attention only.
            _assert_no_unhandled_moe(block, stacked, legacy)
            if asym_strength > 0 and (stacked or legacy):
                raise NotImplementedError(
                    "streaming asym supports dense donors only (MoE pairing needs "
                    "teacher-forced routing; see donor/asym.py)")
            targets = _block_targets(block, extra_skip or [])

            # --- statistics for this block's targets ---
            if asym_strength > 0:
                # (H_q, G) from the batch-synchronized two-stream pass; the fp
                # stream advances here while the fp weights are still resident.
                stats, buffer_fp = _collect_asym_block(
                    block, buffer, buffer_fp, rotary, device, micro_batch, targets)
                hessians = {path: pair[0] for path, pair in stats.items()}
                grads = {path: pair[1] for path, pair in stats.items()}
                fp_out = None
                peak = max(peak, _tensor_bytes(*buffer, *buffer_fp,
                                               *hessians.values(), *grads.values(),
                                               *[p for p in block.parameters()]))
            else:
                hessians: dict[str, torch.Tensor] = {}
                grads = {}
                hooks = []

                def make_hook(path, in_features):
                    def hook(_mod, inputs):
                        x = inputs[0].detach().reshape(-1, in_features).to(torch.float32)
                        h = hessians.get(path)
                        if h is None:
                            h = torch.zeros(in_features, in_features,
                                            dtype=torch.float32, device=x.device)
                            hessians[path] = h
                        h.addmm_(x.t(), x)
                    return hook

                for path in targets:
                    lin = block.get_submodule(path)
                    hooks.append(lin.register_forward_pre_hook(
                        make_hook(path, lin.in_features)))
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
            from .asym import asym_target_weights
            state_out: dict[str, torch.Tensor] = {}
            for path in targets:
                lin = block.get_submodule(path)
                H = hessians.get(path)
                if H is None:                      # never called (dead branch): data-free solve
                    H = torch.eye(lin.in_features, dtype=torch.float32,
                                  device=lin.weight.device)
                    w_target = lin.weight.data.float()
                elif asym_strength > 0:
                    # Residual-form cascade correction, tempered by strength; the
                    # fp weights here ARE w0 (streaming never mutates them).
                    w_target = asym_target_weights(
                        lin.weight.data.float(), H, grads[path],
                        strength=asym_strength)
                else:
                    w_target = lin.weight.data.float()
                state, _ = solve_group_state(w_target, H, group=group, C=C,
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
                grads.pop(path, None)

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

            filename = f"block_{index:04d}.pt"
            block_path = os.path.join(solve_dir, filename)
            atomic_torch_save(block_path, state_out)

            # --- next block's q activations ---
            # cascade: advance through the CONVERTED block so block i+1 calibrates
            # on quantized outputs. Pass 1: the just-swapped counters ARE the frozen
            # states. Later passes: reload the FROZEN pass-1 counters -- pass N's
            # q stream must replay the pass-1 net, not its own in-progress output
            # (in-memory iterated-asym semantics).
            if pass_no > 1:
                _reload_block_counters(block, index, out_dir, kind=kind, group=group,
                                       C=C, counter_kw=counter_kw, stack=stack)
            buffer = (_run_block(block, buffer, rotary, device, micro_batch, collect=True)
                      if cascade else fp_out)

            done.add(index)
            pass_manifest["blocks_done"] = sorted(done)
            pass_manifest["block_files"][str(index)] = {
                "file": filename, **file_fingerprint(block_path)}
            atomic_write_json(save_manifest_path, pass_manifest)
            if pass_no == 1:
                manifest = pass_manifest
            report.blocks_converted += 1

            _release_block(block)
            gc.collect()
            if progress:
                print(f"[stream] pass {pass_no} block {index}: {len(targets)} targets, "
                      f"peak {peak / 2**30:.2f} GiB", flush=True)

    # --- endgame: the LAST pass's states are the artifact; swap them into
    # out_dir and drop the superseded intermediate dirs. ---
    final_pass = max(1, int(asym_passes))
    if final_pass > 1:
        import shutil

        keep = f"{out_dir}_p{final_pass}"
        frozen_dir = f"{out_dir}_final_frozen"
        os.rename(out_dir, frozen_dir)
        os.rename(keep, out_dir)
        shutil.rmtree(frozen_dir, ignore_errors=True)
        for p in range(2, final_pass):
            shutil.rmtree(f"{out_dir}_p{p}", ignore_errors=True)

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


def _release_block(block) -> None:
    """Move a finished block to meta AND drop its private tensor attributes.

    ``Module.to('meta')`` moves registered params/buffers only; the packed
    counter layer caches salient geometry in PLAIN attributes
    (``_salient_perm_flat``, ``_salient_sparse_cache``), which stay resident
    block after block on a deep run (measured ~0.7 GiB commit growth per block
    on the 27B: these caches + heap churn). A block is never re-run after this
    point -- resume rebuilds it from disk, and the layer's load post-hook
    re-derives the salient runtime on reload.
    """
    block.to("meta")
    for module in block.modules():
        for name, value in list(vars(module).items()):
            if isinstance(value, torch.Tensor):
                setattr(module, name, None)


def _reload_block_counters(block, index: int, out_dir: str, *, kind, group, C,
                           counter_kw, stack: str = "model") -> None:
    """Rebuild an already-converted block's counter layers from its saved file."""
    from ..recovery.runtime import restore_counter_structure

    prefix = f"{stack}.layers.{index}."
    saved = _load_block_state(os.path.join(out_dir, f"block_{index:04d}.pt"),
                              index, stack)
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
    if getattr(text, "num_kv_shared_layers", 0):
        raise NotImplementedError(
            f"donor declares num_kv_shared_layers={text.num_kv_shared_layers}: later "
            "layers read key/value produced by earlier ones, so a block cannot be run "
            "in isolation and this driver's premise does not hold"
        )
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


def _block_kwargs(block) -> dict:
    """Extra kwargs this donor's decoder layer requires, from its own signature.

    gemma-4's attention writes its key/value into a caller-supplied
    ``shared_kv_states`` dict when it is the last layer of its attention type
    (``store_full_length_kv``). Running the block bare passes None and it dies
    with "'NoneType' object does not support item assignment" — 46 blocks into a
    48-block run, because only two layers in the model do it.

    A fresh dict per block is correct ONLY while nothing READS it across blocks;
    ``convert_streaming`` refuses donors that declare real KV sharing, where
    block-at-a-time conversion would not be sound in the first place.
    """
    import inspect

    try:
        params = inspect.signature(block.forward).parameters
    except (TypeError, ValueError):
        return {}
    return {"shared_kv_states": {}} if "shared_kv_states" in params else {}


def _collect_asym_block(block, buffer_q, buffer_fp, rotary, device, micro_batch,
                        targets):
    """Batch-synchronized (H_q, G) collection for one block's targets + the
    advanced fp buffer -- the streaming mirror of ``asym.collect_asym_stats``.

    Ordering is the reference's: each micro-chunk runs the FP piece FIRST (its
    target inputs are captured), then the Q piece through the same fp block
    consumes the pairing, accumulating H_q += X_q^T X_q and G += X_q^T X_fp.
    The q stream here enters from the CONVERTED prefix (buffer_q), exactly the
    two-tower semantics; the fp stream (buffer_fp) is the untouched-donor
    activations, advanced for the next block while its fp weights are still
    resident -- the fp tower never needs to exist as a model.
    """
    rotary_kw = _rotary_kwargs(rotary, block)
    block_kw = _block_kwargs(block)
    stats: dict[str, list[torch.Tensor]] = {}
    cap: dict[str, torch.Tensor] = {}
    mode = {"q": False}
    hooks = []

    def fp_hook(path, fin):
        def hook(_m, inputs):
            if mode["q"]:
                return
            cap[path] = inputs[0].detach().reshape(-1, fin).to(torch.float32)
        return hook

    def q_hook(path, fin):
        def hook(_m, inputs):
            if not mode["q"]:
                return
            xq = inputs[0].detach().reshape(-1, fin).to(torch.float32)
            xf = cap.pop(path)
            if xf.shape != xq.shape:
                raise RuntimeError(
                    f"fp/q stream shape mismatch at {path}: {tuple(xf.shape)} "
                    f"vs {tuple(xq.shape)}")
            pair = stats.get(path)
            if pair is None:
                pair = [torch.zeros(fin, fin, dtype=torch.float32, device=xq.device),
                        torch.zeros(fin, fin, dtype=torch.float32, device=xq.device)]
                stats[path] = pair
            pair[0].addmm_(xq.t(), xq)
            pair[1].addmm_(xq.t(), xf)
        return hook

    for path in targets:
        lin = block.get_submodule(path)
        hooks.append(lin.register_forward_pre_hook(fp_hook(path, lin.in_features)))
        hooks.append(lin.register_forward_pre_hook(q_hook(path, lin.in_features)))
    out_fp: list[torch.Tensor] = []
    try:
        for bq, bf in zip(buffer_q, buffer_fp):
            q_pieces = (bq.split(micro_batch, dim=0) if micro_batch and micro_batch > 0
                        else (bq,))
            f_pieces = (bf.split(micro_batch, dim=0) if micro_batch and micro_batch > 0
                        else (bf,))
            produced = []
            for fpiece, qpiece in zip(f_pieces, q_pieces):
                hidden = fpiece.to(device)
                positions = torch.arange(hidden.shape[1], device=device).unsqueeze(0)
                pos_emb = rotary(hidden, positions, **rotary_kw)
                result = block(hidden, position_embeddings=pos_emb, **block_kw)
                result = result[0] if isinstance(result, tuple) else result
                produced.append(result.to("cpu"))
                del hidden, result

                mode["q"] = True
                hidden = qpiece.to(device)
                positions = torch.arange(hidden.shape[1], device=device).unsqueeze(0)
                pos_emb = rotary(hidden, positions, **rotary_kw)
                block(hidden, position_embeddings=pos_emb, **block_kw)
                mode["q"] = False
                del hidden
            out_fp.append(torch.cat(produced, dim=0) if len(produced) > 1 else produced[0])
    finally:
        for hook in hooks:
            hook.remove()
    if cap:
        raise RuntimeError(f"unconsumed fp captures: {sorted(cap)}")
    return stats, out_fp


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
    block_kw = _block_kwargs(block)
    out: list[torch.Tensor] = []
    for chunk in buffer:
        pieces = (chunk.split(micro_batch, dim=0) if micro_batch and micro_batch > 0
                  else (chunk,))
        produced = []
        for piece in pieces:
            hidden = piece.to(device)
            positions = torch.arange(hidden.shape[1], device=device).unsqueeze(0)
            pos_emb = rotary(hidden, positions, **rotary_kw)
            result = block(hidden, position_embeddings=pos_emb, **block_kw)
            result = result[0] if isinstance(result, tuple) else result
            if collect:
                produced.append(result.to("cpu"))
            del hidden, result
        if collect:
            out.append(torch.cat(produced, dim=0) if len(produced) > 1 else produced[0])
    return out if collect else buffer


def load_streamed_state(out_dir: str, *, require_complete: bool = True
                        ) -> dict[str, torch.Tensor]:
    """Merge verified block files for ``restore_counter_structure``.

    Incomplete output is refused by default; ``require_complete=False`` is for
    inspecting interrupted runs. Legacy complete output remains loadable with
    an explicit warning because its original donor/options cannot be verified.
    """
    manifest_path = os.path.join(out_dir, MANIFEST)
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("streamed manifest must be a JSON object")
    done = _validated_blocks(manifest, out_dir, require_complete=require_complete)
    if manifest.get("schema_version") is None:
        warnings.warn("Loading legacy streamed state without verified donor/options "
                      "provenance or block hashes; regenerate it before recovery/resume.",
                      UserWarning, stacklevel=2)
    merged: dict[str, torch.Tensor] = {}
    stack = manifest.get("stack", "model")
    for index in done:
        merged.update(_load_block_state(os.path.join(out_dir, f"block_{index:04d}.pt"),
                                        index, stack))
    return merged
