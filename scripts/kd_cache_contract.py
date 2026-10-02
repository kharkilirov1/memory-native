"""The shared, content-bound contract for the public cached-KD scripts.

Version 1 recorded paths and dimensions only. It cannot establish that a teacher
cache belongs to the current donor/corpus, so it must be rebuilt, not relabelled.
Fingerprints deliberately include file contents (not mtime or folder names).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

SCHEMA_VERSION = 2


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for part in iter(lambda: handle.read(8 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def _file_identity(root: Path, names: list[str]) -> dict:
    return {name: {"bytes": (root / name).stat().st_size,
                   "sha256": sha256_file(root / name)} for name in sorted(names)}


def model_identity(model_dir) -> dict:
    """Bind local safetensors weights, config and available tokenizer assets."""
    from memory_native.donor.provenance import checkpoint_fingerprint
    return checkpoint_fingerprint(model_dir)


def data_identity(data_dir, seq: int) -> dict:
    root = Path(data_dir)
    with open(root / "manifest.json", encoding="utf-8") as handle:
        meta = json.load(handle)
    domains = meta.get("domains")
    if not isinstance(domains, dict) or not domains:
        raise ValueError("corpus manifest must contain nonempty domains")
    if meta.get("dtype", "uint32") != "uint32":
        raise ValueError("DomainMix requires uint32 token bins")
    names, total_share = ["manifest.json"], 0.0
    for name, info in domains.items():
        if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
            raise ValueError("invalid corpus domain name")
        share = info.get("share") if isinstance(info, dict) else None
        if isinstance(share, bool) or not isinstance(share, (int, float)) or not math.isfinite(share) or share < 0:
            raise ValueError(f"invalid sampling share for domain {name}")
        total_share += share
        for split in ("train", "val"):
            file = f"{split}_{name}.bin"
            path = root / file
            if split == "val" and not path.exists():
                continue
            size = path.stat().st_size
            if size % 4 or size < seq * 4:
                raise ValueError(f"{file} must contain at least one complete uint32 sequence")
            names.append(file)
    if total_share <= 0 or not math.isfinite(total_share):
        raise ValueError("corpus sampling shares must have a finite positive sum")
    # Domain order affects numpy's categorical draws; retain it explicitly.
    return {"domain_order": list(domains), "files": _file_identity(root, names)}


def _integer(meta: dict, key: str, minimum=1) -> int:
    value = meta.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"cache manifest {key} must be an integer >= {minimum}")
    return value


def load_cache_manifest(cache_dir) -> dict:
    root = Path(cache_dir)
    with open(root / "cache_manifest.json", encoding="utf-8") as handle:
        meta = json.load(handle)
    if not isinstance(meta, dict) or meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unverified/legacy teacher cache: rebuild with kd_teacher_cache.py (schema v2)")
    for key in ("steps", "batch", "seq", "topk", "shard_steps", "vocab_size"):
        _integer(meta, key, 2 if key == "seq" else 1)
    _integer(meta, "seed", 0)
    _integer(meta, "num_blocks", 0)
    if not isinstance(meta.get("synthetic_calibration", False), bool):
        raise ValueError("cache synthetic_calibration must be boolean")
    if meta["topk"] > meta["vocab_size"]:
        raise ValueError("cache topk exceeds donor vocabulary")
    if meta.get("loss_contract") != "renormalized_teacher_topk_vs_full_vocab_student":
        raise ValueError("unsupported cached KD normalization contract")
    identities = meta.get("identities")
    if not isinstance(identities, dict) or not all(isinstance(identities.get(k), dict) and identities[k] for k in ("model", "data")):
        raise ValueError("cache manifest lacks donor/corpus content identities")
    shards = meta.get("shards")
    expected_count = math.ceil(meta["steps"] / meta["shard_steps"])
    if not isinstance(shards, list) or len(shards) != expected_count:
        raise ValueError("cache shard count does not match steps/shard_steps")
    for sid, shard in enumerate(shards):
        file = f"cache_{sid:04d}.pt"
        rows = min(meta["shard_steps"], meta["steps"] - sid * meta["shard_steps"]) * meta["batch"]
        if not isinstance(shard, dict) or shard.get("file") != file or shard.get("rows") != rows:
            raise ValueError(f"invalid cache shard entry {sid}")
        digest = shard.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"invalid sha256 for {file}")
        if not (root / file).is_file():
            raise ValueError(f"missing cache shard {file}")
    return meta


def validate_cache_context(meta: dict, *, model_dir, data_dir, num_blocks: int,
                           steps: int, batch=None, seq=None, seed=None) -> None:
    if not 0 < steps <= meta["steps"]:
        raise ValueError("requested STEPS must be positive and no longer than the cache")
    if num_blocks != meta["num_blocks"]:
        raise ValueError("NUM_BLOCKS differs from cached teacher depth")
    for key, value in (("batch", batch), ("seq", seq), ("seed", seed)):
        if value is not None and value != meta[key]:
            raise ValueError(f"requested {key.upper()} differs from teacher cache")
    if model_identity(model_dir) != meta["identities"]["model"]:
        raise ValueError("teacher cache donor content differs from MODEL")
    if data_identity(data_dir, meta["seq"]) != meta["identities"]["data"]:
        raise ValueError("teacher cache corpus content differs from DATA_DIR")


def validate_warm_source(state_dir, cache_meta: dict, *, group=None, C=None) -> dict:
    """The warm conversion must attest to the same donor as the teacher cache."""
    path = Path(state_dir) / "manifest.json"
    if not path.is_file():
        raise ValueError("cached recovery requires a streamed warm-state directory with provenance; regenerate it with convert_streaming.py")
    with open(path, encoding="utf-8") as handle:
        meta = json.load(handle)
    if meta.get("schema_version") != 2 or meta.get("source_fingerprint") != cache_meta["identities"]["model"]:
        raise ValueError("warm conversion donor provenance is missing or differs from cached teacher; regenerate conversion")
    if isinstance(meta.get("group"), bool) or not isinstance(meta.get("group"), int) or meta["group"] <= 0 or isinstance(meta.get("C"), bool) or not isinstance(meta.get("C"), int) or not 1 <= meta["C"] <= 11 or meta.get("kind") != "counter_packed":
        raise ValueError("cached recovery requires positive group, packed C in 1..11, kind=counter_packed")
    if (group is not None and group != meta["group"]) or (C is not None and C != meta["C"]):
        raise ValueError("requested GROUP/C differs from warm conversion format")
    total = meta.get("blocks_total")
    done = meta.get("blocks_done")
    required = cache_meta["num_blocks"] or total
    if isinstance(total, bool) or not isinstance(total, int) or total <= 0 or required > total or not isinstance(done, list) or not set(range(required)).issubset(done):
        raise ValueError("warm conversion is incomplete for cached teacher depth")
    return meta


class ShardedCache:
    def __init__(self, cache_dir: str, batch: int, *, meta=None):
        self.dir = Path(cache_dir)
        self.meta = load_cache_manifest(cache_dir) if meta is None else meta
        if batch != self.meta["batch"]:
            raise ValueError("batch differs from cache manifest")
        self.rows_per_step = batch
        self.shard_steps = self.meta["shard_steps"]
        self._shard_id, self._shard = -1, None
        # Verify the complete on-disk cache before loading a large student.
        for entry in self.meta["shards"]:
            if sha256_file(self.dir / entry["file"]) != entry["sha256"]:
                raise ValueError(f"cache shard checksum mismatch: {entry['file']}")

    def _load(self, sid: int):
        entry = self.meta["shards"][sid]
        shard = torch.load(self.dir / entry["file"], map_location="cpu", weights_only=True)
        if not isinstance(shard, dict) or not all(torch.is_tensor(shard.get(k)) for k in ("idx", "val", "tokens")):
            raise ValueError("cache shard must contain idx, val and tokens tensors")
        idx, val, tokens = shard["idx"], shard["val"], shard["tokens"]
        shape = (entry["rows"], self.meta["seq"], self.meta["topk"])
        if tuple(idx.shape) != shape or tuple(val.shape) != shape or tuple(tokens.shape) != shape[:2]:
            raise ValueError("cache shard tensor shape differs from manifest")
        if idx.dtype not in (torch.int32, torch.int64) or tokens.dtype not in (torch.int32, torch.int64) or not val.is_floating_point():
            raise ValueError("cache shard has invalid tensor dtypes")
        if not torch.isfinite(val).all():
            raise ValueError("cached teacher logits must be finite")
        for name, values in (("indices", idx), ("tokens", tokens)):
            if values.min() < 0 or values.max() >= self.meta["vocab_size"]:
                raise ValueError(f"cache {name} outside donor vocabulary")
        ordered = idx.sort(dim=-1).values
        if (ordered[..., 1:] == ordered[..., :-1]).any():
            raise ValueError("cached topk contains duplicate token indices")
        return shard

    def step(self, step: int, device, *, input_ids: torch.Tensor):
        if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step < self.meta["steps"]:
            raise ValueError("step outside cached stream")
        sid = step // self.shard_steps
        if sid != self._shard_id:
            self._shard = self._load(sid)
            self._shard_id = sid
        r0 = (step - sid * self.shard_steps) * self.rows_per_step
        sl = slice(r0, r0 + self.rows_per_step)
        tokens = self._shard["tokens"][sl]
        if not torch.equal(tokens.long(), input_ids.detach().cpu().long()):
            raise ValueError("cached teacher input tokens differ from deterministic training batch")
        return self._shard["idx"][sl].to(device), self._shard["val"][sl].to(device)
