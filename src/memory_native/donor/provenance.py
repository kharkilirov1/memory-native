"""Content identities for local donor snapshots and durable file writes.

Identities contain relative names and file bytes, never the snapshot's absolute
path, so moving an unchanged snapshot does not invalidate a conversion/cache.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


def canonical_json(value) -> bytes:
    """Serialize JSON data deterministically, rejecting NaN/Inf options."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def file_fingerprint(path) -> dict:
    """Hash a file in bounded memory, returning its length and SHA-256."""
    digest = hashlib.sha256()
    count = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            count += len(chunk)
    return {"bytes": count, "sha256": digest.hexdigest()}


def checkpoint_fingerprint(model_path) -> dict:
    """Identify config, referenced safetensors shards, and tokenizer assets.

    A local snapshot is required. Shard names must stay inside that snapshot;
    Hugging Face's ordinary cache symlinks to its blob store remain supported.
    """
    root = Path(model_path)
    if not root.is_dir():
        raise FileNotFoundError(f"local donor snapshot required: {root}")
    names = {"config.json"}
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        names.add(index_path.name)
        with index_path.open(encoding="utf-8") as handle:
            weight_map = json.load(handle).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("safetensors index has no nonempty weight_map")
        for name in weight_map.values():
            if (not isinstance(name, str) or not name.endswith(".safetensors")
                    or Path(name).is_absolute() or ".." in Path(name).parts):
                raise ValueError(f"invalid donor shard path: {name!r}")
            names.add(name)
    else:
        names.add("model.safetensors")
    # Tokenizer semantics also belong to cache provenance when assets exist.
    for pattern in ("tokenizer*.json", "special_tokens_map.json",
                    "added_tokens.json", "vocab.json", "vocab.txt",
                    "merges.txt", "*.model"):
        names.update(p.name for p in root.glob(pattern) if p.is_file())
    files = [{"path": name, **file_fingerprint(root / name)}
             for name in sorted(names)]
    identity = {"schema_version": 1, "files": files}
    return {**identity, "sha256": hashlib.sha256(canonical_json(identity)).hexdigest()}


def atomic_write_json(path, value) -> None:
    """Replace a JSON file only after its complete contents are durable."""
    _atomic_write(path, lambda handle: handle.write(canonical_json(value) + b"\n"))


def atomic_torch_save(path, value) -> None:
    """Atomically replace a local tensor checkpoint."""
    import torch

    _atomic_write(path, lambda handle: torch.save(value, handle))


def _atomic_write(path, write) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp",
                                     dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
