"""Bind corpus token IDs to tokenizer semantics, independently of source paths."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import warnings

from .provenance import canonical_json

SYNTHETIC_PURPOSE = "synthetic plumbing only; no language-quality evidence"


def tokenizer_fingerprint(tokenizer) -> dict:
    """Hash tokenization rules, IDs, special-token behavior, and chat rendering.

    The backend JSON includes vocabulary, merges, normalizer, pre-tokenizer,
    post-processor and added-token flags. Source names/cache paths do not affect
    the result. A fast tokenizer backend is required: a vocabulary alone cannot
    identify a slow tokenizer's segmentation rules or Python preprocessing.
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise ValueError("cannot fingerprint this slow tokenizer's full rules; "
                         "load and use its supported fast tokenizer to build the corpus")
    rules = json.loads(backend.to_str())
    # These are transient per-call settings. The builder uses tokenizer(text)
    # without padding/truncation and transformers resets them for each call.
    rules.pop("padding", None)
    rules.pop("truncation", None)
    special_ids = {
        name: getattr(tokenizer, f"{name}_token_id", None)
        for name in ("bos", "eos", "unk", "sep", "pad", "cls", "mask")
    }
    special_ids["additional"] = list(getattr(tokenizer, "additional_special_tokens_ids", []) or [])
    semantics = {
        "schema_version": 1, "backend": rules, "special_ids": special_ids,
        "chat_template": getattr(tokenizer, "chat_template", None),
        "padding_side": getattr(tokenizer, "padding_side", "right"),
        "truncation_side": getattr(tokenizer, "truncation_side", "right"),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "split_special_tokens": getattr(tokenizer, "split_special_tokens", False),
    }
    return {"schema_version": 1, "vocab_size": len(tokenizer),
            "sha256": hashlib.sha256(canonical_json(semantics)).hexdigest()}


def verify_corpus_tokenizer(manifest: dict, model_path, *, synthetic: bool = False):
    """Check recorded tokenizer semantics against the local donor tokenizer.

    Legacy corpora have no historical identity. Their declared tokenizer source
    must still resolve and match the current donor; a warning makes that weaker
    evidence explicit. Synthetic bypass requires both a flag and a labeled toy
    corpus, so the flag cannot silently skip checks on ordinary language data.
    """
    if not isinstance(manifest, dict):
        raise ValueError("corpus manifest must be a JSON object")
    labeled_synthetic = manifest.get("purpose") == SYNTHETIC_PURPOSE
    if synthetic:
        if not labeled_synthetic:
            raise ValueError("SYNTHETIC_CALIBRATION=1 requires a corpus labeled "
                             f"purpose={SYNTHETIC_PURPOSE!r}")
        return None
    if labeled_synthetic:
        raise ValueError("synthetic corpus requires explicit SYNTHETIC_CALIBRATION=1; "
                         "it provides no language-quality evidence")

    from transformers import AutoTokenizer

    try:
        donor = AutoTokenizer.from_pretrained(model_path, local_files_only=Path(model_path).is_dir())
        current = tokenizer_fingerprint(donor)
    except Exception as exc:
        raise ValueError(f"cannot verify donor tokenizer at {str(model_path)!r}: {exc}") from exc
    recorded = manifest.get("tokenizer_fingerprint")
    if recorded is not None:
        if (not isinstance(recorded, dict) or recorded.get("schema_version") != 1
                or recorded != current):
            raise ValueError("corpus tokenizer fingerprint differs from donor; "
                             "rebuild the corpus with the donor tokenizer")
        return current

    declared = manifest.get("tokenizer")
    if not isinstance(declared, str) or not declared:
        raise ValueError("legacy corpus has no tokenizer identity/source; rebuild it")
    try:
        original = AutoTokenizer.from_pretrained(declared,
                                                 local_files_only=Path(declared).is_dir())
        legacy_identity = tokenizer_fingerprint(original)
    except Exception as exc:
        raise ValueError(f"legacy corpus tokenizer source {declared!r} cannot be verified; "
                         "rebuild the corpus") from exc
    if legacy_identity != current:
        raise ValueError("legacy corpus tokenizer differs from donor; rebuild the corpus "
                         "with the donor tokenizer")
    warnings.warn("Legacy corpus has no saved tokenizer fingerprint: matching current "
                  "tokenizer sources cannot verify which historical bytes built these "
                  "bins. Rebuild the corpus to record that evidence.", UserWarning,
                  stacklevel=2)
    return current
