"""Char-level corpus loading with an offline-safe fallback, so examples run anywhere with no
network and no engine. Resolution: explicit path -> cached file -> download -> deterministic
synthetic corpus (and it prints which source it used)."""
from __future__ import annotations

import os
import random
import urllib.request

import torch

__all__ = ["load_corpus", "get_batch", "synthetic_corpus", "TINY_SHAKESPEARE_URL"]

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def synthetic_corpus(n_chars: int = 200_000, seed: int = 1234) -> str:
    """Deterministic, mildly structured char stream with real next-token signal. NOT
    shakespeare -- absolute losses differ; use a real corpus via data_path for a publishable
    number."""
    rng = random.Random(seed)
    letters = "abcdefghijklmnopqrstuvwxyz"
    words = ["".join(rng.choice(letters) for _ in range(rng.randint(2, 8))) for _ in range(96)]
    out: list[str] = []
    total = 0                       # running length; re-summing the list each step is O(n^2)
    while total < n_chars:
        w = rng.choice(words)
        r = rng.random()
        sep = "\n" if r < 0.08 else (", " if r < 0.2 else (". " if r < 0.32 else " "))
        out.append(w); out.append(sep)
        total += len(w) + len(sep)
    return "".join(out)


def load_corpus(device, data_path: str | None = None, cache_dir: str | None = None,
                download: bool = True, timeout: float = 10.0):
    """Returns (train_ids, val_ids, vocab_size). 90/10 split.

    Resolution order: explicit data_path -> cached file -> download (if enabled) -> synthetic.
    The download is bounded by `timeout` seconds and any failure falls back to the deterministic
    synthetic corpus, so this never hangs offline/behind a firewall. Pass download=False to skip
    the network entirely (cache-or-synthetic only)."""
    cache = os.path.join(cache_dir or os.getcwd(), "tinyshakespeare.txt")
    candidates = ([data_path] if data_path else []) + [cache]

    text = source = None
    for path in candidates:
        if path and os.path.exists(path):
            text, source = open(path, encoding="utf-8").read(), path
            break

    if text is None:
        if data_path:
            raise FileNotFoundError(f"data_path not found: {data_path}")
        if not download:
            text, source = synthetic_corpus(), "SYNTHETIC (download disabled)"
        else:
            try:  # bounded by timeout so a stalled/blocked connection cannot hang the run
                with urllib.request.urlopen(TINY_SHAKESPEARE_URL, timeout=timeout) as resp:
                    raw = resp.read().decode("utf-8")
                with open(cache, "w", encoding="utf-8") as fh:
                    fh.write(raw)
                text, source = raw, cache + " (downloaded)"
            except Exception as exc:  # offline / blocked / slow: deterministic fallback
                text, source = synthetic_corpus(), f"SYNTHETIC fallback (download failed: {exc})"

    print(f"corpus source: {source}")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n].to(device), data[n:].to(device), len(chars)


def get_batch(data, block: int, batch: int, device, generator=None):
    ix = torch.randint(0, len(data) - block - 1, (batch,), generator=generator)
    x = torch.stack([data[i:i + block] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block] for i in ix])
    return x.to(device), y.to(device)
