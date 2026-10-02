"""Corpus checks compare token meanings, never repository/directory names."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import types

import numpy as np
import pytest

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import Whitespace  # noqa: E402
from transformers import AutoTokenizer, PreTrainedTokenizerFast  # noqa: E402

from memory_native.donor.tokenization import (  # noqa: E402
    SYNTHETIC_PURPOSE, tokenizer_fingerprint, verify_corpus_tokenizer,
)


def _tokenizer(path, *, swap=False, chat_template=None):
    vocab = {"[UNK]": 0, "[EOS]": 1, "hello": 3 if swap else 2,
             "world": 2 if swap else 3}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                       eos_token="[EOS]", model_max_length=128,
                                       chat_template=chat_template)
    tokenizer.save_pretrained(path)
    return tokenizer


def _script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_semantic_identity_survives_save_move_and_source_alias(tmp_path):
    original_path = tmp_path / "named-donor"
    original = _tokenizer(original_path)
    moved = tmp_path / "abcdef1234567890"
    shutil.copytree(original_path, moved)
    loaded = AutoTokenizer.from_pretrained(moved, local_files_only=True)
    assert tokenizer_fingerprint(original) == tokenizer_fingerprint(loaded)
    manifest = {"tokenizer": "a/source/name/that/is/no/longer/available",
                "tokenizer_fingerprint": tokenizer_fingerprint(original)}
    assert verify_corpus_tokenizer(manifest, moved) == tokenizer_fingerprint(original)


def test_same_basename_with_different_token_ids_is_rejected(tmp_path):
    original = _tokenizer(tmp_path / "first" / "donor")
    changed_path = tmp_path / "second" / "donor"
    _tokenizer(changed_path, swap=True)
    manifest = {"tokenizer": str(tmp_path / "first" / "donor"),
                "tokenizer_fingerprint": tokenizer_fingerprint(original)}
    with pytest.raises(ValueError, match="fingerprint differs"):
        verify_corpus_tokenizer(manifest, changed_path)


def test_chat_template_and_special_ids_affect_identity(tmp_path):
    tokenizer = _tokenizer(tmp_path / "donor")
    original = tokenizer_fingerprint(tokenizer)
    tokenizer.chat_template = "{% for m in messages %}{{m['content']}}{% endfor %}"
    assert tokenizer_fingerprint(tokenizer) != original
    tokenizer.chat_template = None
    tokenizer.eos_token = "hello"
    assert tokenizer_fingerprint(tokenizer) != original


def test_same_vocab_with_different_normalizer_rules_has_different_identity(tmp_path):
    from tokenizers.normalizers import Lowercase

    tokenizer = _tokenizer(tmp_path / "donor")
    original = tokenizer_fingerprint(tokenizer)
    original_ids = tokenizer("HELLO")["input_ids"]
    tokenizer.backend_tokenizer.normalizer = Lowercase()
    assert tokenizer("HELLO")["input_ids"] != original_ids
    assert tokenizer_fingerprint(tokenizer) != original


def test_transient_backend_padding_does_not_change_semantic_identity(tmp_path):
    tokenizer = _tokenizer(tmp_path / "donor")
    original = tokenizer_fingerprint(tokenizer)
    tokenizer.backend_tokenizer.enable_padding(pad_id=0, pad_token="[UNK]")
    tokenizer.backend_tokenizer.enable_truncation(max_length=8)
    assert tokenizer_fingerprint(tokenizer) == original


def test_unknown_slow_rules_are_not_reduced_to_a_vocabulary_hash():
    with pytest.raises(ValueError, match="slow tokenizer's full rules"):
        tokenizer_fingerprint(types.SimpleNamespace(get_vocab=lambda: {"hello": 0}))


def test_legacy_source_must_resolve_match_and_warn(tmp_path):
    original_path = tmp_path / "original"
    tokenizer = _tokenizer(original_path)
    moved = tmp_path / "moved"
    shutil.copytree(original_path, moved)
    with pytest.warns(UserWarning, match="historical bytes"):
        assert verify_corpus_tokenizer({"tokenizer": str(original_path)}, moved) \
            == tokenizer_fingerprint(tokenizer)
    changed = tmp_path / "changed"
    _tokenizer(changed, swap=True)
    with pytest.raises(ValueError, match="legacy corpus tokenizer differs"):
        verify_corpus_tokenizer({"tokenizer": str(original_path)}, changed)


def test_missing_legacy_identity_does_not_silently_pass(tmp_path, monkeypatch):
    path = tmp_path / "donor"
    _tokenizer(path)
    with pytest.raises(ValueError, match="no tokenizer identity"):
        verify_corpus_tokenizer({}, path)
    real_loader = AutoTokenizer.from_pretrained

    def fail_missing(source, **kwargs):
        if source == "unavailable/source":
            raise OSError("unavailable tokenizer")
        return real_loader(source, **kwargs)

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", fail_missing)
    with pytest.raises(ValueError, match="cannot be verified"):
        verify_corpus_tokenizer({"tokenizer": "unavailable/source"}, path)


def test_synthetic_bypass_requires_flag_and_explicit_label(tmp_path):
    manifest = {"purpose": SYNTHETIC_PURPOSE}
    assert verify_corpus_tokenizer(manifest, tmp_path / "no-tokenizer", synthetic=True) is None
    with pytest.raises(ValueError, match="explicit SYNTHETIC_CALIBRATION"):
        verify_corpus_tokenizer(manifest, tmp_path / "no-tokenizer")
    with pytest.raises(ValueError, match="requires a corpus labeled"):
        verify_corpus_tokenizer({}, tmp_path / "no-tokenizer", synthetic=True)


def test_builder_records_identity_without_fetching_datasets(tmp_path, monkeypatch):
    donor = tmp_path / "donor"
    tokenizer = _tokenizer(donor)
    builder = _script("build_mix_corpus.py")
    monkeypatch.setattr(builder, "MODEL", str(donor))
    monkeypatch.setattr(builder, "SOURCES", [("en", 1.0, "fixture", {"split": "train"}, "text")])
    fake_datasets = types.SimpleNamespace(load_dataset=lambda *a, **kw:
        [{"text": "hello world " * 30} for _ in range(128)])
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    out = tmp_path / "corpus"
    monkeypatch.setattr(sys, "argv", ["build_mix_corpus.py", "--out", str(out),
                                     "--train-tokens", "64", "--val-tokens", "16"])
    builder.main()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["tokenizer_fingerprint"] == tokenizer_fingerprint(tokenizer)
    assert verify_corpus_tokenizer(manifest, donor)
    assert np.fromfile(out / "train_en.bin", dtype=np.uint32).size >= 64


def test_cli_accepts_semantic_identity_in_hash_named_snapshot(tmp_path, monkeypatch):
    donor = tmp_path / "abc1234567890"
    tokenizer = _tokenizer(donor)
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text(json.dumps({"tokenizer": "different-readable-name",
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer)}), encoding="utf-8")
    cli = _script("convert_streaming.py")
    monkeypatch.setattr(cli, "MODEL", str(donor))
    monkeypatch.setattr(cli, "DATA_DIR", str(data))
    monkeypatch.setattr(cli, "SYNTHETIC_CALIBRATION", False)
    monkeypatch.setattr(cli, "CALIB_BATCHES", 2)
    fake_mix = types.SimpleNamespace(DomainMix=lambda *a, **kw:
        types.SimpleNamespace(batch_at=lambda step, device: np.array([step])))
    monkeypatch.setitem(sys.modules, "recovery_session", fake_mix)
    batches = cli.calibration_batches()
    assert len(batches) == 2


def test_cli_random_calibration_requires_explicit_smoke_flag(tmp_path, monkeypatch):
    cli = _script("convert_streaming.py")
    monkeypatch.setattr(cli, "DATA_DIR", "")
    monkeypatch.setattr(cli, "SYNTHETIC_CALIBRATION", False)
    with pytest.raises(SystemExit, match="SYNTHETIC_CALIBRATION=1"):
        cli.calibration_batches()
    donor = tmp_path / "donor"
    donor.mkdir()
    (donor / "config.json").write_text(json.dumps({"model_type": "qwen2", "vocab_size": 4}))
    monkeypatch.setattr(cli, "MODEL", str(donor))
    monkeypatch.setattr(cli, "SYNTHETIC_CALIBRATION", True)
    monkeypatch.setattr(cli, "SEQ", 4)
    monkeypatch.setattr(cli, "CALIB_BATCHES", 2)
    assert all(b.shape == (1, 4) for b in cli.calibration_batches())
