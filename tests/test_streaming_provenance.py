"""A resumed conversion must never mix donor bytes, calibration, or recipes."""
import json
from pathlib import Path
import shutil

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from memory_native.donor.provenance import (  # noqa: E402
    atomic_torch_save, checkpoint_fingerprint, file_fingerprint,
)
from memory_native.donor.streaming import convert_streaming, load_streamed_state  # noqa: E402

OPTIONS = dict(group=8, C=11, refine_iters=0, micro_batch=1, progress=False)


@pytest.fixture(scope="module")
def witness(tmp_path_factory):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    root = tmp_path_factory.mktemp("streamed-provenance")
    torch.manual_seed(23)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=2,
                         num_key_value_heads=1, max_position_embeddings=32)
    donor = root / "donor"
    Qwen2ForCausalLM(config).eval().save_pretrained(donor, safe_serialization=True)
    ids = [torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])]
    state = root / "state"
    convert_streaming(donor, ids, state, **OPTIONS)
    return donor, ids, state


def _copy_state(witness, tmp_path):
    return Path(shutil.copytree(witness[2], tmp_path / "state"))


def _read_manifest(out):
    return json.loads((out / "manifest.json").read_text(encoding="utf-8"))


def _write_manifest(out, manifest):
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_manifest_binds_checkpoint_and_exact_calibration(witness):
    donor, ids, out = witness
    manifest = _read_manifest(out)
    assert manifest["schema_version"] == 2
    assert manifest["source_fingerprint"] == checkpoint_fingerprint(donor)
    assert manifest["calibration_fingerprint"]["tokens"] == ids[0].numel()
    assert manifest["conversion_options"]["solver"]["grid"] == "sym"
    assert set(manifest["block_files"]) == {"0", "1"}


def test_unchanged_snapshot_can_move(witness, tmp_path):
    donor, ids, _ = witness
    moved = Path(shutil.copytree(donor, tmp_path / "moved-donor"))
    assert checkpoint_fingerprint(moved) == checkpoint_fingerprint(donor)
    report = convert_streaming(moved, ids, _copy_state(witness, tmp_path), **OPTIONS)
    assert report.blocks_resumed == 2 and report.blocks_converted == 0


@pytest.mark.parametrize("changed", [
    {"group": 4}, {"C": 10}, {"cascade": False}, {"micro_batch": 2},
    {"grid": "itf"}, {"dtype": torch.float16}, {"extra_skip": ["self_attn.q_proj"]},
    {"stats_scope": "group"}, {"decimation": 2},
])
def test_changed_conversion_recipe_refuses_resume(witness, tmp_path, changed):
    donor, ids, _ = witness
    out = _copy_state(witness, tmp_path)
    before = (out / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="conversion_options mismatch"):
        convert_streaming(donor, ids, out, **{**OPTIONS, **changed})
    assert (out / "manifest.json").read_bytes() == before


def test_changed_tokens_refuse_resume(witness, tmp_path):
    donor, ids, _ = witness
    changed = [ids[0].clone()]
    changed[0][0, -1] += 1
    with pytest.raises(ValueError, match="calibration_fingerprint mismatch"):
        convert_streaming(donor, changed, _copy_state(witness, tmp_path), **OPTIONS)


@pytest.mark.parametrize("bad_ids", [[], [torch.zeros(1, 4)],
                                      [torch.tensor([[-1, 1]])],
                                      [torch.tensor([[1, 32]])]])
def test_invalid_calibration_is_rejected_before_checkpoint_write(witness, tmp_path, bad_ids):
    out = tmp_path / "invalid-state"
    with pytest.raises(ValueError, match="calibration"):
        convert_streaming(witness[0], bad_ids, out, **OPTIONS)
    assert not (out / "manifest.json").exists()


@pytest.mark.parametrize("asset", ["config.json", "model.safetensors"])
def test_changed_donor_bytes_refuse_resume(witness, tmp_path, asset):
    donor, ids, _ = witness
    changed = Path(shutil.copytree(donor, tmp_path / "changed-donor"))
    if asset == "config.json":
        config = json.loads((changed / asset).read_text(encoding="utf-8"))
        config["provenance_test"] = "changed"
        (changed / asset).write_text(json.dumps(config), encoding="utf-8")
    else:
        from safetensors.torch import load_file, save_file

        state = load_file(changed / asset)
        key = next(iter(state))
        state[key] = state[key] + 0.25
        save_file(state, changed / asset, metadata={"format": "pt"})
    with pytest.raises(ValueError, match="source_fingerprint mismatch"):
        convert_streaming(changed, ids, _copy_state(witness, tmp_path), **OPTIONS)


@pytest.mark.parametrize("damage", ["missing", "corrupted"])
def test_missing_or_corrupted_block_refuses_load_and_resume(witness, tmp_path, damage):
    donor, ids, _ = witness
    out = _copy_state(witness, tmp_path)
    block = out / "block_0000.pt"
    if damage == "missing":
        block.unlink()
    else:
        with block.open("ab") as handle:
            handle.write(b"corruption")
    for action in (lambda: load_streamed_state(out),
                   lambda: convert_streaming(donor, ids, out, **OPTIONS)):
        with pytest.raises(ValueError, match="missing checkpoint|integrity mismatch"):
            action()


def test_partial_conversion_requires_explicit_inspection(witness, tmp_path):
    out = _copy_state(witness, tmp_path)
    manifest = _read_manifest(out)
    manifest["blocks_done"] = [0]
    _write_manifest(out, manifest)
    with pytest.raises(ValueError, match="incomplete streamed conversion"):
        load_streamed_state(out)
    partial = load_streamed_state(out, require_complete=False)
    assert partial and all(k.startswith("model.layers.0.") for k in partial)


def test_legacy_load_warns_but_resume_refuses(witness, tmp_path):
    donor, ids, _ = witness
    out = _copy_state(witness, tmp_path)
    manifest = _read_manifest(out)
    for key in ("schema_version", "source_fingerprint", "calibration_fingerprint",
                "conversion_options", "block_files"):
        manifest.pop(key)
    _write_manifest(out, manifest)
    with pytest.warns(UserWarning, match="legacy streamed state"):
        assert load_streamed_state(out)
    with pytest.raises(ValueError, match="legacy streamed manifest"):
        convert_streaming(donor, ids, out, **OPTIONS)


@pytest.mark.parametrize("damage", ["wrong_prefix", "nonfinite"])
def test_structurally_invalid_tensor_state_refuses_load(witness, tmp_path, damage):
    donor, ids, _ = witness
    out = _copy_state(witness, tmp_path)
    block = out / "block_0000.pt"
    state = torch.load(block, weights_only=True)
    if damage == "wrong_prefix":
        state["model.layers.1.injected"] = torch.zeros(1)
    else:
        key = next(k for k, v in state.items() if v.is_floating_point())
        state[key] = torch.full_like(state[key], float("nan"))
    torch.save(state, block)
    manifest = _read_manifest(out)
    manifest["block_files"]["0"].update(file_fingerprint(block))
    _write_manifest(out, manifest)
    with pytest.raises(ValueError, match="invalid tensor state|nonfinite tensor state"):
        load_streamed_state(out)
    with pytest.raises(ValueError, match="invalid tensor state|nonfinite tensor state"):
        convert_streaming(donor, ids, out, **OPTIONS)


def test_atomic_checkpoint_failure_preserves_previous_file(tmp_path, monkeypatch):
    target = tmp_path / "block.pt"
    atomic_torch_save(target, {"value": torch.ones(1)})
    original = target.read_bytes()

    def fail_after_partial_write(value, handle):
        handle.write(b"partial")
        raise OSError("injected interrupted save")

    monkeypatch.setattr(torch, "save", fail_after_partial_write)
    with pytest.raises(OSError, match="interrupted save"):
        atomic_torch_save(target, {"value": torch.zeros(1)})
    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]


def test_group_update_options_are_retained_in_streamed_counters(witness, tmp_path):
    donor, ids, _ = witness
    out = tmp_path / "group-state"
    convert_streaming(donor, ids, out, stats_scope="group", decimation=2, **OPTIONS)
    state = load_streamed_state(out)
    stats = [value for key, value in state.items() if key.endswith(".v")]
    assert stats and all(value.ndim == 2 and value.shape[-1] >= 2 for value in stats)
