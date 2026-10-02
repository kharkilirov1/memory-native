"""CPU regressions for cache identity, KD math and warm-governed selection."""
import json
import math
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import kd_cached_recovery as recovery
import kd_teacher_cache as teacher_cache
from kd_cache_contract import (
    ShardedCache, data_identity, load_cache_manifest, model_identity, sha256_file,
    validate_cache_context, validate_warm_source,
)
from recovery_session import DomainMix


def write_corpus(root, *, vocab=17):
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"dtype": "uint32", "purpose": "synthetic plumbing only; no language-quality evidence", "domains": {"tiny": {"share": 1}}}))
    for split in ("train", "val"):
        (np.arange(64, dtype=np.uint32) % vocab).tofile(root / f"{split}_tiny.bin")


@pytest.fixture
def synthetic_cache(tmp_path):
    model, data, cache = (tmp_path / name for name in ("donor", "data", "cache"))
    model.mkdir()
    (model / "config.json").write_text('{"vocab_size": 17}')
    (model / "model.safetensors").write_bytes(b"synthetic fingerprint-only weights")
    write_corpus(data)
    cache.mkdir()
    mix = DomainMix(data, seq=4, batch=2)
    batches = [mix.batch_at(step, "cpu") for step in range(3)]
    shards = []
    for sid, rows in enumerate((4, 2)):
        file = f"cache_{sid:04d}.pt"
        shard = {"idx": torch.arange(3, dtype=torch.int32).expand(rows, 4, 3).clone(),
                 "val": torch.randn(rows, 4, 3).half(),
                 "tokens": torch.cat(batches[sid * 2:sid * 2 + 2]).int()}
        torch.save(shard, cache / file)
        shards.append({"file": file, "rows": rows, "sha256": sha256_file(cache / file)})
    meta = {"schema_version": 2, "steps": 3, "batch": 2, "seq": 4, "topk": 3,
            "seed": 0, "num_blocks": 0, "shard_steps": 2, "vocab_size": 17,
            "loss_contract": "renormalized_teacher_topk_vs_full_vocab_student",
            "identities": {"model": model_identity(model), "data": data_identity(data, 4)},
            "shards": shards}
    (cache / "cache_manifest.json").write_text(json.dumps(meta))
    return model, data, cache, meta, batches


@pytest.mark.parametrize("temperature", [0.7, 1.0, 2.0])
def test_chunked_kd_ce_values_and_gradients_match_full_vocab_reference(temperature):
    torch.manual_seed(2)
    logits = torch.randn(2, 5, 19, requires_grad=True)
    idx = torch.rand(2, 5, 19).topk(7, dim=-1).indices
    teacher = torch.randn(2, 5, 7) * 8
    targets = torch.randint(19, (2, 4))
    kd, ce = recovery.kd_and_ce_losses(logits, idx, teacher, targets, temperature, chunk_size=3)
    (kd + 0.3 * ce).backward()
    reference = logits.detach().clone().requires_grad_()
    logp = F.log_softmax(teacher / temperature, dim=-1)
    logq = F.log_softmax(reference / temperature, dim=-1).gather(-1, idx)
    expected_kd = (logp.exp() * (logp - logq)).sum(-1).mean() * temperature**2
    expected_ce = F.cross_entropy(reference[:, :-1].reshape(-1, 19), targets.reshape(-1))
    (expected_kd + 0.3 * expected_ce).backward()
    torch.testing.assert_close(kd, expected_kd)
    torch.testing.assert_close(ce, expected_ce)
    torch.testing.assert_close(logits.grad, reference.grad, atol=2e-7, rtol=2e-5)


def test_bf16_large_common_logit_offset_preserves_fp32_normalization():
    # bf16 logsumexp rounds 1024 + log(V) back to 1024, losing both losses.
    logits = torch.full((1, 3, 31), 1024.0, dtype=torch.bfloat16, requires_grad=True)
    idx = torch.arange(5).expand(1, 3, 5)
    kd, ce = recovery.kd_and_ce_losses(logits, idx, torch.zeros(1, 3, 5), torch.zeros(1, 2, dtype=torch.long), 1.0, chunk_size=7)
    assert kd.item() == pytest.approx(math.log(31 / 5), abs=1e-4)
    assert ce.item() == pytest.approx(math.log(31), abs=1e-4)
    (kd + ce).backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


@pytest.mark.parametrize("problem", ["temperature", "indices", "targets", "teacher_nan", "student_nan", "shape"])
def test_loss_rejects_invalid_inputs_before_backward(problem):
    logits = torch.zeros(1, 3, 7, requires_grad=True)
    idx, val, targets, temperature = torch.arange(3).expand(1, 3, 3), torch.zeros(1, 3, 3), torch.zeros(1, 2, dtype=torch.long), 2.0
    if problem == "temperature":
        temperature = 0.0
    elif problem == "indices":
        idx = idx.clone(); idx[0, 0, 0] = 7
    elif problem == "targets":
        targets[0, 0] = -1
    elif problem == "teacher_nan":
        val[0, 0, 0] = float("nan")
    elif problem == "student_nan":
        logits = logits.detach(); logits[0, 0, 0] = float("nan")
    else:
        val = val[:, :-1]
    with pytest.raises(ValueError):
        recovery.kd_and_ce_losses(logits, idx, val, targets, temperature)


def test_cache_content_and_exact_step_tokens_are_verified(synthetic_cache):
    model, data, path, meta, batches = synthetic_cache
    loaded = load_cache_manifest(path)
    validate_cache_context(loaded, model_dir=model, data_dir=data, num_blocks=0, steps=3)
    cache = ShardedCache(path, 2)
    for step, tokens in enumerate(batches):
        idx, val = cache.step(step, "cpu", input_ids=tokens)
        assert idx.shape == val.shape == (2, 4, 3)
    with pytest.raises(ValueError, match="input tokens"):
        cache.step(0, "cpu", input_ids=batches[0] + 1)
    with pytest.raises(ValueError, match="outside cached stream"):
        cache.step(3, "cpu", input_ids=batches[0])
    with pytest.raises(ValueError, match="BATCH"):
        validate_cache_context(meta, model_dir=model, data_dir=data, num_blocks=0, steps=3, batch=1)
    with pytest.raises(ValueError, match="NUM_BLOCKS"):
        validate_cache_context(meta, model_dir=model, data_dir=data, num_blocks=1, steps=3)
    (model / "model.safetensors").write_bytes(b"different donor at the same path")
    with pytest.raises(ValueError, match="donor content"):
        validate_cache_context(meta, model_dir=model, data_dir=data, num_blocks=0, steps=3)


def test_changed_corpus_and_corrupt_shard_fail_closed(synthetic_cache):
    model, data, path, meta, batches = synthetic_cache
    (data / "train_tiny.bin").write_bytes(np.zeros(64, dtype=np.uint32).tobytes())
    with pytest.raises(ValueError, match="corpus content"):
        validate_cache_context(meta, model_dir=model, data_dir=data, num_blocks=0, steps=3)
    with open(path / "cache_0000.pt", "ab") as handle:
        handle.write(b"changed cache")
    with pytest.raises(ValueError, match="checksum mismatch"):
        ShardedCache(path, 2)


@pytest.mark.parametrize("problem", ["shape", "index", "tokens", "nan", "duplicate"])
def test_shard_semantics_are_checked_even_with_matching_hash(synthetic_cache, problem):
    model, data, path, meta, batches = synthetic_cache
    file = path / "cache_0000.pt"
    shard = torch.load(file, weights_only=True)
    if problem == "shape":
        shard["idx"] = shard["idx"][:1]
    elif problem == "index":
        shard["idx"][0, 0, 0] = 17
    elif problem == "tokens":
        shard["tokens"][0, 0] = -1
    elif problem == "nan":
        shard["val"][0, 0, 0] = float("nan")
    else:
        shard["idx"][0, 0, 0] = shard["idx"][0, 0, 1]
    torch.save(shard, file)
    meta["shards"][0]["sha256"] = sha256_file(file)
    (path / "cache_manifest.json").write_text(json.dumps(meta))
    cache = ShardedCache(path, 2)
    with pytest.raises(ValueError):
        cache.step(0, "cpu", input_ids=batches[0])


def test_legacy_cache_and_missing_warm_provenance_are_not_assumed_valid(synthetic_cache):
    model, data, path, meta, _ = synthetic_cache
    legacy = {k: v for k, v in meta.items() if k != "schema_version"}
    (path / "cache_manifest.json").write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match="rebuild"):
        load_cache_manifest(path)
    with pytest.raises(ValueError, match="provenance"):
        validate_warm_source(model, meta)


def test_failed_12b_metric_curve_keeps_explicit_warm_artifact(tmp_path):
    selection = recovery.WarmSelection(3.2585)
    for metric in (10.7481, 10.5228, 10.8415):
        assert not selection.improves(metric)
        with pytest.raises(ValueError, match="does not improve"):
            selection.accept(100, metric)
    recovery.write_selection(tmp_path, selection, model="donor", state_dir="warm", format={"C": 11})
    summary = json.loads((tmp_path / "selection.json").read_text())
    artifact = json.loads((tmp_path / "selected_artifact.json").read_text())
    assert summary["accepted_kd"] is False
    assert summary["selected_step"] == 0
    assert summary["selected_metric"] == 3.2585
    assert artifact["artifact_type"] == "warm_conversion_state"
    assert (tmp_path / "USE_WARM_STATE.txt").exists()
    assert not (tmp_path / "best.pt").exists()


def test_improving_candidate_requires_real_checkpoint_and_switches_marker(tmp_path):
    selection = recovery.WarmSelection(3.0, min_improvement=0.01)
    assert not selection.improves(2.995)
    recovery.write_selection(tmp_path, selection, model="donor", state_dir="warm", format={})
    recovery.save_best_checkpoint({"step": 2, "student": {"w": torch.ones(3)}}, tmp_path, tmp_path / "stage")
    selection.accept(2, 2.9)
    recovery.write_selection(tmp_path, selection, model="donor", state_dir="warm", format={})
    assert (tmp_path / "KD_ACCEPTED.txt").exists()
    assert not (tmp_path / "USE_WARM_STATE.txt").exists()
    assert json.loads((tmp_path / "selection.json").read_text())["selected_step"] == 2


def test_failed_cross_disk_copy_preserves_previously_selected_checkpoint(tmp_path, monkeypatch):
    old = {"step": 1, "student": {"w": torch.ones(3)}}
    torch.save(old, tmp_path / "best.pt")

    def fail_copy(src, dst):
        Path(dst).write_bytes(b"partial new checkpoint")
        raise OSError("simulated full output volume")

    monkeypatch.setattr(recovery.shutil, "copyfile", fail_copy)
    with pytest.raises(OSError, match="full output"):
        recovery.save_best_checkpoint({"step": 2}, tmp_path, tmp_path / "stage")
    assert torch.load(tmp_path / "best.pt", weights_only=True)["step"] == 1
    assert not (tmp_path / "best.pt.tmp").exists()


def test_teacher_logits_include_head_bias_and_gemma_softcap():
    hidden, weight, bias = torch.tensor([[2.0, -1.0]]), torch.tensor([[5.0, 3.0], [-2.0, 4.0]]), torch.tensor([1.0, -1.0])
    config = SimpleNamespace(text_config=SimpleNamespace(final_logit_softcapping=3.0))
    expected = torch.tanh(F.linear(hidden, weight, bias) / 3.0) * 3.0
    torch.testing.assert_close(teacher_cache.final_teacher_logits(hidden, weight, bias, config), expected)


def test_warm_recorded_qwen_bias_materializes_instead_of_donor_bias(tmp_path):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    from memory_native.donor.provenance import file_fingerprint
    from memory_native.donor.streaming import convert_streaming

    model, warm = tmp_path / "donor", tmp_path / "warm"
    model.mkdir()
    config = transformers.Qwen2Config(vocab_size=17, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64)
    donor = transformers.Qwen2ForCausalLM(config).eval()
    donor.save_pretrained(model)
    convert_streaming(str(model), [torch.tensor([[0, 1, 2, 3]])], str(warm),
                      group=32, device="cpu", progress=False)
    file = warm / "block_0000.pt"
    state = torch.load(file, weights_only=True)
    key = "model.layers.0.self_attn.q_proj.bias"
    assert key in state
    recorded = state[key] + 0.25
    assert not torch.equal(recorded, donor.state_dict()[key])
    state[key] = recorded
    torch.save(state, file)
    manifest_path = warm / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["block_files"]["0"] = {"file": file.name, **file_fingerprint(file)}
    manifest_path.write_text(json.dumps(manifest))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        restored = recovery.restore_student(model_path=str(model), state_dir=str(warm),
            num_blocks=0, stats_scope="group", decimation=1)
    assert not any("copying from a non-meta parameter" in str(item.message) for item in caught)
    torch.testing.assert_close(restored.state_dict()[key], recorded)


def test_local_tiny_teacher_conversion_and_recovery_pipeline(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    from memory_native.donor.streaming import convert_streaming

    torch.manual_seed(3)
    monkeypatch.setenv("SYNTHETIC_CALIBRATION", "1")
    model, data, cache, warm, run = (tmp_path / name for name in ("donor", "data", "cache", "warm", "run"))
    model.mkdir()
    config = transformers.LlamaConfig(vocab_size=17, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, attention_dropout=0.3)
    donor = transformers.LlamaForCausalLM(config).eval()
    donor.save_pretrained(model)
    write_corpus(data)
    for key, value in {"MODEL": str(model), "DATA_DIR": str(data), "OUT": str(cache),
            "STEPS": 3, "BATCH": 2, "SEQ": 4, "TOPK": 5, "SEED": 0, "DEVICE": "cpu",
            "DTYPE": torch.float32, "NUM_BLOCKS": 0, "CHUNK_ROWS": 3, "SHARD_STEPS": 2}.items():
        monkeypatch.setattr(teacher_cache, key, value)
    teacher_cache.main()
    mix = DomainMix(data, seq=4, batch=2)
    checked = ShardedCache(cache, 2)
    for step in range(3):
        ids = mix.batch_at(step, "cpu")
        idx, val = checked.step(step, "cpu", input_ids=ids)
        with torch.no_grad():
            expected_val, expected_idx = donor(ids).logits.topk(5, dim=-1)
        assert torch.equal(idx.long(), expected_idx)
        torch.testing.assert_close(val.float(), expected_val, atol=2e-4, rtol=1e-3)
    convert_streaming(str(model), [mix.batch_at(0, "cpu")], str(warm), group=32,
                      dtype=torch.float32, device="cpu", progress=False)
    # Zero learning rates make the trained visible model identical to warm. The
    # tie must select warm, even when EVAL_AT_START=0, after real KD backprop.
    for key, value in {"MODEL": str(model), "STATE_DIR": str(warm), "DATA_DIR": str(data),
            "CACHE": str(cache), "CKPT_DIR": str(run), "DEVICE": "cpu", "GRAD_CKPT": False,
            "FREEZE_EMBED": False, "SPLIT_GPUS": False, "EVAL_EVERY": 1, "LOG_EVERY": 1,
            "EVAL_MAX_TOKENS": 16, "EVAL_AT_START": False, "COUNTER_LR_START": 0.0,
            "COUNTER_LR_END": 0.0, "SCALE_LR_START": 0.0, "SCALE_LR_END": 0.0,
            "FP_LR": 0.0, "MIN_IMPROVEMENT": 0.0, "NUM_BLOCKS": 0}.items():
        monkeypatch.setattr(recovery, key, value)
    for name in ("STEPS", "BATCH", "SEQ", "SEED", "GROUP", "C"):
        monkeypatch.delenv(name, raising=False)
    restored_students = []
    original_restore = recovery.restore_student

    def track_student(**kwargs):
        student = original_restore(**kwargs)
        restored_students.append(student)
        return student

    monkeypatch.setattr(recovery, "restore_student", track_student)
    recovery.main()
    summary = json.loads((run / "selection.json").read_text())
    artifact = json.loads((run / "selected_artifact.json").read_text())
    assert not summary["accepted_kd"]
    assert summary["selected_step"] == 0
    assert artifact["format"]["group"] == 32
    assert artifact["provenance"]["warm_source_fingerprint"] == model_identity(model)
    assert (run / "USE_WARM_STATE.txt").is_file()
    assert not (run / "best.pt").exists()
    assert len(json.loads((run / "metrics.json").read_text())) == 4
    probe_ids = mix.batch_at(0, "cpu")
    from memory_native.recovery.runtime import evaluate_at_alpha
    with torch.no_grad():
        expected_warm = evaluate_at_alpha(restored_students[0], 0.0, lambda: restored_students[0](probe_ids).logits)
        reloaded_warm = recovery.restore_selected_artifact(run / "selected_artifact.json")
        torch.testing.assert_close(reloaded_warm(probe_ids).logits, expected_warm)

    # Exercise the actual partial checkpoint overlay as well as warm fallback.
    selected_logits = []
    original_save = recovery.save_best_checkpoint

    def capture_selected(payload, out_dir, staging_dir=""):
        original_save(payload, out_dir, staging_dir)
        with torch.no_grad():
            selected_logits.append(evaluate_at_alpha(restored_students[-1], 0.0,
                lambda: restored_students[-1](probe_ids).logits.detach().clone()))

    monkeypatch.setattr(recovery, "save_best_checkpoint", capture_selected)
    for key, value in {"CKPT_DIR": str(tmp_path / "improved_run"), "COUNTER_LR_START": 1e-4,
            "COUNTER_LR_END": 1e-4, "SCALE_LR_START": 1e-5, "SCALE_LR_END": 1e-5,
            "FP_LR": 1e-5}.items():
        monkeypatch.setattr(recovery, key, value)
    recovery.main()
    selected_path = tmp_path / "improved_run" / "selected_artifact.json"
    assert json.loads(selected_path.read_text())["accepted_kd"]
    assert selected_logits
    with torch.no_grad():
        reloaded_best = recovery.restore_selected_artifact(selected_path)
        torch.testing.assert_close(reloaded_best(probe_ids).logits, selected_logits[-1])
    # Changing the selected file cannot be mistaken for the evaluated artifact.
    with open(tmp_path / "improved_run" / "best.pt", "ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="checkpoint content mismatch"):
        recovery.restore_selected_artifact(selected_path)
