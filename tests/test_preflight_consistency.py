"""Preflight must never authorize absent or unverified external campaign inputs."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    "preflight_consistency", Path(__file__).resolve().parents[1] / "scripts" / "preflight_consistency.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _evidence(root, names):
    return {"files": {name: {"bytes": (root / name).stat().st_size,
                              "sha256": preflight._sha256(root / name)} for name in names}}


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    """Small content-bound inputs; external execution is stubbed, not certified."""
    yaml = pytest.importorskip("yaml")
    project, donor, data, cache = (tmp_path / name for name in ("project", "donor", "data", "cache"))
    for path in (project / "scripts", project / "tests", project / "src" / "mn_strict_kd",
                 project / "production", donor, data, cache):
        path.mkdir(parents=True, exist_ok=True)
    (project / "src" / "mn_strict_kd" / "__init__.py").write_text("")
    (project / "scripts" / "kd_cached_strict_v3.py").write_text("# SCALE_LR_END STATS_SCOPE DECIMATION\n")
    for name in ("test_strict_sparse_kd.py", "test_strict_gate_v3.py", "test_text_only_mapping.py"):
        (project / "tests" / name).write_text("def test_placeholder(): pass\n")
    for name in ("check_donor_config.py", "validate_kd_cache_v3.py"):
        (project / "scripts" / name).write_text("# external release fixture\n")
    _write_json(donor / "config.json", {"vocab_size": 16})
    (donor / "model.safetensors").write_bytes(b"donor fixture")
    _write_json(donor / "model.safetensors.index.json", {"weight_map": {"weight": "model.safetensors"}})
    _write_json(donor / "tokenizer.json", {"vocab": "fixture"})
    _write_json(data / "manifest.json", {"dtype": "uint32", "domains": {"en": {"share": 1}}})
    (data / "train_en.bin").write_bytes(b"\x00" * 32)
    (data / "val_en.bin").write_bytes(b"\x00" * 32)
    (cache / "part.pt").write_bytes(b"cached logits fixture")
    manifest = {"steps": 7, "topk": 8, "batch": 2, "seq": 4,
                "identities": {"model": _evidence(donor, ["config.json", "model.safetensors", "model.safetensors.index.json", "tokenizer.json"]),
                               "data": {**_evidence(data, ["manifest.json", "train_en.bin", "val_en.bin"]),
                                        "domain_order": ["en"]}},
                "shards": [{"file": "part.pt", "sha256": preflight._sha256(cache / "part.pt")}]}
    _write_json(cache / "cache_manifest.json", manifest)
    policy = {"steps": 21, "micro_batch": 2, "seq_len": 4, "grad_accum": 1,
              "gradient_checkpointing": "AUTO", "alpha": 0.0, "decimation": 1, "stats_scope": "group",
              "kd_weight": 1.0, "ce_weight": 1.0, "temperature": 1.0,
              "max_grad_norm": 0.5, "fp_train_mode": "none",
              "counter_lr": {"start": 0.000125, "end": 0.00001},
              "scale_lr": {"start": 0.000025, "end": 0.000005},
              "eval": {"full_every": 5, "full_max_tokens": 24},
              "early_stop": {"patience": 3, "min_improvement": 0.001},
              "teacher_cache": {"K": 8, "expected_steps": 7, "tail_bucket": True, "logsumexp_saved": True}}
    policy_path = project / "production" / "qwen38_27b_recovery_3k.yaml"
    policy_path.write_text(yaml.safe_dump(policy))
    env = {name: str(preflight._get(policy, keys)) for name, keys, _ in preflight.ENV_FIELDS}
    env.update({"GRAD_CKPT": "0", "SAVE_BEST": "1", "FP_LR": "0", "NUM_BLOCKS": "0",
                "MODEL": str(donor), "DATA_DIR": str(data), "CACHE": str(cache)})
    env_path = tmp_path / "env.json"
    _write_json(env_path, env)
    argv = ["--project", str(project), "--donor", str(donor), "--data", str(data), "--cache", str(cache),
            "--cache-validator", str(project / "scripts" / "validate_kd_cache_v3.py"),
            "--expected-env-json", str(env_path)]
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(preflight.subprocess, "run", run)
    return SimpleNamespace(project=project, donor=donor, data=data, cache=cache, manifest=manifest,
                           policy=policy, policy_path=policy_path, env=env, env_path=env_path,
                           argv=argv, commands=commands)


def test_missing_external_release_is_fail_closed_without_network_or_writes(tmp_path, monkeypatch, capsys):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(preflight.urllib.request, "urlopen", lambda *a, **k: pytest.fail("implicit network access"))
    monkeypatch.setattr(preflight.subprocess, "run", lambda *a, **k: pytest.fail("missing release must not execute"))
    assert preflight.main(["--project", str(project)]) == 2
    output = capsys.readouterr().out
    assert "VERDICT: FAIL" in output
    assert "VERDICT: PASS" not in output
    assert "runner missing" in output
    assert "--donor" in output and "--cache" in output and "--data" in output
    assert list(project.rglob("*")) == []


def test_incomplete_inspection_has_distinct_exit_and_never_pass(tmp_path, capsys):
    assert preflight.main(["--project", str(tmp_path), "--inspection-only"]) == 3
    output = capsys.readouterr().out
    assert "VERDICT: INCOMPLETE" in output
    assert "VERDICT: PASS" not in output


def test_complete_static_preflight_runs_external_gates_and_uses_policy_step_pin(campaign, capsys):
    assert preflight.main(campaign.argv) == 0
    output = capsys.readouterr().out
    assert "VERDICT: PASS (static preflight" in output
    assert "covers 7 unique steps" in output
    assert len(campaign.commands) == 3
    assert "pytest" in campaign.commands[0]
    command = campaign.commands[-1]
    assert command[command.index("--expected-steps") + 1] == "7"
    assert command[command.index("--model-index") + 1] == str(campaign.donor / "model.safetensors.index.json")
    assert command[command.index("--data-manifest") + 1] == str(campaign.data / "manifest.json")


def test_complete_inspection_never_runs_external_code_or_returns_launch_pass(campaign, capsys):
    assert preflight.main(campaign.argv + ["--inspection-only"]) == 0
    assert campaign.commands == []
    output = capsys.readouterr().out
    assert "VERDICT: INSPECTION COMPLETE" in output
    assert "VERDICT: PASS" not in output


def test_numeric_env_errors_are_reported_instead_of_crashing(campaign, capsys):
    campaign.env["COUNTER_LR_START"] = "not-a-number"
    campaign.env["SCALE_LR_END"] = "NaN"
    campaign.env["BATCH"] = "2.0"
    _write_json(campaign.env_path, campaign.env)
    assert preflight.main(campaign.argv) == 2
    output = capsys.readouterr().out
    assert "invalid numeric value" in output
    assert "must be finite" in output
    assert "invalid integer" in output


def test_scientific_lr_strings_compare_by_numeric_value(campaign):
    campaign.env["COUNTER_LR_END"] = "1.0e-5"
    campaign.env["SCALE_LR_START"] = "2.5e-5"
    _write_json(campaign.env_path, campaign.env)
    assert preflight.main(campaign.argv) == 0


def test_missing_env_not_filled_from_unrelated_shell_snapshot(campaign, monkeypatch, capsys):
    del campaign.env["STEPS"]
    monkeypatch.setenv("STEPS", "21")
    _write_json(campaign.env_path, campaign.env)
    assert preflight.main(campaign.argv) == 2
    assert "runtime environment STEPS is unset" in capsys.readouterr().out


@pytest.mark.parametrize("field,value", [("steps", None), ("steps", "7"), ("steps", True),
                                          ("topk", 0), ("batch", 3), ("seq", 8)])
def test_missing_or_invalid_cache_dimensions_fail(campaign, field, value, capsys):
    campaign.manifest[field] = value
    _write_json(campaign.cache / "cache_manifest.json", campaign.manifest)
    assert preflight.main(campaign.argv) == 2
    assert "cache unverified" in capsys.readouterr().out
    assert not any("validate_kd_cache_v3.py" in str(command) for command in campaign.commands)


def test_legacy_dimension_only_manifest_cannot_launch(campaign, capsys):
    del campaign.manifest["identities"]
    _write_json(campaign.cache / "cache_manifest.json", campaign.manifest)
    assert preflight.main(campaign.argv) == 2
    assert "legacy cache cannot launch" in capsys.readouterr().out


@pytest.mark.parametrize("target", ["donor", "data", "cache"])
def test_changed_donor_corpus_or_cached_shard_is_rejected(campaign, target, capsys):
    changed = {"donor": campaign.donor / "model.safetensors", "data": campaign.data / "train_en.bin",
               "cache": campaign.cache / "part.pt"}[target]
    original = changed.read_bytes()
    changed.write_bytes(b"X" + original[1:])  # Equal size: content hash, not stat, must decide.
    assert preflight.main(campaign.argv) == 2
    assert "fingerprint mismatch" in capsys.readouterr().out


def test_missing_cache_step_pin_is_not_silently_assumed_400(campaign, capsys):
    yaml = pytest.importorskip("yaml")
    del campaign.policy["teacher_cache"]["expected_steps"]
    campaign.policy_path.write_text(yaml.safe_dump(campaign.policy))
    assert preflight.main(campaign.argv) == 2
    assert "expected cache steps must be a positive integer" in capsys.readouterr().out
    assert preflight.main(campaign.argv + ["--expected-cache-steps", "7"]) == 0


def test_explicit_cache_pin_cannot_override_recipe(campaign, capsys):
    assert preflight.main(campaign.argv + ["--expected-cache-steps", "400"]) == 2
    assert "conflicts with the local policy pin" in capsys.readouterr().out


def test_cache_validator_must_be_supplied_explicitly(campaign, capsys):
    argv = campaign.argv.copy()
    i = argv.index("--cache-validator")
    del argv[i:i + 2]
    assert preflight.main(argv) == 2
    assert "--cache-validator must name" in capsys.readouterr().out


def test_bad_yaml_and_json_report_failures(campaign, capsys):
    campaign.policy_path.write_text("steps: [1\n")
    campaign.env_path.write_text("[]")
    assert preflight.main(campaign.argv) == 2
    output = capsys.readouterr().out
    assert "invalid YAML" in output
    assert "must be a JSON object" in output


def test_failed_external_tests_or_validator_prevent_pass(campaign, monkeypatch, capsys):
    monkeypatch.setattr(preflight.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))
    assert preflight.main(campaign.argv) == 2
    assert "failed with exit 1" in capsys.readouterr().out


def test_provenance_cannot_read_outside_input_directory(campaign, capsys):
    campaign.manifest["identities"]["model"]["files"]["../env.json"] = {"bytes": 0, "sha256": "0" * 64}
    _write_json(campaign.cache / "cache_manifest.json", campaign.manifest)
    assert preflight.main(campaign.argv) == 2
    assert "escapes its directory" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["MODEL", "DATA_DIR", "CACHE"])
def test_checked_paths_must_match_actual_runtime_snapshot(campaign, name, capsys):
    campaign.env[name] = str(campaign.project)  # Valid directory, wrong content source.
    _write_json(campaign.env_path, campaign.env)
    assert preflight.main(campaign.argv) == 2
    assert "%s=" % name in capsys.readouterr().out


@pytest.mark.parametrize("name", ["MODEL", "DATA_DIR", "CACHE"])
def test_actual_runtime_paths_cannot_be_missing(campaign, name, capsys):
    del campaign.env[name]
    _write_json(campaign.env_path, campaign.env)
    assert preflight.main(campaign.argv) == 2
    assert "runtime path %s is unset" % name in capsys.readouterr().out


def test_relative_runtime_paths_are_resolved_against_runner_project(campaign):
    campaign.env.update({"MODEL": "../donor", "DATA_DIR": "../data", "CACHE": "../cache"})
    _write_json(campaign.env_path, campaign.env)
    assert preflight.main(campaign.argv) == 0


@pytest.mark.parametrize("name,value", [("SAVE_BEST", "1.0"), ("NUM_BLOCKS", "0.0")])
def test_runner_integer_flags_reject_decimal_strings(campaign, name, value, capsys):
    campaign.env[name] = value
    _write_json(campaign.env_path, campaign.env)
    assert preflight.main(campaign.argv) == 2
    assert "invalid integer" in capsys.readouterr().out


def test_zero_temperature_cannot_pass_matched_policy_env(campaign, capsys):
    yaml = pytest.importorskip("yaml")
    campaign.policy["temperature"] = 0
    campaign.env["KD_T"] = "0"
    campaign.policy_path.write_text(yaml.safe_dump(campaign.policy))
    _write_json(campaign.env_path, campaign.env)
    assert preflight.main(campaign.argv) == 2
    assert "invalid nonpositive policy value for KD_T" in capsys.readouterr().out


def test_content_evidence_adapter_accepts_file_list_representation(campaign):
    original = campaign.manifest["identities"]["model"]["files"]
    campaign.manifest["identities"]["model"]["files"] = [
        {"path": name, **value} for name, value in original.items()]
    _write_json(campaign.cache / "cache_manifest.json", campaign.manifest)
    assert preflight.main(campaign.argv) == 0


def test_content_evidence_adapter_rejects_duplicate_file_list_entries(campaign, capsys):
    evidence = campaign.manifest["identities"]["model"]["files"]
    entries = [{"path": name, **value} for name, value in evidence.items()]
    campaign.manifest["identities"]["model"]["files"] = entries + entries[:1]
    _write_json(campaign.cache / "cache_manifest.json", campaign.manifest)
    assert preflight.main(campaign.argv) == 2
    assert "duplicate donor file evidence" in capsys.readouterr().out


def test_huggingface_snapshot_symlinks_are_hashed_as_donor_content(campaign):
    hf_cache = campaign.project.parent / "hf_cache"
    snapshot = hf_cache / "snapshots" / "ref"
    blobs = hf_cache / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    for path in campaign.donor.iterdir():
        if path.name == "model.safetensors":
            (blobs / "weights").write_bytes(path.read_bytes())
            (snapshot / path.name).symlink_to("../../blobs/weights")
        else:
            (snapshot / path.name).write_bytes(path.read_bytes())
    campaign.env["MODEL"] = str(snapshot)
    _write_json(campaign.env_path, campaign.env)
    argv = campaign.argv.copy()
    argv[argv.index("--donor") + 1] = str(snapshot)
    assert preflight.main(argv) == 0


def test_donor_symlink_support_does_not_allow_metadata_traversal(campaign, capsys):
    campaign.manifest["identities"]["model"]["files"]["../donor/model.safetensors"] = {
        "bytes": len(b"donor fixture"), "sha256": preflight._sha256(campaign.donor / "model.safetensors")}
    _write_json(campaign.cache / "cache_manifest.json", campaign.manifest)
    assert preflight.main(campaign.argv) == 2
    assert "escapes its directory" in capsys.readouterr().out
