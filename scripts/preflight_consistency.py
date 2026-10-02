#!/usr/bin/env python3
"""Fail-closed static preflight for the archived, external strict-v3 campaign.

This is NOT the public kd_cached_recovery.py runner's validator. The external
release, its strict tests, donor checker and cache validator must be supplied
locally. --cache-validator must explicitly name a validator accepting CACHE,
--expected-topk K, --expected-steps N, --model-index INDEX and --data-manifest FILE
(the archived full-cache notebook CLI). That validator must check strict-v3
tail-bucket/logsumexp tensor semantics. No external format
compatibility is inferred from metadata alone. No code is downloaded or written
by this checker. Local policy defaults to
production/qwen38_27b_recovery_3k.yaml under --project; a remote policy is read only
when --policy-url is explicitly provided.

A launch requires --donor, --cache, --data and every mapped runtime environment
value (or a complete --expected-env-json snapshot). MODEL/DATA_DIR/CACHE must
resolve to the corresponding checked CLI directories (relative env paths use
--project, the runner working directory). This checker introduces a required
provenance evidence adapter: identities.model.files and identities.data.files
contain either {path: {bytes, sha256}} or [{path, bytes, sha256}] evidence; corpus
identity also records domain_order and shards contain file/sha256 entries. This
is a new validation requirement, NOT a claim about the absent strict-v3 format.
Legacy path-only manifests cannot establish that binding and are rejected, even
if dimensions match. The external validator remains responsible for its private
tensor/loss format.

--inspection-only reports available evidence without running release unit tests
or external validators. Its verdict is INSPECTION COMPLETE or INCOMPLETE, never
launch PASS. Exit codes: 0 = static PASS / complete inspection, 2 = launch FAIL,
3 = incomplete inspection. A static PASS does not certify warm-state provenance,
quality gates, 400-to-3000 cache reuse semantics, piecewise LR, proxy eval, full
resume support, or GPU memory capacity. Those require the external source and
runtime checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path


class Report:
    def __init__(self):
        self.failures = []

    def check(self, condition, ok_message, fail_message):
        print("  [%s] %s" % ("ok" if condition else "FAIL",
                                ok_message if condition else fail_message))
        if not condition:
            self.failures.append(fail_message)
        return bool(condition)

    def info(self, message):
        print("  [info] %s" % message)

    def section(self, title):
        print("\n== %s ==" % title)


def load_policy(path=None, url=None):
    """Read YAML accurately, with no network fallback for a missing local file."""
    import yaml

    if url is not None:
        with urllib.request.urlopen(url, timeout=30) as response:
            text = response.read().decode("utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    try:
        policy = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError("policy is invalid YAML: %s" % exc) from exc
    if not isinstance(policy, dict):
        raise ValueError("policy must be a YAML mapping")
    return policy


def _load_object(path, name):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("%s must be a JSON object" % name)
    return value


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("%s must be a positive integer" % name)
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(8 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def _safe_file(root, name, *, allow_symlinks=False):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("invalid provenance file path %r" % name)
    # Metadata paths stay lexical children even when a donor snapshot uses
    # ordinary Hugging Face symlinks into its sibling blobs directory.
    if ".." in Path(name).parts:
        raise ValueError("provenance file escapes its directory: %s" % name)
    target = (root / name).resolve()
    if not allow_symlinks and not target.is_relative_to(root.resolve()):
        raise ValueError("provenance file escapes its directory: %s" % name)
    if not target.is_file():
        raise ValueError("provenance file missing: %s" % target)
    return target


def _check_files(identity, root, required, label, *, allow_symlinks=False):
    if not isinstance(identity, dict):
        raise ValueError("cache lacks %s content identity (legacy cache is unverified)" % label)
    files = identity.get("files")
    if isinstance(files, list):
        mapped = {}
        for evidence in files:
            if not isinstance(evidence, dict) or not isinstance(evidence.get("path"), str):
                raise ValueError("invalid %s file evidence entry" % label)
            name = evidence["path"]
            if name in mapped:
                raise ValueError("duplicate %s file evidence: %s" % (label, name))
            mapped[name] = evidence
        files = mapped
    if not isinstance(files, dict):
        raise ValueError("cache lacks %s content identity (legacy cache is unverified)" % label)
    if not files or not required.issubset(files):
        raise ValueError("cache %s identity omits required files: %s" %
                         (label, sorted(required - files.keys())))
    for name, evidence in files.items():
        path = _safe_file(root, name, allow_symlinks=allow_symlinks)
        if not isinstance(evidence, dict):
            raise ValueError("invalid %s file evidence: %s" % (label, name))
        digest = evidence.get("sha256")
        size = evidence.get("bytes")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("invalid %s sha256: %s" % (label, name))
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("invalid %s byte count: %s" % (label, name))
        if path.stat().st_size != size or _sha256(path) != digest:
            raise ValueError("cache %s fingerprint mismatch: %s" % (label, name))


def validate_cache_provenance(manifest, *, donor, data, cache, seq):
    """Verify recorded content evidence; delegate private tensor format to release."""
    identities = manifest.get("identities")
    if not isinstance(identities, dict):
        raise ValueError("cache has no donor/corpus content identities; legacy cache cannot launch")
    donor_files = {p.name for p in donor.glob("*.safetensors") if p.is_file()}
    if not donor_files:
        raise ValueError("donor has no safetensors weight files")
    donor_files.add("config.json")
    for name in ("model.safetensors.index.json", "tokenizer.json", "tokenizer.model",
                 "tokenizer_config.json", "special_tokens_map.json", "vocab.json",
                 "merges.txt", "spiece.model", "added_tokens.json"):
        if (donor / name).is_file():
            donor_files.add(name)
    _check_files(identities.get("model"), donor, donor_files, "donor", allow_symlinks=True)

    corpus = _load_object(data / "manifest.json", "corpus manifest")
    domains = corpus.get("domains")
    if not isinstance(domains, dict) or not domains:
        raise ValueError("corpus manifest must contain nonempty domains")
    if corpus.get("dtype", "uint32") != "uint32":
        raise ValueError("expected uint32 corpus bins")
    data_files, total_share = {"manifest.json"}, 0.0
    for name, spec in domains.items():
        if (not isinstance(name, str) or Path(name).name != name
                or name in ("", ".", "..")):
            raise ValueError("invalid corpus domain name")
        share = spec.get("share") if isinstance(spec, dict) else None
        if (isinstance(share, bool) or not isinstance(share, (int, float))
                or not math.isfinite(share) or share < 0):
            raise ValueError("invalid corpus sampling share: %s" % name)
        total_share += share
        for split in ("train", "val"):
            filename = "%s_%s.bin" % (split, name)
            path = _safe_file(data, filename)
            if path.stat().st_size % 4 or path.stat().st_size < seq * 4:
                raise ValueError("%s lacks a complete uint32 sequence" % filename)
            data_files.add(filename)
    if total_share <= 0 or not math.isfinite(total_share):
        raise ValueError("corpus sampling shares must have finite positive sum")
    data_id = identities.get("data")
    _check_files(data_id, data, data_files, "corpus")
    if data_id.get("domain_order") != list(domains):
        raise ValueError("cache corpus domain order differs from current sampler")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("cache lacks shard integrity evidence")
    names = []
    for shard in shards:
        if not isinstance(shard, dict):
            raise ValueError("invalid cache shard evidence")
        name, digest = shard.get("file"), shard.get("sha256")
        path = _safe_file(cache, name)
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("invalid cache shard sha256")
        if _sha256(path) != digest:
            raise ValueError("cache shard fingerprint mismatch: %s" % name)
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("duplicate cache shard provenance entry")


# Fields the archived notebook actually exports to the strict-v3 runner. Policy
# intentions requiring unavailable patches (proxy eval/piecewise LR/periodic full
# resume) are not certified by these static environment comparisons.
ENV_FIELDS = (
    ("STEPS", ("steps",), "int"),
    ("BATCH", ("micro_batch",), "int"),
    ("SEQ", ("seq_len",), "int"),
    ("DECIMATION", ("decimation",), "int"),
    ("STATS_SCOPE", ("stats_scope",), "str"),
    ("KD_WEIGHT", ("kd_weight",), "float"),
    ("CE_WEIGHT", ("ce_weight",), "float"),
    ("KD_T", ("temperature",), "float"),
    ("GRAD_CLIP", ("max_grad_norm",), "float"),
    ("FP_TRAIN_MODE", ("fp_train_mode",), "str"),
    ("EVAL_EVERY", ("eval", "full_every"), "int"),
    ("EVAL_MAX_TOKENS", ("eval", "full_max_tokens"), "int"),
    ("MIN_IMPROVEMENT", ("early_stop", "min_improvement"), "float"),
    ("EARLY_STOP_PATIENCE", ("early_stop", "patience"), "int"),
    ("COUNTER_LR_START", ("counter_lr", "start"), "float"),
    ("COUNTER_LR_END", ("counter_lr", "end"), "float"),
    ("SCALE_LR_START", ("scale_lr", "start"), "float"),
    ("SCALE_LR_END", ("scale_lr", "end"), "float"),
)


def _get(policy, keys):
    value = policy
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError("policy missing %s" % ".".join(keys))
        value = value[key]
    return value


def _number(value, kind):
    if isinstance(value, bool) or value is None:
        raise ValueError("expected a finite %s, got %r" % (kind, value))
    if kind == "int":
        if isinstance(value, int):
            return value
        if not isinstance(value, str):
            raise ValueError("expected an integer, got %r" % value)
        # Match the runner's int() parsing; e.g. '2.0' is not valid BATCH.
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError("invalid integer %r" % value) from exc
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("invalid numeric value %r" % value) from exc
    if not result.is_finite():
        raise ValueError("numeric value must be finite, got %r" % value)
    return result


def check_environment(report, policy, env):
    for name, keys, kind in ENV_FIELDS:
        try:
            expected = _get(policy, keys)
            if name not in env or env[name] in (None, ""):
                raise ValueError("required runtime environment %s is unset" % name)
            actual = env[name]
            if kind == "str":
                if not isinstance(expected, str) or not isinstance(actual, str):
                    raise ValueError("%s must be a string" % name)
                matches = actual == expected
            else:
                expected_value, actual_value = _number(expected, kind), _number(actual, kind)
                if (expected_value < 0 or (kind == "int" and expected_value == 0
                                          and name != "EARLY_STOP_PATIENCE")
                        or (name == "KD_T" and expected_value == 0)):
                    raise ValueError("invalid nonpositive policy value for %s" % name)
                matches = expected_value == actual_value
            report.check(matches, "%s=%s matches policy" % (name, actual),
                         "%s=%s but policy says %s" % (name, actual, expected))
        except (ValueError, TypeError) as exc:
            report.check(False, "", str(exc))
    report.check(policy.get("fp_train_mode") == "none", "policy freezes FP weights",
                 "strict recovery campaign requires fp_train_mode=none")
    for key, expected in (("grad_accum", 1), ("alpha", 0.0)):
        try:
            actual = _number(_get(policy, (key,)), "float")
            report.check(actual == Decimal(str(expected)), "%s=%s is supported" % (key, expected),
                         "%s=%s is unsupported by this strict counter-update campaign" % (key, actual))
        except (ValueError, TypeError) as exc:
            report.check(False, "", str(exc))
    mode = policy.get("gradient_checkpointing")
    resolved = env.get("GRAD_CKPT")
    try:
        choice = _number(resolved, "int")
        valid = choice in (0, 1) and (mode == "AUTO" or choice == _number(mode, "int"))
        report.check(valid, "GRAD_CKPT=%s is an explicit checkpointing choice" % choice,
                     "GRAD_CKPT=%s is invalid for policy %s" % (resolved, mode))
    except (ValueError, TypeError) as exc:
        report.check(False, "", "checkpointing choice invalid: %s" % exc)
    for name, expected, kind in (("SAVE_BEST", "1", "int"), ("FP_LR", "0", "float"),
                                 ("NUM_BLOCKS", "0", "int")):
        try:
            matches = _number(env.get(name), kind) == Decimal(expected)
            report.check(matches, "%s=%s is campaign-compatible" % (name, expected),
                         "%s=%s must equal %s for the full frozen-FP campaign" %
                         (name, env.get(name), expected))
        except (ValueError, TypeError) as exc:
            report.check(False, "", "%s: %s" % (name, exc))
    if mode == "AUTO":
        report.info("AUTO choice must come from a successful hardware probe; static preflight does not certify VRAM headroom")


def _run_checked(report, command, label, project):
    try:
        result = subprocess.run(command, cwd=project, check=False)
        report.check(result.returncode == 0, "%s succeeded" % label,
                     "%s failed with exit %d" % (label, result.returncode))
    except OSError as exc:
        report.check(False, "", "%s could not run: %s" % (label, exc))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--donor", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--data", type=Path)
    policy_source = parser.add_mutually_exclusive_group()
    policy_source.add_argument("--policy", type=Path)
    policy_source.add_argument("--policy-url", help="explicitly fetch this YAML; never used as fallback")
    parser.add_argument("--expected-env-json", type=Path)
    parser.add_argument("--cache-validator", type=Path,
                        help="explicit local strict-v3 validator using the documented archived full-cache CLI")
    parser.add_argument("--expected-cache-steps", type=int,
                        help="required if policy teacher_cache.expected_steps is absent")
    parser.add_argument("--inspection-only", action="store_true")
    args = parser.parse_args(argv)
    report = Report()
    project = args.project.resolve()
    report.check(project.is_dir(), "project directory exists", "project directory missing: %s" % project)

    report.section("archived external release sanity")
    runner = project / "scripts" / "kd_cached_strict_v3.py"
    package = project / "src" / "mn_strict_kd"
    tests = [project / "tests" / name for name in
             ("test_strict_sparse_kd.py", "test_strict_gate_v3.py", "test_text_only_mapping.py")]
    runner_ok = report.check(runner.is_file(), "strict-v3 runner present",
                             "external strict-v3 runner missing: %s" % runner)
    package_ok = report.check(package.is_dir() and (package / "__init__.py").is_file(),
                              "strict-v3 package present", "external mn_strict_kd package missing/incomplete")
    missing = [path.name for path in tests if not path.is_file()]
    tests_ok = report.check(not missing, "strict release unit-test files present", "missing strict release tests: %s" % missing)
    if runner_ok:
        try:
            source = runner.read_text(encoding="utf-8")
            for name in ("SCALE_LR_END", "STATS_SCOPE", "DECIMATION"):
                report.check(name in source, "runner mentions %s (tests must verify semantics)" % name,
                             "runner lacks required %s support" % name)
        except (OSError, UnicodeError) as exc:
            runner_ok = False
            report.check(False, "", "runner source is unreadable: %s" % exc)
    if runner_ok and package_ok and tests_ok and not args.inspection_only:
        _run_checked(report, [sys.executable, "-m", "pytest", "-q", *map(str, tests)],
                     "external strict release unit tests", project)
    elif args.inspection_only:
        report.info("inspection mode: external unit tests and validators are not executed")

    report.section("policy and complete runtime environment")
    policy = None
    try:
        policy = load_policy(args.policy or project / "production" / "qwen38_27b_recovery_3k.yaml",
                             args.policy_url)
        report.info("policy source: %s" % (args.policy_url or args.policy or
                                           project / "production" / "qwen38_27b_recovery_3k.yaml"))
    except (ImportError, OSError, ValueError) as exc:
        report.check(False, "", "policy unavailable/invalid (PyYAML required): %s" % exc)
    # A supplied snapshot is authoritative, not filled from unrelated shell env.
    env = dict(os.environ)
    if args.expected_env_json:
        try:
            env = _load_object(args.expected_env_json, "resolved environment snapshot")
        except (OSError, ValueError) as exc:
            report.check(False, "", "runtime environment snapshot unreadable: %s" % exc)
            env = {}
    if policy is not None:
        check_environment(report, policy, env)
        report.info("piecewise LR/proxy eval/full-resume intentions are not verified by environment matching")

    for name in ("donor", "cache", "data"):
        path = getattr(args, name)
        report.check(path is not None and path.is_dir(), "%s directory supplied" % name,
                     "--%s must point to an existing local directory" % name)
    donor = args.donor.resolve() if args.donor else None
    cache = args.cache.resolve() if args.cache else None
    data = args.data.resolve() if args.data else None
    for env_name, checked in (("MODEL", donor), ("DATA_DIR", data), ("CACHE", cache)):
        try:
            value = env.get(env_name)
            if not isinstance(value, str) or not value:
                raise ValueError("required runtime path %s is unset or invalid" % env_name)
            runtime_path = Path(value)
            if not runtime_path.is_absolute():
                runtime_path = project / runtime_path
            report.check(checked is not None and runtime_path.resolve() == checked,
                         "%s resolves to the checked input directory" % env_name,
                         "%s=%s differs from checked directory %s" % (env_name, value, checked))
        except (OSError, ValueError) as exc:
            report.check(False, "", str(exc))
    report.info("STATE_DIR/warm-state provenance must be checked by the external release; this preflight does not certify it")

    report.section("local donor gate")
    checker = project / "scripts" / "check_donor_config.py"
    checker_ok = report.check(checker.is_file(), "local donor checker present",
                              "local donor checker missing; no mutable code will be downloaded")
    if checker_ok and donor and donor.is_dir() and not args.inspection_only:
        _run_checked(report, [sys.executable, str(checker), "--donor", str(donor)], "donor gate", project)

    report.section("existing teacher cache provenance and private-format validator")
    validator = args.cache_validator.resolve() if args.cache_validator else None
    validator_ok = report.check(validator is not None and validator.is_file(),
                                "explicit local strict-v3 cache validator present",
                                "--cache-validator must name a local strict-v3 validator; private loss format is unverified")
    cache_ok = False
    topk = expected_steps = None
    if cache and cache.is_dir() and policy is not None:
        try:
            settings = _get(policy, ("teacher_cache",))
            if not isinstance(settings, dict):
                raise ValueError("teacher_cache policy must be a mapping")
            pinned = settings.get("expected_steps")
            if args.expected_cache_steps is not None and pinned is not None and args.expected_cache_steps != pinned:
                raise ValueError("--expected-cache-steps conflicts with the local policy pin")
            expected_steps = _positive_integer(args.expected_cache_steps if args.expected_cache_steps is not None
                                               else pinned, "expected cache steps")
            topk = _positive_integer(settings.get("K"), "policy teacher_cache.K")
            manifest = _load_object(cache / "cache_manifest.json", "cache manifest")
            built = _positive_integer(manifest.get("steps"), "cache steps")
            actual_topk = _positive_integer(manifest.get("topk", manifest.get("K")), "cache topk")
            if built != expected_steps or actual_topk != topk:
                raise ValueError("cache dimensions differ: steps=%d/K=%d, policy expects %d/%d" %
                                 (built, actual_topk, expected_steps, topk))
            for flag in ("tail_bucket", "logsumexp_saved"):
                if settings.get(flag) is not True:
                    raise ValueError("strict-v3 policy requires explicit %s=true" % flag)
                if flag in manifest and manifest[flag] is not True:
                    raise ValueError("cache explicitly contradicts required %s=true" % flag)
            batch = _positive_integer(manifest.get("batch"), "cache batch")
            seq = _positive_integer(manifest.get("seq"), "cache seq")
            if batch != _get(policy, ("micro_batch",)) or seq != _get(policy, ("seq_len",)):
                raise ValueError("cache batch/sequence differs from policy")
            if donor is None or data is None:
                raise ValueError("donor and corpus paths are needed to verify cache identities")
            validate_cache_provenance(manifest, donor=donor, data=data, cache=cache, seq=seq)
            cache_ok = report.check(True, "cache content identities and shard hashes match", "")
            report.info("cache covers %d unique steps; requested %s training steps requires the external runner's documented reuse contract" %
                        (built, policy.get("steps")))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            report.check(False, "", "cache unverified: %s" % exc)
    if cache_ok and validator_ok:
        model_index = donor / "model.safetensors.index.json"
        index_ok = report.check(model_index.is_file(), "donor index available for archived validator CLI",
                                "archived full-cache validator requires model.safetensors.index.json")
        if index_ok and not args.inspection_only:
            _run_checked(report, [sys.executable, str(validator), str(cache), "--expected-topk", str(topk),
                                  "--expected-steps", str(expected_steps), "--model-index", str(model_index),
                                  "--data-manifest", str(data / "manifest.json")],
                         "strict-v3 cache format gate", project)

    report.section("corpus")
    if data:
        report.check((data / "manifest.json").is_file(), "corpus manifest present",
                     "corpus manifest missing: %s" % (data / "manifest.json"))
    if args.inspection_only:
        verdict = "INCOMPLETE (%d issues; inspection only)" % len(report.failures) if report.failures else "INSPECTION COMPLETE (not launch approval)"
        code = 3 if report.failures else 0
    else:
        verdict = "FAIL (%d issues; DO NOT LAUNCH)" % len(report.failures) if report.failures else "PASS (static preflight only; external runtime semantics and warm provenance unverified)"
        code = 2 if report.failures else 0
    print("\nVERDICT:", verdict)
    return code


if __name__ == "__main__":
    sys.exit(main())
