#!/usr/bin/env python3
"""Pre-flight consistency check for the v4 recovery-3k campaign.

Run ON the Colab runtime (after stage B of the v4 notebook) for full coverage;
outside a built release it degrades to SKIP sections instead of failing.

  1. release sanity     -> runner + package + unit tests; static answers:
                           SCALE_LR_END support, STATS_SCOPE/DECIMATION wiring
  2. donor gate         -> scripts/check_donor_config.py --donor
  3. cache/data restore -> cache_manifest present; steps==400 archived pin
  4. policy <-> env     -> production yaml numbers vs current/resolved env

Usage (runtime):
  python preflight_consistency.py \
      --project /content/mn_recovery_3k/project \
      --donor /content/drive/MyDrive/colab_models/Qwen3.8-27B \
      --cache /content/mn_recovery_3k/teacher_cache_v3 \
      --data /content/mn_recovery_3k/mix_qwen38_v3 \
      [--policy-url ...] [--expected-env-json resolved_env.json]
Exit: 0 all-pass, 2 any FAIL.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

POLICY_DEFAULT_URL = ("https://raw.githubusercontent.com/kharkilirov1/"
                      "memory-native/main/production/qwen38_27b_recovery_3k.yaml")

fails = []


def check(cond, ok_msg, fail_msg):
    if cond:
        print("  [ok]  %s" % ok_msg)
    else:
        print("  [FAIL] %s" % fail_msg)
        fails.append(fail_msg)


def info(msg):
    print("  [info] %s" % msg)


def section(title):
    print("\n== %s ==" % title)


def load_policy(path=None, url=None):
    text = None
    if path and Path(path).is_file():
        text = Path(path).read_text(encoding="utf-8")
    elif url:
        try:
            text = urllib.request.urlopen(url, timeout=30).read().decode("utf-8")
        except Exception as exc:
            info("cannot fetch policy (%r)" % exc)
    if not text:
        return None
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError:
        def grab(key):
            m = re.search(r"^%s:\s*(\S+)" % key, text, re.M)
            return m.group(1) if m else None
        try:
            return {"steps": int(grab("steps") or 0),
                    "micro_batch": int(grab("micro_batch") or 0),
                    "seq_len": int(grab("seq_len") or 0)}
        except Exception:
            return None
    except Exception as exc:
        info("yaml parse failed: %r" % exc)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--donor", type=Path)
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--data", type=Path)
    ap.add_argument("--policy", type=Path)
    ap.add_argument("--policy-url", default=POLICY_DEFAULT_URL)
    ap.add_argument("--expected-env-json")
    args = ap.parse_args()

    project = args.project.resolve()

    section("release sanity")
    runner = project / "scripts" / "kd_cached_strict_v3.py"
    if not runner.is_file():
        info("no release/runner under %s -> release sections SKIPPED "
             "(run inside the v4 notebook environment for full coverage)" % project)
    else:
        src = project / "src"
        check((src / "mn_strict_kd").is_dir(), "mn_strict_kd package present",
              "package dir missing under %s" % src)
        tests = [project / "tests" / t for t in
                 ("test_strict_sparse_kd.py", "test_strict_gate_v3.py",
                  "test_text_only_mapping.py")]
        missing = [t.name for t in tests if not t.is_file()]
        check(not missing, "strict unit-test files present",
              "missing tests: %s" % missing)
        sc = runner.read_text(encoding="utf-8", errors="ignore")
        check("SCALE_LR_END" in sc,
              "runner honors SCALE_LR_END (per-knob end value)",
              "runner has NO SCALE_LR_END - scale LR stays flat at START (2.5e-5); "
              "acceptable but record it in the run manifest")
        for env_name in ("STATS_SCOPE", "DECIMATION"):
            check(env_name in sc,
                  "runner wires %s through" % env_name,
                  "%s silently dropped by some kw-filter site!" % env_name)

    if args.donor:
        section("donor gate")
        checker = project / "scripts" / "check_donor_config.py"
        if not checker.is_file():
            try:
                url = ("https://raw.githubusercontent.com/kharkilirov1/memory-native/"
                       "main/scripts/check_donor_config.py")
                checker.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(url, checker)
                info("fetched checker from main: %s" % checker)
            except Exception as exc:
                check(False, "-", "checker absent and fetch failed: %r" % exc)
                checker = None
        if checker and Path(checker).is_file():
            rc = subprocess.run([sys.executable, str(checker), "--donor", str(args.donor)],
                                check=False).returncode
            check(rc == 0, "check_donor_config PASSed", "DONOR GATE FAILED rc=%d" % rc)

    if args.cache:
        section("teacher cache (restore-only)")
        cm = args.cache / "cache_manifest.json"
        if not cm.is_file():
            hits = list(args.cache.rglob("cache_manifest.json"))
            cm = hits[0] if len(hits) == 1 else None
        if cm is None or not Path(cm).is_file():
            check(False, "cache_manifest.json exists under the cache dir",
                  "teacher cache manifest NOT found under %s - teacher is NEVER "
                  "rebuilt in v4; stop and resolve" % args.cache)
        else:
            try:
                man = json.loads(Path(cm).read_text(encoding="utf-8"))
                topk = man.get("topk") or man.get("K") or man.get("expected_topk")
                built = man.get("steps") or man.get("expected_steps")
                info("cache provenance: topk=%s steps_built=%s "
                     "(archived v3 build: steps=400)" % (topk, built))
                if isinstance(built, int) and built != 400:
                    check(False, "-",
                          "cache built with steps=%d != archived 400 - Drive copy "
                          "changed since v3; STOP and investigate" % built)
                else:
                    check(True, "cache provenance consistent with archived v3 build", "-")
            except Exception as exc:
                check(False, "-", "cache manifest unreadable: %r" % exc)

    if args.data:
        section("corpus (restore-only)")
        check((args.data / "manifest.json").is_file(),
              "corpus manifest exists",
              "corpus manifest missing at %s - corpus is never rebuilt in v4" % args.data)

    section("policy <-> env consistency")
    pol = load_policy(args.policy, args.policy_url)
    if pol is None:
        info("policy unavailable - cross-check skipped")
    else:
        env = {}
        if args.expected_env_json and Path(args.expected_env_json).is_file():
            env = json.loads(Path(args.expected_env_json).read_text(encoding="utf-8"))
        ev = pol.get("eval") if isinstance(pol.get("eval"), dict) else {}
        es = pol.get("early_stop") if isinstance(pol.get("early_stop"), dict) else {}

        def norm(v):
            s = str(v)
            return s[:-2] if re.fullmatch(r"\d+\.0", s) else s

        flat = [
            ("STEPS", pol.get("steps")),
            ("BATCH", pol.get("micro_batch")),
            ("SEQ", pol.get("seq_len")),
            ("DECIMATION", pol.get("decimation")),
            ("EVAL_EVERY", ev.get("full_every")),
            ("EVAL_MAX_TOKENS", ev.get("full_max_tokens")),
            ("MIN_IMPROVEMENT", es.get("min_improvement")),
            ("EARLY_STOP_PATIENCE", es.get("patience")),
        ]
        lr = [("COUNTER_LR_START", (pol.get("counter_lr") or {}).get("start")),
              ("COUNTER_LR_END", (pol.get("counter_lr") or {}).get("end")),
              ("SCALE_LR_START", (pol.get("scale_lr") or {}).get("start")),
              ("SCALE_LR_END", (pol.get("scale_lr") or {}).get("end"))]

        unset = 0
        checks_run = 0
        for name, expected in flat + lr:
            if expected is None:
                continue
            actual = os.environ.get(name) or env.get(name)
            if actual is None:
                unset += 1
                continue
            checks_run += 1
            if name.startswith(("COUNTER", "SCALE")):
                ok = abs(float(actual) - float(expected)) < 1e-12
            else:
                ok = norm(actual) == norm(expected)
            check(ok, "%s=%s matches policy" % (name, actual),
                  "%s=%s but policy says %s" % (name, actual, expected))
        if checks_run == 0:
            info("no run env set yet (notebook cells will export it)")

    print("\nVERDICT:", "PASS" if not fails else "FAIL (%d)" % len(fails))
    return 0 if not fails else 2


if __name__ == "__main__":
    sys.exit(main())
