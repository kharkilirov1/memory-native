#!/usr/bin/env python3
"""Donor verification gate for the Qwen3.8-27B recovery campaign.

Run BEFORE any expensive work. Static-file checks only (no weights loading):
  1. config.json exists and is a Qwen3-family text-only causal LM;
  2. HARD GUARD: no 'qwen3_5'/'qwen35'/'qwen3.5' naming anywhere in the config
     (the 3.5-vs-3.8 mismatch already shipped one wrong commit);
  3. tokenizer files present;
  4. safetensors index covers all shards on disk; total size plausible for ~27B bf16.

Exit codes: 0 = PASS, 2 = FAIL (do not launch), 1 = usage/selftest error.
Self-test:  python check_donor_config.py --selftest
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

FAIL = "FAIL"
INFO = "info"
OK = "ok"
QWEN35_RX = re.compile(r"qwen[\s_\.]?3[\s_\.]?5", re.IGNORECASE)


class Report:
    def __init__(self):
        self.items = []

    def add(self, status, text):
        self.items.append((status, text))

    def fail(self, text):
        self.add(FAIL, text)

    def ok(self, text):
        self.add(OK, text)

    @property
    def passed(self):
        return not any(s == FAIL for s, _ in self.items)

    def dump(self):
        for s, t in self.items:
            print("[%s] %s" % (s.upper().ljust(5), t))
        print("VERDICT:", "PASS" if self.passed else "FAIL - DO NOT LAUNCH")


def sha256_file(path, cap=None):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def check_donor(donor: Path, min_total_gib=46.0, warn_only=False):
    rep = Report()
    cfg_path = donor / "config.json"

    # --- config.json -------------------------------------------------------
    if not cfg_path.is_file():
        rep.fail("config.json missing in %s" % donor)
        rep.dump()
        return rep
    raw_cfg = cfg_path.read_text(encoding="utf-8")
    try:
        cfg = json.loads(raw_cfg)
    except json.JSONDecodeError as exc:
        rep.fail("config.json is not valid JSON: %s" % exc)
        rep.dump()
        return rep

    # hard guard against the 3.5/3.8 mixup anywhere in the file
    if QWEN35_RX.search(raw_cfg):
        rep.fail("config.json mentions Qwen3.5 somewhere — the 3.8/3.5 mixup guard "
                 "tripped. Inspect the file before launching anything.")
    else:
        rep.ok("no Qwen3.5 references in config.json")

    arch = (cfg.get("architectures") or ["<none>"])[0]
    mt = str(cfg.get("model_type", "<none>"))
    if "qwen3" not in arch.lower() or "qwen3" not in mt.lower():
        rep.fail("architectures=%r model_type=%r do not look like the Qwen3 family — "
                 "wrong donor dir?" % (arch, mt))
    else:
        rep.ok("architectures=%s model_type=%s" % (arch, mt))
    if "qwen3moe" in mt.lower() or "moe" in mt.lower():
        rep.fail("model_type suggests MoE — this campaign expects DENSE 27B")

    # text-only student contract
    bad_keys = [k for k in cfg if re.search(r"(vision|visual|mtp|tower)", k, re.I)]
    if bad_keys:
        rep.fail("text-only contract violated: extra modalities present: %s "
                 "(student never materializes vision/MTP)" % bad_keys)
    else:
        rep.ok("text-only config (no vision/MTP keys)")

    n_layers = cfg.get("num_hidden_layers")
    if not isinstance(n_layers, int) or n_layers < 32:
        rep.fail("num_hidden_layers=%r implausible for 27B dense" % n_layers)
    else:
        rep.ok("num_hidden_layers=%d hidden_size=%s heads=%s kv_heads=%s" % (
            n_layers, cfg.get("hidden_size"), cfg.get("num_attention_heads"),
            cfg.get("num_key_value_heads")))
    dtype = cfg.get("torch_dtype") or cfg.get("dtype")
    rep.add(INFO, "dtype=%s vocab=%s rope_theta=%s tie_word_embeddings=%s" % (
        dtype, cfg.get("vocab_size"), cfg.get("rope_theta"), cfg.get("tie_word_embeddings")))

    # --- tokenizer ---------------------------------------------------------
    tok_ok = (donor / "tokenizer_config.json").is_file() and (
        (donor / "tokenizer.json").is_file() or (donor / "tokenizer.model").is_file())
    if tok_ok:
        rep.ok("tokenizer files present")
    else:
        rep.fail("tokenizer_config.json + tokenizer.json|tokenizer.model missing")

    # --- safetensors index & shard coverage --------------------------------
    idx = donor / "model.safetensors.index.json"
    total = None
    if not idx.is_file():
        single = list(donor.glob("*.safetensors"))
        if len(single) != 1:
            rep.fail("no model.safetensors.index.json and %d stray shards" % len(single))
        else:
            total = single[0].stat().st_size
    else:
        try:
            index = json.loads(idx.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            rep.fail("index json unreadable: %s" % exc)
            rep.dump()
            return rep
        wm = index.get("weight_map") or {}
        if not wm:
            rep.fail("index has empty weight_map")
        shard_names = sorted(set(wm.values()))
        total_bytes = 0
        missing = []
        for name in shard_names:
            p = donor / name
            if p.is_file():
                total_bytes += p.stat().st_size
            else:
                missing.append(name)
        if missing:
            rep.fail("missing %d/%d shards, e.g. %s" % (len(missing), len(shard_names), missing[:3]))
        else:
            rep.ok("all %d shards present on disk" % len(shard_names))
        total = total_bytes
        rep.add(INFO, "index sha256=%s (record into run manifest)" % sha256_file(idx))
        meta_cfg_sha = rep.add
        meta_cfg_sha(INFO, "config.json sha256=%s" % sha256_file(cfg_path))
    if total is not None:
        gib = total / 2**30
        if gib < min_total_gib:
            rep.fail("weights total %.1f GiB < expected >=%.0f GiB for ~27B bf16" % (gib, min_total_gib))
        else:
            rep.ok("weights total %.2f GiB plausible for ~27B bf16" % gib)

    rep.dump()
    return rep


def selftest(tmp=Path("/tmp/donor_selftest")):
    import shutil
    if tmp.exists():
        shutil.rmtree(tmp)
    good = tmp / "good_donordir"
    good.mkdir(parents=True)
    layers = 64
    cfg = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
           "num_hidden_layers": layers, "hidden_size": 5120,
           "num_attention_heads": 40, "num_key_value_heads": 8,
           "torch_dtype": "bfloat16", "vocab_size": 151936}
    (good / "config.json").write_text(json.dumps(cfg))
    (good / "tokenizer_config.json").write_text("{}")
    (good / "tokenizer.json").write_text("{}")
    idx = {"weight_map": {"layer0.w": "shard-a.safetensors"}}
    (good / "model.safetensors.index.json").write_text(json.dumps(idx))
    (good / "shard-a.safetensors").write_bytes(b"\0" * 1024)  # tiny stub
    rep_good = check_donor(good, min_total_gib=0.0)
    assert rep_good.passed, "selftest good-donor must PASS"

    bad = tmp / "bad_donordir"
    bad.mkdir(parents=True)
    cfg_bad = dict(cfg, architectures=["Qwen3MoEForCausalLM"], model_type="qwen3_moe")
    raw = json.dumps(cfg_bad) + "\n# trained from qwen3.5 base snapshot"
    (bad / "config.json").write_text(raw)
    (bad / "tokenizer_config.json").write_text("{}")
    rep_bad = check_donor(bad)
    assert not rep_bad.passed, "selftest bad-donor must FAIL"
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor", type=Path, help="path to the donor model dir")
    ap.add_argument("--min-total-gib", type=float, default=46.0)
    ap.add_argument("--warn-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return 0
    if not args.donor:
        ap.error("--donor is required")
    rep = check_donor(args.donor, args.min_total_gib)
    return 0 if rep.passed or args.warn_only else 2


if __name__ == "__main__":
    sys.exit(main())
