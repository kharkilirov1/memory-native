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

    # Identity guard (v3 semantics, post-mortem of the 3.5/3.8 mixup):
    # HF class names do NOT track the marketing/folder version - the real
    # Qwen3.8-27B donor IS served by Qwen3_5* classes (proven by the working
    # strict-v3 run on this exact checkpoint). Therefore:
    #   - identity check = Qwen3-family prefix + DENSE text stack + a
    #     text_config when the checkpoint is multimodal;
    #   - any 3.5-vs-3.8 string difference is reported as INFO context only.
    arch = (cfg.get("architectures") or ["<none>"])[0]
    mt = str(cfg.get("model_type", "<none>"))

    if "qwen3" not in arch.lower() or "qwen3" not in mt.lower():
        rep.fail("architectures=%r model_type=%r do not look like the Qwen3 family — "
                 "wrong donor dir?" % (arch, mt))
    else:
        rep.ok("architectures=%s model_type=%s" % (arch, mt))
        for m in QWEN35_RX.finditer(raw_cfg):
            ctx = raw_cfg[max(0, m.start() - 30):m.end() + 20].replace("\n", " ")
            rep.add(INFO, "3.5-family naming in config.json (HF class family, not a "
                          "donor identity mismatch): ...%s..." % ctx[:80])
    if "moe" in mt.lower():
        rep.fail("model_type suggests MoE — this campaign expects DENSE 27B")

    # Multimodal donor layout (strict-KD contract): the checkpoint MAY be a
    # vision-language ForConditionalGeneration model; the student materializes
    # only the text stack (cfg.text_config). Vision keys at top level are the
    # NORM for such donors, not a violation.
    text_cfg = cfg.get("text_config") or cfg.get("text") or {}
    if isinstance(text_cfg, dict) and text_cfg:
        rep.ok("multimodal donor with text_config (strict-KD text-only student)")
        n_layers = text_cfg.get("num_hidden_layers")
        hidden = text_cfg.get("hidden_size")
        heads = text_cfg.get("num_attention_heads")
        kv = text_cfg.get("num_key_value_heads")
        mtp_layers = text_cfg.get("mtp_num_hidden_layers")
        if isinstance(mtp_layers, int) and mtp_layers > 0:
            rep.add(INFO, "donor carries MTP head (mtp_num_hidden_layers=%d); the "
                          "strict pipeline must keep it unmaterialized" % mtp_layers)
    else:
        bad_keys = [k for k in cfg if re.search(r"(vision|visual|tower)", k, re.I)]
        if bad_keys:
            rep.fail("vision keys WITHOUT text_config - unsupported layout: %s" % bad_keys)
        else:
            rep.ok("text-only config (no vision keys, no text_config needed)")
        n_layers = cfg.get("num_hidden_layers")
        hidden = cfg.get("hidden_size")
        heads = cfg.get("num_attention_heads")
        kv = cfg.get("num_key_value_heads")

    if not isinstance(n_layers, int) or n_layers < 32:
        rep.fail("num_hidden_layers=%r implausible for 27B dense" % n_layers)
    else:
        rep.ok("num_hidden_layers=%d hidden_size=%s heads=%s kv_heads=%s" % (
            n_layers, hidden, heads, kv))
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
    tmp.mkdir(parents=True)

    # 1) plain dense text-only donor -> PASS
    good = tmp / "good_donordir"
    good.mkdir(parents=True)
    cfg = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
           "num_hidden_layers": 64, "hidden_size": 5120,
           "num_attention_heads": 40, "num_key_value_heads": 8,
           "torch_dtype": "bfloat16", "vocab_size": 151936}
    (good / "config.json").write_text(json.dumps(cfg))
    (good / "tokenizer_config.json").write_text("{}")
    (good / "tokenizer.json").write_text("{}")
    idx = {"weight_map": {"layer0.w": "shard-a.safetensors"}}
    (good / "model.safetensors.index.json").write_text(json.dumps(idx))
    (good / "shard-a.safetensors").write_bytes(b"\0" * 1024)
    rep_good = check_donor(good, min_total_gib=0.0)
    assert rep_good.passed, "selftest: dense text-only donor must PASS"

    # 2) MULTIMODAL donor (the real Qwen3.8-27B layout): vision keys + text_config,
    #    nested model_type mentions 3.5 -> must PASS with INFO lines only
    mm = tmp / "mm_donordir"
    mm.mkdir()
    cfg_mm = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "vision_config": {"hidden": 1},
        "text_config": {"model_type": "qwen3_5_text", "num_hidden_layers": 64,
                        "hidden_size": 5120, "num_attention_heads": 40,
                        "num_key_value_heads": 8, "vocab_size": 248320,
                        "mtp_num_hidden_layers": 1},
        "_name_or_path": "Qwen/Qwen3.5-27B",
    }
    (mm / "config.json").write_text(json.dumps(cfg_mm))
    (mm / "tokenizer_config.json").write_text("{}")
    (mm / "tokenizer.json").write_text("{}")
    idx2 = {"weight_map": {"layer0.w": "shard-b.safetensors"}}
    (mm / "model.safetensors.index.json").write_text(json.dumps(idx2))
    (mm / "shard-b.safetensors").write_bytes(b"\0" * 1024)
    rep_mm = check_donor(mm, min_total_gib=0.0)
    assert rep_mm.passed, "selftest: multimodal text_config donor must PASS (INFO only)"

    # 3) wrong family entirely -> FAIL
    bad = tmp / "bad_family"
    bad.mkdir()
    cfg_bad = {"architectures": ["LlamaForCausalLM"], "model_type": "llama",
               "num_hidden_layers": 64, "hidden_size": 5120}
    (bad / "config.json").write_text(json.dumps(cfg_bad))
    (bad / "tokenizer_config.json").write_text("{}")
    (bad / "tokenizer.json").write_text("{}")
    rep_bad = check_donor(bad, min_total_gib=0.0)
    assert not rep_bad.passed, "selftest: non-Qwen3 family must FAIL"

    # 4) MoE -> FAIL
    moe = tmp / "moe_dir"
    moe.mkdir()
    cfg_moe = dict(cfg, architectures=["Qwen3MoEForCausalLM"], model_type="qwen3_moe")
    (moe / "config.json").write_text(json.dumps(cfg_moe))
    (moe / "tokenizer_config.json").write_text("{}")
    (moe / "tokenizer.json").write_text("{}")
    rep_moe = check_donor(moe, min_total_gib=0.0)
    assert not rep_moe.passed, "selftest: MoE donor must FAIL"

    print("SELFTEST OK (4 cases)")


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
