#!/usr/bin/env python
"""E1 — quality at *measured* bits, on a causal transformer (not a per-position MLP).

The first version of this probe reused the smoke-test architecture from
tests/test_packed_na.py, which has no context mixing: every arm converged to the
bigram floor (3.49 bits/char vs 3.47 for an ideal bigram) and the comparison was
meaningless. This version uses the same attention block as probes/charlm_phase2.py,
which is what the project's historical quality tables were measured with.

Loss is reported in bits/char (divided by ln 2), matching those tables.

Validation is HELD OUT: the corpus (--chars, default whole file) is split contiguously into
train = first (1 - --val-frac) and val = last --val-frac (default 0.1); batches are sampled from
train only. Every eval reports val, a train-loss estimate (fixed train subset) and gap = val - train;
the header prints train_tokens / val_tokens / epochs. `--val-frac 0 --chars 200000` reproduces the
legacy numbers (val measured on the training data = memorisation).

Arms (measured bits/param at d=128):
    dense_bf16   96.00   fp32 master + bf16 autocast + AdamW
    s9_nibble     4.41   MXFP4 E2M1 ladder, stateless SR, nibble-packed
    na5           5.41   PackedNA5
    na6           6.41   PackedNA6
    na_<ladder>_c<C>    PackedNA(LADDERS[ladder], C): ladder in {wide, e2m1, fp4}. fp4 = the full E2M1
                        magnitudes {0,.5,1,1.5,2,3,4,6} (15 signed rungs x (2C-1) accumulator levels):
                        c1 = 15 codes (4 bit), c2 = 45 (6), c4 = 105 (7), c8 = 225 (8), c9 = 255 (8),
                        c10..c16 > 256 codes (9 bit; the unpacked code is held in int16, not uint8).
                        Bits/param = code bits + 0.5 (E4M3/16 scale, --scale-format nvfp4) + RMS.
    tern_counter        main project's ternary counter layer (memory_native.PackedRMSCounterLinear,
                        6-bit state, fp activations saved -> comparable with the mx6 arms);
                        tern_counter_int4 = same + act_save_bits=4 (SHOOTOUT.md "counter+int4").
                        See the TERN_* block below for every hyperparameter and its source.
                        Needs `memory_native` importable (PYTHONPATH=<dir containing memory_native>).
                        --tern-lr sets its lr (default 3e-3); --lr is for the mx6 arms only.

Gap-decomposition arms (opt-in; all linears BIAS-FREE; see the DENSE_/QAT_ block below):
    dense_nobias        nn.Linear(bias=False) + AdamW (incl. weight_decay 0.01): dense_bf16 minus the bias.
    dense_rms           fp32 weights, mx6's RMS-normalised step w -= lr*g_hat (no momentum, no decay),
                        NOT in AdamW. --dense-rms-lr (default 0.01), follows --sched.
    qat_tern            BitNet b1.58 fake-quant (per-row absmean), latent fp32 + AdamW, STE.
    qat_fp4             NVFP4 fake-quant, full E2M1 {0,.5,1,1.5,2,3,4,6}, block 16, E4M3 scale + fp32.
    qat_e2m1lad         same NVFP4 scaling, values restricted to mx6's e2m1 ladder {0,.5,1,6}.
                        --qat-lr (default = --adam-lr) is the lr of the qat_* latent weights.
Opt-in na_* flags: --counter-wd (restoring force toward 0), --scale-lr (learnable E4M3 block scale; with
nvfp4 a block that would pass E4M3's 448 doubles the per-tensor scale first, so it can keep growing),
--scale-growth pressure, --headroom-mode ladder --headroom h (nvfp4: block amax -> L/h on the ladder,
leaving room before the outer rung).
Every arm prints a deployed-weight distribution summary (kurtosis, fraction |w| > 3*rms_row) at the end.

Data / model size / memory (scaling study; every default = the previous behaviour):
    --dataset {tinyshakespeare,enwik8}   enwik8: http://mattmahoney.net/dc/enwik8.zip (mirror data.deepai.org) is
                        downloaded to data/enwik8 if absent, unzipped, and read as BYTES (latin-1: 1 byte = 1 symbol,
                        vocab <= 256). --chars 20000000 reads only the first 20 M bytes; --val-frac splits as always.
    --data-encoding {utf-8,latin-1}      default utf-8 (enwik8: latin-1). latin-1 = byte-level, no newline translation.
    --layers L          transformer blocks (default 2).      --heads H   attention heads (default 1 = the legacy single
                        head with 1/sqrt(d) scaling; H > 1 splits d into H heads of d/H with 1/sqrt(d/H)). MLP = 4d, seq <= 512.
    The header prints the parameter counts (total, block linears = "counter-layer params", other = emb/pos/LN/head).
    Memory (per run; printed per run and in a table at the end):
      n_lin  = 12 d^2 per block (qkv 3d^2 + proj d^2 + fc 4d^2 + out 4d^2)         [counter-layer params]
      state  = measured_bits / 8 * n_lin      bytes of PERSISTENT training state of the block linears.
               AdamW arms (dense_*, qat_*): 96 bit = 12 B/param (fp32 master + AdamW m + v);
               mx6 / lg / tern arms: measured bits (code + block scales + fp32 RMS rows), e.g. 6.69 bit = 0.836 B/param.
      +grad  = state + 4 B * n_lin (AdamW arms: every fp32 gradient is resident until opt.step) or
               state + 4 B * (largest single layer) (in-backward-update arms: grad_w is consumed layer by layer).
      other  = 12 B * (emb + pos + LN + head): fp32 + AdamW in EVERY arm, a constant offset.
      peak   = torch.cuda.max_memory_allocated, reset after the initial telemetry snapshot and after each eval, read before
               the eval / telemetry => a TRAINING peak, minus the flip/edge telemetry snapshot the harness keeps (int8,
               1 B per counter weight). It includes the corpus tensor on the GPU (peak-run subtracts what was already
               allocated at run start). nan on CPU. It is the honest equal-memory number: unlike `state` it contains the
               run-time caches (decoded fp32 weights, unpacked codes, lg's fp32 expanded scale) and the update temporaries.

Usage:
    python probes/e1_quality.py --d 128 --steps 1500 --seeds 0,1
    python probes/e1_quality.py --dataset enwik8 --chars 20000000 --d 256 --layers 4 --heads 4 --steps 6000 --sched cosine
    python probes/e1_quality.py --d 128 --steps 1500 --sched cosine --log-curve

--sched cosine applies one schedule (2 % linear warmup, cosine to 10 %) to the counter lr and
to AdamW (dense arm / embeddings, LN, head), mirroring scripts/run_tail.py. Every
--eval-every steps the probe logs val, flip_rate (fraction of decoded weight values changed
since the previous eval, from the codes, not the weight_flips counters), edge (fraction of
weights on an outermost rung) and clip%.
"""
from __future__ import annotations

import argparse
import contextlib
import math
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mx6 import update as mx_update                 # noqa: E402
from mx6.layer import MX6Linear, MXNLinear          # noqa: E402
from mx6.packed_na import PackedNA, PackedNA5, PackedNA6  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SHAKESPEARE = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
ENWIK8_URLS = ("http://mattmahoney.net/dc/enwik8.zip", "https://data.deepai.org/enwik8.zip")   # primary, mirror
ENWIK8_BYTES = 100_000_000
DATASETS = ("tinyshakespeare", "enwik8")
MAX_SEQ = 512            # CharLM.pos = nn.Embedding(512, d)

# Magnitude ladders, so ladder and counter depth can be varied independently.
LADDERS = {
    "wide": [0.0, 0.5, 3.0],            # NA5's ladder
    "e2m1": [0.0, 0.5, 1.0, 6.0],       # NA6's ladder
    "fp4": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],   # the full FP4 E2M1 magnitudes (15 signed rungs)
}

# Code bits only; the E8M0 scale (+0.25) and the per-row RMS (32/in) are added per width.
# na_<ladder>_c<C> arms are not listed: expected_bits() asks the format (PackedNA.bits =
# ceil(log2(len(signed) * (2C - 1)))), e.g. na_fp4_c1/c2/c4/c8/c16 = 4/6/7/8/9 code bits.
CODE_BITS = {"dense_bf16": 96.0, "s9_nibble": 4.0}

# Set from --rescale / --scale-mode / --clip.
RESCALE_EVERY = 0
SCALE_MODE = "headroom"
CLIP_FRAC = 0.0
# Set from --scale-format / --headroom-mode / --headroom / --scale-growth / --growth-tau.
SCALE_FORMAT = "e8m0"
HEADROOM_MODE = "absolute"
SDPA = False             # True (--sdpa / bench_train_step): Block attention = F.scaled_dot_product_attention(is_causal=True)
HEADROOM = None          # None -> MXNLinear default (1.5; ladder mode: h = 1.0)
SCALE_GROWTH = "off"
GROWTH_TAU = 1.0 / 16
CLIP_WINDOW = 100        # steps averaged for the reported clip fraction
# Set from --counter-wd / --scale-lr (na_* arms only; 0 = off = bit-identical to the old behaviour).
COUNTER_WD = 0.0
SCALE_LR = 0.0

# --- Main project's ternary counter arms ("tern_counter", "tern_counter_int4") -----------------
# Layer: memory_native.packed.PackedRMSCounterLinear == make_linear("counter_packed", ...)
#        (src/memory_native/baselines.py make_linear), the kind named in SHOOTOUT.md:8
#        ("counter_packed + int4 acts"). Bias-free (CLAUDE.md "Counter layers are bias-free");
#        6-bit state packed 4 codes / 3 bytes (packed.py:1-10), requires in_features % 4 == 0.
# NOTE the SHOOTOUT runner itself (a Kaggle kernel) is NOT in the repo, so its exact lr is
# unrecoverable; the values below are the in-repo recipe it corresponds to:
TERN_C = 11              # C=11: tests/test_reversible_gpt.py:11, research/experiments/gpu_acceleration_bench.py:39,
                         #   MotifCL docs/TERNARY15M_PRETRAIN_RESULTS.md:39. Max for packed (packed.py _validate_packed_C).
                         #   (cli.py:100 charlm default is C=8; class default C_DEFAULT=8, counter.py:58.)
TERN_LR = 3e-3           # counter lr (peak). cli.py:102 --fs-lr-rms 3e-3 (the RMS kinds), tests/test_reversible_gpt.py:11.
                         #   NOT the class default 0.04 (counter.py:366, that is for the non-RMS kind). MotifCL used 4e-3
                         #   cosine->4e-4. Override with --tern-lr. Schedule = harness --sched (warmup 2%, cosine to 10%),
                         #   same shape as MotifCL (2% warmup, min 10%).
TERN_LR_SCALE = 2e-4     # per-row scale lr, held CONSTANT (not scheduled): counter.py:367, cli.py:103,
                         #   tests/test_reversible_gpt.py:11. (MotifCL used lr_scale=0.)
TERN_RMS_BETA = 0.9      # counter.py:718 (RMSCounterLinear default; _rms_eager_pre_sr counter.py:99-121); MotifCL same
TERN_RMS_EPS = 1e-3      # counter.py:718; MotifCL same
TERN_INIT_GAIN = 1.0     # counter.py:368; models.py Block passes 1.0 for qkv/fc (0.5/sqrt(L) only for proj/fc2 there - not used here)
# Scales: row-scale layer = one fp32 scale per OUTPUT ROW (counter.py `scale` buffer, [out,1]), no group scales
#   (GroupScale* is a separate kind, not used in the shootout). Persistent per layer: 6 b/param + (scale+v+s_base) 12 B/row.
TERN_ARMS = {            # arm -> act_save_bits (counter.py:372: 0 = save fp x, 4 = save unbiased int4 Q(x) for grad_w only;
                         #   the forward always uses fp x, so act quant only perturbs the update)
    "tern_counter": 0, "tern_counter_noact": 0, "tern_counter_int4": 4,
}
# update path is eager-only: exactly one forward per backward (CLAUDE.md gotchas); evaluate() is @torch.no_grad().


def is_tern_arm(arm: str) -> bool:
    return arm in TERN_ARMS


def is_tern(m: nn.Module) -> bool:
    """A main-project row-scale RMS counter layer (duck-typed so no memory_native import is needed)."""
    return hasattr(m, "rms_beta") and hasattr(m, "_decode_rows") and hasattr(m, "C") \
        and hasattr(m, "in_features")


def _counter_cls():
    """Import memory_native lazily (mx6-only runs must not need it). Needs PYTHONPATH to contain
    a directory with memory_native/; falls back to <ancestor>/src/memory_native in a repo checkout."""
    try:
        from memory_native.packed import PackedRMSCounterLinear
    except ImportError:
        for p in Path(__file__).resolve().parents:
            if (p / "src" / "memory_native" / "__init__.py").exists():
                sys.path.insert(0, str(p / "src"))
                break
        from memory_native.packed import PackedRMSCounterLinear
    return PackedRMSCounterLinear


def tern_persistent_bytes(m: nn.Module) -> int:
    """Persistent bytes of a main-project counter layer: every buffer that goes into state_dict
    (packed state + fp32 row scale + fp32 RMS second moment v + fp32 s_base), i.e. what a
    checkpoint holds. The layer exposes no persistent_bytes(); non-persistent diagnostics
    (weight_flips, update_events, _t_cache) are excluded."""
    skip = getattr(m, "_non_persistent_buffers_set", set())
    return sum(b.numel() * b.element_size() for n, b in m.named_buffers(recurse=False)
               if n not in skip)


def layer_persistent_bytes(m: nn.Module):
    if hasattr(m, "persistent_bytes") and hasattr(m, "in_features"):
        return m.persistent_bytes()
    if is_tern(m):
        return tern_persistent_bytes(m)
    return None


def parse_na(arm: str):
    """'na_wide_c3' -> (ladder name, C)."""
    _, ladder, c = arm.split("_")
    return ladder, int(c[1:])


def _is_latin1(encoding: str) -> bool:
    return encoding.lower().replace("_", "-") in ("latin-1", "latin1", "iso-8859-1")


def load_text(path: Path, chars: int | None = None, encoding: str = "utf-8") -> str:
    """Whole file by default (chars=None). The old probe used chars=200_000 (--chars 200000).

    encoding "utf-8" (default = legacy): text mode (universal newlines), first `chars` characters.
    encoding "latin-1": BYTE-LEVEL, every byte is one character (vocab <= 256), no newline translation,
    and only the first `chars` bytes are read from disk (enwik8 is 100 MB)."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SHAKESPEARE, path)
    if _is_latin1(encoding):
        with open(path, "rb") as f:
            return f.read(-1 if chars is None else chars).decode("latin-1")
    return path.read_text(encoding=encoding)[:chars]


def _download(url: str, dest: Path, timeout: float = 60.0) -> None:
    """Stream `url` to `dest` (socket timeout per read, so a dead mirror fails instead of hanging)."""
    with urllib.request.urlopen(url, timeout=timeout) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)


def ensure_enwik8(path: Path, urls=ENWIK8_URLS, fetch=_download) -> Path:
    """Make sure the raw 100 MB enwik8 file exists at `path` (no extension, as in the zip): download the zip from
    the first URL that works (primary, then mirror), unzip it, delete the zip. `fetch(url, dest)` is injectable
    (tests never touch the network). Does nothing if `path` exists."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    zpath = path.with_name(path.name + ".zip")
    tmp = path.with_name(path.name + ".part")
    errors = []
    for url in urls:
        try:
            print(f"enwik8: downloading {url} -> {zpath}", flush=True)
            fetch(url, zpath)
            with zipfile.ZipFile(zpath) as zf:
                names = [n for n in zf.namelist() if not n.endswith("/")]
                name = next((n for n in names if Path(n).name == "enwik8"), names[0] if names else None)
                if name is None:
                    raise ValueError("empty zip")
                with zf.open(name) as src, open(tmp, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
            tmp.replace(path)
            zpath.unlink(missing_ok=True)
            break
        except Exception as e:                      # URLError, timeout, BadZipFile, ... -> try the mirror
            errors.append(f"{url}: {type(e).__name__}: {e}")
            zpath.unlink(missing_ok=True)
            tmp.unlink(missing_ok=True)
    else:
        raise RuntimeError("could not download enwik8:\n  " + "\n  ".join(errors))
    size = path.stat().st_size
    if size != ENWIK8_BYTES:
        print(f"enwik8: WARNING {path} is {size} bytes, expected {ENWIK8_BYTES}", flush=True)
    return path


def default_data_path(dataset: str) -> Path:
    root = Path(__file__).resolve().parents[1] / "data"
    return root / ("enwik8" if dataset == "enwik8" else "tinyshakespeare.txt")


def load_corpus(dataset: str, path: Path, chars: int | None = None, encoding: str | None = None):
    """-> (text, encoding used). "tinyshakespeare" (utf-8; downloaded if absent) or "enwik8" (downloaded to `path`
    if absent; latin-1 = byte-level). `encoding` overrides the dataset's default."""
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")
    enc = encoding or ("latin-1" if dataset == "enwik8" else "utf-8")
    if dataset == "enwik8":
        ensure_enwik8(path)
    return load_text(path, chars, enc), enc


def encode_text(text: str):
    """text -> (ids int64 [len(text)] on the CPU, chars): chars = sorted(set(text)), ids[i] = chars.index(text[i]).
    The legacy `{c: i}` comprehension, vectorised (20 M chars in about a second instead of a minute)."""
    if not text:
        raise ValueError("empty corpus")
    cp = torch.frombuffer(bytearray(text.encode("utf-32-le", errors="surrogatepass")), dtype=torch.int32)
    uniq, inv = torch.unique(cp, return_inverse=True)
    return inv.to(torch.long), [chr(c) for c in uniq.tolist()]


def split_ids(ids: torch.Tensor, val_frac: float):
    """Contiguous split: train = first (1 - val_frac), val = the rest; no overlap, and the train
    sampler (starts in [0, len(train) - seq - 1)) never reaches into val. val_frac == 0 returns
    (ids, ids): the legacy behaviour where "val" is measured on the training data."""
    if not 0.0 <= val_frac < 1.0:
        raise ValueError("val_frac must be in [0, 1)")
    if val_frac == 0.0:
        return ids, ids
    n_train = int(len(ids) * (1.0 - val_frac))
    return ids[:n_train], ids[n_train:]


# --- Gap-decomposition arms: dense_nobias, dense_rms, qat_tern, qat_fp4, qat_e2m1lad --------------
# The G4 comparison dense_bf16+AdamW vs the counters confounds FORMAT with OPTIMIZER: dense has a bias,
# momentum (AdamW m) and weight decay (AdamW default 0.01); the counters are bias-free, take RMS-
# normalised steps with no momentum and no decay, and the mx6 block scales are frozen. These arms
# change one factor at a time. ALL their linears are bias-free (like every quantized arm; emb / LN /
# head are exactly as in the other arms: fp32 on AdamW at --adam-lr).
#   dense_bf16    nn.Linear(bias=True)  + AdamW                      (existing)
#   dense_nobias  nn.Linear(bias=False) + AdamW                      -> bias effect
#   dense_rms     fp32 weights, mx6's RMS step, no quantisation       -> optimizer effect (vs dense_nobias)
#                                                                      and format effect (vs na_*)
#   qat_*         fp32 latent + AdamW, quantised forward, STE         -> capacity of the FORMAT under a good
#                                                                      optimizer (vs na_* / tern_counter)
DENSE_ADAM_ARMS = ("dense_bf16", "dense_nobias")                 # all weights in one AdamW group
DENSE_RMS_ARM = "dense_rms"
QAT_ARMS = {"qat_tern": "tern", "qat_fp4": "fp4", "qat_e2m1lad": "e2m1lad"}   # arm -> fake-quant mode
# Training-time state of an AdamW arm: fp32 master + m + v = 96 bit/param. For the qat_* arms this is the
# LATENT (training) width, NOT the deployed width of the quantised format (1.58 / ~4.5 bit).
ADAM_STATE_BITS = 96.0
E2M1_FULL = LADDERS["fp4"]                                        # the real FP4 E2M1 magnitudes


def is_qat_arm(arm: str) -> bool:
    return arm in QAT_ARMS


class _DenseRMSFunction(torch.autograd.Function):
    """x @ W^T whose backward also applies the update (the same in-backward pattern as _MXNFunction)."""

    @staticmethod
    def forward(ctx, x, module, token):
        ctx.save_for_backward(x)
        ctx.module = module
        ctx.version = module.weight._version
        return x @ module.weight.t()

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        module = ctx.module
        if module.weight._version != ctx.version:
            raise RuntimeError("weights changed between forward and backward")
        grad_x = grad_out @ module.weight
        grad_w = grad_out.t() @ x
        with torch.no_grad():
            module.apply_update_(grad_w)
        module._pending = False
        return grad_x, None, None


class DenseRMSLinear(nn.Module):
    """Bias-free fp32 linear updated by EXACTLY the mx6 rule, minus the quantisation.

    mx6 (MXNLinear.apply_update_): v_row <- beta*v_row + (1-beta)*mean_in(g^2) per OUTPUT ROW
    (init v_row = rms_eps, beta = 0.9), g_hat = g / max(sqrt(v_row), rms_eps) [mx6/update.py
    rms_update, called here verbatim], then the counter integrates -lr*g_hat. Here the weight
    simply does w <- w - lr*g_hat: no momentum, no weight decay, no bias, init U(+-1/sqrt(in)) (the
    same RNG call as MXNLinear's init). It is a buffer, not a Parameter, so it is NOT in AdamW.

    Why the absolute step lr*g_hat equals the counter's EXPECTED value step: in PackedNA.step the
    accumulator is a progress meter in TICKS, one rung = C ticks, so a tick is gap*s/C in weight units
    (gap = directional ladder gap, s = block scale). The pre-SR tick count added by one update is
    (-lr*g_hat) / (gap*s) * C, i.e. the value-unit step -lr*g_hat divided by the tick size, and the
    stochastic rounding to an integer is unbiased. So E[ticks] * (gap*s/C) = -lr*g_hat: gap, s and C
    cancel, and the counter's latent weight (value + acc/C*gap*s) moves by -lr*g_hat in expectation
    (tests/test_gap_arms.py measures this). The only differences left are the SR noise, the 4-bit
    readout (the forward sees the rung, not the latent) and the clamp at the outer rungs.
    """

    def __init__(self, in_features: int, out_features: int, lr: float = 0.01,
                 rms_beta: float = 0.9, rms_eps: float = 1e-3):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.lr, self.rms_beta, self.rms_eps = lr, rms_beta, rms_eps
        bound = 1.0 / in_features ** 0.5
        self.register_buffer("weight", torch.empty(out_features, in_features).uniform_(-bound, bound))
        self.register_buffer("v_rms", torch.full((out_features, 1), rms_eps, dtype=torch.float32))
        self._token = torch.zeros(0, requires_grad=True)
        self._pending = False

    def persistent_bytes(self) -> int:
        return self.weight.numel() * 4 + self.v_rms.numel() * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected {self.in_features}, got {x.shape[-1]}")
        shape = x.shape
        if torch.is_grad_enabled() and self.training:
            if self._pending:
                raise RuntimeError("DenseRMSLinear forwarded twice before backward")
            self._pending = True
            out = _DenseRMSFunction.apply(x.reshape(-1, shape[-1]), self, self._token)
            return out.reshape(*shape[:-1], self.out_features)
        return x @ self.weight.t()

    def apply_update_(self, grad_w: torch.Tensor) -> None:
        self.v_rms, grad_hat = mx_update.rms_update(
            grad_w.detach().float(), self.v_rms, self.rms_beta, self.rms_eps)
        self.weight.sub_(grad_hat, alpha=self.lr)          # w <- w - lr * g_hat

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, lr={self.lr}"


@torch.no_grad()
def fq_tern(w: torch.Tensor) -> torch.Tensor:
    """BitNet b1.58 weight fake-quant, per OUTPUT ROW: gamma = mean_in|w|, w_q = gamma*clamp(round(w/gamma),-1,1).
    (The paper uses one absmean per tensor; per row is what the task asks for. torch.round = half to even;
    ties have measure zero.) Computed without autograd: the gradient comes from the STE wrapper."""
    g = w.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
    return g * torch.clamp(torch.round(w / g), -1.0, 1.0)


_QAT_FMT = {   # C=1: pure value codebook (no accumulator) -- only fit / round_idx / signed are used
    "fp4": PackedNA(E2M1_FULL, C=1, scale_format="nvfp4"),
    "e2m1lad": PackedNA(LADDERS["e2m1"], C=1, scale_format="nvfp4"),
}


@torch.no_grad()
def fq_nvfp4(w: torch.Tensor, fmt: PackedNA) -> torch.Tensor:
    """NVFP4 fake-quant with the mx6 helpers: block 16 along the input dim, E4M3 block scale (RTN) and one
    fp32 per-tensor scale, both re-fitted from `w` on every call (PackedNA.nvfp4_fit: tscale =
    amax/(L*448), byte = RTN_e4m3(block_amax/(L*tscale)), L = outer rung = 6 for both codebooks); values
    RTN to the codebook with ties AWAY from zero (PackedNA.round_idx, an odd map)."""
    sb, ts = fmt.nvfp4_fit(w)
    s = fmt.expand_scale(sb, ts)
    return fmt.signed.to(w.device)[fmt.round_idx(w / s)] * s


QAT_FQ = {"tern": fq_tern,
          "fp4": lambda w: fq_nvfp4(w, _QAT_FMT["fp4"]),
          "e2m1lad": lambda w: fq_nvfp4(w, _QAT_FMT["e2m1lad"])}


class _STEQuant(torch.autograd.Function):
    """Straight-through estimator: forward = fn(w) EXACTLY (no w + (q - w) round-off), backward = identity."""

    @staticmethod
    def forward(ctx, w, fn):
        return fn(w)

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None


class QATLinear(nn.Linear):
    """Bias-free linear with fp32 latent weights (nn.Parameter -> AdamW) and a quantised forward.

    forward uses quantized_weight() = fake-quant of the latent weight; with ste=True the gradient wrt the
    quantised weight is passed to the latent weight unchanged. `ste=False` is the null twin: the quantiser
    is applied as a plain (non-differentiable) function and the latent weight gets no gradient.
    Same init as nn.Linear(bias=False) (so the same seed gives the same latent weights as dense_nobias)."""

    def __init__(self, in_features: int, out_features: int, mode: str, ste: bool = True):
        super().__init__(in_features, out_features, bias=False)
        if mode not in QAT_FQ:
            raise ValueError(mode)
        self.mode, self.ste = mode, ste

    def quantized_weight(self) -> torch.Tensor:
        fn = QAT_FQ[self.mode]
        return _STEQuant.apply(self.weight, fn) if self.ste else fn(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.quantized_weight())

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, mode={self.mode}"


def make_linear(arm: str, i: int, o: int, lr: float) -> nn.Module:
    if arm == "dense_bf16":
        return nn.Linear(i, o)
    if arm == "dense_nobias":
        return nn.Linear(i, o, bias=False)
    if arm == DENSE_RMS_ARM:
        return DenseRMSLinear(i, o, lr=lr)       # beta 0.9 / eps 1e-3 = MXNLinear's defaults (not the tern ones)
    if arm in QAT_ARMS:
        return QATLinear(i, o, QAT_ARMS[arm])
    if arm in TERN_ARMS:
        return _counter_cls()(i, o, C=TERN_C, lr=lr, lr_scale=TERN_LR_SCALE,
                              rms_beta=TERN_RMS_BETA, rms_eps=TERN_RMS_EPS,
                              init_gain=TERN_INIT_GAIN, act_save_bits=TERN_ARMS[arm])
    if arm == "s9_nibble":
        return MX6Linear(i, o, integrator=0, compact_state=True, lr=lr)
    if arm == "na5":
        return MXNLinear(i, o, PackedNA5, lr=lr)
    if arm == "na6":
        return MXNLinear(i, o, PackedNA6, lr=lr)
    if arm.startswith("na_"):
        ladder, c = parse_na(arm)
        h = HEADROOM if HEADROOM is not None else (1.0 if HEADROOM_MODE == "ladder" else 1.5)
        return MXNLinear(i, o, PackedNA(mags_unsigned=LADDERS[ladder], C=c,
                                        scale_format=SCALE_FORMAT), lr=lr,
                         rescale_every=RESCALE_EVERY, scale_mode=SCALE_MODE,
                         clip_frac=CLIP_FRAC, scale_headroom=h, headroom_mode=HEADROOM_MODE,
                         scale_growth=SCALE_GROWTH, growth_tau=GROWTH_TAU, track_clip=True,
                         weight_decay=COUNTER_WD, scale_lr=SCALE_LR)
    raise ValueError(arm)


class Block(nn.Module):
    """Causal self-attention + MLP, same shape as probes/charlm_phase2.py."""

    def __init__(self, arm: str, d: int, lr: float, heads: int = 1):
        super().__init__()
        if heads < 1 or d % heads:
            raise ValueError(f"heads={heads} must divide d={d}")
        self.d, self.heads = d, heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = make_linear(arm, d, 3 * d, lr)
        self.proj = make_linear(arm, d, d, lr)
        self.ln2 = nn.LayerNorm(d)
        self.fc = make_linear(arm, d, 4 * d, lr)
        self.out = make_linear(arm, 4 * d, d, lr)

    def forward(self, x):
        b, t, _ = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        if SDPA:                     # --sdpa: F.scaled_dot_product_attention (flash / mem-efficient), same maths as below
            hd = self.d // self.heads
            q, k, v = (z.reshape(b, t, self.heads, hd).transpose(1, 2) for z in (q, k, v))
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)      # scale = 1/sqrt(hd) (= 1/sqrt(d) at 1 head)
            y = y.transpose(1, 2).reshape(b, t, self.d)
        elif self.heads == 1:          # the legacy single head (bit-identical to the probe before --heads existed)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)
            mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
            att = att.masked_fill(mask, float("-inf")).softmax(dim=-1)
            y = att @ v
        else:                        # H heads of d/H, scaled by 1/sqrt(d/H)
            hd = self.d // self.heads
            q, k, v = (z.reshape(b, t, self.heads, hd).transpose(1, 2) for z in (q, k, v))
            att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
            mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
            att = att.masked_fill(mask, float("-inf")).softmax(dim=-1)
            y = (att @ v).transpose(1, 2).reshape(b, t, self.d)
        x = x + self.proj(y)
        return x + self.out(torch.nn.functional.gelu(self.fc(self.ln2(x))))


class CharLM(nn.Module):
    def __init__(self, arm: str, vocab: int, d: int = 128, blocks: int = 2, lr: float = 0.01, heads: int = 1):
        super().__init__()
        self.arm = arm
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(512, d)
        self.blocks = nn.ModuleList([Block(arm, d, lr, heads) for _ in range(blocks)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        t = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))[None]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))


def measured_bits(model: nn.Module, arm: str) -> float:
    """Bits per counter parameter actually held; embedding/head stay dense on both sides."""
    if arm in DENSE_ADAM_ARMS or arm in QAT_ARMS:
        return ADAM_STATE_BITS          # training state (fp32 master + AdamW m, v); see ADAM_STATE_BITS
    total = 0
    params = 0
    for m in model.modules():
        b = layer_persistent_bytes(m)
        if b is not None:
            total += b
            params += m.in_features * m.out_features
    return total * 8 / params


def expected_bits(model: nn.Module, arm: str) -> float:
    if arm in DENSE_ADAM_ARMS or arm in QAT_ARMS:
        return ADAM_STATE_BITS
    total = 0.0
    params = 0
    for m in model.modules():
        if is_tern(m):                          # 6-bit packed code + fp32 {scale, v, s_base} per row
            p = m.in_features * m.out_features
            total += (6.0 + 3 * 32.0 / m.in_features) * p
            params += p
        elif isinstance(m, DenseRMSLinear):     # fp32 weight + fp32 v_rms per row (no quantisation)
            p = m.in_features * m.out_features
            total += (32.0 + 32.0 / m.in_features) * p
            params += p
        elif hasattr(m, "persistent_bytes") and hasattr(m, "in_features"):
            p = m.in_features * m.out_features
            cb = CODE_BITS.get(arm)
            if cb is None:                      # a PackedNA variant: ask the format
                cb = m.fmt.bits
            sb = getattr(m.fmt, "scale_bits", 0.25) if hasattr(m, "fmt") else 0.25
            ts = 32.0 / p if getattr(getattr(m, "fmt", None), "nvfp4", False) else 0.0
            vs = 32.0 / m.in_features if getattr(m, "v_scale", None) is not None else 0.0  # --scale-lr: fp32/row
            total += (cb + sb + 32.0 / m.in_features + ts + vs) * p
            params += p
    return total / params


ADAMW_STATE_BYTES = ADAM_STATE_BITS / 8.0       # 12 B/param: fp32 master + AdamW m + v


def block_linears(model: nn.Module) -> list:
    """The 4 linears (qkv, proj, fc, out) of every block, in order = the "counter layers" of the quantised arms."""
    return [lin for blk in model.blocks for lin in (blk.qkv, blk.proj, blk.fc, blk.out)]


def autocast_ctx(precision: str = "fp32", device: str | None = None):
    """--precision bf16: torch.autocast(bfloat16) around the model FORWARD (the weights stay fp32 masters; backward runs
    outside it, autograd replays the forward casts). fp32 (default): a no-op context = the old behaviour."""
    if precision == "fp32":
        return contextlib.nullcontext()
    if precision != "bf16":
        raise ValueError(f"precision must be fp32 or bf16, got {precision!r}")
    return torch.autocast(device_type=device or DEV, dtype=torch.bfloat16)


def fp32_islands(model: nn.Module) -> int:
    """bf16 support for the CUSTOM-autograd block linears (mx6 / latent-grid / tern / dense_rms: x @ W^T lives inside a
    torch.autograd.Function whose backward runs OUTSIDE autocast -> bf16 grad_out vs fp32 saved x = dtype mismatch).
    Each such linear's forward becomes: autocast OFF, input cast to fp32, fp32 output (instance attribute; the module
    and its apply_update_ / weight() are untouched). Consequence: the counter arms' OWN matmuls stay fp32 (no tensor
    cores) while attention / LN / head follow autocast; nn.Linear / QATLinear arms are plain autocast. Exception:
    LatentGridLinear layers on the native path (`native_enabled()`, i.e. triton backend on CUDA) run bf16 end to end
    (bf16 decode + cuBLAS GEMMs + bf16 grad into the fused update) and are NOT islands. Returns #islands."""
    n = 0
    for lin in block_linears(model):
        if isinstance(lin, nn.Linear) or "forward" in lin.__dict__:
            continue
        native = getattr(lin, "native_enabled", None)
        if callable(native) and native():        # LatentGridLinear native path: casts x to the autocast dtype itself
            continue

        def fwd(x, _orig=lin.forward):
            with torch.autocast(device_type=x.device.type, enabled=False):
                return _orig(x.float())
        lin.forward = fwd
        n += 1
    return n


def compile_kind(arm: str):
    """--compile support: "lg" (latent grid: opaque custom ops, fullgraph), "plain" (dense_* / qat_*: plain torch.compile),
    None (na_* / tern_* / dense_rms / s9 ...: the in-backward state mutation cannot be traced -> refused)."""
    if arm.startswith("lg"):
        return "lg"
    if arm in DENSE_ADAM_ARMS or is_qat_arm(arm):
        return "plain"
    return None


def compile_refusal(arm: str):
    """None if `arm` can be compiled, else the message that --compile prints for it."""
    if compile_kind(arm) is not None:
        return None
    return (f"--compile is not supported for arm {arm!r}: its state is updated by a custom autograd.Function inside backward "
            f"(na_* counters, tern_*, dense_rms, s9_*), which torch.compile cannot trace without graph breaks / double "
            f"updates. Compilable arms: lg*_fp4*, dense_bf16 / dense_nobias, qat_*. Drop this arm or --compile.")


def compile_model(model: nn.Module, arm: str, backend: str | None = None):
    """-> (callable to use for the TRAINING forward, info dict). lg arms: every LatentGridLinear is switched to the native
    path if it is not yet (CPU / no Triton: the torch backend, so the ops path exists) and to torch.ops.mx6.lg_linear
    (prepare_lg_for_compile), then torch.compile(fullgraph=True, dynamic=False): a graph break raises. The cosine schedule
    changes lr every step WITHOUT recompiling (lr lives in a device buffer, see mx6/latent_grid.py). Evaluation under
    no_grad should keep using the eager `model` (same modules; the compiled one is never run in eval mode)."""
    msg = compile_refusal(arm)
    if msg:
        raise ValueError(msg)
    import torch._dynamo as dynamo
    dynamo.reset()
    dynamo.utils.counters.clear()
    info = {"kind": compile_kind(arm), "backend": backend or "inductor"}
    if info["kind"] == "lg":
        from mx6.latent_grid import HAS_LG_OPS, lg_layers, prepare_lg_for_compile
        if not HAS_LG_OPS:
            raise RuntimeError("--compile for lg arms needs torch.library.register_autograd / register_fake (torch >= 2.4)")
        for m in lg_layers(model):
            if not m.native_enabled():
                m.native = True
        prepare_lg_for_compile(model, "op")
        kw = dict(fullgraph=True)
    else:
        kw = {}
    if backend:
        kw["backend"] = backend
    return torch.compile(model, dynamic=False, **kw), info


def compile_stats() -> dict:
    """Dynamo counters of the current process: unique graphs (1 = no recompiles over the whole run) and graph breaks."""
    import torch._dynamo.utils as du
    c = du.counters
    return {"graphs": int(c["stats"]["unique_graphs"]), "graph_breaks": int(sum(c["graph_break"].values()))}


def fwd_logits(model: nn.Module, x: torch.Tensor, precision: str = "fp32") -> torch.Tensor:
    """model(x) under the precision; logits returned in fp32 for the loss."""
    with autocast_ctx(precision, x.device.type):
        out = model(x)
    return out.float()


def count_params(model: nn.Module) -> dict:
    """{'total', 'lin', 'other'}. lin = in*out of the block linears (12 d^2 per block; dense_bf16 biases NOT included),
    other = every other nn.Parameter (token / position embedding, LayerNorms, head, dense_bf16 biases)."""
    lins = block_linears(model)
    w_ids = {id(m.weight) for m in lins if isinstance(getattr(m, "weight", None), nn.Parameter)}
    n_lin = sum(m.in_features * m.out_features for m in lins)
    n_other = sum(p.numel() for p in model.parameters() if id(p) not in w_ids)
    return {"total": n_lin + n_other, "lin": n_lin, "other": n_other}


def analytic_params(d: int, blocks: int, vocab: int) -> dict:
    """count_params without building a model (bias-free arms): lin = 12 d^2 per block; other = token vocab*d +
    position MAX_SEQ*d + LayerNorms (2 per block + ln_f) * 2d + head d*vocab."""
    n_lin = blocks * 12 * d * d
    n_other = 2 * vocab * d + MAX_SEQ * d + (2 * blocks + 1) * 2 * d
    return {"total": n_lin + n_other, "lin": n_lin, "other": n_other}


def memory_accounting(model: nn.Module, arm: str, bits: float) -> dict:
    """Analytic TRAINING-state bytes of the block linears (formulas in the module docstring; no CUDA needed).
    state_bytes = bits/8 * n_lin (bits = measured_bits: 96 for AdamW arms, the persistent bits for the counters);
    grad_bytes = 4*n_lin for the AdamW arms (all gradients resident), 4*(largest layer) for the in-backward arms;
    other_state_bytes = 12 * n_other (emb / pos / LN / head are fp32 + AdamW in every arm)."""
    pc = count_params(model)
    adam_arm = arm in DENSE_ADAM_ARMS or arm in QAT_ARMS
    biggest = max(m.in_features * m.out_features for m in block_linears(model))
    return {"n_total": pc["total"], "n_lin": pc["lin"], "n_other": pc["other"], "bits": bits,
            "state_bytes": bits / 8.0 * pc["lin"],
            "grad_bytes": 4.0 * (pc["lin"] if adam_arm else biggest),
            "other_state_bytes": ADAMW_STATE_BYTES * pc["other"]}


def clip_frac_now(model: nn.Module):
    """Parameter-weighted fraction of weights clipped at an outer rung this step (0-dim
    tensor, no host sync), or None if no layer tracks clipping."""
    num, den = None, 0
    for m in model.modules():
        f = getattr(m, "last_clip_frac", None)
        if f is not None:
            p = m.in_features * m.out_features
            num = f * p if num is None else num + f * p
            den += p
    return None if num is None else num / den


def grown_blocks(model: nn.Module) -> int:
    return sum(getattr(m, "total_grown", 0) for m in model.modules())


def magnitude_histogram(model: nn.Module):
    """Fraction of codes at each magnitude index, summed over all counter layers.

    Shows whether the outlier slot of a ladder is actually reachable during
    training, or whether the bulk sits on a coarse sub-grid.
    """
    counts = None
    fmt = None
    tern = None
    for m in model.modules():
        if hasattr(m, "fmt") and hasattr(m, "codes"):
            idx = (m.codes.to(torch.long) // m.fmt.acc_levels).flatten()
            c = torch.bincount(idx, minlength=m.fmt.n_vals).float()
            counts = c if counts is None else counts + c
            fmt = m.fmt
        elif is_tern(m):                         # ternary counter: fraction at -1 / 0 / +1
            t, _ = m._decode_rows(0, m.out_features)
            c = torch.bincount((t.to(torch.long) + 1).flatten(), minlength=3).float()
            tern = c if tern is None else tern + c
    if counts is None and tern is not None:
        return [-1, 0, 1], (tern / tern.sum()).tolist()
    if counts is None:
        return None
    counts = counts / counts.sum()
    return fmt.signed.tolist(), counts.tolist()


def format_histogram(pair) -> str:
    if pair is None:
        return "(no counter layers)"
    values, fracs = pair
    return "  ".join(f"{v:+g}:{f*100:.1f}%" for v, f in zip(values, fracs) if f > 0.0005)


def sched_lr(sched: str, base: float, step: int, total: int,
             warmup_frac: float = 0.02, min_frac: float = 0.1) -> float:
    """Learning rate at `step` (0-based) of `total`.

    const : base.
    cosine: linear warmup 0 -> base over W = round(warmup_frac * total) steps
            (lr = base * (step + 1) / W for step < W), then
            lr_min + 0.5 * (base - lr_min) * (1 + cos(pi * t / T)),
            t = step - W, T = total - W, lr_min = min_frac * base.
            So lr(W) == base and lr(total) == lr_min (the last real step, total - 1,
            is one tick above it). Same shape as scripts/run_tail.py (cosine) and the
            MotifCL ternary-15M pretrain (2 % warmup, min 10 %).
    """
    if sched == "const":
        return base
    if sched != "cosine":
        raise ValueError(sched)
    w = int(round(warmup_frac * total))
    if step < w:
        return base * (step + 1) / w
    lo = min_frac * base
    t, big_t = step - w, max(total - w, 1)
    return lo + 0.5 * (base - lo) * (1.0 + math.cos(math.pi * t / big_t))


def set_counter_lr(model: nn.Module, lr: float, scale_frac: float | None = None) -> None:
    """Counter layers read `self.lr` at every apply_update_, so setting it per step is enough.
    `scale_frac` (= scheduled lr / peak lr): layers with a learnable block scale (--scale-lr) get
    scale_lr = scale_lr_base * scale_frac, so it follows the same schedule; None leaves it alone."""
    for m in model.modules():
        if hasattr(m, "lr") and (hasattr(m, "apply_update_") or is_tern(m)):
            m.lr = lr            # main-project counter layer: lr read in _eff_lr()/_fused_update (counter.py:516)
        if scale_frac is not None and getattr(m, "scale_lr_base", 0.0) > 0.0:
            m.scale_lr = m.scale_lr_base * scale_frac


@torch.no_grad()
def deployed_weight(m: nn.Module) -> torch.Tensor:
    """The weight the forward pass actually multiplies by: fake-quantised (QAT), decoded rung * scale
    (mx6), row-scale * ternary (tern counter), or the fp32 weight (dense arms)."""
    if hasattr(m, "quantized_weight"):
        return m.quantized_weight().detach().float()
    if is_tern(m):
        return m._dense_weight(torch.float32).detach()
    w = m.weight() if callable(m.weight) else m.weight
    return w.detach().float()


@torch.no_grad()
def weight_stats(model: nn.Module) -> dict:
    """Distribution of the DEPLOYED weights of the 4 block linears (qkv/proj/fc/out; emb/LN/head are fp32 in
    every arm), pooled over rows and layers, comparable across arms (dense, QAT, counters):
      kurt   mean over rows of E[w^4] / E[w^2]^2 (4th moment about 0 = Pearson kurtosis for zero-mean rows):
             Gaussian 3, uniform 1.8, a two-point +-a distribution 1; heavy tails / peaked >> 3.
             All-zero rows are skipped.
      tail3  fraction of weights with |w| > 3 * rms_row (Gaussian: 0.27 %).
      zero   fraction of exactly-zero weights (quantised arms: mass on the zero rung)."""
    k_sum, k_n, n_tail, n_zero, n = 0.0, 0, 0.0, 0.0, 0
    for blk in model.blocks:
        for lin in (blk.qkv, blk.proj, blk.fc, blk.out):
            w = deployed_weight(lin)
            ms = w.square().mean(dim=1)
            ok = ms > 0
            k_sum += float((w.pow(4).mean(dim=1)[ok] / ms[ok].square()).sum())
            k_n += int(ok.sum())
            n_tail += float((w.abs() > 3.0 * ms.sqrt().unsqueeze(1)).sum())
            n_zero += float((w == 0).sum())
            n += w.numel()
    return {"kurt": k_sum / max(k_n, 1), "tail3": n_tail / n, "zero": n_zero / n}


@torch.no_grad()
def snapshot_values(model: nn.Module):
    """Decoded value index (magnitude rung, accumulator residue stripped) of every counter
    weight, one int8 tensor per layer, plus the rung count n_vals (None: no 'edge' notion)."""
    snap = []
    for m in model.modules():
        if hasattr(m, "fmt") and hasattr(m, "codes"):            # MXNLinear (PackedNA*)
            idx = (m.codes.to(torch.long) // m.fmt.acc_levels).clamp(0, m.fmt.n_vals - 1)
            snap.append((idx.to(torch.int8), m.fmt.n_vals))
        elif hasattr(m, "apply_update_") and hasattr(m, "codes"):  # MX6Linear: codes are values
            snap.append((m.codes.to(torch.int8), None))
        elif is_tern(m):                     # ternary counter: value = t in {-1,0,+1}; nv=-1 marks "ternary"
            t, _ = m._decode_rows(0, m.out_features)
            snap.append((t.to(torch.int8), -1))
    return snap


def flip_edge(prev, cur):
    """(flip_rate, edge): fraction of weights whose decoded value changed between two
    snapshots, and fraction currently on an outermost rung (nan if not defined)."""
    if not cur:
        return float("nan"), float("nan")
    flips = torch.zeros((), device=cur[0][0].device)
    edge = torch.zeros((), device=cur[0][0].device)
    n = n_edge = 0
    for (a, _), (b, nv) in zip(prev, cur):
        flips += (a != b).sum()
        n += b.numel()
        if nv == -1:                      # ternary counter: "edge" = fraction of NONZERO weights (t = +-1)
            edge += (b != 0).sum()
            n_edge += b.numel()
        elif nv is not None:
            edge += ((b == 0) | (b == nv - 1)).sum()
            n_edge += b.numel()
    return float(flips) / n, (float(edge) / n_edge if n_edge else float("nan"))


@torch.no_grad()
def evaluate(model, data, seq=128, iters=10, precision="fp32") -> float:
    model.eval()
    gen = torch.Generator(device="cpu").manual_seed(1234)
    total = 0.0
    for _ in range(iters):
        starts = torch.randint(0, data.numel() - seq - 1, (32,), generator=gen)
        x = torch.stack([data[s:s + seq] for s in starts]).to(DEV)
        y = torch.stack([data[s + 1:s + seq + 1] for s in starts]).to(DEV)
        total += float(nn.functional.cross_entropy(
            fwd_logits(model, x, precision).reshape(-1, model.head.out_features), y.reshape(-1)))
    model.train()
    return total / iters / math.log(2)   # bits/char


def _cuda_mem() -> bool:
    """Track torch.cuda memory in run_arm? (a function so tests can fake the CUDA calls on a CPU box)"""
    return DEV == "cuda"


def build_model_opt(arm: str, vocab: int, d: int, lr: float, adam_lr: float, blocks: int = 2, heads: int = 1):
    """(model on DEV, AdamW) exactly as run_arm trains them (probes/bench_train_step.py builds through this too).
    Seed with torch.manual_seed BEFORE calling (the layer inits draw from the global RNG)."""
    model = CharLM(arm, vocab, d=d, blocks=blocks, lr=lr, heads=heads).to(DEV)
    # AdamW groups. Default: ONE group with every trainable parameter at adam_lr (dense arms: all
    # weights; counter / dense_rms arms: embeddings / LN / head -- their linears are buffers). The qat_*
    # arms add a second group with the latent linear weights at `lr` (= --qat-lr); same AdamW settings
    # (default weight_decay 0.01, betas) in both groups, i.e. identical to dense_nobias when qat_lr == adam_lr.
    qat_w = [m.weight for m in model.modules() if isinstance(m, QATLinear)]
    qat_ids = {id(p) for p in qat_w}
    groups = [{"params": [p for p in model.parameters() if p.requires_grad and id(p) not in qat_ids],
               "base_lr": adam_lr}]
    if qat_w:
        groups.append({"params": qat_w, "base_lr": lr})
    return model, torch.optim.AdamW(groups, lr=adam_lr)


def run_arm(arm: str, seed: int, d: int, steps: int, batch: int, seq: int,
            ids: torch.Tensor, vocab: int, lr: float, adam_lr: float,
            sched: str = "const", warmup_frac: float = 0.02, lr_min_frac: float = 0.1,
            eval_every: int = 100, val_ids: torch.Tensor | None = None,
            blocks: int = 2, heads: int = 1, precision: str = "fp32", compile: bool = False,
            compile_backend: str | None = None, accum: int = 1, accum_mode: str = "buffer",
            accum_dtype: str = "bf16", momentum: float = 0.0, momentum_dtype: str = "bf16",
            nesterov: bool = False) -> dict:
    """compile: torch.compile the TRAINING forward (see compile_model); eval stays eager. `ids` is the TRAINING tensor (batches are sampled from it only). `val_ids` is the held-out
    tensor "val" is measured on; None -> `ids` itself (legacy: val == train). Each eval also
    estimates the train loss on a fixed subset of `ids` (same fixed starts every time)."""
    same = val_ids is None or val_ids is ids
    ids = ids.to(DEV)                    # data built on CPU while DEV=cuda (tests, notebooks): sample indices live on DEV
    val_ids = ids if same else val_ids.to(DEV)
    cuda = _cuda_mem()
    base_mem, peak = 0, 0
    if cuda:                       # per-run peak: what is allocated now (corpus tensors, ...) is the baseline
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base_mem = torch.cuda.memory_allocated()
    torch.manual_seed(seed)
    model, opt = build_model_opt(arm, vocab, d, lr, adam_lr, blocks, heads)
    acc_state = None
    if accum > 1:                     # opt-in gradient accumulation; `batch` is then the MICRO-batch (effective = batch*accum)
        from mx6.latent_grid import lg_layers
        if lg_layers(model):          # lg arms: mx6/lg_accum (buffer | latent); other arms: plain autograd accumulation
            from mx6.lg_accum import set_accum
            acc_state = set_accum(model, accum, accum_mode, accum_dtype, loss_scaled=True)
    if momentum > 0.0:                # opt-in per-weight momentum for lg arms (mx6/lg_accum.set_momentum); no-op for others
        from mx6.latent_grid import lg_layers
        if lg_layers(model):
            if accum > 1:
                raise ValueError("--momentum and --accum both use the per-layer lg buffer; pick one")
            from mx6.lg_accum import set_momentum
            set_momentum(model, momentum, momentum_dtype, nesterov)
    train_fwd, cinfo = model, None
    if compile:                       # BEFORE fp32_islands: it must see the lg layers as native (no fp32 island)
        train_fwd, cinfo = compile_model(model, arm, compile_backend)
    if precision != "fp32":
        fp32_islands(model)
    bits = measured_bits(model, arm)
    want = expected_bits(model, arm)
    if abs(bits - want) > 0.02:
        raise AssertionError(f"{arm}: measured {bits:.3f} bits != expected {want:.3f}")
    mem = memory_accounting(model, arm, bits)
    model.train()
    t0 = time.time()
    clips = []
    # One schedule shape for everything: counter lr and every AdamW group (dense arm: all
    # weights; quantized arms: embeddings / LN / head; qat arms: + the latent linear weights).
    prev = snapshot_values(model)
    tele_bytes = sum(t.numel() * t.element_size() for t, _ in prev)   # flip/edge snapshot: telemetry, not training memory
    if cuda:                         # the training window starts here (the snapshot above used int64 temporaries)
        torch.cuda.reset_peak_memory_stats()
    curve = []                       # (step, val, lr, flip_rate, edge, clip) per eval
    for step in range(steps):
        cur_lr = sched_lr(sched, lr, step, steps, warmup_frac, lr_min_frac)
        set_counter_lr(model, cur_lr, scale_frac=(cur_lr / lr if lr > 0 else 0.0))
        for g in opt.param_groups:
            g["lr"] = sched_lr(sched, g["base_lr"], step, steps, warmup_frac, lr_min_frac)
        idx = torch.randint(0, len(ids) - seq - 1, (batch * accum,), device=DEV)
        x = ids[idx.unsqueeze(1) + torch.arange(seq, device=DEV)]
        y = ids[idx.unsqueeze(1) + torch.arange(1, seq + 1, device=DEV)]
        if accum == 1:
            loss = nn.functional.cross_entropy(fwd_logits(train_fwd, x, precision).reshape(-1, vocab), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        else:                         # ONE optimizer step over `accum` micro-batches of the SAME windows a batch*accum run uses
            opt.zero_grad(set_to_none=True)
            ctx = contextlib.nullcontext()
            if acc_state is not None:
                from mx6.lg_accum import lg_accumulate
                ctx = lg_accumulate(model, accum)
            with ctx as acc:
                loss = 0.0
                for i in (acc.micro_batches() if acc is not None else range(accum)):
                    li = nn.functional.cross_entropy(
                        fwd_logits(train_fwd, x.chunk(accum)[i], precision).reshape(-1, vocab), y.chunk(accum)[i].reshape(-1))
                    (li / accum).backward()
                    loss = loss + li.detach() / accum
            opt.step()
        c = clip_frac_now(model)
        if c is not None:
            clips.append(c)
            if len(clips) > CLIP_WINDOW:
                clips.pop(0)
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            if cuda:               # training-window peak, read BEFORE the telemetry / eval below allocate
                peak = max(peak, torch.cuda.max_memory_allocated())
            cur = snapshot_values(model)
            flip, edge = flip_edge(prev, cur)       # vs the previous eval (or init)
            prev = cur
            clip_now = float(torch.stack(clips).mean()) if clips else float("nan")
            v = evaluate(model, val_ids, seq=seq, precision=precision)
            tr = v if val_ids is ids else evaluate(model, ids, seq=seq, precision=precision)   # train-loss estimate
            curve.append((step + 1, v, cur_lr, flip, edge, clip_now, tr))
            if cuda:
                torch.cuda.reset_peak_memory_stats()
    elapsed = time.time() - t0
    clip = float(torch.stack(clips).mean()) if clips else float("nan")
    if cinfo is not None:
        cinfo.update(compile_stats())
        del train_fwd

    val = curve[-1][1]
    flips = [c[3] for c in curve]
    return {"arm": arm, "seed": seed, "bits": bits, "val": val, "sec": elapsed, "compile": cinfo,
            "hist": magnitude_histogram(model), "clip": clip,
            "grown": grown_blocks(model), "curve": curve,
            "best": min(c[1] for c in curve),
            "flip": (flips[0], flips[len(flips) // 2], flips[-1]),
            "edge": curve[-1][4],
            "train": curve[-1][6], "gap": curve[-1][1] - curve[-1][6],   # gap = val - train
            "wstats": weight_stats(model),
            "scale_moves": sum(getattr(m, "total_scale_moves", 0) for m in model.modules()),
            "rebased": sum(getattr(m, "total_rebased", 0) for m in model.modules()),
            # parameter counts + analytic training-state MB (module docstring) + measured CUDA peak (nan on CPU)
            "n_total": mem["n_total"], "n_lin": mem["n_lin"], "n_other": mem["n_other"],
            "state_mb": mem["state_bytes"] / 2**20,
            "state_grad_mb": (mem["state_bytes"] + mem["grad_bytes"]) / 2**20,
            "other_state_mb": mem["other_state_bytes"] / 2**20,
            "peak_mem_mb": (peak - tele_bytes) / 2**20 if cuda else float("nan"),
            "peak_mem_delta_mb": (peak - tele_bytes - base_mem) / 2**20 if cuda else float("nan")}


def arm_lr(arm: str, args) -> float:
    """Peak lr handed to run_arm for `arm`: tern_* -> --tern-lr, dense_rms -> --dense-rms-lr, qat_* ->
    --qat-lr (default --adam-lr; it is the AdamW lr of the latent weights), everything else -> --lr."""
    if is_tern_arm(arm):
        return args.tern_lr
    if arm == DENSE_RMS_ARM:
        return args.dense_rms_lr
    if is_qat_arm(arm):
        return args.qat_lr if args.qat_lr is not None else args.adam_lr
    return args.lr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1,
                    help="gradient accumulation: N micro-batches of --batch per OPTIMIZER step (effective batch = batch*N). "
                         "Default 1 = off (unchanged). lg arms: mx6/lg_accum (see --accum-mode); other arms: plain autograd")
    ap.add_argument("--accum-mode", default="buffer", choices=["buffer", "latent"],
                    help="lg arms: buffer = per-layer grad buffer, ONE update per optimizer step (== big batch); latent = every "
                         "micro-batch updates with lr/N and beta^(1/N), no buffer")
    ap.add_argument("--accum-dtype", default="bf16", choices=["bf16", "fp8"],
                    help="lg buffer mode: bf16 (+2 B/param) or fp8 e4m3 + per-row scale + stochastic rounding (~ +1 B/param)")
    ap.add_argument("--momentum", type=float, default=0.0,
                    help="lg arms: per-weight momentum mu (EMA of the gradient fed to the RMS-normalised update). 0 = off")
    ap.add_argument("--momentum-dtype", default="bf16", choices=["bf16", "fp8"],
                    help="lg momentum buffer: bf16 (+2 B/param) or fp8 e4m3 + row scale + SR (~ +1 B/param)")
    ap.add_argument("--nesterov", action="store_true", help="lg momentum: nesterov direction mu*buf + (1-mu)*g")
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--lr", type=float, default=0.01,
                    help="counter lr of the mx6 arms (NOT used by tern_* arms, see --tern-lr)")
    ap.add_argument("--tern-lr", type=float, default=TERN_LR,
                    help="peak counter lr of the tern_* arms (main-project ternary counter); "
                         "follows --sched like --lr. Default = recipe value 3e-3")
    ap.add_argument("--adam-lr", type=float, default=3e-3)
    ap.add_argument("--dense-rms-lr", type=float, default=0.01,
                    help="peak lr of the dense_rms arm (fp32 weights, mx6's RMS-normalised step, no "
                         "quantisation); follows --sched like --lr")
    ap.add_argument("--qat-lr", type=float, default=None,
                    help="peak AdamW lr of the qat_* latent weights (default: --adam-lr); follows --sched")
    ap.add_argument("--counter-wd", type=float, default=0.0,
                    help="na_* arms: restoring force g_hat += lambda * (w/s), w/s = the dimensionless "
                         "ladder value of each weight (0 = off = old behaviour)")
    ap.add_argument("--scale-lr", type=float, default=0.0,
                    help="na_* arms, --scale-format nvfp4 only: learnable E4M3 block scale; the log-scale "
                         "step lr (RMS-normalised grad wrt ln(scale)), follows --sched (0 = frozen scales)")
    ap.add_argument("--arms", default="na_wide_c3,na_wide_c5,na_e2m1_c3,na_e2m1_c5")
    ap.add_argument("--sched", default="const", choices=["const", "cosine"],
                    help="lr schedule for counter lr AND AdamW (dense / FP params), same shape: "
                         "const (legacy) or linear warmup + cosine to lr_min")
    ap.add_argument("--warmup-frac", type=float, default=0.02, help="cosine: warmup fraction of steps")
    ap.add_argument("--lr-min-frac", type=float, default=0.1, help="cosine: lr_min = frac * lr")
    ap.add_argument("--eval-every", type=int, default=100,
                    help="val + flip_rate/edge telemetry every N steps (and at the last step)")
    ap.add_argument("--log-curve", action="store_true", help="print the val curve of every run")
    ap.add_argument("--hist", action="store_true", help="print magnitude histograms")
    ap.add_argument("--scale-mode", default="headroom", choices=["headroom", "mse"])
    ap.add_argument("--clip", type=float, default=0.0, help="clip fraction for the MSE scale")
    ap.add_argument("--rescale", type=int, default=0,
                    help="refresh the E8M0 block scale every N updates (0 = frozen)")
    ap.add_argument("--scale-format", default="e8m0", choices=["e8m0", "nvfp4"],
                    help="block scale of the na_* arms: E8M0/32 (default) or NVFP4 E4M3/16 + fp32")
    ap.add_argument("--headroom-mode", default="absolute", choices=["absolute", "ladder"],
                    help="absolute: --headroom is amax/s (legacy); ladder: h relative to ladder max")
    ap.add_argument("--headroom", type=float, default=None,
                    help="scale_headroom (default 1.5; with --headroom-mode ladder default h=1.0)")
    ap.add_argument("--scale-growth", default="off", choices=["off", "pressure"],
                    help="bump a block scale one step when its clip count exceeds tau*block")
    ap.add_argument("--growth-tau", type=float, default=1.0 / 16)
    ap.add_argument("--chars", type=int, default=None,
                    help="use only the first N chars of the corpus (default: whole file; the legacy "
                         "probe used 200000)")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="held-out fraction: contiguous split, train = first 1-frac, val = last frac "
                         "(no overlap). 0 = legacy behaviour, val measured on the TRAINING data")
    ap.add_argument("--dataset", default="tinyshakespeare", choices=DATASETS,
                    help="corpus: tinyshakespeare (default, utf-8) or enwik8 (downloaded to data/enwik8 if absent, "
                         "read as bytes = latin-1; use --chars 20000000 for the first 20 M bytes)")
    ap.add_argument("--data", default=None,
                    help="corpus file (default: data/tinyshakespeare.txt, or data/enwik8 with --dataset enwik8)")
    ap.add_argument("--data-encoding", default=None, choices=["utf-8", "latin-1"],
                    help="utf-8 (default; enwik8: latin-1). latin-1 = BYTE-LEVEL: one byte = one symbol, vocab <= 256")
    ap.add_argument("--layers", type=int, default=2, help="transformer blocks (default 2)")
    ap.add_argument("--heads", type=int, default=1,
                    help="attention heads (default 1 = the legacy single head, 1/sqrt(d) scaling); must divide --d; "
                         "H > 1 gives H heads of d/H scaled by 1/sqrt(d/H). e.g. d/64")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "bf16"],
                    help="fp32 (default, unchanged) or bf16 = torch.autocast(bfloat16) for forward/backward, fp32 master "
                         "weights; custom-autograd counter linears stay fp32 inside (fp32_islands)")
    ap.add_argument("--sdpa", action="store_true",
                    help="attention via F.scaled_dot_product_attention(is_causal=True) instead of the naive T x T "
                         "softmax (same maths, different rounding; default OFF so quality numbers stay unchanged)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the TRAINING forward: lg arms = opaque custom ops + fullgraph=True (graph break "
                         "raises), dense_* / qat_* = plain compile; na_* / tern_* / dense_rms are refused. Works with "
                         "--precision bf16, --sdpa and --sched cosine (lr is a device buffer: no per-step recompiles; the "
                         "graph count is printed after each run). Evaluation stays eager (no_grad, never updates).")
    ap.add_argument("--compile-backend", default=None,
                    help="torch.compile backend (default inductor; aot_eager for a CPU smoke test without a C++ compiler)")
    args = ap.parse_args(argv)
    global SDPA
    SDPA = bool(args.sdpa)
    if args.compile:
        for a in (x.strip() for x in args.arms.split(",") if x.strip()):
            msg = compile_refusal(a)
            if msg:
                ap.error(msg)
    if args.layers < 1:
        ap.error("--layers must be >= 1")
    if args.heads < 1 or args.d % args.heads:
        ap.error(f"--heads {args.heads} must be >= 1 and divide --d {args.d}")
    if args.seq > MAX_SEQ:
        ap.error(f"--seq {args.seq} > {MAX_SEQ} (the position embedding has {MAX_SEQ} rows)")

    if args.counter_wd < 0:
        ap.error("--counter-wd must be >= 0")
    if args.scale_lr < 0:
        ap.error("--scale-lr must be >= 0")
    if args.scale_lr > 0 and (args.scale_format != "nvfp4" or args.scale_growth != "off" or args.rescale):
        ap.error("--scale-lr needs --scale-format nvfp4 and neither --scale-growth nor --rescale "
                 "(they rewrite the same E4M3 byte)")
    global RESCALE_EVERY, SCALE_MODE, CLIP_FRAC, SCALE_FORMAT, HEADROOM_MODE, HEADROOM
    global SCALE_GROWTH, GROWTH_TAU, COUNTER_WD, SCALE_LR
    COUNTER_WD, SCALE_LR = args.counter_wd, args.scale_lr
    SCALE_FORMAT, HEADROOM_MODE, HEADROOM = args.scale_format, args.headroom_mode, args.headroom
    SCALE_GROWTH, GROWTH_TAU = args.scale_growth, args.growth_tau
    RESCALE_EVERY = args.rescale
    SCALE_MODE = args.scale_mode
    CLIP_FRAC = args.clip

    data_path = Path(args.data) if args.data else default_data_path(args.dataset)
    text, enc = load_corpus(args.dataset, data_path, args.chars, args.data_encoding)
    n_chars = len(text)
    ids_cpu, chars = encode_text(text)          # vocab from the FULL text: both splits share it
    del text
    ids_all = ids_cpu.to(DEV)
    del ids_cpu
    ids, val_ids = split_ids(ids_all, args.val_frac)
    if args.accum < 1:
        ap.error("--accum must be >= 1")
    epochs = args.steps * args.batch * args.accum * args.seq / len(ids)

    print(f"device={torch.cuda.get_device_name(0) if DEV == 'cuda' else 'cpu'} "
          f"d={args.d} steps={args.steps} batch={args.batch} seq={args.seq} "
          + (f"ACCUM={args.accum} (effective batch {args.batch * args.accum}, mode={args.accum_mode}, dtype={args.accum_dtype}) "
             if args.accum > 1 else "")
          + f"vocab={len(chars)} tokens={len(ids_all)} train_tokens={len(ids)} "
          f"val_tokens={len(val_ids) if args.val_frac > 0 else 0} val_frac={args.val_frac:g} "
          f"epochs={epochs:.2f}"
          + (" [LEGACY: val measured on train data]" if args.val_frac == 0 else "")
          + " (loss in bits/char) "
          f"sched={args.sched}"
          + (f"(warmup={args.warmup_frac:g} min={args.lr_min_frac:g})" if args.sched == "cosine" else "")
          + f" lr={args.lr:g} adam_lr={args.adam_lr:g} eval_every={args.eval_every}"
          + f" scale_format={args.scale_format}", flush=True)
    pc = analytic_params(args.d, args.layers, len(chars))
    print(f"data: dataset={args.dataset} encoding={enc}{' (byte-level)' if _is_latin1(enc) else ''} "
          f"path={data_path} chars={n_chars} (--chars {args.chars}) vocab={len(chars)}", flush=True)
    print(f"model: d={args.d} layers={args.layers} heads={args.heads} head_dim={args.d // args.heads} mlp=4d "
          f"seq={args.seq} | params total={pc['total']:,} block_linears(counter layers)={pc['lin']:,} "
          f"other(emb/pos/LN/head)={pc['other']:,} | AdamW state {ADAMW_STATE_BYTES:g} B/param "
          f"(fp32 master+m+v)", flush=True)
    arm_set = [a.strip() for a in args.arms.split(",")]
    extra = []
    if DENSE_RMS_ARM in arm_set:
        extra.append(f"dense_rms_lr={args.dense_rms_lr:g}")
    if any(is_qat_arm(a) for a in arm_set):
        extra.append(f"qat_lr={arm_lr('qat_tern', args):g}")
    if any(a.startswith("na_") for a in arm_set):
        extra.append(f"counter_wd={args.counter_wd:g} scale_lr={args.scale_lr:g}")
    if extra:
        print("gap arms: " + " ".join(extra), flush=True)
    if any(is_tern_arm(a.strip()) for a in args.arms.split(",")):
        _counter_cls()                          # may add <repo>/src to sys.path
        import memory_native
        print(f"tern arms: memory_native={memory_native.__file__} tern_lr={args.tern_lr:g} C={TERN_C} "
              f"lr_scale={TERN_LR_SCALE:g} rms_beta={TERN_RMS_BETA:g} rms_eps={TERN_RMS_EPS:g}", flush=True)

    results = []
    for arm in args.arms.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            r = run_arm(arm.strip(), seed, args.d, args.steps, args.batch, args.seq,
                        ids, len(chars), arm_lr(arm.strip(), args),
                        args.adam_lr, sched=args.sched,
                        warmup_frac=args.warmup_frac, lr_min_frac=args.lr_min_frac,
                        eval_every=args.eval_every, val_ids=val_ids,
                        blocks=args.layers, heads=args.heads, precision=args.precision,
                        compile=args.compile, compile_backend=args.compile_backend,
                        accum=args.accum, accum_mode=args.accum_mode, accum_dtype=args.accum_dtype,
                        momentum=args.momentum, momentum_dtype=args.momentum_dtype, nesterov=args.nesterov)
            results.append(r)
            f0, f1, f2 = r["flip"]
            print(f"  {r['arm']:17} seed={r['seed']} bits={r['bits']:.3f} "
                  f"val={r['val']:.4f} train={r['train']:.4f} gap={r['gap']:+.4f} "
                  f"best={r['best']:.4f} clip={r['clip']*100:.2f}% "
                  f"flip%(start/mid/end)={f0*100:.2f}/{f1*100:.2f}/{f2*100:.2f} "
                  f"edge={r['edge']*100:.2f}% grown={r['grown']} ({r['sec']:.0f}s)"
                  + (f" scale_moves={r['scale_moves']}" if r.get("scale_moves") else "")
                  + (f" rebased={r['rebased']}" if r.get("rebased") else ""), flush=True)
            if r.get("compile"):
                ci = r["compile"]
                print(f"      compile: kind={ci['kind']} backend={ci['backend']} graphs={ci['graphs']} "
                      f"graph_breaks={ci['graph_breaks']}" + ("" if ci["graphs"] == 1 else
                      "  <-- expected 1 graph (a recompile happened)"), flush=True)
            ws = r.get("wstats")
            if ws:
                print(f"      weights: kurt={ws['kurt']:.2f} tail3={ws['tail3']*100:.2f}% "
                      f"zero={ws['zero']*100:.1f}%", flush=True)
            if r.get("n_lin") is not None:
                print(f"      memory: params total={r['n_total']:,} block_linears={r['n_lin']:,} | "
                      f"state={r['state_mb']:.1f}MB (+grad {r['state_grad_mb']:.1f}MB) | "
                      f"peak={r['peak_mem_mb']:.0f}MB (+{r['peak_mem_delta_mb']:.0f}MB over run start)", flush=True)
            if args.log_curve:
                print("      curve: " + " ".join(f"{s}:{v:.3f}" for s, v, *_ in r["curve"]), flush=True)
            if args.hist:
                print(f"      magnitudes: {format_histogram(r['hist'])}", flush=True)

    print(f"\n{'arm':17} {'bits':>6} {'mean val':>9} {'train':>7} {'gap':>7} {'sd':>7} {'clip%':>6} {'best':>7} "
          f"{'flip%e':>7} {'edge%':>6}  seeds")
    for arm in args.arms.split(","):
        arm = arm.strip()
        rows = [r for r in results if r["arm"] == arm]
        vals = [r["val"] for r in rows]
        mean = sum(vals) / len(vals)
        sd = (sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
        cl = sum(r["clip"] for r in rows) / len(rows) * 100
        best = sum(r["best"] for r in rows) / len(rows)
        fl = sum(r["flip"][2] for r in rows) / len(rows) * 100
        ed = sum(r["edge"] for r in rows) / len(rows) * 100
        print(f"{arm:17} {rows[0]['bits']:6.3f} {mean:9.4f} {sum(r['train'] for r in rows) / len(rows):7.4f} {sum(r['gap'] for r in rows) / len(rows):+7.4f} {sd:7.4f} {cl:6.2f} {best:7.4f} "
              f"{fl:7.2f} {ed:6.2f}  {[round(v, 4) for v in vals]}")

    # Deployed-weight distribution, comparable across dense / QAT / counter arms (see weight_stats).
    print("\ndeployed weights of the 4 block linears (pooled over rows & layers; mean over seeds): "
          "kurt = E[w^4]/E[w^2]^2 per row (Gaussian 3, uniform 1.8); tail3 = frac |w| > 3*rms_row "
          "(Gaussian 0.27%); zero = frac exactly 0")
    print(f"{'arm':17} {'kurt':>7} {'tail3%':>7} {'zero%':>6}")
    for arm in args.arms.split(","):
        arm = arm.strip()
        ws = [r["wstats"] for r in results if r["arm"] == arm and r.get("wstats")]
        if ws:
            print(f"{arm:17} {sum(w['kurt'] for w in ws) / len(ws):7.2f} "
                  f"{sum(w['tail3'] for w in ws) / len(ws) * 100:7.2f} "
                  f"{sum(w['zero'] for w in ws) / len(ws) * 100:6.1f}")

    # Memory per arm (MB). state = measured_bits/8 x block-linear params; +grad = state + resident gradient bytes;
    # other = 12 B x (emb+pos+LN+head); peak = torch.cuda.max_memory_allocated of the training windows (max over
    # seeds; nan on CPU), peak-run = peak minus what was allocated at run start (the corpus on the GPU).
    if any(r.get("n_lin") is not None for r in results):
        print(f"\nmemory per arm, MB (block linears: {pc['lin']:,} params; formulas in the module docstring)")
        print(f"{'arm':17} {'bits':>6} {'B/param':>8} {'state':>9} {'+grad':>9} {'other':>8} {'peak':>9} {'peak-run':>9}")
        for arm in args.arms.split(","):
            arm = arm.strip()
            rows = [r for r in results if r["arm"] == arm and r.get("n_lin") is not None]
            if rows:
                r0 = rows[0]
                print(f"{arm:17} {r0['bits']:6.3f} {r0['bits'] / 8:8.3f} {r0['state_mb']:9.2f} {r0['state_grad_mb']:9.2f} "
                      f"{r0['other_state_mb']:8.2f} {max(r['peak_mem_mb'] for r in rows):9.0f} "
                      f"{max(r['peak_mem_delta_mb'] for r in rows):9.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
