#!/usr/bin/env python
"""Real ~1B training run (TinyStories, byte-level, vocab 256): dense (strong baseline) vs lg6_fp4_h2 under COMPARABLE conditions.

Reuses probes/e1_quality.py (model builder, sched_lr, set_counter_lr, fwd_logits, fp32_islands, compile_model) and
probes/lg_ab.py (lg arms) and probes/bench_train_step.py (grad checkpoint) by IMPORT; nothing is copied or edited there.

Arms (--arm):
    dense                 = e1 "dense_nobias" (bias-free nn.Linear) + bf16 autocast + fp32 master + fused AdamW + torch.compile
                            + SDPA + grad-checkpoint per block.   (aliases: dense_strong)
    lg6_fp4_h2            = LatentGridLinear block linears (native custom-op path) + the SAME everything else.
    lg6_fp4_h2_nvfp4fwd   = optional: forward GEMM = cuBLASLt NVFP4 (approximate activations).
Identical model code (CharLM from e1_quality), data order (batch i = f(seed, i), same bytes for every arm), schedule
(2 % warmup, cosine to 10 %), AdamW for emb/pos/LN/head (decay only ndim>=2 params), val subset.

Optimizer notes
    * dense: ONE AdamW (betas .9/.95, wd --weight-decay on matrices, 0 on LN), global grad-norm clip --clip over ALL params.
    * lg: block linears update INSIDE backward (RMS-normalised counter step, lr = --lr, no AdamW, no momentum), so a GLOBAL grad
      clip is impossible: --clip only covers the FP params (emb/pos/LN/head, AdamW). This is an inherent asymmetry; the logged
      gnorm of lg is therefore NOT comparable with dense (it excludes the 24*4 big matrices).
    * lg => by default NO gradient accumulation (each backward applies the update): the whole --batch is ONE micro-batch, hence
      the per-block gradient checkpointing (default on for both arms; --no-grad-checkpoint to disable).
      OPT-IN accumulation (mx6/lg_accum.py): --accum N [--accum-mode buffer|latent] [--accum-dtype bf16|fp8]: --batch is then the
      MICRO-batch, effective batch = batch*N (the data of step s is the same windows as a no-accum run with --batch batch*N).
      buffer: ONE lg update per optimizer step on the accumulated gradient (== one big batch); latent: N updates with lr/N.
    * data: batch for step s = default_rng([seed, s]) windows => stateless, resume needs no data-RNG state.

Modes
    --prepare-data     download TinyStories train/valid into --data-dir (lock file + atomic rename of .part; run ONCE)
    --probe-batch      for each arm in --arm (comma list) and --batches: 3 steps, report peak mem / ms per step / OOM
    --lr-probe         --probe-steps steps per lr (warmup+cosine over those steps) for --lrs; val at end; prints the best
    (default)          train: --steps, log every --log-every, eval every --eval-every, checkpoint every --ckpt-every, --resume

Loss unit: bits per byte (cross-entropy / ln 2).

Colab sequence (see the final report of the task for timing):
    python probes/train_1b.py --prepare-data
    python probes/train_1b.py --probe-batch --arm dense,lg6_fp4_h2
    python probes/train_1b.py --lr-probe --arm dense --batch 128 --lrs 1e-4,3e-4,6e-4
    python probes/train_1b.py --lr-probe --arm lg6_fp4_h2 --batch 128 --adam-lr <dense winner> --lrs 0.003,0.01,0.02
    python probes/train_1b.py --arm dense --batch 128 --adam-lr <w> --out /content/runs/dense
    python probes/train_1b.py --arm lg6_fp4_h2 --batch 128 --adam-lr <w> --lr <w2> --out /content/runs/lg6
    (re-run the same command with --resume after a disconnect)
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lg_ab                                                       # noqa: E402

VOCAB = 256
LN2 = math.log(2.0)
HF_BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
DATA_FILES = {"train": "TinyStoriesV2-GPT4-train.txt", "valid": "TinyStoriesV2-GPT4-valid.txt"}
ARM_ALIASES = {"dense": "dense_nobias", "dense_strong": "dense_nobias"}
VAL_SEED = 1234
CKPT_NAME = "ckpt.pt"

_E1 = None


def get_e1():
    """e1_quality with the lg arms installed (one shared module object per process)."""
    global _E1
    if _E1 is None:
        _E1 = lg_ab.install(lg_ab.load_e1())
    return _E1


def _bts():
    import bench_train_step
    return bench_train_step


def is_oom(e: BaseException) -> bool:
    oom_t = getattr(torch, "OutOfMemoryError", None) or getattr(torch.cuda, "OutOfMemoryError", ())
    return isinstance(e, oom_t) or "out of memory" in str(e).lower()


def cuda_ok(dev: str) -> bool:
    return dev == "cuda" and torch.cuda.is_available()


# ── data: atomic, locked download; uint8 memmap; seeded windows ─────────────────────────────────────────────────

def download(url: str, dest: Path, timeout: float = 60.0, retries: int = 3) -> None:
    """Stream url -> dest (a .part file; the caller renames). Checks Content-Length; retries with backoff."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "train_1b/1.0"})
            n = 0
            with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
                total = r.headers.get("Content-Length")
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    n += len(chunk)
                f.flush()
                os.fsync(f.fileno())
            if n == 0 or (total is not None and n != int(total)):
                raise OSError(f"truncated download: got {n} bytes, expected {total}")
            return
        except Exception as e:                                     # noqa: BLE001
            last = e
            Path(dest).unlink(missing_ok=True)
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f"download failed after {retries} tries: {url}: {type(last).__name__}: {last}")


def ensure_file(path, url: str, fetch=download, wait_s: float = 7200.0, poll: float = 1.0, stale_s: float = 1800.0) -> Path:
    """Make sure `path` exists. Safe against concurrent processes/threads: ONE holder of <path>.lock (O_CREAT|O_EXCL)
    downloads to <path>.part and os.replace()s it into place (atomic); the others wait for `path` to appear. A lock whose
    mtime AND .part mtime are older than stale_s (dead holder) is broken. fetch(url, dest) is injectable (tests)."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    part = path.with_name(path.name + ".part")
    t0 = time.time()
    while True:
        if path.exists():
            return path
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}:{threading.get_ident()}".encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                newest = max(lock.stat().st_mtime, part.stat().st_mtime if part.exists() else 0.0)
                if time.time() - newest > stale_s:
                    lock.unlink(missing_ok=True)
                    part.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.time() - t0 > wait_s:
                raise TimeoutError(f"waited {wait_s:.0f}s for another process to download {path}")
            time.sleep(poll)
    try:
        if not path.exists():
            part.unlink(missing_ok=True)
            print(f"downloading {url} -> {path}", flush=True)
            fetch(url, part)
            os.replace(part, path)
    finally:
        part.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)
    return path


def data_paths(args):
    tr = Path(args.train_file) if args.train_file else Path(args.data_dir) / DATA_FILES["train"]
    va = Path(args.val_file) if args.val_file else Path(args.data_dir) / DATA_FILES["valid"]
    return tr, va


def ensure_data(args, fetch=download):
    tr, va = data_paths(args)
    for p, key, given in ((tr, "train", args.train_file), (va, "valid", args.val_file)):
        if given:
            if not p.exists():
                raise FileNotFoundError(p)
        else:
            ensure_file(p, HF_BASE + DATA_FILES[key], fetch)
    return tr, va


def open_corpus(path) -> np.memmap:
    return np.memmap(str(path), dtype=np.uint8, mode="r")


def windows(mm, starts, seq: int) -> np.ndarray:
    """[len(starts), seq+1] uint8 contiguous windows."""
    return np.stack([np.asarray(mm[int(s):int(s) + seq + 1]) for s in starts])


def batch_starts(n_bytes: int, seed: int, step: int, batch: int, seq: int) -> np.ndarray:
    if n_bytes < seq + 2:
        raise ValueError(f"corpus of {n_bytes} bytes is too short for seq={seq}")
    return np.random.default_rng([seed, 0xDA7A, step]).integers(0, n_bytes - seq, size=batch)


def to_xy(arr: np.ndarray, dev: str):
    t = torch.from_numpy(arr).to(dev).long()
    return t[:, :-1].contiguous(), t[:, 1:].contiguous()


def get_batch(mm, seed: int, step: int, batch: int, seq: int, dev: str):
    return to_xy(windows(mm, batch_starts(len(mm), seed, step, batch, seq), seq), dev)


def val_starts(n_bytes: int, seq: int, n_batches: int, n_windows: int) -> np.ndarray:
    """Fixed validation windows [n_batches, n_windows] (seeded, identical for every arm and every eval)."""
    if n_bytes < seq + 2:
        raise ValueError(f"validation file of {n_bytes} bytes is too short for seq={seq}")
    return np.random.default_rng([VAL_SEED]).integers(0, n_bytes - seq, size=(n_batches, n_windows))


@torch.no_grad()
def evaluate(eng, mm, starts: np.ndarray, seq: int) -> float:
    """Mean val bits/byte on the fixed windows (eager model, no_grad, never updates)."""
    e1 = get_e1()
    eng.model.eval()
    tot = 0.0
    try:
        for st in starts:
            x, y = to_xy(windows(mm, st, seq), eng.dev)
            tot += float(F.cross_entropy(e1.fwd_logits(eng.model, x, eng.args.precision).reshape(-1, VOCAB), y.reshape(-1)))
    finally:
        eng.model.train()
    return tot / len(starts) / LN2


# ── model / optimizer engine ─────────────────────────────────────────────────────────────────────────────────

def make_optimizer(model, adam_lr: float, args, fused: bool):
    """AdamW over every nn.Parameter (dense: all weights; lg: emb/pos/LN/head). Weight decay only on ndim>=2 params."""
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    plain = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    groups = [{"params": decay, "weight_decay": args.weight_decay, "base_lr": adam_lr}]
    if plain:
        groups.append({"params": plain, "weight_decay": 0.0, "base_lr": adam_lr})
    return torch.optim.AdamW(groups, lr=adam_lr, betas=(args.beta1, args.beta2), eps=1e-8, fused=fused)


class Engine:
    """model + (compiled) forward + optimizer + one training step. The same setup for both arms."""

    def __init__(self, arm: str, args, batch: int | None = None, dev: str | None = None):
        e1 = get_e1()
        self.e1, self.arm, self.args = e1, arm, args
        self.base = ARM_ALIASES.get(arm, arm)
        self.is_lg = self.base.startswith("lg")
        self.dev = dev or e1.DEV
        self.batch = batch or args.batch
        self.adam_lr, self.lr, self.total = args.adam_lr, args.lr, args.steps
        e1.SDPA = bool(args.sdpa)
        e1.SCALE_FORMAT = "nvfp4"                       # lg arms are NVFP4-only (lg_ab._check_flags)
        e1.COUNTER_WD = float(args.counter_wd)
        torch.manual_seed(args.seed)
        model, _unused_opt = e1.build_model_opt(self.base, VOCAB, args.d, args.lr, args.adam_lr, args.layers, args.heads)
        self.model = model
        self.lgs = []
        if self.is_lg:
            from mx6.latent_grid import HAS_LG_OPS, lg_layers, prepare_lg_for_compile
            if not HAS_LG_OPS:
                raise RuntimeError("lg arms need torch.library.register_autograd / register_fake (torch >= 2.4)")
            self.lgs = lg_layers(model)
            for m in self.lgs:
                if not m.native_enabled():              # CPU / no Triton: torch backend, same op path
                    m.native = True
            prepare_lg_for_compile(model, "op")         # update lives only in the backward op => checkpoint-safe
        #: opt-in gradient accumulation (default --accum 1 = off, nothing below runs). lg: mx6/lg_accum.set_accum (buffer or
        #: latent); dense: plain autograd accumulation. --batch is the MICRO-batch; effective batch = batch * accum.
        self.accum = int(getattr(args, "accum", 1))
        self.acc_state = None
        if self.accum > 1 and self.is_lg:
            from mx6.lg_accum import set_accum
            self.acc_state = set_accum(model, self.accum, getattr(args, "accum_mode", "buffer"),
                                       getattr(args, "accum_dtype", "bf16"), loss_scaled=True)
        #: opt-in per-weight momentum for lg (mx6/lg_accum.set_momentum): the EMA of grad_w feeds the RMS-normalised update.
        self.momentum = float(getattr(args, "momentum", 0.0) or 0.0)
        if self.momentum > 0.0 and self.is_lg:
            from mx6.lg_accum import set_momentum      # accum > 1: micro-batches accumulate INTO the momentum buffer
            st = set_momentum(model, self.momentum, getattr(args, "momentum_dtype", "bf16"),
                              bool(getattr(args, "nesterov", False)), n=self.accum)
            self.acc_state = st if self.accum > 1 else None
        if args.grad_checkpoint:
            why = _bts().checkpoint_refusal(model, "compile" if args.compile else "eager", "op")
            if why:
                raise RuntimeError(why)
            _bts().enable_grad_checkpoint(model)
        if args.precision != "fp32":
            e1.fp32_islands(model)
        self.fwd, self.cinfo = model, None
        if args.compile:
            self.fwd, self.cinfo = e1.compile_model(model, self.base, args.compile_backend)
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.fused = bool(args.fused_adam) and cuda_ok(self.dev)
        self.opt = make_optimizer(model, self.adam_lr, args, self.fused)
        model.train()

    def sched(self, base: float, step: int) -> float:
        a = self.args
        return self.e1.sched_lr("cosine", base, step, self.total, a.warmup_frac, a.lr_min_frac)

    def lrs_at(self, step: int):
        return self.sched(self.lr, step), self.sched(self.adam_lr, step)

    def train_step(self, x, y, step: int):
        """-> (loss, grad_norm|None) as detached device tensors (no host sync)."""
        a = self.args
        c_lr, a_lr = self.lrs_at(step)
        for m in self.lgs:
            m.lr = c_lr
        for g in self.opt.param_groups:
            g["lr"] = self.sched(g["base_lr"], step)
        if self.accum > 1:
            return self._train_step_accum(x, y)
        with self.e1.autocast_ctx(a.precision, self.dev):
            logits = self.fwd(x)
        loss = F.cross_entropy(logits.float().reshape(-1, VOCAB), y.reshape(-1))
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(self.params, a.clip) if a.clip > 0 else None
        self.opt.step()
        return loss.detach(), (gn.detach() if gn is not None else None)

    def _train_step_accum(self, x, y):
        """x, y: [batch*accum, seq] = the ONE effective batch of this optimizer step (same windows as a no-accum run with
        --batch batch*accum); split into `accum` micro-batches. Loss of each micro-batch / accum (lg_accum convention and FP
        params: autograd sums the grads = the big-batch mean); ONE AdamW step per call, lg buffer-mode updates once (on the last
        micro-batch), latent mode accum times. -> (mean loss, grad_norm|None)."""
        a, n = self.args, self.accum
        xs, ys = x.chunk(n), y.chunk(n)
        self.opt.zero_grad(set_to_none=True)
        tot = torch.zeros((), device=x.device)
        from contextlib import nullcontext
        ctx = None
        if self.acc_state is not None:
            from mx6.lg_accum import lg_accumulate
            ctx = lg_accumulate(self.model, n)
        with (ctx if ctx is not None else nullcontext()) as acc:
            for i in (acc.micro_batches() if acc is not None else range(n)):
                with self.e1.autocast_ctx(a.precision, self.dev):
                    logits = self.fwd(xs[i])
                loss = F.cross_entropy(logits.float().reshape(-1, VOCAB), ys[i].reshape(-1))
                (loss / n).backward()
                tot += loss.detach() / n
        gn = torch.nn.utils.clip_grad_norm_(self.params, a.clip) if a.clip > 0 else None
        self.opt.step()
        return tot, (gn.detach() if gn is not None else None)

    def new_optimizer(self):
        self.opt = None
        self.opt = make_optimizer(self.model, self.adam_lr, self.args, self.fused)

    def clip_frac(self):
        c = self.e1.clip_frac_now(self.model)
        return float(c) if c is not None else None


def cleanup(*objs):
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    try:
        import torch._dynamo as dynamo
        dynamo.reset()
    except Exception:                                           # noqa: BLE001
        pass


# ── logging / checkpoints ───────────────────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self, out: Path, name: str = "train"):
        out.mkdir(parents=True, exist_ok=True)
        self.txt = open(out / f"{name}.log", "a", buffering=1)
        self.jsonl = open(out / f"{name}.jsonl", "a", buffering=1)

    def line(self, s: str):
        print(s, flush=True)
        self.txt.write(s + "\n")

    def rec(self, d: dict, s: str | None = None):
        self.jsonl.write(json.dumps({k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                                     for k, v in d.items()}) + "\n")
        if s:
            self.line(s)

    def close(self):
        self.txt.close()
        self.jsonl.close()


CKPT_KEYS = ("arm", "d", "layers", "heads", "seq", "batch", "steps", "seed", "precision")


def plain_args(args) -> dict:
    return {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, type(None)))}


def lg_momentum_buffers(model) -> list:
    """CPU copies of every lg layer's momentum buffer (+ fp8 row scale), in lg_layers order; [] if momentum is off."""
    from mx6.latent_grid import lg_layers
    out = []
    for m in lg_layers(model):
        st = getattr(m, "_acc", None)
        if st is None or st.mode != "ema":
            return []
        out.append([m._acc_buf.detach().cpu(), m._acc_scale.detach().cpu()])
    return out


def load_lg_momentum_buffers(model, bufs) -> None:
    from mx6.latent_grid import lg_layers
    layers = list(lg_layers(model))
    if len(layers) != len(bufs):
        raise ValueError(f"momentum buffers: ckpt has {len(bufs)} layers, model {len(layers)}")
    for m, (b, sc) in zip(layers, bufs):
        m._acc_buf.copy_(b.to(m._acc_buf.device))
        if sc.numel():
            m._acc_scale.copy_(sc.to(m._acc_scale.device))


def save_ckpt(path: Path, eng: Engine, step: int, args) -> float:
    """Atomic: write <path>.tmp, fsync, os.replace. Holds model (incl. lg idx/v_rms/scale buffers), AdamW state, RNG states."""
    t0 = time.time()
    obj = {"model": eng.model.state_dict(), "opt": eng.opt.state_dict(), "step": int(step),
           "rng_cpu": torch.get_rng_state(), "args": plain_args(args), "torch": str(torch.__version__),
           "data": {"seed": args.seed, "scheme": "default_rng([seed, 0xDA7A, step])"}}
    if torch.cuda.is_available():
        obj["rng_cuda"] = torch.cuda.get_rng_state_all()
    mom = lg_momentum_buffers(eng.model)
    if mom:
        obj["lg_momentum"] = mom
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return time.time() - t0


def load_ckpt(path: Path, eng: Engine, args) -> int:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    old = ck["args"]
    bad = [f"{k}: ckpt={old.get(k)!r} now={getattr(args, k)!r}" for k in CKPT_KEYS if old.get(k) != getattr(args, k)]
    # accumulation keys: absent in older checkpoints (= 1 / buffer / bf16); mode / dtype only matter when accum > 1
    if int(old.get("accum", 1)) != int(getattr(args, "accum", 1)):
        bad.append(f"accum: ckpt={old.get('accum', 1)!r} now={getattr(args, 'accum', 1)!r}")
    elif int(getattr(args, "accum", 1)) > 1:
        bad += [f"{k}: ckpt={old.get(k, d)!r} now={getattr(args, k, d)!r}"
                for k, d in (("accum_mode", "buffer"), ("accum_dtype", "bf16")) if old.get(k, d) != getattr(args, k, d)]
    for k, d in (("momentum", 0.0), ("momentum_dtype", "bf16"), ("nesterov", False)):
        if old.get(k, d) != getattr(args, k, d):
            bad.append(f"{k}: ckpt={old.get(k, d)!r} now={getattr(args, k, d)!r}")
    if bad:
        raise ValueError("--resume with different settings: " + "; ".join(bad))
    eng.model.load_state_dict(ck["model"])
    if "lg_momentum" in ck:
        load_lg_momentum_buffers(eng.model, ck["lg_momentum"])
    eng.opt.load_state_dict(ck["opt"])
    torch.set_rng_state(ck["rng_cpu"].cpu())
    if "rng_cuda" in ck and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in ck["rng_cuda"]])
    step = int(ck["step"])
    del ck
    gc.collect()
    return step


# ── modes ───────────────────────────────────────────────────────────────────────────────────────────────────

def _sync(dev):
    if cuda_ok(dev):
        torch.cuda.synchronize()


def _peak_gb(dev) -> float:
    return torch.cuda.max_memory_allocated() / 2 ** 30 if cuda_ok(dev) else float("nan")


def _reset_peak(dev):
    if cuda_ok(dev):
        torch.cuda.reset_peak_memory_stats()


def run_train(args, record: list | None = None) -> dict:
    """Train --arm for --steps. record (tests): list that receives float(loss) of every step."""
    e1 = get_e1()
    tr_p, va_p = ensure_data(args)
    tr, va = open_corpus(tr_p), open_corpus(va_p)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = Logger(out)
    eng = Engine(args.arm, args)
    dev = eng.dev
    vst = val_starts(len(va), args.seq, args.eval_batches, args.eval_windows)
    ck_path = out / CKPT_NAME
    start = 0
    if args.resume:
        if ck_path.exists():
            start = load_ckpt(ck_path, eng, args)
            log.line(f"RESUMED from {ck_path} at step {start}")
        else:
            log.line(f"--resume: no {ck_path}, starting from step 0")
    end = args.steps if not args.stop_after else min(args.steps, args.stop_after)
    pc = e1.analytic_params(args.d, args.layers, VOCAB)
    tok_step = args.batch * eng.accum * args.seq
    if eng.accum > 1:
        from mx6.lg_accum import accum_resident_bytes
        acc_desc = (f"batch={args.batch} x accum={eng.accum} = effective batch {args.batch * eng.accum} "
                    f"(accum_mode={args.accum_mode if eng.is_lg else 'plain autograd (dense)'}"
                    + (f", accum_dtype={args.accum_dtype}, buffer {accum_resident_bytes(eng.model) / 2 ** 20:.0f} MiB"
                       if eng.is_lg and args.accum_mode == "buffer" else "") + ")")
    else:
        acc_desc = f"batch={args.batch} (one micro-batch, no accumulation)"
    log.line(f"arm={args.arm} ({eng.base}) device={dev} torch={torch.__version__} "
             f"{torch.cuda.get_device_name(0) if cuda_ok(dev) else 'cpu'}")
    log.line(f"model d={args.d} layers={args.layers} heads={args.heads} seq={args.seq} params={pc['total']:,} "
             f"| {acc_desc} tokens/step={tok_step} steps={args.steps} "
             f"tokens={tok_step * args.steps:,} epochs={tok_step * args.steps / len(tr):.3f} "
             f"train_bytes={len(tr):,} val_bytes={len(va):,}")
    log.line(f"optim: adam_lr={args.adam_lr:g} lr(counter)={args.lr:g} wd={args.weight_decay:g} clip={args.clip:g} "
             f"betas=({args.beta1:g},{args.beta2:g}) sched=cosine warmup={args.warmup_frac:g} min={args.lr_min_frac:g} "
             f"precision={args.precision} sdpa={args.sdpa} compile={args.compile} grad_checkpoint={args.grad_checkpoint} "
             f"fused_adam={eng.fused} lg_layers={len(eng.lgs)}"
             + ("  [lg: grad clip covers FP params only]" if eng.is_lg else ""))
    zero = lambda: torch.zeros((), device=dev)                    # noqa: E731
    acc, gacc, n_win, n_g = zero(), zero(), 0, 0
    last_losses, ms_hist = [], []
    _sync(dev)
    _reset_peak(dev)
    t_win, t_run = time.perf_counter(), time.perf_counter()
    first_window = True
    result = {"arm": args.arm, "steps": args.steps, "batch": args.batch, "diverged": False}
    try:
        for step in range(start, end):
            x, y = get_batch(tr, args.seed, step, args.batch * eng.accum, args.seq, dev)   # accum>1: the SAME windows as --batch B*accum
            loss, gn = eng.train_step(x, y, step)
            if record is not None:
                record.append(float(loss))
            acc += loss
            n_win += 1
            if gn is not None:
                gacc += gn
                n_g += 1
            done = step + 1
            if done % args.log_every == 0 or done == end:
                _sync(dev)
                dt = time.perf_counter() - t_win
                l_bpb = float(acc) / n_win / LN2
                gnorm = float(gacc) / n_g if n_g else float("nan")
                last_losses.append(l_bpb)
                c_lr, a_lr = eng.lrs_at(step)
                ms = dt / n_win * 1e3
                if not first_window:
                    ms_hist.append(ms)
                rec = {"type": "train", "step": done, "loss_bpb": l_bpb, "lr": a_lr, "counter_lr": c_lr if eng.is_lg else None,
                       "gnorm": gnorm, "tok_s": tok_step * n_win / dt, "ms_step": ms, "peak_gb": _peak_gb(dev),
                       "clip_frac": eng.clip_frac() if eng.is_lg else None}
                log.rec(rec, f"step {done}/{args.steps} | loss {l_bpb:.4f} bpb | lr {a_lr:.2e}"
                             + (f" ctr {c_lr:.2e}" if eng.is_lg else "")
                             + f" | gnorm {gnorm:.3f} | {rec['tok_s']:.0f} tok/s | {ms:.0f} ms/step | peak {rec['peak_gb']:.1f} GB"
                             + (" (incl. compile)" if first_window else ""))
                if not math.isfinite(l_bpb):
                    log.line("DIVERGED: non-finite loss; stopping WITHOUT overwriting the last checkpoint")
                    result["diverged"] = True
                    return result
                acc, gacc, n_win, n_g = zero(), zero(), 0, 0
                first_window = False
                _reset_peak(dev)
                t_win = time.perf_counter()
            if done % args.eval_every == 0 or done == args.steps:
                _sync(dev)
                te = time.perf_counter()
                v = evaluate(eng, va, vst, args.seq)
                log.rec({"type": "eval", "step": done, "val_bpb": v, "eval_s": time.perf_counter() - te},
                        f"EVAL step {done}: val {v:.4f} bpb ({time.perf_counter() - te:.1f}s)")
                result["val_bpb"] = v
                t_win = time.perf_counter()
                if n_win:
                    n_win = 0
                    acc, gacc, n_g = zero(), zero(), 0
                _reset_peak(dev)
            if done % args.ckpt_every == 0 or done == end:
                if not math.isfinite(float(loss)):
                    log.line("DIVERGED at checkpoint time; not saving")
                    result["diverged"] = True
                    return result
                s = save_ckpt(ck_path, eng, done, args)
                log.rec({"type": "ckpt", "step": done, "save_s": s}, f"checkpoint step {done} -> {ck_path} ({s:.0f}s)")
                t_win = time.perf_counter()
                n_win = 0
                acc, gacc, n_g = zero(), zero(), 0
    except BaseException as e:                                   # noqa: BLE001
        if is_oom(e):
            log.line(f"OUT OF MEMORY with --batch {args.batch} arm={args.arm} (checkpointing={args.grad_checkpoint}). "
                     f"Re-run with a smaller --batch (see --probe-batch).")
            result["oom"] = True
            result["error"] = "oom"
            return result
        raise
    finally:
        log.close()
    result.update({"final_step": end, "train_bpb_last": last_losses[-1] if last_losses else None,
                   "wall_s": time.perf_counter() - t_run, "ms_step_median": float(np.median(ms_hist)) if ms_hist else None,
                   "compile": eng.cinfo})
    if end == args.steps:
        (out / "result.json").write_text(json.dumps(result, indent=1, default=str))
    return result


def parse_floats(s: str):
    return [float(v) for v in s.split(",") if v.strip()]


def run_probe_batch(args) -> dict:
    """3 steps (default) per (arm, batch) on random bytes: peak mem, ms/step (steps after the first), OOM. A fresh engine per
    trial; once a batch OOMs the larger ones of that arm are skipped."""
    arms = [a.strip() for a in args.arm.split(",") if a.strip()]
    batches = sorted(int(b) for b in args.batches.split(","))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for arm in arms:
        oom_seen = False
        for B in batches:
            rec = {"arm": arm, "batch": B, "ok": False, "oom": False, "ms_step": None, "peak_gb": None, "reserved_gb": None}
            results[(arm, B)] = rec
            if oom_seen:
                rec["skipped"] = True
                print(f"{arm:22} batch {B:4d}: skipped (a smaller batch already OOMed)", flush=True)
                continue
            eng = None
            try:
                eng = Engine(arm, args, B)
                dev = eng.dev
                _sync(dev)
                _reset_peak(dev)
                g = torch.Generator().manual_seed(0)
                times = []
                for i in range(max(args.probe_n, 2)):
                    t = torch.randint(0, VOCAB, (B * eng.accum, args.seq + 1), generator=g)
                    x, y = t[:, :-1].contiguous().to(dev), t[:, 1:].contiguous().to(dev)
                    _sync(dev)
                    t0 = time.perf_counter()
                    loss, _ = eng.train_step(x, y, i)
                    _sync(dev)
                    times.append(time.perf_counter() - t0)
                rec.update(ok=bool(torch.isfinite(loss).item()), ms_step=float(np.mean(times[1:])) * 1e3,
                           first_step_s=times[0], peak_gb=_peak_gb(dev),
                           reserved_gb=torch.cuda.max_memory_reserved() / 2 ** 30 if cuda_ok(dev) else float("nan"))
                print(f"{arm:22} batch {B:4d}: OK  {rec['ms_step']:.0f} ms/step  peak {rec['peak_gb']:.1f} GB "
                      f"(reserved {rec['reserved_gb']:.1f} GB)  first step {times[0]:.0f}s", flush=True)
            except BaseException as e:                           # noqa: BLE001
                if not is_oom(e):
                    rec["error"] = f"{type(e).__name__}: {e}"
                    print(f"{arm:22} batch {B:4d}: ERROR {rec['error']}", flush=True)
                    raise
                rec["oom"] = True
                oom_seen = True
                print(f"{arm:22} batch {B:4d}: OOM", flush=True)
            finally:
                eng = None
                cleanup()
    fits = [B for B in batches if all(results[(a, B)]["ok"] for a in arms)]
    print("\nsummary (peak GB / ms per step):")
    for arm in arms:
        print(f"  {arm:22} " + "  ".join(
            f"B{B}: " + ("OOM" if results[(arm, B)]["oom"] else "skip" if results[(arm, B)].get("skipped") else
                         f"{results[(arm, B)]['peak_gb']:.1f}GB/{results[(arm, B)]['ms_step']:.0f}ms") for B in batches))
    best = max(fits) if fits else None
    print(f"largest batch that fits ALL arms: {best}" + ("  (leave 5-10 % memory margin if you run both arms in parallel)"
                                                        if best else "  (none: lower --batches)"))
    (out / "probe_batch.json").write_text(json.dumps(
        {"results": [results[k] for k in sorted(results)], "largest_common_batch": best}, indent=1, default=str))
    return {"largest_common_batch": best, "results": results}


def run_lr_probe(args) -> dict:
    """--probe-steps steps (warmup + cosine over those steps) for each lr, from the SAME init and the SAME batches; dense: lrs =
    --adam-lr values; lg: lrs = counter --lr values with --adam-lr fixed. Picks the best final val."""
    tr_p, va_p = ensure_data(args)
    tr, va = open_corpus(tr_p), open_corpus(va_p)
    arms = [a.strip() for a in args.arm.split(",") if a.strip()]
    if len(arms) != 1:
        raise SystemExit("--lr-probe takes ONE --arm")
    arm = arms[0]
    is_lg = ARM_ALIASES.get(arm, arm).startswith("lg")
    lrs = parse_floats(args.lrs) if args.lrs else ([0.003, 0.01, 0.02] if is_lg else [1e-4, 3e-4, 6e-4])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    eng = Engine(arm, args)
    eng.total = args.probe_steps
    dev = eng.dev
    init_sd = {k: v.detach().to("cpu", copy=True) for k, v in eng.model.state_dict().items()}
    vst = val_starts(len(va), args.seq, args.eval_batches, args.eval_windows)
    rows = []
    print(f"lr-probe arm={arm} batch={args.batch} steps={args.probe_steps} "
          f"{'counter --lr' if is_lg else '--adam-lr'} in {lrs}" + (f" (adam-lr fixed {args.adam_lr:g})" if is_lg else ""), flush=True)
    for v in lrs:
        eng.model.load_state_dict(init_sd)
        eng.adam_lr = args.adam_lr if is_lg else v
        eng.lr = v if is_lg else args.lr
        eng.new_optimizer()
        torch.manual_seed(args.seed + 1)
        losses = []
        _sync(dev)
        t0 = time.perf_counter()
        for step in range(args.probe_steps):
            x, y = get_batch(tr, args.seed, step, args.batch * eng.accum, args.seq, dev)
            loss, _ = eng.train_step(x, y, step)
            losses.append(loss)
        _sync(dev)
        el = time.perf_counter() - t0
        ls = torch.stack(losses).float().cpu() / LN2
        tail = ls[-min(20, len(ls)):]
        finite = bool(torch.isfinite(ls).all())
        val = evaluate(eng, va, vst, args.seq) if finite else float("nan")
        row = {"lr": v, "train_bpb_last20": float(tail.mean()), "val_bpb": val, "finite": finite, "sec": el,
               "ms_step": el / args.probe_steps * 1e3}
        rows.append(row)
        print(f"  lr={v:g}: train(last20)={row['train_bpb_last20']:.4f} val={val:.4f} bpb "
              f"{'' if finite else 'DIVERGED '}({el:.0f}s, {row['ms_step']:.0f} ms/step incl. compile)", flush=True)
    ok = [r for r in rows if r["finite"] and math.isfinite(r["val_bpb"])]
    best = min(ok, key=lambda r: r["val_bpb"]) if ok else None
    flag = "--lr" if is_lg else "--adam-lr"
    print(f"BEST: {flag} {best['lr']:g}  (val {best['val_bpb']:.4f} bpb)" if best else "BEST: none (all diverged)", flush=True)
    if best and best is rows[0] or best and best is rows[-1]:
        print("  note: the winner is at the edge of the grid; consider extending it", flush=True)
    res = {"arm": arm, "batch": args.batch, "probe_steps": args.probe_steps, "flag": flag, "rows": rows,
           "best": best["lr"] if best else None, "adam_lr_fixed": args.adam_lr if is_lg else None}
    (out / f"lr_probe_{arm}.json").write_text(json.dumps(res, indent=1))
    eng = None
    cleanup()
    return res


def run_prepare(args) -> int:
    tr_p, va_p = ensure_data(args)
    for name, p in (("train", tr_p), ("valid", va_p)):
        mm = open_corpus(p)
        print(f"{name}: {p} {len(mm):,} bytes ({len(mm) / 2 ** 20:.1f} MiB) head={bytes(mm[:60])!r} max_byte={int(mm[:1 << 20].max())}")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    m = ap.add_argument_group("mode (default: train)")
    m.add_argument("--prepare-data", action="store_true")
    m.add_argument("--probe-batch", action="store_true")
    m.add_argument("--lr-probe", action="store_true")
    ap.add_argument("--arm", default="dense", help="dense | lg6_fp4_h2 | lg6_fp4_h2_nvfp4fwd (comma list in --probe-batch)")
    ap.add_argument("--d", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--accum", type=int, default=1,
                    help="gradient accumulation: N micro-batches of --batch per OPTIMIZER step (effective batch = batch*N); "
                         "1 = off (default, unchanged behaviour). lg arms need --accum-mode; dense = plain autograd accumulation")
    ap.add_argument("--accum-mode", default="buffer", choices=["buffer", "latent"],
                    help="lg only. buffer: per-layer grad buffer, ONE update per optimizer step (== a big batch); latent: no "
                         "buffer, every micro-batch updates with lr/N and beta^(1/N) (see mx6/lg_accum.py)")
    ap.add_argument("--accum-dtype", default="bf16", choices=["bf16", "fp8"],
                    help="lg buffer mode: bf16 (+2 B/param) or fp8 e4m3 with per-row scale + stochastic rounding (+~1 B/param)")
    ap.add_argument("--momentum", type=float, default=0.0,
                    help="lg only: per-weight momentum mu (EMA of grad fed to the update), 0 = off. d512 best: 0.9 with ~3x lr")
    ap.add_argument("--momentum-dtype", default="bf16", choices=["bf16", "fp8", "int4"],
                    help="lg momentum buffer: bf16 (+2 B/param) or fp8 e4m3 + row scale + SR (+~1 B/param)")
    ap.add_argument("--nesterov", action="store_true", help="lg momentum: nesterov direction")
    ap.add_argument("--warmup-frac", type=float, default=0.02)
    ap.add_argument("--lr-min-frac", type=float, default=0.1)
    ap.add_argument("--adam-lr", type=float, default=6e-4, help="AdamW lr: dense weights AND lg's emb/pos/LN/head")
    ap.add_argument("--lr", type=float, default=0.01, help="lg counter lr")
    ap.add_argument("--weight-decay", type=float, default=0.1, help="AdamW wd on ndim>=2 params (not LN)")
    ap.add_argument("--counter-wd", type=float, default=0.0, help="lg restoring force (default off)")
    ap.add_argument("--clip", type=float, default=1.0, help="grad-norm clip of the AdamW params (lg: FP params only)")
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    B = argparse.BooleanOptionalAction
    ap.add_argument("--sdpa", action=B, default=True)
    ap.add_argument("--compile", action=B, default=True)
    ap.add_argument("--compile-backend", default=None)
    ap.add_argument("--grad-checkpoint", action=B, default=True)
    ap.add_argument("--fused-adam", action=B, default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="/content/data")
    ap.add_argument("--train-file", default=None, help="override (tests): byte file used instead of the TinyStories train")
    ap.add_argument("--val-file", default=None)
    ap.add_argument("--out", default=None, help="run dir (default /content/runs/<arm>)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--eval-windows", type=int, default=64)
    ap.add_argument("--stop-after", type=int, default=0, help="stop (and checkpoint) after this step, e.g. to simulate preemption")
    ap.add_argument("--batches", default="64,128,192,256", help="--probe-batch sizes")
    ap.add_argument("--probe-n", type=int, default=3, help="--probe-batch steps per trial (first = compile, not timed)")
    ap.add_argument("--lrs", default=None, help="--lr-probe list (dense: adam-lr; lg: counter lr)")
    ap.add_argument("--probe-steps", type=int, default=300)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.out is None:
        args.out = f"/content/runs/{args.arm.split(',')[0]}"
    if args.seq > 512:
        raise SystemExit("--seq must be <= 512 (CharLM position table)")
    if args.prepare_data:
        return run_prepare(args)
    if args.probe_batch:
        run_probe_batch(args)
        return 0
    if args.lr_probe:
        try:
            run_lr_probe(args)
        except BaseException as e:                               # noqa: BLE001
            if is_oom(e):
                print(f"OUT OF MEMORY with --batch {args.batch}: use a smaller --batch (see --probe-batch).", flush=True)
                return 3
            raise
        return 0
    res = run_train(args)
    if res.get("oom"):
        return 3
    return 4 if res.get("diverged") else 0


if __name__ == "__main__":
    raise SystemExit(main())
