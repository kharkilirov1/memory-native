#!/usr/bin/env python
"""Training-step speed + memory benchmark of the e1_quality arms (run it ALONE on an idle GPU).

One timed unit = ONE FULL TRAINING STEP of the same char-LM that probes/e1_quality.py trains
(model built by e1_quality.build_model_opt: embeddings / LN / head on AdamW, block linears per arm):
    forward + cross-entropy + zero_grad + backward (incl. the in-backward counter / index update of the
    mx6, lg and tern arms) + optimizer.step (AdamW for dense / qat arms and for emb / LN / head everywhere).
Synthetic uniform-random byte tokens (speed does not depend on the data), batch 32 x seq 128 by default, fp32
(torch matmul precision "highest", like the quality runs; --matmul-precision high enables TF32).

Per (arm, d):  --warmup 30 steps, then --steps 100 timed steps. Reported:
  ms/step   wall clock of the synchronized 100-step window (headline) and the CUDA-event mean / median / min per step
  tok/s     batch * seq / (ms/step)
  phases    median of --phase-steps steps with CUDA events around forward | backward | optimizer.step, plus
            update_ms = time inside the layers' apply_update_ (a SUBSET of backward) and decode_ms = time inside the
            layers' weight() / quantized_weight() (fake-quant / unpack+decode, a subset of forward)
  kernels   torch.profiler on one step: top-10 CUDA kernels by self time, a category split (gemm / rng / ... ) and the
            number of kernel launches per step (CPU-only machines list the top CPU ops instead)
  memory    peak = torch.cuda.max_memory_allocated over warmup + timed steps (stats reset after the model is built),
            resident = memory_allocated between steps, both minus what was allocated before the model; plus the
            ANALYTIC training state of the block linears (see e1_quality's docstring: bits/8 x block-linear params).
Each (arm, d) is fail-soft: an exception / OOM is recorded in the table and the JSON and the bench goes on.

Arms (defaults): dense_nobias, qat_fp4, tern_counter (--tern-lr 3e-2), na_fp4_c8 (nvfp4, ladder headroom 2),
lg6_fp4_h2, lg8_fp4_h2 (the latent grid, mx6/latent_grid.py via probes/lg_ab.py). Any other e1_quality / lg_ab arm works.

  --compile     also run "<arm>+compile": na_* -> mx6's packed_na.enable_compile() (torch.compile of the decode / step
                helper functions); dense_nobias / qat_fp4 -> torch.compile(model); lg* -> REGIONAL compile of the whole
                model: every LatentGridLinear is an opaque custom op (torch.ops.mx6.lg_linear / lg_linear_bwd with the
                in-backward update declared as a mutation), everything else (embeddings, LN, attention, GELU, residuals,
                head, loss inputs) is compiled by inductor with fullgraph=True (a graph break raises). --lg-compile
                disable = fallback: torch.compiler.disable around each lg forward (graph breaks, always works).
                tern_counter has no compile path (in-backward state mutation). A compile failure is reported on that row
                only. The 30 warmup steps absorb the compile time. Rows record `graphs` / `graph_breaks` (dynamo
                counters) and `updates_per_step` (real update-kernel launches in one step; must equal #lg layers).
  --sdpa X      attention = F.scaled_dot_product_attention(is_causal=True) (flash) instead of the naive T x T softmax, for
                ALL arms (same model code): auto (default) = on for --preset d512/d1024/d2048/1b, off otherwise; on | off.
                A bare `--sdpa` means on.
  --fused X     na_* only: MXNLinear.fused = off (default) | auto | on. "auto" uses the Triton decode-in-GEMM kernel only
                for M <= 8 rows, so at batch*seq = 4096 it is the same as off; "on" forces it (CUDA + Triton only).
  --fast-pack   EXPERIMENT (na arms whose code is 8 bit, e.g. na_fp4_c8): also run "<arm>+fastpack" with PackedNA.pack_state /
                unpack_state replaced by the identity for bits == 8 (an 8-bit MSB-first packed byte IS the code; the stock
                generic path expands every code to 8 int64 bits). Bit-exactness vs the stock path is checked first.
                mx6 itself is not modified. See probes/SPEED_NOTES.md for why.

Strong-baseline mode (see the flags): --precision bf16 (autocast of the whole model; custom-autograd counter linears stay
fp32 islands), --adam-impl fused, arm "dense_strong" (= dense_nobias + bf16 + fused AdamW + torch.compile; the reference
every other row is divided by in "vs dense_strong"), --preset tiny|d512|d1024|d2048|1b, --grad-checkpoint (dense/qat arms only).

Outputs: <out>.json (everything, incl. per-config kernels) and <out>.md (the tables below); both also on stdout.

Colab:
    python probes/bench_train_step.py --out bench_out/bench                       # the standard bench (all arms, d 128/256/512)
    python probes/bench_train_step.py --compile --fast-pack --out bench_out/bench_x   # + compile / fast-pack variants

Estimates of where the time SHOULD go (passes over the weights, bytes moved, fused-kernel potential) are in
probes/SPEED_NOTES.md; this script measures them.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import itertools
import json
import statistics
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lg_ab                                            # noqa: E402  (probes/lg_ab.py: e1_quality + latent-grid arms)
from mx6 import packed_na                               # noqa: E402
from mx6.layer import MXNLinear                         # noqa: E402

DEFAULT_ARMS = "dense_strong,dense_nobias,qat_fp4,tern_counter,na_fp4_c8,lg6_fp4_h2,lg8_fp4_h2"
MODEL_COMPILE_ARMS = ("dense_nobias", "dense_bf16", "qat_fp4", "qat_tern", "qat_e2m1lad")

# dense_strong = THE reference to beat: dense_nobias (nn.Linear bias=False) + bf16 autocast + fused AdamW + torch.compile.
# It is an alias handled here (e1_quality has no such arm): built as dense_nobias, always bf16 / fused / compiled.
STRONG_ARM = "dense_strong"
ARM_ALIASES = {STRONG_ARM: "dense_nobias", "lg6_fp4_h2_nvfp4fwd": "lg6_fp4_h2", "lg8_fp4_h2_nvfp4fwd": "lg8_fp4_h2"}
# lg arms whose forward GEMM is NOT the exact cuBLAS bf16 one: cuBLASLt NVFP4 with x quantized on the fly (APPROXIMATE).
ARM_LG_GEMM = {"lg6_fp4_h2_nvfp4fwd": "nvfp4_scaled_mm", "lg8_fp4_h2_nvfp4fwd": "nvfp4_scaled_mm"}


def configure_lg(model, args, arm: str) -> dict:
    """Apply --lg-native / --lg-gemm / --lg-save-weight (and the arm's own gemm) to every LatentGridLinear BEFORE
    fp32_islands decides which linears are islands. -> {'lg_layers', 'native', 'gemm'}."""
    from mx6.latent_grid import LatentGridLinear
    native = {"auto": "auto", "on": True, "off": False}[getattr(args, "lg_native", "auto")]
    gemm = ARM_LG_GEMM.get(arm, getattr(args, "lg_gemm", "cublas_bf16"))
    n = 0
    for m in model.modules():
        if isinstance(m, LatentGridLinear):
            m.native, m.gemm, m.save_weight = native, gemm, bool(getattr(args, "lg_save_weight", False))
            n += 1
    return {"lg_layers": n, "native_lg": sum(1 for m in model.modules()
                                              if isinstance(m, LatentGridLinear) and m.native_enabled()),
            "lg_gemm": gemm if n else None}

# preset -> (d, layers, heads, seq, batch). Explicit --d/--layers/--heads/--seq/--batch override single fields.
PRESETS = {
    "tiny":  (128, 2, 2, 64, 8),        # CPU-sized smoke test
    "d512":  (512, 8, 8, 256, 32),
    "d1024": (1024, 8, 16, 512, 16),
    "d2048": (2048, 16, 32, 512, 8),
    "1b":    (2048, 24, 32, 512, 8),    # ~1.2B params with the 4d MLP, byte-level vocab 256
}
LEGACY_DEFAULTS = {"d": "128,256,512", "layers": 4, "heads": 0, "seq": 128, "batch": 32}


def resolve_shape(args) -> None:
    """Fill args.d / layers / heads / seq / batch (None = unset) from --preset, else from the legacy defaults."""
    pre = dict(zip(("d", "layers", "heads", "seq", "batch"), PRESETS[args.preset])) if args.preset else {}
    for k, legacy in LEGACY_DEFAULTS.items():
        if getattr(args, k) is None:
            setattr(args, k, str(pre[k]) if k == "d" and k in pre else pre.get(k, legacy))


def param_count(d: int, layers: int, vocab: int) -> int:
    """Exact parameter count of the bias-free CharLM (tok + pos(512) + blocks + LNs + head); equals the built model."""
    return _analytic(d, layers, vocab)["total"]


def _analytic(d: int, layers: int, vocab: int) -> dict:
    n_lin = layers * 12 * d * d
    n_other = 2 * vocab * d + 512 * d + (2 * layers + 1) * 2 * d
    return {"total": n_lin + n_other, "lin": n_lin, "other": n_other}


def amp_ctx(precision: str, dev: str):
    """bf16 -> torch.autocast(dev, bfloat16) (forward only; backward replays the casts); fp32 -> no-op."""
    if precision == "bf16":
        return torch.autocast(device_type=dev, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def has_custom_autograd_linears(model) -> bool:
    """True if any block linear is not a plain nn.Linear (mx6 / lg / tern / dense_rms: stateful in-backward update)."""
    return any(not isinstance(lin, torch.nn.Linear) for blk in model.blocks for lin in (blk.qkv, blk.proj, blk.fc, blk.out))


def enable_grad_checkpoint(model) -> int:
    """torch.utils.checkpoint (non-reentrant) around every block. The recompute re-runs forward during backward, which
    breaks the counter layers' one-forward-per-backward guard -> callers must refuse arms with custom linears."""
    from torch.utils.checkpoint import checkpoint
    for blk in model.blocks:
        blk.forward = (lambda x, _f=blk.forward: checkpoint(_f, x, use_reentrant=False))
    return len(model.blocks)

# kernel-name -> category (first match wins; lower-cased name)
CATEGORIES = (
    ("gemm", ("gemm", "cutlass", "cublas", "nvjet", "xmma", "sm80_", "sm90_", "sm100_", "sm120_", "ampere_", "magma")),
    ("rng", ("philox", "distribution", "rand", "curand")),
    ("index/gather", ("index", "gather", "scatter", "bucketize", "searchsorted", "embedding")),
    ("reduce", ("reduce", "softmax", "layer_norm", "layernorm", "norm_", "cunn_")),
    ("copy/cast", ("copy", "cast", "to_copy", "memcpy", "memset", "fill")),
    ("triton", ("triton_",)),
    ("elementwise", ("elementwise", "vectorized", "unrolled", "foreach", "multi_tensor", "binary", "unary")),
)


def classify_kernel(name: str) -> str:
    n = name.lower()
    for cat, keys in CATEGORIES:
        if any(k in n for k in keys):
            return cat
    return "other"


class Clock:
    """CUDA events on a GPU, perf_counter on the CPU (CPU timings are plumbing only)."""

    def __init__(self, cuda: bool):
        self.cuda = cuda

    def mark(self):
        if self.cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            return e
        return time.perf_counter()

    def ms(self, a, b) -> float:
        return a.elapsed_time(b) if self.cuda else (b - a) * 1e3

    def sync(self):
        if self.cuda:
            torch.cuda.synchronize()


class Spans:
    """Accumulates (start, end) marks around selected methods of the layers (instance-attribute wrappers)."""

    def __init__(self, clock: Clock):
        self.clock, self.spans, self._undo = clock, {}, []

    def wrap(self, obj, attr: str, key: str) -> bool:
        fn = getattr(obj, attr, None)
        if not callable(fn):
            return False
        clock, bucket = self.clock, self.spans.setdefault(key, [])

        def wrapped(*a, **k):
            s = clock.mark()
            out = fn(*a, **k)
            bucket.append((s, clock.mark()))
            return out

        setattr(obj, attr, wrapped)
        self._undo.append((obj, attr))
        return True

    def total_ms(self, key: str) -> float | None:
        if key not in self.spans or not self.spans[key]:
            return None
        return sum(self.clock.ms(s, e) for s, e in self.spans[key])

    def restore(self):
        for obj, attr in self._undo:
            with contextlib.suppress(AttributeError):
                delattr(obj, attr)
        self._undo.clear()


def instrument(model, spans: Spans) -> dict:
    """update: apply_update_ of mx6 / lg / dense_rms layers (tern: the fused-kernel call); decode: weight() of
    mx6 / lg layers, quantized_weight() of qat layers."""
    found = {"update": 0, "decode": 0}
    for blk in model.blocks:
        for lin in (blk.qkv, blk.proj, blk.fc, blk.out):
            for attr in ("apply_update_", "_fused_update"):
                if spans.wrap(lin, attr, "update"):
                    found["update"] += 1
                    break
            for attr in ("quantized_weight", "decode_weight", "weight"):      # lg native path: decode_weight (fwd + bwd re-decode)
                if spans.wrap(lin, attr, "decode"):
                    found["decode"] += 1
                    break
    return found


# ── experiments: compile hooks, fast pack ──────────────────────────────────────

def _dynamo_reset():
    try:
        import torch._dynamo as dynamo
        dynamo.reset()
        dynamo.config.cache_size_limit = max(int(dynamo.config.cache_size_limit), 64)
    except Exception:                                      # pragma: no cover - torch without dynamo
        pass


@contextlib.contextmanager
def compile_hook(kind: str):
    """kind: 'na' (packed_na.enable_compile), 'lg' (EXPERIMENTAL compiled LatentGrid.move), 'model' (caller wraps)."""
    _dynamo_reset()
    if kind == "na":
        packed_na.enable_compile()
        try:
            yield
        finally:
            packed_na.disable_compile()
    elif kind == "lg":
        from mx6.latent_grid import LatentGrid
        eager = LatentGrid.move
        LatentGrid.move = torch.compile(eager, dynamic=False)
        try:
            yield
        finally:
            LatentGrid.move = eager
    else:
        yield
    _dynamo_reset()


@contextlib.contextmanager
def fast_pack_hook():
    """PackedNA.pack_state / unpack_state := identity for 8-bit uint8 codes (checked bit-exact against the stock path)."""
    cls = packed_na.PackedNA
    orig_pack, orig_unpack = cls.pack_state, cls.unpack_state

    def fast_pack(self, codes):
        if self.bits == 8 and self.code_dtype == torch.uint8:
            return codes.to(torch.uint8).contiguous()
        return orig_pack(self, codes)

    def fast_unpack(self, packed, in_features):
        if self.bits == 8 and self.code_dtype == torch.uint8:
            return packed.reshape(packed.shape[0], in_features).clone()
        return orig_unpack(self, packed, in_features)

    fmt = packed_na.PackedNA([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], C=8, scale_format="nvfp4")   # 225 codes, 8 bit
    codes = torch.randint(0, fmt.n_codes, (6, 48), generator=torch.Generator().manual_seed(0)).to(torch.uint8)
    ref = orig_pack(fmt, codes)
    cls.pack_state, cls.unpack_state = fast_pack, fast_unpack
    try:
        assert fmt.bits == 8 and torch.equal(fast_pack(fmt, codes), ref), "fast pack != stock pack"
        assert torch.equal(fast_unpack(fmt, ref, 48), orig_unpack(fmt, ref, 48)), "fast unpack != stock unpack"
        yield
    finally:
        cls.pack_state, cls.unpack_state = orig_pack, orig_unpack


# ── measurement ────────────────────────────────────────────────────────────────

def count_updates_per_step(step, model):
    """Run ONE step with the fused update kernel wrapped by a counter -> (launches, #LatentGridLinear layers).
    A correct step launches exactly one update per layer (compile must neither skip nor repeat it)."""
    import mx6.latent_grid as LG
    n = {"c": 0}
    orig = LG.lg_update_

    def counted(*a, **k):
        n["c"] += 1
        return orig(*a, **k)

    LG.lg_update_ = counted
    try:
        step()
    finally:
        LG.lg_update_ = orig
    return n["c"], len(LG.lg_layers(model))


def make_batches(n: int, batch: int, seq: int, vocab: int, dev: str, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        t = torch.randint(0, vocab, (batch, seq + 1), generator=g)
        out.append((t[:, :-1].contiguous().to(dev), t[:, 1:].contiguous().to(dev)))
    return out


def train_step_fn(fwd, opt, batches, vocab: int, amp=None):
    """amp: a zero-arg factory of the forward's context manager (bf16 autocast), None = fp32. Backward is outside it."""
    it = itertools.cycle(batches)
    amp = amp or contextlib.nullcontext

    def step():
        x, y = next(it)
        with amp():
            logits = fwd(x)
        loss = F.cross_entropy(logits.float().reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss
    return step


def time_steps(step, n: int, clock: Clock):
    """-> (per-step ms from events / perf_counter, wall ms per step over the synchronized window)."""
    clock.sync()
    t0 = time.perf_counter()
    marks = []
    for _ in range(n):
        s = clock.mark()
        step()
        marks.append((s, clock.mark()))
    clock.sync()
    wall = (time.perf_counter() - t0) * 1e3 / n
    return [clock.ms(s, e) for s, e in marks], wall


def phase_breakdown(fwd, opt, batches, vocab: int, n: int, clock: Clock, spans: Spans | None, amp=None):
    """Median ms of forward | backward | optimizer.step over n steps; update / decode = mean per step of the wrapped
    spans (subsets of backward / forward)."""
    it = itertools.cycle(batches)
    amp = amp or contextlib.nullcontext
    rows = []
    for _ in range(n):
        x, y = next(it)
        a = clock.mark()
        with amp():
            logits = fwd(x)
        loss = F.cross_entropy(logits.float().reshape(-1, vocab), y.reshape(-1))
        b = clock.mark()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        c = clock.mark()
        opt.step()
        d = clock.mark()
        rows.append((a, b, c, d))
    clock.sync()
    ph = [[clock.ms(r[i], r[i + 1]) for r in rows] for i in range(3)]
    out = {"fwd_ms": statistics.median(ph[0]), "bwd_ms": statistics.median(ph[1]), "opt_ms": statistics.median(ph[2])}
    for key, name in (("update", "update_ms"), ("decode", "decode_ms")):
        tot = spans.total_ms(key) if spans is not None else None
        out[name] = None if tot is None else tot / n
    return out


def profile_step(step, clock: Clock, topk: int) -> dict:
    """torch.profiler over one step -> top-k device kernels by self time (CPU ops on a CPU-only machine)."""
    from torch.profiler import ProfilerActivity, profile
    cuda = clock.cuda
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
    with profile(activities=acts) as prof:
        step()
        clock.sync()
    return summarize_events(list(prof.key_averages()), cuda, topk)


def summarize_events(avgs, cuda: bool, topk: int) -> dict:
    """key_averages() rows -> {unit, total_ms, launches, top[{name, ms, pct, count}], categories{name: ms}}.
    cuda: only device (kernel) rows by self device time; else CPU ops by self CPU time."""
    dev_type = getattr(torch.autograd, "DeviceType", None)

    def collect(filter_kernels: bool):
        out = []
        for e in avgs:
            if cuda:
                if filter_kernels and dev_type is not None and getattr(e, "device_type", None) != dev_type.CUDA:
                    continue
                t_us = getattr(e, "self_device_time_total", None)
                if t_us is None:
                    t_us = getattr(e, "self_cuda_time_total", 0.0)
            else:
                t_us = getattr(e, "self_cpu_time_total", 0.0)
            if t_us and t_us > 0:
                out.append((e.key, float(t_us) / 1e3, int(e.count)))
        return out

    rows = collect(True) or collect(False)          # second chance if this torch labels kernels differently
    rows.sort(key=lambda r: -r[1])
    total = sum(r[1] for r in rows) or float("nan")
    cats: dict[str, float] = {}
    for name, ms, _ in rows:
        c = classify_kernel(name) if cuda else "cpu-op"
        cats[c] = cats.get(c, 0.0) + ms
    return {"unit": "cuda_kernels" if cuda else "cpu_ops", "total_ms": total, "launches": sum(r[2] for r in rows),
            "top": [{"name": n[:110], "ms": ms, "pct": 100.0 * ms / total, "count": c} for n, ms, c in rows[:topk]],
            "categories": {k: v for k, v in sorted(cats.items(), key=lambda kv: -kv[1])}}


def cuda_cleanup(cuda: bool):
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()


def bench_one(arm: str, d: int, variant: str, args, e1, vocab: int, dev: str, cuda: bool | None = None) -> dict:
    """variant: 'eager' | 'compile' | 'fastpack'. Never raises: failures land in rec['error'].
    cuda: use CUDA events / memory counters (default: dev == 'cuda'; tests fake torch.cuda and pass True)."""
    if cuda is None:
        cuda = dev == "cuda"
    clock = Clock(cuda)
    heads = args.heads if args.heads > 0 else max(1, d // 64)
    precision = getattr(args, "precision", "fp32")
    adam_impl = getattr(args, "adam_impl", "foreach")
    use_ckpt = bool(getattr(args, "grad_checkpoint", False))
    strong = arm == STRONG_ARM
    if strong:                                              # the reference is bf16 + fused AdamW + compile, always
        precision, adam_impl = "bf16", "fused"
    base_arm = ARM_ALIASES.get(arm, arm)                    # what e1_quality actually builds
    label = arm + ("" if variant == "eager" or strong else "+" + variant)
    if arm.startswith("na_") and args.fused != "off":
        label += f"+fused-{args.fused}"
    rec = {"arm": arm, "label": label, "variant": variant, "d": d, "layers": args.layers, "heads": heads,
           "batch": args.batch, "seq": args.seq, "precision": precision, "adam_impl": adam_impl,
           "grad_checkpoint": use_ckpt, "error": None}
    model = opt = spans = fwd = step = batches = None
    try:
        lr = e1.arm_lr(base_arm, SimpleNamespace(lr=args.lr, tern_lr=args.tern_lr, dense_rms_lr=0.01, qat_lr=None,
                                            adam_lr=args.adam_lr))
        kind = ("na" if base_arm.startswith("na_") else "lgmodel" if base_arm.startswith("lg") else
                "model" if base_arm in MODEL_COMPILE_ARMS else None)
        if variant == "compile" and kind is None:
            rec["unsupported"] = True
            raise RuntimeError("UNSUPPORTED: no compile path for this arm (in-backward state mutation is eager-only; "
                               "torch.compile cannot trace the counter update) -- not a failure")
        if variant == "fastpack" and not arm.startswith("na_"):
            raise RuntimeError("--fast-pack only applies to na_* arms")
        with contextlib.ExitStack() as stack:
            if variant == "compile":
                stack.enter_context(compile_hook(kind))
            if variant == "fastpack":
                stack.enter_context(fast_pack_hook())
            if cuda:
                cuda_cleanup(True)
                torch.cuda.synchronize()
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(args.seed)
            model, opt = e1.build_model_opt(base_arm, vocab, d, lr, args.adam_lr, args.layers, heads)
            model.train()
            if use_ckpt:
                if has_custom_autograd_linears(model):
                    raise RuntimeError(
                        "--grad-checkpoint refused for this arm: checkpoint re-runs the block forward inside backward, "
                        "which breaks the counter layers' one-forward-per-backward guard (_pending / weight._version) and "
                        "would apply the in-backward update twice. Use it only for dense_* / dense_strong / qat_* arms.")
                rec["checkpointed_blocks"] = enable_grad_checkpoint(model)
            if base_arm.startswith("lg"):
                rec.update(configure_lg(model, args, arm))
            if precision != "fp32":
                # custom-autograd counter linears (mx6 / lg / tern) cannot take bf16 (backward runs outside autocast ->
                # dtype mismatch): their forward becomes an fp32 island (autocast off, fp32 in/out); nn.Linear/QAT: autocast
                rec["fp32_islands"] = e1.fp32_islands(model)
            if adam_impl == "fused":                       # same groups / lr / defaults, fused multi-tensor kernel
                groups = [{"params": g["params"], "base_lr": g.get("base_lr", args.adam_lr)} for g in opt.param_groups]
                opt = torch.optim.AdamW(groups, lr=args.adam_lr, fused=True)
            if base_arm.startswith("na_"):
                for m in model.modules():
                    if isinstance(m, MXNLinear):
                        m.fused = args.fused
            fwd = model
            counters = None
            if variant == "compile" and kind == "model":
                fwd = torch.compile(model, dynamic=False)
            elif variant == "compile" and kind == "lgmodel":
                from mx6.latent_grid import HAS_LG_OPS, lg_layers, prepare_lg_for_compile
                want = getattr(args, "lg_compile", "op")
                if want == "op" and not (HAS_LG_OPS and any(m.native_enabled() for m in lg_layers(model))):
                    want = "disable"                       # no ops / lg not on the native path: eager islands instead
                rec["lg_compile"] = want
                prepare_lg_for_compile(model, want)
                fwd = torch.compile(model, dynamic=False, fullgraph=(want == "op"))
            if variant == "compile":
                import torch._dynamo.utils as dyn_utils
                counters = dyn_utils.counters
                counters.clear()
            bits = e1.measured_bits(model, base_arm)
            rec.update(e1.memory_accounting(model, base_arm, bits))
            rec["state_mb"] = rec["state_bytes"] / 2**20
            rec["state_grad_mb"] = (rec["state_bytes"] + rec["grad_bytes"]) / 2**20
            rec["other_state_mb"] = rec["other_state_bytes"] / 2**20
            batches = make_batches(8, args.batch, args.seq, vocab, dev, args.seed)
            amp = lambda: amp_ctx(precision, dev)          # noqa: E731
            step = train_step_fn(fwd, opt, batches, vocab, amp=amp)
            for _ in range(args.warmup):
                loss = step()
            clock.sync()
            if cuda:
                rec["resident_mb"] = (torch.cuda.memory_allocated() - base) / 2**20
            if counters is not None:
                rec["graphs"] = int(counters["stats"]["unique_graphs"])
                rec["graph_breaks"] = int(sum(counters["graph_break"].values()))
            if base_arm.startswith("lg") and cuda:
                rec["updates_per_step"], rec["lg_layers_n"] = count_updates_per_step(step, model)
            ev_ms, wall_ms = time_steps(step, args.steps, clock)
            loss = step()
            clock.sync()
            rec["final_loss"] = float(loss.detach())
            rec["finite"] = bool(torch.isfinite(loss).item())
            if cuda:
                rec["peak_mem_mb"] = (torch.cuda.max_memory_allocated() - base) / 2**20
                rec["peak_mem_abs_mb"] = torch.cuda.max_memory_allocated() / 2**20
            else:
                rec["resident_mb"] = rec["peak_mem_mb"] = rec["peak_mem_abs_mb"] = float("nan")
            rec["ms_step"] = wall_ms
            rec["ms_step_ev_mean"] = statistics.fmean(ev_ms)
            rec["ms_step_ev_median"] = statistics.median(ev_ms)
            rec["ms_step_ev_min"] = min(ev_ms)
            rec["tokens_per_s"] = args.batch * args.seq / (wall_ms / 1e3)
            rec["model_tflops"] = 6.0 * rec["n_total"] * rec["tokens_per_s"] / 1e12   # 6*N*tokens/time (N = all params)
            if args.phase_steps > 0:
                spans = Spans(clock)
                rec["instrumented"] = instrument(model, spans) if variant != "compile" or kind not in ("model", "lgmodel") else {}
                rec.update(phase_breakdown(fwd, opt, batches, vocab, args.phase_steps, clock, spans, amp=amp))
                spans.restore()
            if not args.no_profile:
                try:
                    rec["profile"] = profile_step(step, clock, args.topk)
                except Exception as e:                       # profiler problems must not lose the timing
                    rec["profile"] = {"error": f"{type(e).__name__}: {e}"}
    except Exception as e:                                   # incl. CUDA OOM, compile failures
        rec["error"] = (str(e) if rec.get("unsupported") else
                        f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}")
        rec["traceback"] = traceback.format_exc()[-1800:]
        rec["oom"] = "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"
        if cuda and rec["oom"]:
            rec["error"] = "OOM: " + rec["error"] + (" (try a smaller --batch or --grad-checkpoint)" if not use_ckpt else "")
    finally:
        if spans is not None:
            spans.restore()
        # drop EVERY reference to the model (compiled wrapper / step closure / batches hold it) before the next arm builds
        del model, opt, spans, fwd, step, batches
        rec.setdefault("oom", False)
        cuda_cleanup(cuda)
    return rec


# ── report ─────────────────────────────────────────────────────────────────────

def _json_clean(o):
    """NaN / inf -> null so the file is strict JSON (CPU runs have nan memory columns)."""
    if isinstance(o, float):
        return o if o == o and o not in (float("inf"), float("-inf")) else None
    if isinstance(o, dict):
        return {k: _json_clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_clean(v) for v in o]
    return o


def _f(x, nd=2, none="-"):
    return none if x is None or (isinstance(x, float) and x != x) else f"{x:.{nd}f}"


def markdown(meta: dict, recs: list[dict]) -> str:
    L = [f"### bench_train_step -- {meta['device']} | torch {meta['torch']} | triton {meta['triton']} | "
         f"batch {meta['batch']} x seq {meta['seq']} | {meta['layers']} layers | warmup {meta['warmup']} + {meta['steps']} timed steps | "
         f"matmul precision {meta['matmul_precision']} | --precision {meta.get('precision', 'fp32')} | "
         f"adam {meta.get('adam_impl', 'foreach')} | grad-checkpoint {meta.get('grad_checkpoint', False)} | "
         f"sdpa {meta.get('sdpa', False)}"
         + (f" | preset {meta['preset']}" if meta.get("preset") else ""), ""]
    ref = {r["d"]: r["ms_step"] for r in recs if r["arm"] == "dense_nobias" and r["variant"] == "eager" and not r["error"]}
    strong = {r["d"]: r["ms_step"] for r in recs if r["arm"] == STRONG_ARM and not r["error"]}
    L += ["| arm | d | heads | total params | block-linear params | bits | ms/step | (ev median) | tok/s | model TFLOPS | "
          "vs dense_strong | vs dense | fwd | bwd | opt | update (in bwd) | decode (in fwd) | peak MB | resident MB | "
          "state MB | +grad MB | note |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    vorder = {"eager": 0, "compile": 1, "fastpack": 2}
    for r in sorted(recs, key=lambda r: (r["d"], meta["arms"].index(r["arm"]) if r["arm"] in meta["arms"] else 99,
                                         vorder.get(r["variant"], 9))):
        if r["error"]:
            L.append(f"| {r['label']} | {r['d']} | {r['heads']} | | | | "
                     f"{'UNSUPPORTED' if r.get('unsupported') else 'FAILED'} | | | | | | | | | | | | | | "
                     f"{r['error'][:140].replace('|', '/')} |")
            continue
        vs = r["ms_step"] / ref[r["d"]] if r["d"] in ref else None
        vss = r["ms_step"] / strong[r["d"]] if r["d"] in strong else None      # >1 = slower than dense_strong
        note = "" if r.get("finite", True) else "NON-FINITE LOSS"
        if r.get("fp32_islands"):
            note = (note + " fp32 linears").strip()
        if r.get("native_lg"):
            note = (note + f" native bf16 lg ({r.get('lg_gemm')})").strip()
        if r.get("graphs") is not None:
            note = (note + f" compile: {r['graphs']} graph(s), {r['graph_breaks']} break(s)"
                    + (f", lg={r['lg_compile']}" if r.get("lg_compile") else "")).strip()
        if r.get("updates_per_step") is not None:
            ok_u = r["updates_per_step"] == r["lg_layers_n"]
            note = (note + f" updates/step {r['updates_per_step']}/{r['lg_layers_n']}" + ("" if ok_u else " WRONG")).strip()
        if r.get("lg_gemm") == "nvfp4_scaled_mm":
            note += " APPROX: NVFP4 activations"
        L.append(f"| {r['label']} | {r['d']} | {r['heads']} | {r['n_total']:,} | {r['n_lin']:,} | {_f(r['bits'])} | "
                 f"{_f(r['ms_step'])} | {_f(r['ms_step_ev_median'])} | {r['tokens_per_s']:,.0f} | "
                 f"{_f(r.get('model_tflops'), 1)} | {_f(vss, 2) + 'x' if vss else '-'} | {_f(vs, 2) + 'x' if vs else '-'} | "
                 f"{_f(r.get('fwd_ms'))} | {_f(r.get('bwd_ms'))} | {_f(r.get('opt_ms'))} | {_f(r.get('update_ms'))} | "
                 f"{_f(r.get('decode_ms'))} | {_f(r.get('peak_mem_mb'), 0)} | {_f(r.get('resident_mb'), 0)} | "
                 f"{_f(r['state_mb'])} | {_f(r['state_grad_mb'])} | {note} |")
    L += ["", "ms/step = wall clock of the synchronized timed window / steps (headline); ev median = median of per-step CUDA "
          "events. fwd/bwd/opt = median of the phase steps; update = time inside apply_update_ (tern: the fused-kernel call "
          "only), decode = weight()/quantized_weight(); both are subsets of bwd / fwd. peak/resident = CUDA memory above "
          "what was allocated before the model (nan on CPU); state/+grad = analytic (bits/8 x block-linear params; "
          "+grad = resident gradient bytes), see e1_quality's docstring. model TFLOPS = 6 x total params x tok/s (attention "
          "FLOPs not counted). vs dense_strong / vs dense = ms/step ratio (<1 = faster). 'fp32 linears' = under --precision "
          "bf16 this arm's custom-autograd block linears run as fp32 islands (no tensor cores), the rest is autocast.", ""]
    for r in recs:
        p = r.get("profile")
        if r["error"] or not p or "top" not in p:
            continue
        L += [f"#### {r['label']} d={r['d']}: top {len(p['top'])} {p['unit']} of one step "
              f"(total {_f(p['total_ms'])} ms in {p['launches']} launches; "
              + ", ".join(f"{k} {v:.2f} ms" for k, v in list(p["categories"].items())[:6]) + ")", "",
              "| kernel | ms | % | calls |", "|---|---|---|---|"]
        L += [f"| `{t['name']}` | {t['ms']:.3f} | {t['pct']:.1f} | {t['count']} |" for t in p["top"]]
        L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    ap.add_argument("--preset", default=None, choices=sorted(PRESETS),
                    help="model size (d, layers, heads, seq, batch): " + ", ".join(f"{k}={v}" for k, v in PRESETS.items())
                    + ". Explicit --d/--layers/--heads/--seq/--batch override single fields")
    ap.add_argument("--d", default=None, help="comma list of model widths (default 128,256,512, or the preset's d)")
    ap.add_argument("--batch", type=int, default=None, help="default 32 (or the preset's)")
    ap.add_argument("--seq", type=int, default=None, help="default 128 (or the preset's); max 512 (position embedding)")
    ap.add_argument("--layers", type=int, default=None, help="blocks (default 4 = the scaling-study config, or the preset's)")
    ap.add_argument("--heads", type=int, default=None, help="attention heads; 0 = d/64 (min 1) (default 0, or the preset's)")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "bf16"],
                    help="fp32 (default, as before) or bf16 = torch.autocast(bfloat16) around the forward of the WHOLE model "
                         "(fp32 master weights + AdamW). Custom-autograd counter linears (na_*/lg*/tern) stay fp32 islands. "
                         "dense_strong is always bf16")
    ap.add_argument("--adam-impl", default="foreach", choices=["foreach", "fused"],
                    help="AdamW implementation of the AdamW arms and of emb/LN/head everywhere: foreach (default, as before) "
                         "or fused=True. dense_strong is always fused")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="torch.utils.checkpoint per block (fits big presets). REFUSED (row FAILED) for counter arms "
                         "(na_*/lg*/tern/dense_rms): the recompute breaks their one-forward-per-backward guard")
    ap.add_argument("--vocab", type=int, default=256, help="byte-level vocabulary")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--phase-steps", type=int, default=20, help="steps for the fwd/bwd/opt breakdown (0 = skip)")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--no-profile", action="store_true", help="skip the torch.profiler kernel table")
    ap.add_argument("--lr", type=float, default=0.01, help="counter lr of the mx6 / lg arms")
    ap.add_argument("--tern-lr", type=float, default=3e-2)
    ap.add_argument("--adam-lr", type=float, default=3e-3)
    ap.add_argument("--headroom", type=float, default=2.0, help="na_* arms: --headroom-mode ladder --headroom h")
    ap.add_argument("--lg-native", default="auto", choices=["auto", "on", "off"],
                    help="lg* arms: native bf16 path (auto = on when the triton backend is active; no fp32 island). "
                         "off = legacy path (fp32 island under --precision bf16) for A/B")
    ap.add_argument("--lg-gemm", default="cublas_bf16", choices=["cublas_bf16", "triton_bf16", "nvfp4_scaled_mm"],
                    help="lg* arms: forward GEMM (nvfp4_scaled_mm = approximate NVFP4 activations; the "
                         "<lg>_nvfp4fwd arms always use it)")
    ap.add_argument("--lg-save-weight", action="store_true",
                    help="lg* arms: keep the decoded bf16 weight from forward to backward (+2 B/param) instead of re-decoding")
    ap.add_argument("--sdpa", nargs="?", const="on", default="auto", choices=["auto", "on", "off"],
                    help="attention via F.scaled_dot_product_attention (flash) for ALL arms: auto = on for --preset "
                         "d512/d1024/d2048/1b, else off; bare --sdpa = on")
    ap.add_argument("--lg-compile", default="op", choices=["op", "disable"],
                    help="lg* arms with --compile: op = opaque custom op, no graph breaks (default); disable = "
                         "torch.compiler.disable around each lg forward (graph breaks)")
    ap.add_argument("--compile", action="store_true", help="also run <arm>+compile where a compile path exists")
    ap.add_argument("--fused", default="off", choices=["off", "auto", "on"], help="na_* arms: MXNLinear.fused")
    ap.add_argument("--fast-pack", action="store_true", help="EXPERIMENT: also run na_* 8-bit arms with identity pack/unpack")
    ap.add_argument("--matmul-precision", default="highest", choices=["highest", "high", "medium"])
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output prefix for <out>.json / <out>.md (default bench_train_step_<time>)")
    args = ap.parse_args(argv)
    resolve_shape(args)
    if args.seq > 512:
        ap.error(f"--seq {args.seq} > 512 (CharLM position embedding has 512 rows)")

    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if dev == "cuda" and not torch.cuda.is_available():
        ap.error("--device cuda but CUDA is not available")
    cuda = dev == "cuda"
    torch.set_float32_matmul_precision(args.matmul_precision)
    if args.precision == "bf16" or STRONG_ARM in args.arms:
        try:                                                   # CPU torch without bf16 autocast matmul: skip, don't crash
            with torch.autocast(device_type=dev, dtype=torch.bfloat16):
                torch.randn(4, 4, device=dev) @ torch.randn(4, 4, device=dev)
        except Exception as e:
            print(f"SKIPPED: bf16 autocast is not usable on {dev} in this torch ({type(e).__name__}: {e})", flush=True)
            return 0

    e1 = lg_ab.install(lg_ab.load_e1(register=False))
    e1.DEV = dev                                               # build_model_opt / CharLM place the model here
    e1.SCALE_FORMAT, e1.HEADROOM_MODE, e1.HEADROOM = "nvfp4", "ladder", args.headroom
    args.sdpa_on = (args.sdpa == "on") or (args.sdpa == "auto" and args.preset in ("d512", "d1024", "d2048", "1b"))
    e1.SDPA = bool(args.sdpa_on)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    widths = [int(x) for x in args.d.split(",") if x.strip()]
    for w in widths:
        h = args.heads if args.heads > 0 else max(1, w // 64)
        if w % h:
            ap.error(f"heads {h} must divide d {w}")
    try:
        import triton
        triton_v = triton.__version__
    except Exception:
        triton_v = "none"
    meta = {"device": torch.cuda.get_device_name(0) if cuda else "cpu", "torch": torch.__version__, "triton": triton_v,
            "cuda": torch.version.cuda if cuda else None, "batch": args.batch, "seq": args.seq, "layers": args.layers,
            "warmup": args.warmup, "steps": args.steps, "matmul_precision": args.matmul_precision, "arms": arms,
            "widths": widths, "fused": args.fused, "headroom": args.headroom, "lr": args.lr, "tern_lr": args.tern_lr,
            "adam_lr": args.adam_lr, "precision": args.precision, "adam_impl": args.adam_impl,
            "grad_checkpoint": args.grad_checkpoint, "preset": args.preset, "sdpa": bool(args.sdpa_on),
            "lg_compile": args.lg_compile,
            "params": {str(w): param_count(w, args.layers, args.vocab) for w in widths},
            "time": datetime.now().isoformat(timespec="seconds"), "args": vars(args)}
    print(f"bench_train_step on {meta['device']} (torch {meta['torch']}, triton {triton_v}), arms={arms}, d={widths}, "
          f"{args.layers} layers, batch {args.batch} x seq {args.seq}, warmup {args.warmup} + {args.steps} steps"
          + ("" if cuda else "  [CPU: plumbing only, timings are meaningless]"), flush=True)
    print(f"precision={args.precision} adam={args.adam_impl} grad_checkpoint={args.grad_checkpoint} preset={args.preset} | "
          f"params: " + ", ".join(f"d={w}: {n:,}" for w, n in meta["params"].items()), flush=True)

    recs = []
    for d in widths:
        for arm in arms:
            variants = ["compile"] if arm == STRONG_ARM else ["eager"]      # dense_strong = compiled, always
            if args.compile and arm != STRONG_ARM:
                variants.append("compile")
            if args.fast_pack and arm.startswith("na_"):
                variants.append("fastpack")
            for variant in variants:
                r = bench_one(arm, d, variant, args, e1, args.vocab, dev)
                recs.append(r)
                if r["error"]:
                    tag = "SKIPPED" if r.get("unsupported") else "FAILED"
                    print(f"  d={d:<4} {r['label']:<28} {tag}: {r['error']}", flush=True)
                else:
                    print(f"  d={d:<4} {r['label']:<28} {r['ms_step']:8.2f} ms/step  {r['tokens_per_s']:>10,.0f} tok/s  "
                          f"fwd {_f(r.get('fwd_ms'))} bwd {_f(r.get('bwd_ms'))} opt {_f(r.get('opt_ms'))} "
                          f"(update {_f(r.get('update_ms'))})  peak {_f(r.get('peak_mem_mb'), 0)} MB", flush=True)

    prefix = args.out if args.out else f"bench_train_step_{datetime.now():%Y%m%d_%H%M%S}"
    jpath, mpath = Path(prefix + ".json"), Path(prefix + ".md")
    jpath.parent.mkdir(parents=True, exist_ok=True)
    md = markdown(meta, recs)
    jpath.write_text(json.dumps(_json_clean({"meta": meta, "results": recs}), indent=1, default=str, allow_nan=False),
                     encoding="utf-8")
    mpath.write_text(md + "\n", encoding="utf-8")
    print("\n" + md)
    print(f"\nwrote {jpath} and {mpath}", flush=True)
    return 0 if any(not r["error"] for r in recs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
