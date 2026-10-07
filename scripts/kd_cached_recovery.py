"""Cached-KD recovery: train the counter student against a PRECOMPUTED teacher cache.

The 12B-on-T4 recovery loop: no live teacher (see kd_teacher_cache.py), loss =
top-K logit KD (T=KD_T, renormalized over the cached K) + CE_ALPHA * data CE.
Production counter recipe by default: stats_scope=group + decimation=4 + fused kernel,
cosine counter lr, homotopy alpha schedule, strict-alpha=0 selection — mirrors
run_ptq_recovery.py minus feature-KD (uncacheable) and minus the live teacher.

The student is restored WITHOUT the donor ever fully resident (witness machinery), and
its fp tail is cast to bf16 on CUDA (AdamW on the tail; counter layers self-update in
backward through their own kernels).

Run: MODEL=<donor> STATE_DIR=<streamed state> DATA_DIR=<mix bins> CACHE=<cache dir> \
     CKPT_DIR=<out> PYTHONPATH=src python scripts/kd_cached_recovery.py
env: STEPS (from cache), BATCH/SEQ/SEED (must match cache), KD_T (2.0), CE_ALPHA (0.3),
     COUNTER_LR_START (0.002) / COUNTER_LR_END (1e-4), FP_LR (1e-4), GRAD_CLIP (1.0),
     SCALE_LR_START / SCALE_LR_END (both 2e-4; historical flat scale rate),
     STATS_SCOPE (group), DECIMATION (4), HOMOTOPY_HOLD (0.2) / HOMOTOPY_END (0.9),
     HOMOTOPY_ALPHA_START (1; set 0 for strict-only control),
     STRICT_EXPOSURE_EVERY (0=off; 4 runs one strict alpha=0 step per four steps),
     EVAL_EVERY (300), EVAL_MAX_TOKENS (60000), NUM_BLOCKS (0; smoke only), DEVICE,
     GRAD_CKPT (1), FREEZE_EMBED (1), SPLIT_GPUS (1; 2-GPU model-parallel when >=2
     CUDA devices — see split_across_gpus), SPLIT_AT (0 = n_layers//2),
     GROUP/C (optional consistency checks against the streamed warm manifest),
     CKPT_TMP (initial checkpoint staging dir; atomic replacement still requires
     space for old + new checkpoints on the output filesystem). Checkpoints are
     slim and best-only. SYNTHETIC_CALIBRATION=1 requires a labeled toy corpus.

The warm strict-alpha=0 baseline is always evaluated. selection.json and
selected_artifact.json govern use: if no candidate improves it, the selected
artifact remains the original STATE_DIR + MODEL, and no degraded best.pt is made.
Legacy path-only caches must be rebuilt with kd_teacher_cache.py (schema v2).
MIN_IMPROVEMENT (0) is the required strict-metric margin; LOGSUMEXP_CHUNK (8192)
bounds fp32 vocabulary chunks for numerically stable KD/CE losses.

The two 12B-on-T4 fit levers (both default ON, both no-ops at small scale):
- GRAD_CKPT: REENTRANT activation checkpointing on the decoder blocks. Reentrant is
  the only mode compatible with the eager-only counter layers: its first pass runs
  under no_grad, where the counter forward takes the plain path (no reuse guard, no
  autograd Function), and the backward-time recompute builds the Function graph
  exactly once -- its backward fires the counter update as usual. Non-reentrant
  checkpointing re-invokes the Function with the guard already set and raises.
  Without this the 48-block forward alone carries ~5 GiB of activations on top of a
  ~12 GiB resident student (v3 died at 14.31 GiB allocated, inside the HF forward).
- FREEZE_EMBED: the tied 262k x 3840 embedding is ~1.0B fp params; AdamW would
  lazily allocate ~8.5 GiB of moments at the FIRST step (after the forward already
  fits), plus a ~2 GiB dense lm_head weight grad every step. Frozen embeddings keep
  the fp AdamW tail = norms (+ never-exercised multimodal linears); recovery
  capacity rides the counter layers, which is the point of the method. Documented
  delta vs the 1.5B production recipe (there the tail includes embeddings).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from memory_native.recovery.strict_exposure import strict_exposure_alpha
from kd_cache_contract import (
    ShardedCache, load_cache_manifest, model_identity, sha256_file,
    validate_cache_context, validate_warm_source,
)

MODEL = os.environ.get("MODEL", "")
STATE_DIR = os.environ.get("STATE_DIR", "")
DATA_DIR = os.environ.get("DATA_DIR", "")
CACHE = os.environ.get("CACHE", "")
CKPT_DIR = os.environ.get("CKPT_DIR", "ckpt_cached_kd")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
KD_T = float(os.environ.get("KD_T", "2.0"))
CE_ALPHA = float(os.environ.get("CE_ALPHA", "0.3"))
COUNTER_LR_START = float(os.environ.get("COUNTER_LR_START", "0.002"))
COUNTER_LR_END = float(os.environ.get("COUNTER_LR_END", "0.0001"))
SCALE_LR_START = float(os.environ.get("SCALE_LR_START", "0.0002"))
SCALE_LR_END = float(os.environ.get("SCALE_LR_END", "0.0002"))
FP_LR = float(os.environ.get("FP_LR", "0.0001"))
GRAD_CLIP = float(os.environ.get("GRAD_CLIP", "1.0"))
STATS_SCOPE = os.environ.get("STATS_SCOPE", "group")
DECIMATION = int(os.environ.get("DECIMATION", "4"))
HOMOTOPY_ALPHA_START = float(os.environ.get("HOMOTOPY_ALPHA_START", "1.0"))
STRICT_EXPOSURE_EVERY = int(os.environ.get("STRICT_EXPOSURE_EVERY", "0"))
HOMOTOPY_HOLD = float(os.environ.get("HOMOTOPY_HOLD", "0.2"))
HOMOTOPY_END = float(os.environ.get("HOMOTOPY_END", "0.9"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "300"))
EVAL_MAX_TOKENS = int(os.environ.get("EVAL_MAX_TOKENS", "60000"))
NUM_BLOCKS = int(os.environ.get("NUM_BLOCKS", "0"))
LOG_EVERY = int(os.environ.get("LOG_EVERY", "25"))
GRAD_CKPT = os.environ.get("GRAD_CKPT", "1") == "1"
FREEZE_EMBED = os.environ.get("FREEZE_EMBED", "1") == "1"
SPLIT_GPUS = os.environ.get("SPLIT_GPUS", "1") == "1"
SPLIT_AT = int(os.environ.get("SPLIT_AT", "0"))
CKPT_TMP = os.environ.get("CKPT_TMP", "")
EVAL_AT_START = os.environ.get("EVAL_AT_START", "1") == "1"
MIN_IMPROVEMENT = float(os.environ.get("MIN_IMPROVEMENT", "0"))
LOGSUMEXP_CHUNK = int(os.environ.get("LOGSUMEXP_CHUNK", "8192"))
SAVE_BEST = os.environ.get("SAVE_BEST", "1") == "1"


def log(msg: str) -> None:
    print(f"[kd {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cosine(start: float, end: float, progress: float) -> float:
    progress = min(1.0, max(0.0, progress))
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))


def homotopy_alpha(progress: float) -> float:
    if progress <= HOMOTOPY_HOLD:
        return HOMOTOPY_ALPHA_START
    if progress >= HOMOTOPY_END:
        return 0.0
    span = (progress - HOMOTOPY_HOLD) / (HOMOTOPY_END - HOMOTOPY_HOLD)
    return HOMOTOPY_ALPHA_START * 0.5 * (1.0 + math.cos(math.pi * span))


def _tree_to(obj, dev):
    if torch.is_tensor(obj):
        return obj.to(dev)
    if isinstance(obj, (tuple, list)):
        moved = [_tree_to(o, dev) for o in obj]
        return tuple(moved) if isinstance(obj, tuple) else moved
    return obj


def split_across_gpus(student) -> int:
    """Model-parallel over 2 GPUs: layers[split:] move to cuda:1; embeddings, final
    norm and lm_head STAY on cuda:0 so the embed/head tie survives untouched. Device
    movers ride forward pre-hooks (inside the checkpointed region, so both the
    no-grad pass and the recompute cross the boundary identically, and autograd's
    .to nodes route grads back). The two hidden-state boundaries (block split-1 ->
    split, last block -> final norm) are the only real transfers -- [B, S, d] each.

    The v4 lesson forcing this: the 12B counter buffers alone are ~11.7 GiB resident
    (packed state 8.2 + salient idx/val 1.3 + salient perm 1.7 + scales/v 0.5) plus
    the frozen embedding 2.0 -- ~13.8 GiB before a single activation on a 14.56 GiB
    T4. No activation trick fixes resident state; a second T4 does."""
    from memory_native.donor.streaming import _resolve_decoder

    inner, _ = _resolve_decoder(student)
    n = len(inner.layers)
    split = SPLIT_AT if SPLIT_AT > 0 else n // 2
    if not 0 < split < n:
        raise ValueError("2-GPU SPLIT_AT must divide the decoder into two nonempty stacks")
    dev0, dev1 = torch.device("cuda", 0), torch.device("cuda", 1)

    def mover(dev):
        def hook(module, args, kwargs):
            return _tree_to(args, dev), {k: _tree_to(v, dev) for k, v in kwargs.items()}
        return hook

    for i in range(split, n):
        inner.layers[i].to(dev1)
        inner.layers[i].register_forward_pre_hook(mover(dev1), with_kwargs=True)
    inner.norm.register_forward_pre_hook(mover(dev0), with_kwargs=True)
    log(f"2-GPU split: layers {split}..{n - 1} -> cuda:1; embed/norm/head + "
        f"layers 0..{split - 1} on cuda:0")
    return split


class _ChunkedLogsumexp(torch.autograd.Function):
    """FP32 normalization without retaining a full fp32 vocabulary tensor."""

    @staticmethod
    def forward(ctx, logits, temperature, chunk_size):
        dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        lse = torch.full(logits.shape[:-1], -float("inf"), device=logits.device, dtype=dtype)
        for start in range(0, logits.shape[-1], chunk_size):
            chunk = logits[..., start:start + chunk_size].to(dtype) / temperature
            lse = torch.logaddexp(lse, torch.logsumexp(chunk, dim=-1))
        ctx.save_for_backward(logits, lse)
        ctx.temperature, ctx.chunk_size = temperature, chunk_size
        return lse

    @staticmethod
    def backward(ctx, grad_output):
        logits, lse = ctx.saved_tensors
        grad = torch.empty_like(logits)
        for start in range(0, logits.shape[-1], ctx.chunk_size):
            chunk = logits[..., start:start + ctx.chunk_size].to(lse.dtype) / ctx.temperature
            values = torch.exp(chunk - lse.unsqueeze(-1))
            values *= grad_output.unsqueeze(-1) / ctx.temperature
            grad[..., start:start + ctx.chunk_size] = values.to(logits.dtype)
        return grad, None, None


def kd_and_ce_losses(logits: torch.Tensor, idx: torch.Tensor, val: torch.Tensor,
                     targets: torch.Tensor, T: float, *, chunk_size: int = LOGSUMEXP_CHUNK):
    """Top-K KD (teacher renormalized over its cached K) + CE, WITHOUT any full-vocab
    fp32 copy: gathered logits minus chunked FP32 logsumexp. At 262k vocab log_softmax(float())
    route costs ~1.6-2 GiB of transient+retained on top of a 12 GiB-resident 12B student
    — the difference between OOM and fitting a T4. The teacher distribution is
    renormalized over K, with zero mass elsewhere; student normalization includes
    the full vocabulary (it is NOT a softmax over only the gathered K)."""
    if not math.isfinite(T) or T <= 0 or isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("KD temperature and normalization chunk size must be positive")
    if logits.ndim != 3 or logits.shape[1] < 2 or logits.shape[0] < 1 or logits.shape[-1] < 1:
        raise ValueError("student logits must have nonempty [batch, seq>=2, vocab] shape")
    if idx.ndim != 3 or idx.shape[:2] != logits.shape[:2] or val.shape != idx.shape or not 0 < idx.shape[-1] <= logits.shape[-1]:
        raise ValueError("cached topk tensor shapes differ from student logits")
    if idx.dtype not in (torch.int32, torch.int64) or targets.dtype not in (torch.int32, torch.int64) or not val.is_floating_point():
        raise ValueError("invalid cached index, target or logit dtypes")
    if tuple(targets.shape) != (logits.shape[0], logits.shape[1] - 1):
        raise ValueError("CE targets must have [batch, seq-1] shape")
    if any(t.device != logits.device for t in (idx, val, targets)):
        raise ValueError("student logits, cached tensors and targets must share a device")
    if not torch.isfinite(val).all():
        raise ValueError("cached teacher logits must be finite")
    for values in (idx, targets):
        if values.min() < 0 or values.max() >= logits.shape[-1]:
            raise ValueError("KD index or CE target outside student vocabulary")
    lse_T = _ChunkedLogsumexp.apply(logits, T, chunk_size).unsqueeze(-1)
    logq_T = logits.gather(-1, idx.long()).float() / T - lse_T
    logp = F.log_softmax(val.float() / T, dim=-1)
    kd = (logp.exp() * (logp - logq_T)).sum(-1).mean() * (T * T)
    shifted = logits[:, :-1]
    lse_1 = _ChunkedLogsumexp.apply(shifted, 1.0, chunk_size)
    tgt = shifted.gather(-1, targets.long().unsqueeze(-1)).squeeze(-1).float()
    ce = (lse_1 - tgt).mean()
    if not torch.isfinite(kd) or not torch.isfinite(ce):
        raise ValueError("nonfinite KD/CE loss; training step was not applied")
    return kd, ce


class WarmSelection:
    """Accept only evaluated checkpoints that improve the untouched warm model."""

    def __init__(self, metric: float, min_improvement: float = 0.0):
        if not math.isfinite(metric) or not math.isfinite(min_improvement) or min_improvement < 0:
            raise ValueError("warm metric and nonnegative improvement margin must be finite")
        self.warm_metric = self.best_metric = float(metric)
        self.min_improvement = float(min_improvement)
        self.best_step = 0

    def improves(self, metric: float) -> bool:
        return math.isfinite(metric) and metric < self.best_metric - self.min_improvement

    def accept(self, step: int, metric: float) -> None:
        if step <= 0 or not self.improves(metric):
            raise ValueError("candidate does not improve the selected warm/KD artifact")
        self.best_metric, self.best_step = float(metric), int(step)

    def summary(self) -> dict:
        return {"accepted_kd": self.best_step > 0, "selected_step": self.best_step,
                "warm_metric": self.warm_metric, "selected_metric": self.best_metric,
                "min_improvement": self.min_improvement,
                "selection_rule": "strict alpha=0 mean log perplexity; lower is better"}


def _write_json(path, payload) -> None:
    path = Path(path)
    with open(str(path) + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    os.replace(str(path) + ".tmp", path)


def write_selection(out_dir, selection: WarmSelection, *, model, state_dir, format,
                    provenance=None) -> None:
    """Publish concrete reload instructions and distinguish warm fallback from KD."""
    root = Path(out_dir)
    summary = selection.summary()
    artifact = {**summary, "model": str(Path(model).resolve()),
                "conversion_state": str(Path(state_dir).resolve()),
                "format": format, "residual_alpha": 0.0,
                "artifact_type": "cached_kd_partial_checkpoint" if summary["accepted_kd"] else "warm_conversion_state"}
    if provenance is not None:
        artifact["provenance"] = provenance
    if summary["accepted_kd"]:
        if not (root / "best.pt").is_file():
            raise ValueError("accepted KD artifact is missing best.pt")
        artifact["checkpoint"] = str((root / "best.pt").resolve())
        artifact["checkpoint_fingerprint"] = {"bytes": (root / "best.pt").stat().st_size,
                                              "sha256": sha256_file(root / "best.pt")}
    _write_json(root / "selected_artifact.json", artifact)
    _write_json(root / "selection.json", summary)
    marker = "KD_ACCEPTED.txt" if summary["accepted_kd"] else "USE_WARM_STATE.txt"
    other = "USE_WARM_STATE.txt" if summary["accepted_kd"] else "KD_ACCEPTED.txt"
    (root / marker).write_text("selection.json and selected_artifact.json are authoritative.\n", encoding="utf-8")
    (root / other).unlink(missing_ok=True)


def save_best_checkpoint(payload: dict, out_dir, staging_dir="") -> None:
    """Never overwrite the previously selected file with a partial cross-disk copy."""
    root, stage = Path(out_dir), Path(staging_dir or out_dir)
    stage.mkdir(parents=True, exist_ok=True)
    staged = stage / "best.pt.tmp"
    destination_tmp = root / "best.pt.tmp"
    torch.save(payload, staged)
    try:
        if staged.resolve() != destination_tmp.resolve():
            shutil.copyfile(staged, destination_tmp)
        os.replace(destination_tmp, root / "best.pt")
    except BaseException:
        destination_tmp.unlink(missing_ok=True)
        raise
    finally:
        staged.unlink(missing_ok=True)


@torch.no_grad()
def restore_student(*, format=None, model_path=None, state_dir=None, num_blocks=None,
                    stats_scope=None, decimation=None):
    from memory_native.donor.streaming import (
        _WeightSource, _build_probe, _resolve_decoder, _skeleton_for_checkpoint,
        load_streamed_state,
    )
    from memory_native.recovery.runtime import restore_counter_structure
    from transformers import AutoConfig
    import re

    model_path = MODEL if model_path is None else model_path
    state_dir = STATE_DIR if state_dir is None else state_dir
    num_blocks = NUM_BLOCKS if num_blocks is None else num_blocks
    stats_scope = STATS_SCOPE if stats_scope is None else stats_scope
    decimation = DECIMATION if decimation is None else decimation
    if format is None:
        manifest_file = Path(state_dir) / "manifest.json"
        if not manifest_file.is_file():
            raise SystemExit(
                f"no manifest at {manifest_file}: pass an explicit format= for a "
                "single-file (.pt) state directory")
        with open(manifest_file, encoding="utf-8") as handle:
            format = json.load(handle)
    if str(state_dir).endswith(".pt"):
        state = torch.load(state_dir, map_location="cpu", weights_only=True)
    else:
        state = load_streamed_state(state_dir, require_complete=(num_blocks == 0))
    config = AutoConfig.from_pretrained(model_path)
    src = _WeightSource(model_path)
    model, inner, _ = _skeleton_for_checkpoint(src, config)
    if num_blocks:
        inner, _ = _resolve_decoder(model)
        inner.layers = nn.ModuleList(list(inner.layers)[:num_blocks])
        text_config = getattr(model.config, "text_config", model.config)
        text_config.num_hidden_layers = num_blocks
        # smoke truncation: drop state keys of layers beyond the cut
        keep = re.compile(r"\.layers\.(\d+)\.")
        state = {k: v for k, v in state.items()
                 if not keep.search(k) or int(keep.search(k).group(1)) < num_blocks}
    uncovered = [
        path for path, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
        and f"{path}.state" not in state and f"{path}.counter.state" not in state
        and path != "lm_head"
    ]
    report = restore_counter_structure(
        model, state, kind=format["kind"], group=format["group"], C=format["C"],
        kernel_mode="auto", strict_update=True, flip_sample_size=4096,
        lr=COUNTER_LR_START, lr_scale=SCALE_LR_START, local_grad_clip=1.0, residual_alpha=1.0,
        stats_scope=stats_scope, decimation=decimation,
        extra_skip=uncovered,
    )
    log(f"restored {len(report.swapped)} counter linears; {len(uncovered)} fp-only linears")
    # The RMS second moment is scope-shaped ([out,1] row vs [out,G] group) and a WARM
    # state carries zeros there anyway: drop mismatching .v keys so the fresh zeros of
    # the restored scope stand (cross-scope v is meaningless, not restorable).
    dropped_v = 0
    for key in [k for k in state if k.endswith(".v")]:
        try:
            buf = model.get_submodule(key.rsplit(".", 1)[0]).v
        except AttributeError:
            continue
        if buf.shape != state[key].shape:
            del state[key]
            dropped_v += 1
    if dropped_v:
        log(f"  dropped {dropped_v} scope-mismatched .v keys (fresh zeros stand)")
    # The model skeleton still has meta FP biases/parameters. Assignment installs
    # recorded warm tensors directly; copying would be a no-op and later reread
    # the donor bias instead. No optimizer exists yet, so replacing tensor objects
    # is safe, and the counter load hooks still rebuild derived runtime buffers.
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if unexpected:
        raise SystemExit(f"unexpected keys: {unexpected[:5]}")

    src = _WeightSource(model_path)
    probe = _build_probe(config, "cpu", cls=type(model))
    # Only buffers can be reconstructed from config. A missing donor parameter
    # must never silently become a randomly initialized probe parameter.
    probe_tensors = dict(probe.named_buffers())
    _, stack = _resolve_decoder(model)
    head = model.get_output_embeddings()
    head_path = next((name for name, module in model.named_modules() if module is head), None)
    tied_head = bool(getattr(getattr(config, "text_config", config), "tie_word_embeddings", False))
    mat_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[
        os.environ.get("DTYPE", "fp32")]
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if not tensor.is_meta:
            continue
        leaf = name.rsplit(".", 1)[-1]
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        if src.has(name):
            value = src.get(name, mat_dtype if tensor.is_floating_point() else None)
        elif tied_head and name == f"{head_path}.weight" and src.has(f"{stack}.embed_tokens.weight"):
            value = src.get(f"{stack}.embed_tokens.weight", mat_dtype)
        else:
            if isinstance(getattr(parent, leaf, None), nn.Parameter):
                raise ValueError(f"donor checkpoint is missing required parameter {name}")
            source = probe_tensors.get(name)
            if source is None:
                source = probe_tensors.get(
                    re.sub(r"\.layers\.\d+\.", ".layers.0.", name))
            if source is None or source.is_meta:
                raise SystemExit(f"no source for meta tensor {name}")
            value = source.detach().clone()
        if isinstance(getattr(parent, leaf, None), nn.Parameter):
            setattr(parent, leaf, nn.Parameter(value, requires_grad=True))
        else:
            setattr(parent, leaf, value)
    src.close()
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
            if t.is_meta]
    if left:
        raise SystemExit(f"still-meta after materialization: {left[:5]}")
    return model


@torch.no_grad()
def restore_selected_artifact(path, *, model_dir=None, state_dir=None,
                              checkpoint=None, device="cpu"):
    """Reload the selected warm/partial-KD artifact with verified base provenance.

    Paths may be overridden after moving artifacts to another machine; matching
    content fingerprints remain mandatory. The returned model is in eval mode
    with residual_alpha=0. Example: model = restore_selected_artifact(path);
    with torch.no_grad(): logits = model(token_ids).logits.
    """
    with open(path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    provenance = artifact.get("provenance", {})
    fmt = artifact.get("format", {})
    if not isinstance(provenance, dict) or provenance.get("cache_schema_version") != 2 or provenance.get("warm_schema_version") != 2:
        raise ValueError("selected artifact lacks verified cache/warm provenance")
    model_dir = model_dir or artifact["model"]
    state_dir = state_dir or artifact["conversion_state"]
    expected_model = provenance.get("warm_source_fingerprint")
    if model_identity(model_dir) != expected_model:
        raise ValueError("selected artifact donor content mismatch")
    if sha256_file(Path(state_dir) / "manifest.json") != provenance.get("warm_manifest_sha256"):
        raise ValueError("selected artifact warm manifest changed")
    num_blocks = fmt.get("num_blocks", 0)
    verified_warm = validate_warm_source(state_dir, {"identities": {"model": expected_model}, "num_blocks": num_blocks})
    if any(fmt.get(key) != verified_warm[key] for key in ("kind", "group", "C")):
        raise ValueError("selected checkpoint format differs from warm conversion")
    stats_scope, decimation = fmt.get("stats_scope"), fmt.get("decimation")
    if stats_scope not in ("row", "group") or isinstance(decimation, bool) or not isinstance(decimation, int) or decimation < 1:
        raise ValueError("invalid selected counter statistics format")
    model = restore_student(format=fmt, model_path=model_dir, state_dir=state_dir,
                            num_blocks=num_blocks, stats_scope=stats_scope, decimation=decimation)
    fp_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}.get(fmt.get("fp_dtype"))
    if fp_dtype is None:
        raise ValueError("unsupported selected floating-parameter dtype")
    for parameter in model.parameters():
        parameter.data = parameter.data.to(fp_dtype)
    accepted = artifact.get("accepted_kd")
    if not isinstance(accepted, bool):
        raise ValueError("selected artifact accepted_kd must be boolean")
    if accepted:
        if artifact.get("artifact_type") != "cached_kd_partial_checkpoint" or artifact.get("selected_step", 0) <= 0:
            raise ValueError("invalid selected KD artifact metadata")
        checkpoint = Path(checkpoint or artifact["checkpoint"])
        actual = {"bytes": checkpoint.stat().st_size, "sha256": sha256_file(checkpoint)}
        if actual != artifact.get("checkpoint_fingerprint"):
            raise ValueError("selected KD checkpoint content mismatch")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("step") != artifact["selected_step"] or payload.get("strict_metric") != artifact["selected_metric"] or payload.get("format") != fmt or payload.get("model_fingerprint") != expected_model or payload.get("warm_manifest_sha256") != provenance["warm_manifest_sha256"]:
            raise ValueError("selected KD payload does not match artifact metadata/base")
        state = payload.get("student")
        reference = model.state_dict()
        if not isinstance(state, dict) or not state:
            raise ValueError("selected KD checkpoint has no partial student state")
        for key, tensor in state.items():
            if key not in reference or not torch.is_tensor(tensor) or tensor.shape != reference[key].shape or tensor.dtype != reference[key].dtype:
                raise ValueError(f"invalid selected state tensor {key}")
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError(f"nonfinite selected state tensor {key}")
        _, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise ValueError("unexpected keys in selected KD state")
    elif artifact.get("artifact_type") != "warm_conversion_state" or artifact.get("selected_step") != 0 or artifact.get("selected_metric") != artifact.get("warm_metric"):
        raise ValueError("invalid selected warm fallback metadata")
    model.to(device)
    for module in model.modules():
        if hasattr(module, "set_residual_alpha"):
            module.set_residual_alpha(0.0)
    model.eval()
    return model


def main() -> None:
    from recovery_session import DomainMix
    from memory_native.eval import perplexity
    from memory_native.donor.tokenization import verify_corpus_tokenizer
    from memory_native.group_scale_packed import PackedGroupScaleCounterLinear
    from memory_native.recovery.runtime import evaluate_at_alpha, metric_from_ppl

    if not all((MODEL, STATE_DIR, DATA_DIR, CACHE)):
        raise ValueError("MODEL, STATE_DIR, DATA_DIR and CACHE are required")
    if min(EVAL_EVERY, LOG_EVERY, EVAL_MAX_TOKENS, LOGSUMEXP_CHUNK) <= 0 or not math.isfinite(MIN_IMPROVEMENT) or MIN_IMPROVEMENT < 0:
        raise ValueError("evaluation/log/chunk sizes must be positive and MIN_IMPROVEMENT finite/nonnegative")
    if not 0 <= HOMOTOPY_HOLD < HOMOTOPY_END <= 1:
        raise ValueError("homotopy schedule requires 0 <= HOLD < END <= 1")
    if not math.isfinite(HOMOTOPY_ALPHA_START) or not 0 <= HOMOTOPY_ALPHA_START <= 1:
        raise ValueError("HOMOTOPY_ALPHA_START must be finite in [0,1]")
    if STRICT_EXPOSURE_EVERY < 0:
        raise ValueError("STRICT_EXPOSURE_EVERY must be nonnegative")
    if not math.isfinite(KD_T) or KD_T <= 0 or not math.isfinite(CE_ALPHA) or CE_ALPHA < 0:
        raise ValueError("KD_T must be positive and CE_ALPHA nonnegative, both finite")
    if any(not math.isfinite(x) or x < 0 for x in (COUNTER_LR_START, COUNTER_LR_END, SCALE_LR_START, SCALE_LR_END, FP_LR, GRAD_CLIP)):
        raise ValueError("learning rates and gradient clip must be finite and nonnegative")
    os.makedirs(CKPT_DIR, exist_ok=True)
    if any((Path(CKPT_DIR) / name).exists() for name in ("metrics.json", "selection.json", "best.pt", "selected_artifact.json", "USE_WARM_STATE.txt", "KD_ACCEPTED.txt")):
        raise ValueError("CKPT_DIR already contains a run; choose a fresh directory")
    cache_meta = load_cache_manifest(CACHE)
    STEPS = int(os.environ.get("STEPS", str(cache_meta["steps"])))
    BATCH, SEQ, SEED = cache_meta["batch"], cache_meta["seq"], cache_meta["seed"]
    validate_cache_context(cache_meta, model_dir=MODEL, data_dir=DATA_DIR,
                           num_blocks=NUM_BLOCKS, steps=STEPS,
                           **{name.lower(): int(os.environ[name]) if name in os.environ else None
                              for name in ("BATCH", "SEQ", "SEED")})
    warm_meta = validate_warm_source(STATE_DIR, cache_meta,
                                     **{name: int(os.environ[name.upper()]) if name.upper() in os.environ else None
                                        for name in ("group", "C")})
    mix = DomainMix(DATA_DIR, seq=SEQ, batch=BATCH, seed=SEED)
    synthetic = os.environ.get("SYNTHETIC_CALIBRATION", "0") == "1"
    if cache_meta.get("synthetic_calibration", False) != synthetic:
        raise ValueError("SYNTHETIC_CALIBRATION differs from cached teacher mode")
    verify_corpus_tokenizer(mix.manifest, MODEL, synthetic=synthetic)
    cache = ShardedCache(CACHE, BATCH, meta=cache_meta)

    student = restore_student(format=warm_meta)
    student.train()
    if FREEZE_EMBED:
        seen, frozen = set(), 0
        for mod in (student.get_input_embeddings(), student.get_output_embeddings()):
            for p in (mod.parameters() if mod is not None else []):
                if id(p) in seen:
                    continue
                seen.add(id(p))
                p.requires_grad_(False)
                frozen += p.numel()
        log(f"froze embeddings/lm_head ({frozen / 1e6:.0f}M params)")
    device = torch.device(DEVICE)
    if device.type == "cuda":
        for p in student.parameters():
            p.data = p.data.to(torch.bfloat16)
    student = student.to(device)
    if device.type == "cuda" and SPLIT_GPUS and device.index in (None, 0) and torch.cuda.device_count() >= 2:
        split_across_gpus(student)
    student.config.use_cache = False    # never needed: KD forward + labels-CE eval only
    if GRAD_CKPT:
        # reentrant ONLY -- see module docstring for the counter-guard interaction
        student.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": True})
        # frozen embeddings would leave the checkpointed blocks with no grad-requiring
        # input -> no backward would ever reach the counter layers; the hook restores
        # the grad path without unfreezing the weight
        student.enable_input_require_grads()
    counters = [m for m in student.modules()
                if isinstance(m, PackedGroupScaleCounterLinear)]
    fp_params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(fp_params, lr=FP_LR, weight_decay=0.0) if fp_params else None
    log(f"student on {DEVICE}: {len(counters)} counter layers, "
        f"{sum(p.numel() for p in fp_params) / 1e6:.0f}M fp params; "
        f"scope={STATS_SCOPE} dec={DECIMATION} steps={STEPS} batch={BATCH}x{SEQ} "
        f"ckpt={int(GRAD_CKPT)} freeze_embed={int(FREEZE_EMBED)}")

    val = mix.val_batches(DEVICE, max_tokens=EVAL_MAX_TOKENS)
    if not val or any(not batches for batches in val.values()):
        raise ValueError("warm selection requires nonempty validation batches in every validation domain")

    def strict_eval():
        res = {f"ppl_{n}": perplexity(student, b) for n, b in val.items()}
        return res

    # Slim checkpoint: a full 12B student state_dict is ~13-15 GiB; best+latest+tmp
    # would blow Kaggle's ~20 GiB output cap at the FIRST eval. Keep best.pt only and
    # drop everything reconstructible: frozen fp params (donor has them; tied lm_head
    # is caught via remove_duplicate=False) and the immutable salient/perm/v buffers
    # (the conversion state has them; v is an optimizer stat, not model state).
    frozen_fp = {name for name, p in student.named_parameters(remove_duplicate=False)
                 if not p.requires_grad}
    drop_suffix = (".salient_idx", ".salient_val", ".perm", ".v")
    checkpoint_format = {"kind": warm_meta["kind"], "group": warm_meta["group"], "C": warm_meta["C"], "stats_scope": STATS_SCOPE,
                         "homotopy_alpha_start": HOMOTOPY_ALPHA_START, "strict_exposure_every": STRICT_EXPOSURE_EVERY,
                         "decimation": DECIMATION, "num_blocks": NUM_BLOCKS,
                         "fp_dtype": "bfloat16" if device.type == "cuda" else "float32"}
    provenance = {"cache_schema_version": cache_meta["schema_version"],
                  "synthetic_calibration": synthetic,
                  "cache_identities": cache_meta["identities"],
                  "warm_schema_version": warm_meta["schema_version"],
                  "warm_source_fingerprint": warm_meta["source_fingerprint"],
                  "warm_manifest_sha256": sha256_file(Path(STATE_DIR) / "manifest.json")}

    def slim_payload(step: int, metric: float) -> dict:
        keep = {k: v.cpu() for k, v in student.state_dict().items()
                if k not in frozen_fp and not k.endswith(drop_suffix)}
        return {"step": step, "student": keep, "strict_metric": metric,
                "format": checkpoint_format,
                "model_fingerprint": cache_meta["identities"]["model"],
                "warm_manifest_sha256": provenance["warm_manifest_sha256"],
                "partial_state": "frozen fp + salient/perm/v dropped; rebase on the "
                                 "conversion state + donor to reload"}

    history = []
    if not EVAL_AT_START:
        log("EVAL_AT_START=0 cannot disable the mandatory warm selection baseline")
    res = evaluate_at_alpha(student, 0.0, strict_eval)
    metric = metric_from_ppl(res)
    selection = WarmSelection(metric, MIN_IMPROVEMENT)
    log(f"strict alpha=0 WARM {res} metric={metric:.4f}")
    history.append({"step": 0, "metric": metric, "selected": True,
                    **{k: float(v) for k, v in res.items()}})
    _write_json(Path(CKPT_DIR) / "metrics.json", history)
    write_selection(CKPT_DIR, selection, model=MODEL, state_dir=STATE_DIR, format=checkpoint_format,
                    provenance=provenance)
    t0 = time.time()
    for step in range(STEPS):
        progress = step / max(STEPS - 1, 1)
        clr = cosine(COUNTER_LR_START, COUNTER_LR_END, progress)
        slr = cosine(SCALE_LR_START, SCALE_LR_END, progress)
        base_alpha = homotopy_alpha(progress)
        alpha = strict_exposure_alpha(base_alpha, step, STRICT_EXPOSURE_EVERY)
        for m in counters:
            m.set_lr(clr)
            m.lr_scale = slr
            m.set_residual_alpha(alpha)
        for g in opt.param_groups if opt is not None else []:
            g["lr"] = cosine(FP_LR, FP_LR * 0.1, progress)

        ids_cpu = mix.batch_at(step, "cpu")
        idx, valp = cache.step(step, DEVICE, input_ids=ids_cpu)
        ids = ids_cpu.to(DEVICE)
        out = student(ids).logits
        loss_kd, loss_ce = kd_and_ce_losses(out, idx, valp, ids[:, 1:], KD_T)
        loss = loss_kd + CE_ALPHA * loss_ce
        loss.backward()
        if opt is not None:
            torch.nn.utils.clip_grad_norm_(fp_params, GRAD_CLIP)
            opt.step()
            opt.zero_grad(set_to_none=True)

        if (step + 1) % LOG_EVERY == 0:
            log(f"step {step + 1}/{STEPS} kd={loss_kd.item():.4f} ce={loss_ce.item():.4f} "
                f"clr={clr:.5f} alpha={alpha:.2f} base_alpha={base_alpha:.2f} "
                f"strict_exposure_every={STRICT_EXPOSURE_EVERY} {(time.time() - t0) / (step + 1):.2f}s/step")
        if (step + 1) % EVAL_EVERY == 0 or step + 1 == STEPS:
            res = evaluate_at_alpha(student, 0.0, strict_eval)
            metric = metric_from_ppl(res)
            if not math.isfinite(metric):
                raise ValueError("evaluation metric is nonfinite; selected warm/best artifact is retained")
            log(f"strict alpha=0 {res} metric={metric:.4f}")
            improves = selection.improves(metric)
            history.append({"step": step + 1, "metric": metric, "selected": improves,
                            **{k: float(v) for k, v in res.items()}})
            _write_json(Path(CKPT_DIR) / "metrics.json", history)
            if improves:
                # Selection bookkeeping is in-memory and always updated, so the
                # final report names the right step even when SAVE_BEST gates the
                # disk write (e.g. a host whose output cap cannot hold the ckpt).
                selection.accept(step + 1, metric)
                if SAVE_BEST:
                    save_best_checkpoint(slim_payload(step + 1, metric), CKPT_DIR, CKPT_TMP)
                    write_selection(CKPT_DIR, selection, model=MODEL, state_dir=STATE_DIR, format=checkpoint_format,
                                    provenance=provenance)
                    log(f"new best metric={metric:.4f} (slim ckpt)")
                else:
                    log(f"new best metric={metric:.4f} (SAVE_BEST=0, no ckpt write)")
    source = f"KD step {selection.best_step}" if selection.best_step else "original warm conversion"
    log(f"done: selected {source}, strict metric={selection.best_metric:.4f}; "
        f"warm={selection.warm_metric:.4f}, accepted_kd={selection.best_step > 0}")


if __name__ == "__main__":
    main()
