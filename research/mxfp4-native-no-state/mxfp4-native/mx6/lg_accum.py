"""Opt-in gradient accumulation for `LatentGridLinear` (the update lives INSIDE backward, so plain autograd accumulation
cannot work: every backward would already move the weights). Nothing here is active unless `set_accum(model, n, mode)` is called.

Two modes (N = micro-batches per optimizer step)
  "buffer"  per-layer gradient buffer (bf16 [out,in] +2 B/param, or float8_e4m3fn + fp32 per-row scale ~ +1 B/param, or fp32
            +4 B/param for tests). Micro-batches 1..N-1 only ADD grad_w into the buffer (no update, RMS EMA untouched); micro-batch N
            forms the total gradient, calls the (all-reduce hook and the) normal ONE-launch update with it and zeroes the buffer.
            => ONE update and ONE RMS-EMA step per optimizer step, identical to a single batch of N x the size (up to the summation
            order / the buffer rounding). All N micro-batches see the SAME weights (as in a big batch).
  "latent"  no buffer. Every micro-batch applies its own update with lr / N and a per-micro RMS-EMA factor beta' = beta^(1/N):
            the latent grid integrates the micro-updates. N updates per optimizer step. NOT equivalent to a big batch (see below).

Loss convention (`loss_scaled=True`, default): the training loop divides every micro-batch loss by N (standard practice, so the
grads of ordinary FP params accumulated by autograd are the big-batch mean). Then buffer mode uses SUM(buffer) as the gradient (that
already is the mean) and latent mode multiplies each micro gradient back by N (restores the scale of a normal batch gradient;
matters only through rms_eps). With `loss_scaled=False` (loss NOT divided) buffer mode divides the sum by N and latent mode uses
the micro gradient as is.

Latent mode, the choices
  * lr/N: the N micro-steps add up to about one optimizer step of size lr when the micro-gradients agree.
  * beta' = beta^(1/N): N EMA steps decay old information by beta in total, so the EMA horizon IN OPTIMIZER STEPS is unchanged
    (1/(1-beta) steps). The EMA sees micro-batch gradients, whose second moment is larger than the big-batch one by up to N x in
    the noise-dominated regime (E[g_micro^2] = mu^2 + sigma^2/B vs mu^2 + sigma^2/(N B)): v is biased UP, g/sqrt(v) is smaller, the
    effective lr lies between lr/sqrt(N) (pure noise) and lr (pure signal) relative to buffer mode. Tune lr (or scale it by sqrt(N))
    if you compare to buffer mode. Micro-batches 2..N also see weights already moved by micro-batch 1 (SGD-like, not a big batch).
  * SR noise: each micro-update is stochastically rounded to the integer index grid. For a step delta (in index units) the SR
    variance is d^2 p(1-p), p = frac(delta). N draws of delta/N: sum of variances ~ N * d * (delta/N) * (1 - delta/(N d)) =
    d*delta*(1 - delta/(N d)), versus one draw d^2 frac(delta)(1-frac(delta)) <= d^2/4. For |delta| << d (the usual regime) they are
    nearly equal (about d*delta; the single draw loses a factor (1-p), the N draws (1-p/N)); for |delta| >> d, N small draws are
    noisier (grows like d*delta, a single draw is bounded by d^2/4). Every draw is unbiased, so the expectation is unaffected.
    The MC test measures the ratio (tests/test_lg_accum.py).

fp8 buffer (float8_e4m3fn + per-row fp32 scale): total t = deq(buf) + grad is formed in fp32 per row-chunk, the new row scale is
amax(t)/448 and t/scale is STOCHASTICALLY ROUNDED to the e4m3 grid (mantissa 3 bits; subnormal spacing 2^-9) -> E[deq] = t exactly
(unbiased; the only exception is the clamp at +-448 which never triggers because the scale is set by amax). Each requantisation
adds zero-mean noise of ~2.5-5 % of |value| rms (relative spacing 2^-4..2^-3, rms of SR = 0.41 * spacing); N-1 requantisations for
N micro-batches (the last one is NOT stored: the total is formed in fp32 and goes to the update). The noise is far below the
element-wise sampling noise of a micro-batch gradient. Cost: transient fp32 chunk (<= `FP8_CHUNK` elements, 32 MB), several small
torch ops per chunk (not fused; a Triton kernel is the obvious follow-up).

DDP hook: `grad_allreduce(grad) -> grad|None` is called inside the backward op right before the update: buffer mode on the LAST
micro-batch only (the accumulated gradient: ONE all-reduce per layer per optimizer step), latent mode on EVERY micro-batch (every
update needs a synchronised gradient). `dist_allreduce_mean` is a ready-made version (sum all-reduce / world_size, guarded by
`dist.is_initialized()`). Keep the lg state identical on all ranks: same global CPU seed (the Philox seed is drawn from it) and
DDP(broadcast_buffers=False) (otherwise DDP re-broadcasts idx every forward).

State is read at RUN time inside the opaque backward op (like the Philox seed), never traced by torch.compile: the micro-batch
index is a plain Python int on `AccumState`, the compiled graph only sees the int `acc_id` (constant) => no recompiles, no host
sync. Not CUDA-graph-replay safe (mode="reduce-overhead"), same as the seed.
"""
from __future__ import annotations

import contextlib
import os

import torch

KINDS = ("bf16", "fp8", "fp32", "int4")
Q4_GROUP = 32                            # int4 momentum: one bf16 scale per 32 weights of a row
MODES = ("buffer", "latent", "ema")
FP8_MAX = 448.0
FP8_CHUNK = 1 << 23                      # elements per fp32 chunk in the fp8 buffer arithmetic

_STATES: dict[int, "AccumState"] = {}
_NEXT_ID = [0]


def dist_allreduce_mean(grad: torch.Tensor) -> torch.Tensor:
    """Mean all-reduce of `grad` over the default process group, in place; a no-op if torch.distributed is not initialised."""
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        ws = dist.get_world_size()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        if ws > 1:
            grad.div_(ws)
    return grad


class AccumState:
    """Shared by all LatentGridLinear layers of one model (see set_accum). `idx` = index of the micro-batch whose backward is
    running (set by `lg_accumulate`); the layers read it at run time."""

    def __init__(self, n: int, mode: str, kind: str, loss_scaled: bool, grad_allreduce, sparse: bool = False):
        if sparse and (mode != "ema" or grad_allreduce is not None):
            raise ValueError("sparse accumulation is supported for momentum (ema) mode without a grad_allreduce hook only "
                             "(ranks may touch different layers: a per-layer all-reduce would deadlock)")
        if mode == "ema":
            n = max(1, int(n))
        elif int(n) < 2:
            raise ValueError(f"accumulation needs n >= 2, got {n}")
        if mode not in MODES:
            raise ValueError(f"accum mode must be one of {MODES}, got {mode!r}")
        if kind not in KINDS:
            raise ValueError(f"accum dtype must be one of {KINDS}, got {kind!r}")
        if kind == "int4" and (mode != "ema" or n > 1 or sparse):
            raise ValueError("int4 buffer: momentum only (set_momentum, n=1, not sparse)")
        if grad_allreduce == "dist":
            grad_allreduce = dist_allreduce_mean
        if grad_allreduce is not None and not callable(grad_allreduce):
            raise ValueError("grad_allreduce must be None, 'dist' or a callable")
        self.n, self.mode, self.kind = int(n), mode, kind
        self.mu = 0.0                    # ema mode: momentum factor (set_momentum)
        #: sparse (MoE experts): a layer may run 0..n backwards per step. Decay on its first backward, ONE update at the exit
        #: of lg_accumulate for every touched layer; untouched layers are left alone (like sparse Adam).
        self.sparse = bool(sparse)
        self.touched: dict = {}          # sparse: layer key -> (finish(grad), buf, bscale), insertion order = first backward
        self.micro_seen: dict = {}       # sparse: layer key -> micro-batch index of its last backward (<= 1 per micro-batch)
        self.nesterov = False
        self.loss_scaled, self.grad_allreduce = bool(loss_scaled), grad_allreduce
        self.idx = 0                     # micro-batch index of the running backward
        self.active = False              # inside lg_accumulate
        self.n_layers = 0
        self.n_bwd = 0                   # backward calls of lg layers since the context was entered
        self.n_updates = 0               # updates applied (all layers) since set_accum
        self.n_hook = 0                  # grad_allreduce calls since set_accum
        self.id = _NEXT_ID[0]
        _NEXT_ID[0] += 1
        _STATES[self.id] = self

    @property
    def lr_mult(self) -> float:
        return 1.0 / self.n if self.mode == "latent" else 1.0

    def beta_eff(self, beta: float) -> float:
        return float(beta) ** (1.0 / self.n) if self.mode == "latent" else float(beta)

    @property
    def last(self) -> bool:
        return self.idx == self.n - 1


def get_state(acc_id: int) -> AccumState:
    return _STATES[int(acc_id)]


# ── fp8 stochastic rounding / buffer arithmetic ───────────────────────────────────────────────────────────────

def sr_fp8(x: torch.Tensor) -> torch.Tensor:
    """Stochastic rounding of float32 `x` (|x| <= 448) to the float8_e4m3fn grid: E[result] = x. Spacing of the grid at x:
    2^(floor(log2|x|) - 3), but at least 2^-9 (subnormals). Returns float8_e4m3fn."""
    x = x.to(torch.float32)
    _, e = torch.frexp(x)                                           # |x| = m 2^e, m in [0.5, 1)
    d = torch.exp2((e - 4).clamp_(min=-9).to(torch.float32))        # floor(log2|x|) = e - 1
    a = x.abs() / d
    q = torch.floor(a + torch.rand_like(a)) * d
    return torch.copysign(q, x).clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


def _rows_per_chunk(cols: int) -> int:
    return max(1, FP8_CHUNK // max(cols, 1))


def fp8_add_(buf: torch.Tensor, bscale: torch.Tensor, grad: torch.Tensor) -> None:
    """buf (fp8) <- SR_fp8(deq(buf) + grad), per-row scale amax/448 (bscale [rows,1] fp32)."""
    rows, cols = buf.shape
    step = _rows_per_chunk(cols)
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        t = buf[r0:r1].to(torch.float32).mul_(bscale[r0:r1]).add_(grad[r0:r1])
        amax = t.abs().amax(dim=1, keepdim=True)
        ns = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
        buf[r0:r1] = sr_fp8(t / ns)
        bscale[r0:r1] = ns


def acc_add_(buf: torch.Tensor, bscale: torch.Tensor, grad: torch.Tensor, kind: str) -> None:
    """Micro-batches 1..N-1 (buffer mode): buffer += grad. bf16/fp32: one in-place add, no fp32 copy."""
    if kind == "fp8":
        fp8_add_(buf, bscale, grad)
    else:
        buf.add_(grad)


def acc_finish_(buf: torch.Tensor, bscale: torch.Tensor, grad: torch.Tensor, kind: str, mult: float) -> torch.Tensor:
    """Last micro-batch: grad <- (buffer + grad) * mult IN PLACE (grad is the layer's fresh grad_w), buffer zeroed. -> grad."""
    if kind == "fp8":
        rows, cols = buf.shape
        step = _rows_per_chunk(cols)
        for r0 in range(0, rows, step):
            r1 = min(rows, r0 + step)
            t = buf[r0:r1].to(torch.float32).mul_(bscale[r0:r1]).add_(grad[r0:r1])
            grad[r0:r1] = (t * mult).to(grad.dtype) if mult != 1.0 else t.to(grad.dtype)
        buf.zero_()
        bscale.zero_()
    else:
        grad.add_(buf)
        if mult != 1.0:
            grad.mul_(mult)
        buf.zero_()
    return grad



# ── int4 momentum buffer (packed nibbles, group-32 bf16 scales, stochastic rounding) ───────────────────────────

def q4_group(cols: int) -> int:
    return Q4_GROUP if cols % Q4_GROUP == 0 else cols


def q4_shapes(rows: int, cols: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """(packed uint8 buffer shape, bf16 scale shape) of an int4 momentum buffer for a [rows, cols] weight (cols even)."""
    if cols % 2:
        raise ValueError(f"int4 momentum needs an even in_features, got {cols}")
    return (rows, cols // 2), (rows, cols // q4_group(cols))


def q4_dequant(buf: torch.Tensor, scale: torch.Tensor, r0: int, r1: int, cols: int) -> torch.Tensor:
    """Rows r0:r1 of the int4 buffer as fp32 [r1-r0, cols]. Nibble = q + 8, q in [-7, 7]; low nibble = even column."""
    b = buf[r0:r1]
    q = torch.stack(((b & 15).to(torch.int8) - 8, (b >> 4).to(torch.int8) - 8), dim=-1).reshape(r1 - r0, cols)
    g = q4_group(cols)
    return (q.to(torch.float32).view(r1 - r0, cols // g, g) * scale[r0:r1].to(torch.float32).unsqueeze(-1)).view(r1 - r0, cols)


def q4_store_(buf: torch.Tensor, scale: torch.Tensor, r0: int, r1: int, t: torch.Tensor) -> None:
    """buf/scale rows r0:r1 <- SR_int4(t) (t fp32 [r1-r0, cols]): group scale = amax/7 rounded UP to bf16, q = floor(t/s + U)
    clamped to [-7, 7] (never active: |t/s| <= 7) -> E[deq] = t (unbiased)."""
    rows, cols = t.shape
    g = q4_group(cols)
    tg = t.view(rows, cols // g, g)
    amax = tg.abs().amax(dim=-1)
    s = torch.where(amax > 0, amax / 7.0 * (1.0 + 2.0 ** -7), torch.ones_like(amax)).to(torch.bfloat16)
    sf = s.to(torch.float32).unsqueeze(-1)
    q = torch.floor(tg / sf + torch.rand_like(tg)).clamp_(-7, 7).to(torch.int16).view(rows, cols) + 8
    q = q.view(rows, cols // 2, 2)
    buf[r0:r1] = (q[..., 0] | (q[..., 1] << 4)).to(torch.uint8)
    scale[r0:r1] = s

def ema_step_(st: AccumState, grad_w: torch.Tensor, buf: torch.Tensor, bscale) -> torch.Tensor:
    """Momentum ("ema" mode): buf <- mu*buf + (1-mu)*g; returns the direction handed to the RMS-normalised update:
    buf (heavy-ball EMA) or mu*buf + (1-mu)*g (nesterov). bf16/fp32 buffer: in-place lerp; fp8: fp32 chunk + SR requant.
    The update's row-RMS normaliser then sees the EMA, so the step stays ~unit-scale where the gradient sign is consistent
    and shrinks where it flips (noise averages out): Adam's first-moment effect at 1 or 2 B/param."""
    mu = float(st.mu)
    if st.kind == "int4":
        rows, cols = grad_w.shape
        step = _rows_per_chunk(cols)
        for r0 in range(0, rows, step):
            r1 = min(rows, r0 + step)
            g = grad_w[r0:r1].to(torch.float32)
            t = q4_dequant(buf, bscale, r0, r1, cols).mul_(mu).add_(g, alpha=1.0 - mu)
            q4_store_(buf, bscale, r0, r1, t)
            out = t.mul_(mu).add_(g, alpha=1.0 - mu) if st.nesterov else t
            grad_w[r0:r1] = out.to(grad_w.dtype)
        return grad_w
    if st.kind == "fp8":
        rows, cols = buf.shape
        step = _rows_per_chunk(cols)
        for r0 in range(0, rows, step):
            r1 = min(rows, r0 + step)
            g = grad_w[r0:r1].to(torch.float32)
            t = buf[r0:r1].to(torch.float32).mul_(bscale[r0:r1]).mul_(mu).add_(g, alpha=1.0 - mu)
            amax = t.abs().amax(dim=1, keepdim=True)
            ns = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
            buf[r0:r1] = sr_fp8(t / ns)
            bscale[r0:r1] = ns
            out = t.mul_(mu).add_(g, alpha=1.0 - mu) if st.nesterov else t
            grad_w[r0:r1] = out.to(grad_w.dtype)
        return grad_w
    buf.lerp_(grad_w.to(buf.dtype), 1.0 - mu)
    if st.nesterov:
        return grad_w.mul_(1.0 - mu).add_(buf.to(grad_w.dtype), alpha=mu)
    return grad_w.copy_(buf)


def _fp8_store_(buf: torch.Tensor, bscale: torch.Tensor, src: torch.Tensor) -> None:
    """buf (fp8) <- SR_fp8(src) with per-row scale amax/448."""
    rows, cols = buf.shape
    step = _rows_per_chunk(cols)
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        t = src[r0:r1].to(torch.float32)
        amax = t.abs().amax(dim=1, keepdim=True)
        ns = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
        buf[r0:r1] = sr_fp8(t / ns)
        bscale[r0:r1] = ns


def ema_accum_(st: AccumState, grad_w: torch.Tensor, buf: torch.Tensor, bscale):
    """Momentum WITH gradient accumulation over st.n micro-batches, in the SAME buffer (no extra memory):
    micro 0 decays buf by mu (fp8: exact, only the row scale is multiplied), every micro adds c*g_i
    (c = 1-mu; loss_scaled => sum of g_i is already the big-batch mean), the last micro forms
    mu*buf_old + (1-mu)*G, runs the all-reduce hook on it (identical buffers on all ranks => the mean of the
    per-rank totals is exactly the global EMA), stores it back and returns it for the update. -> grad | None."""
    mu = float(st.mu)
    c = (1.0 - mu) if st.loss_scaled else (1.0 - mu) / st.n
    if st.idx == 0:
        if st.kind == "fp8":
            bscale.mul_(mu)
        else:
            buf.mul_(mu)
    if not st.last:
        if st.kind == "fp8":
            fp8_add_(buf, bscale, grad_w.mul_(c))
        else:
            buf.add_(grad_w.to(buf.dtype), alpha=c)
        return None
    if st.kind == "fp8":
        rows, cols = buf.shape
        step = _rows_per_chunk(cols)
        for r0 in range(0, rows, step):
            r1 = min(rows, r0 + step)
            t = buf[r0:r1].to(torch.float32).mul_(bscale[r0:r1]).add_(grad_w[r0:r1].to(torch.float32), alpha=c)
            grad_w[r0:r1] = t.to(grad_w.dtype)
    else:
        grad_w.mul_(c).add_(buf.to(grad_w.dtype))
    if st.grad_allreduce is not None:
        st.n_hook += 1
        out = st.grad_allreduce(grad_w)
        if out is not None:
            grad_w = out
    if st.kind == "fp8":
        _fp8_store_(buf, bscale, grad_w)
    else:
        buf.copy_(grad_w.to(buf.dtype))
    st.n_updates += 1
    return grad_w


def fused_int4_ok(st, cols: int, triton_on: bool) -> bool:
    """True when the int4 momentum EMA can run INSIDE the Triton update kernel (lg_update_(momentum=...)): plain momentum
    (n=1, not sparse, no nesterov, no all-reduce hook), Triton path, cols % 32 == 0. MX6_FUSE_M4=0 forces the unfused
    torch path (A/B)."""
    return (st is not None and st.mode == "ema" and st.kind == "int4" and st.n == 1 and not st.sparse
            and not st.nesterov and st.grad_allreduce is None and bool(triton_on) and cols % 32 == 0
            and os.environ.get("MX6_FUSE_M4", "1") != "0")


def _dense_of(buf: torch.Tensor, bscale, kind: str) -> torch.Tensor:
    """The momentum buffer as a dense fp32 tensor (fp8: dequantised with the per-row scale)."""
    if kind == "fp8":
        return buf.to(torch.float32).mul_(bscale)
    return buf.to(torch.float32)


def ema_accum_sparse_(st: AccumState, grad_w: torch.Tensor, buf: torch.Tensor, bscale, finish, key) -> None:
    """Sparse momentum accumulation of ONE layer backward. First backward of the layer in this step: buf *= mu (fp8: row scale);
    every backward: buf += (1-mu) * g. NO update here: `flush_sparse_` applies one update per touched layer at the context exit
    (the layer may not appear in the last micro-batch, or at all). At most one backward per layer per micro-batch."""
    if finish is None or key is None:
        raise RuntimeError("sparse accumulation needs the layer's update callback and key")
    if st.micro_seen.get(key) == st.idx:
        raise RuntimeError("sparse lg_accumulate: a layer ran two backwards in the same micro-batch (an expert must be called "
                           "ONCE per forward with all its tokens batched)")
    st.micro_seen[key] = st.idx
    mu = float(st.mu)
    c = (1.0 - mu) if st.loss_scaled else (1.0 - mu) / st.n
    if key not in st.touched:
        if st.kind == "fp8":
            bscale.mul_(mu)
        else:
            buf.mul_(mu)
        st.touched[key] = (finish, buf, bscale)
    if st.kind == "fp8":
        fp8_add_(buf, bscale, grad_w.mul_(c))
    else:
        buf.add_(grad_w.to(buf.dtype), alpha=c)


def flush_sparse_(st: AccumState) -> int:
    """Apply ONE update (direction = the momentum buffer) to every layer touched in this step; -> number of updates."""
    n = 0
    touched, st.touched, st.micro_seen = st.touched, {}, {}
    for finish, buf, bscale in touched.values():
        finish(_dense_of(buf, bscale, st.kind))
        st.n_updates += 1
        n += 1
    return n


def accum_grad_(st: AccumState, grad_w: torch.Tensor, buf, bscale, finish=None, key=None):
    """The accumulation step of one layer's backward. -> the gradient to hand to the update (after the all-reduce hook), or None
    (buffer mode, not the last micro-batch: nothing to update). `grad_w` is consumed (modified in place)."""
    if st.mode == "ema" and (st.n > 1 or st.sparse):
        if not st.active:
            raise RuntimeError("momentum with accumulation (set_momentum(..., n>1)) needs the lg_accumulate context")
        st.n_bwd += 1
        if st.sparse:
            ema_accum_sparse_(st, grad_w, buf, bscale, finish, key)
            return None
        return ema_accum_(st, grad_w, buf, bscale)
    if st.mode == "ema":
        if st.grad_allreduce is not None:            # sync the raw gradient first: the EMA stays identical on all ranks
            st.n_hook += 1
            out = st.grad_allreduce(grad_w)
            if out is not None:
                grad_w = out
        st.n_updates += 1
        return ema_step_(st, grad_w, buf, bscale)
    if not st.active:
        raise RuntimeError("LatentGridLinear accumulation is configured (set_accum) but backward ran outside "
                           "`with lg_accumulate(model, n) as acc: for i in acc.micro_batches(): ...`")
    st.n_bwd += 1
    if st.mode == "latent":
        g = grad_w.mul_(float(st.n)) if st.loss_scaled else grad_w
    else:
        if not st.last:
            acc_add_(buf, bscale, grad_w, st.kind)
            return None
        g = acc_finish_(buf, bscale, grad_w, st.kind, 1.0 if st.loss_scaled else 1.0 / st.n)
    if st.grad_allreduce is not None:
        st.n_hook += 1
        out = st.grad_allreduce(g)
        if out is not None:
            g = out
    st.n_updates += 1
    return g


# ── user API ───────────────────────────────────────────────────────────────────────────────────────────────

def _lg_layers(model):
    from .latent_grid import lg_layers
    return lg_layers(model)


def get_accum(model) -> AccumState | None:
    for m in _lg_layers(model):
        if m._acc is not None:
            return m._acc
    return None


def clear_accum(model) -> None:
    for m in _lg_layers(model):
        m.disable_accum_()


def set_accum(model, n: int, mode: str = "buffer", dtype: str = "bf16", loss_scaled: bool = True,
              grad_allreduce=None) -> AccumState | None:
    """Enable gradient accumulation over `n` micro-batches on every LatentGridLinear of `model` (n <= 1 disables it and frees
    the buffers). Call BEFORE torch.compile. Allocates the per-layer buffers (buffer mode). Returns the shared AccumState."""
    layers = _lg_layers(model)
    clear_accum(model)
    if n is None or int(n) <= 1:
        return None
    if not layers:
        raise ValueError("set_accum: the model has no LatentGridLinear layer")
    st = AccumState(n, mode, dtype, loss_scaled, grad_allreduce)
    for m in layers:
        m.enable_accum_(st)
    st.n_layers = len(layers)
    return st


def set_momentum(model, mu: float, dtype: str = "bf16", nesterov: bool = False, grad_allreduce=None,
                 n: int = 1, sparse: bool = False) -> AccumState | None:
    """Per-weight momentum for every LatentGridLinear (reuses the accumulation buffer: bf16 +2 B/param, fp8 ~ +1 B/param).
    Every backward: buf <- mu*buf + (1-mu)*grad and the update uses buf. No lg_accumulate context needed. mu <= 0 disables.
    n > 1: ALSO accumulates n micro-batches into the same buffer (run them inside lg_accumulate(model, n)); ONE update
    per optimizer step on mu*buf + (1-mu)*G, G = the big-batch gradient. Replaces set_accum (same buffer, no extra
    memory). Call BEFORE torch.compile.
    sparse=True (MoE experts; always needs the lg_accumulate context, any n >= 1): a layer may run 0..n backwards per step
    (at most one per micro-batch). Its buffer decays by mu on its first backward, accumulates (1-mu)*g on each, and ONE update
    with the buffer is applied at the EXIT of lg_accumulate; layers with 0 backwards are untouched (no decay, no update).
    Not combinable with nesterov / grad_allreduce."""
    if mu is None or float(mu) <= 0.0:
        clear_accum(model)
        return None
    if not 0.0 < float(mu) < 1.0:
        raise ValueError(f"momentum must be in (0, 1), got {mu}")
    layers = _lg_layers(model)
    if not layers:
        raise ValueError("set_momentum: the model has no LatentGridLinear layer")
    if (int(n) > 1 or sparse) and nesterov:
        raise ValueError("nesterov momentum with accumulation is not supported")
    clear_accum(model)
    st = AccumState(int(n), "ema", dtype, True, grad_allreduce, sparse=sparse)
    st.mu, st.nesterov = float(mu), bool(nesterov)
    for m in layers:
        m.enable_accum_(st)
    st.n_layers = len(layers)
    return st


class _Accumulate:
    def __init__(self, st: AccumState):
        self.st = st
        self.n = st.n

    def micro_batches(self):
        """Iterate 0..n-1; the layers know which micro-batch's backward is running (the last one triggers the buffer-mode
        update). Run forward AND backward of micro-batch i inside the loop body."""
        for i in range(self.n):
            self.st.idx = i
            yield i
        self.st.idx = self.n

    @contextlib.contextmanager
    def micro(self, i: int):
        """Explicit alternative: `with acc.micro(i): fwd; bwd`."""
        if not 0 <= i < self.n:
            raise ValueError(f"micro-batch {i} out of range 0..{self.n - 1}")
        self.st.idx = i
        yield i
        self.st.idx = i + 1


@contextlib.contextmanager
def lg_accumulate(model, n: int | None = None):
    """One optimizer step worth of micro-batches. On exit it checks that every lg layer ran exactly n backwards (otherwise a
    buffer would be left half-full: RuntimeError) and, after an exception, zeroes the buffers.

        set_accum(model, 4, "buffer")
        for step in ...:
            with lg_accumulate(model, 4) as acc:
                for i in acc.micro_batches():
                    loss = f(model, batch_i) / 4        # loss_scaled convention
                    loss.backward()
            optimizer.step()                            # FP params: normal autograd accumulation
    """
    st = get_accum(model)
    if st is None:
        raise RuntimeError("lg_accumulate: call set_accum(model, n, mode) first")
    if n is not None and int(n) != st.n:
        raise ValueError(f"lg_accumulate(n={n}) != set_accum n={st.n}")
    if st.active:
        raise RuntimeError("lg_accumulate is not re-entrant")
    st.active, st.idx, st.n_bwd = True, 0, 0
    st.touched, st.micro_seen = {}, {}
    ok = False
    try:
        yield _Accumulate(st)
        ok = True
    finally:
        st.active = False
        want = st.n * st.n_layers
        got = st.n_bwd
        st.idx = 0
        st.n_bwd = 0
        if not ok:
            if st.sparse:                       # only the touched layers hold a half-built buffer
                for _, buf, bscale in st.touched.values():
                    buf.zero_()
                    if bscale is not None:
                        bscale.zero_()
                st.touched, st.micro_seen = {}, {}
            else:
                for m in _lg_layers(model):
                    m.zero_accum_()
    if st.sparse:
        with torch.no_grad():
            flush_sparse_(st)
        return
    if got != want:
        for m in _lg_layers(model):
            m.zero_accum_()
        raise RuntimeError(f"lg_accumulate: {got} lg backward calls, expected {want} = {st.n} micro-batches x "
                           f"{st.n_layers} layers (a layer unused in a micro-batch, used twice, or fewer micro-batches run); "
                           f"accumulation buffers were reset")


def accum_bytes_per_param(kind: str, in_features: int, mode: str = "buffer") -> float:
    """Resident bytes per weight added by the accumulation buffer."""
    if mode == "latent":
        return 0.0
    # "ema" (momentum) uses the same buffer as "buffer"
    return {"bf16": 2.0, "fp32": 4.0, "fp8": 1.0 + 4.0 / in_features,
            "int4": 0.5 + 2.0 / q4_group(in_features)}[kind]


def accum_resident_bytes(model) -> int:
    n = 0
    for m in _lg_layers(model):
        for t in (m._acc_buf, m._acc_scale):
            if t is not None:
                n += t.numel() * t.element_size()
    return n
