"""Non-redundant LATENT GRID: one k-bit index per weight into a fine, FP4-aligned grid, no hysteresis counter.

Why. PackedNA stores (rung, signed counter in +-(C-1)). Every rung has its own counter, so the counter
ranges of neighbouring rungs overlap ("rung 1, +7" ~ "rung 1.5, -7"): roughly half of the code space is
redundant hysteresis, and qat_fp4 (fp32 latent, deploy = RTN to E2M1 re-rounded every step, no hysteresis)
shows hysteresis is not needed. Here the whole k-bit word is ONE latent position.

Format (units of the block scale s, NVFP4 scaling: E4M3 byte per 16 + one fp32 per tensor):
  V = signed E2M1 values (-6,-4,-3,-2,-1.5,-1,-.5,0,.5,1,1.5,2,3,4,6): 15 points, 14 gaps.
  G = V with every gap cut into m equal sub-steps; n = 14*m + 1 points, odd-symmetric, i = 7*m is 0.
  m = (2^k - 1) // 14:  k=8 -> m=18 (253 pts), k=7 -> 9 (127), k=6 -> 4 (57), k=5 -> 2 (29), k=4 -> 1 (15:
  the grid IS E2M1, i.e. plain stateless stochastic rounding). Resolution is relative (fine near 0).
  stored word  : idx in [0, n-1]            latent value  G[idx] * s
  deployed w   : RTN_E2M1(G[idx]) * s       ties AWAY from zero (PackedNA.round_idx), a pure function of idx

Update (inside the autograd backward, like MXNLinear; no optimizer, no momentum, no counter):
  v_row, g_hat = rms_update(grad_w, v_row, beta, eps)       per-row RMS normalisation (mx6.update, verbatim)
  Delta        = -lr * g_hat   [+ weight_decay]             absolute step in weight units
  delta        = Delta / s                                  in E2M1 (block-scale) units, s = e4m3(byte) * tscale
  u1           = idx + (U(G[idx] + delta) - U(G[idx]))      U = piecewise-linear value -> index map
  idx'         = SR(clamp(u1, 0, n-1))                      stochastic rounding to an integer index
U is linear on every sub-step interval and every E2M1 value is a grid point, so a sub-step never straddles a
gap boundary: E[G[idx']] = G[idx] + delta EXACTLY for any |delta| (multi-gap moves included), up to the clamp
at the grid ends (the excess is dropped: no wind-up) and float32 resolution. update_mode "dir" (delta divided
by the sub-step of the gap moved INTO, no boundary handling: right inside one gap, biased across several) and
"min" (the legacy min-gap tick: biased outward drift) are kept as null twins for tests.

Bits/param = k + 0.5 (E4M3 per 16) + 32/in (fp32 RMS per row) + 32/numel (fp32 tensor scale). The index is held
in a dense uint8 buffer at run time; `pack_state()` is the bit-packed k-bit layout that `persistent_bytes()` counts.
Backends: backend="torch" is the eager reference (SR draws torch.rand once per update from `generator`; None = global
RNG). backend="triton" (auto = triton when the module is on CUDA, Triton imports and update_mode="exact") runs the whole
update as ONE fused kernel (mx6/lg_update_kernel.py: row RMS + value step + U map + Philox SR, in place on the uint8
index) and decodes the forward weight on the fly into x.dtype (bf16) without any fp32 cache. Same distribution, not the
same random numbers (Philox seeded per update from `generator`/the global CPU RNG).
Resident bytes/param (triton, cache_weight=False): idx 1 + scale 1/16 + v_rms 4/in  ~ 1.07 (torch path: + fp32
_scale_cache 4 + fp32 _weight_cache 4 = ~9.07). Transient inside a step: bf16 deployed weight 2 + grad 2-4.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from torch import nn

from . import update as mx_update
from . import lg_accum
from .lg_update_kernel import HAS_TRITON, lg_decode, lg_update_
from .packed_na import NVFP4_BLOCK, PackedNA

E2M1_MAGS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1_SIGNED = [-v for v in reversed(E2M1_MAGS[1:])] + E2M1_MAGS      # 15 signed values
N_RUNGS = len(E2M1_SIGNED)                                           # 15
N_GAPS = N_RUNGS - 1                                                 # 14
UPDATE_MODES = ("exact", "dir", "min")
BACKENDS = ("torch", "triton", "auto")


def pack_bits(idx: torch.Tensor, k: int) -> torch.Tensor:
    """[..., n] ints in [0, 2^k) -> [..., n*k/8] uint8, MSB first. n*k must be a multiple of 8."""
    n = idx.shape[-1]
    if (n * k) % 8:
        raise ValueError(f"n*k = {n * k} is not a multiple of 8")
    shifts = torch.arange(k - 1, -1, -1, device=idx.device)
    bits = ((idx.to(torch.long).unsqueeze(-1) >> shifts) & 1).reshape(*idx.shape[:-1], n * k // 8, 8)
    weights = 1 << torch.arange(7, -1, -1, device=idx.device)
    return (bits * weights).sum(-1).to(torch.uint8)


def unpack_bits(packed: torch.Tensor, k: int, n: int) -> torch.Tensor:
    """Inverse of `pack_bits` -> [..., n] int64."""
    shifts8 = torch.arange(7, -1, -1, device=packed.device)
    bits = ((packed.to(torch.long).unsqueeze(-1) >> shifts8) & 1).reshape(*packed.shape[:-1], n, k)
    return (bits << torch.arange(k - 1, -1, -1, device=packed.device)).sum(-1)


class LatentGrid:
    """The k-bit latent grid G (values in block-scale units) and its update arithmetic."""

    def __init__(self, k: int = 8, m: int | None = None):
        if not 4 <= k <= 12:
            raise ValueError(f"k must be in [4, 12], got {k}")
        if m is None:
            m = (2 ** k - 1) // N_GAPS
        elif m < 1 or N_GAPS * m + 1 > 2 ** k:
            raise ValueError(f"m={m}: {N_GAPS * m + 1} grid points do not fit in k={k} bits")
        self.k, self.m = int(k), int(m)
        self.n_pts = N_GAPS * self.m + 1
        # interface the e1_quality harness reads from `layer.fmt` (expected_bits)
        self.bits = self.k
        self.scale_bits = 8.0 / NVFP4_BLOCK            # E4M3 per 16 elements = 0.5
        self.nvfp4 = True
        self.block = NVFP4_BLOCK
        self.ladder_max = float(max(E2M1_MAGS))
        self.dtype = torch.uint8 if self.n_pts <= 256 else torch.int16
        #: PackedNA(C=1) is used ONLY for its NVFP4 helpers (nvfp4_fit, expand_scale) and round_idx.
        self.na = PackedNA(E2M1_MAGS, C=1, scale_format="nvfp4")
        signed = self.na.signed.to(torch.float32)
        gaps = signed[1:] - signed[:-1]
        m, n = self.m, self.n_pts
        # grid values: non-negative half in float64, the negative half mirrored => exactly odd
        v64 = torch.tensor(E2M1_SIGNED, dtype=torch.float64)
        vals = torch.zeros(n, dtype=torch.float64)
        for j in range(N_GAPS // 2, N_GAPS):
            vals[j * m:(j + 1) * m] = v64[j] + (v64[j + 1] - v64[j]) * torch.arange(m, dtype=torch.float64) / m
        vals[N_GAPS * m] = v64[-1]
        half = N_GAPS // 2 * m
        vals[:half] = -vals[half + 1:].flip(0)
        self.values = vals.to(torch.float32)
        self.rung = self.na.round_idx(self.values)                 # RTN to E2M1, ties away from zero
        self.deploy = signed[self.rung]
        i = torch.arange(n)
        self.sub_up = gaps[(i // m).clamp(max=N_GAPS - 1)] / m     # width of one sub-step moving up from i
        self.sub_dn = gaps[((i - 1) // m).clamp(min=0)] / m        # ... moving down from i
        self.sub_min = torch.minimum(self.sub_up, self.sub_dn)     # legacy min-gap tick
        self._cpu = SimpleNamespace(
            values=self.values, deploy=self.deploy, rung=self.rung, sub_up=self.sub_up, sub_dn=self.sub_dn,
            sub_min=self.sub_min, vint=signed[1:-1].contiguous(), vlo=signed[:-1].contiguous(),
            mg=(m / gaps).contiguous())
        self._dev: dict[str, SimpleNamespace] = {}

    def tables(self, device) -> SimpleNamespace:
        key = str(device)
        if key not in self._dev:
            self._dev[key] = SimpleNamespace(**{n: t.to(device) for n, t in vars(self._cpu).items()})
        return self._dev[key]

    def bits_per_param(self, numel: int | None = None) -> float:
        """k + 0.5 (+ 32/numel fp32 tensor scale); per-row RMS excluded (same convention as PackedNA)."""
        return float(self.bits) + self.scale_bits + (32.0 / numel if numel else 0.0)

    def _u(self, w: torch.Tensor, T: SimpleNamespace, clamp: bool = True) -> torch.Tensor:
        """U: value (block-scale units) -> continuous index; piecewise linear, U(G[i]) = i. Clamped to the grid,
        or (clamp=False) extrapolated linearly with the slope of the outermost gap."""
        j = torch.bucketize(w, T.vint, right=True)                  # gap index 0..13
        u = j.to(torch.float32) * self.m + (w - T.vlo[j]) * T.mg[j]
        return u.clamp_(0.0, float(self.n_pts - 1)) if clamp else u

    def nearest_index(self, a: torch.Tensor) -> torch.Tensor:
        """Index of the grid point nearest to the dimensionless values `a`; ties away from zero (odd)."""
        u = self._u(a.to(torch.float32), self.tables(a.device))
        c = float(N_GAPS // 2 * self.m)
        r = torch.where(u >= c, torch.floor(u + 0.5), torch.ceil(u - 0.5))
        return r.clamp_(0, self.n_pts - 1).to(torch.long)

    def move(self, idx: torch.Tensor, delta: torch.Tensor, gen: torch.Generator | None = None,
             mode: str = "exact", return_clip: bool = False):
        """New index after the value step `delta` (block-scale units, same shape as idx), by stochastic rounding.

        exact: u1 = idx + U(G[idx] + delta) - U(G[idx]);  E[G[SR(u1)]] = G[idx] + delta for any delta (see module doc).
        dir  : u1 = idx + delta / (sub-step of the gap moved into)   (exact only while the move stays in one gap)
        min  : u1 = idx + delta / (smallest adjacent sub-step)        (legacy, biased)
        NaN deltas do not move; |delta| is capped at 16 (the grid spans 12). The index is clamped to [0, n-1] BEFORE
        the rounding, so the excess is dropped. -> new index (int64) [, over: bool mask of requests beyond the grid].
        """
        if mode not in UPDATE_MODES:
            raise ValueError(f"update mode must be one of {UPDATE_MODES}, got {mode!r}")
        T = self.tables(idx.device)
        i = idx.to(torch.long)
        delta = torch.nan_to_num(delta.to(torch.float32), nan=0.0, posinf=16.0, neginf=-16.0).clamp_(-16.0, 16.0)
        if mode == "exact":
            w0 = T.values[i]
            d = self._u(w0 + delta, T, clamp=False) - self._u(w0, T, clamp=False)
        elif mode == "dir":
            d = delta / torch.where(delta >= 0, T.sub_up[i], T.sub_dn[i])
        else:
            d = delta / T.sub_min[i]
        u = i.to(torch.float32) + d
        top = float(self.n_pts - 1)
        over = (u < 0.0) | (u > top)
        u = u.clamp(0.0, top)
        fl = torch.floor(u)
        up = torch.rand(u.shape, generator=gen, device=u.device) < (u - fl)
        new = (fl + up.to(u.dtype)).to(torch.long).clamp_(0, self.n_pts - 1)
        return (new, over) if return_clip else new


_GRIDS: dict[tuple[int, int | None], LatentGrid] = {}


def get_grid(k: int, m: int | None = None) -> LatentGrid:
    if (k, m) not in _GRIDS:
        _GRIDS[(k, m)] = LatentGrid(k, m)
    return _GRIDS[(k, m)]


class _LGFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, module, token):
        weight = module.decode_weight(x.dtype)
        ctx.save_for_backward(x, weight)
        ctx.module = module
        ctx.version = module.idx._version
        return x @ weight.t()

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        module = ctx.module
        if module.idx._version != ctx.version:
            raise RuntimeError("weights changed between forward and backward")
        grad_x = grad_out @ weight
        grad_w = (grad_out.t() @ x).contiguous()
        with torch.no_grad():
            module._backward_update_(grad_w)
        module._pending = False
        return grad_x, None, None


GEMMS = ("cublas_bf16", "triton_bf16", "nvfp4_scaled_mm")
NATIVE_MODES = ("auto", True, False)


class _LGNativeFunction(torch.autograd.Function):
    """Native low-precision path (no fp32 anywhere). x arrives already cast to the compute dtype `cdt` (bf16 under
    autocast; the cast is outside, so autograd casts dx back). Forward: decode the deployed weight straight to `cdt`
    (or run an fp4 GEMM on the nibbles), y = x W^T. Saves x (cdt) and, only if module.save_weight, the decoded W;
    otherwise backward re-decodes (1 B read + 2 B write per param, ~free next to the GEMMs). Backward: dx = dy W,
    grad_w = dy^T x in `cdt` (fp32 accumulation inside cuBLAS, rounded once to `cdt`), then ONE fused update; grad_w is
    freed right after the kernel. Backward does not depend on the autocast state (everything is explicitly `cdt`)."""

    @staticmethod
    def forward(ctx, x, module, token):
        y, w = module._gemm_fwd(x)
        ctx.save_for_backward(x, w if module.save_weight else None)
        ctx.module = module
        ctx.version = module.idx._version
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, w = ctx.saved_tensors
        module = ctx.module
        if module.idx._version != ctx.version:
            raise RuntimeError("weights changed between forward and backward")
        dy = grad_out.to(x.dtype)
        dx = module._gemm_dx(dy, w) if ctx.needs_input_grad[0] else None
        grad_w = dy.t() @ x                                       # [out, in] in x.dtype, contiguous
        with torch.no_grad():
            module._backward_update_(grad_w)
        del grad_w
        module._pending = False
        return dx, None, None


# ── opaque custom ops for torch.compile (regional compile: everything EXCEPT the lg linear is compiled) ─────────────────
#
# `torch.ops.mx6.lg_linear` (forward, differentiable) and `torch.ops.mx6.lg_linear_bwd` (backward) wrap the SAME arithmetic
# as `_LGNativeFunction`, as functions of explicit tensors (x, idx, scale, tscale, v_rms, clip_buf) + python scalars. The
# backward op declares `Tensor(a!) idx, v_rms, clip_buf` as mutated, so dynamo / AOTAutograd see an opaque node with a
# declared side effect: no graph break, the surrounding LN / attention / GELU / residual code fuses normally, and the
# update runs exactly once per backward execution (it lives in the backward graph, never re-traced at run time).
# lr / weight_decay are NOT schema constants: they are 0-dim fp32 buffers of the module (`_lr_t`, `_wd_t`, set in place by
# the `lr` / `weight_decay` properties, i.e. by e1_quality.set_counter_lr) read by the update kernel by pointer, so a
# cosine schedule never recompiles. Only `has_wd` (bool, wd > 0) is a guard. Checkpoint-safe: the forward op has no side
# effect, the update lives only in the backward op (executed once per backward), so a non-reentrant torch.utils.checkpoint
# recompute of the forward is harmless. State the ops do NOT see: the RNG (the Philox seed is drawn from the global CPU
# RNG inside the op body, i.e. at run time), `module.generator` (ops path requires generator=None), `save_weight` (always re-decodes in backward), and
# `fp4_fallbacks` (counted in the module-level FP4_FALLBACKS instead). Defined via torch.library.Library (explicit
# schema strings: this file has `from __future__ import annotations`, which torch.library.custom_op cannot infer from).

FP4_FALLBACKS = {"n": 0}
_CFG_SCHEMA = ("int k, int m, str gemm, str dx_gemm, bool triton_on, str umode, float beta, float eps, "
               "bool has_wd, bool track_clip")


def _decode_dense(idx, scale, tscale, grid: LatentGrid, triton_on: bool, dtype):
    if triton_on:
        return lg_decode(idx, scale, tscale, grid.m, dtype)
    w = grid.tables(idx.device).deploy[idx.long()] * grid.na.expand_scale(scale, tscale)
    return w if w.dtype == dtype else w.to(dtype)


def _fp4_weight_of(idx, scale, tscale, grid: LatentGrid, triton_on: bool):
    from . import fp4_gemm as fg
    from .lg_update_kernel import lg_pack_nibbles, pack_nibbles_cf
    nib = lg_pack_nibbles(idx, grid.m) if triton_on else pack_nibbles_cf(idx, grid.m)
    return fg.Fp4Weight(nib, scale, tscale)


def _op_fwd_impl(x, idx, scale, tscale, v_rms, clip_buf, token, lr_t, wd_t, k, m, gemm, dx_gemm, triton_on, umode, beta,
                 eps, has_wd, track_clip):
    grid = get_grid(k, m)
    if gemm != "cublas_bf16":
        from . import fp4_gemm as fg
        try:
            if not x.is_cuda:
                raise fg.Fp4Unsupported("fp4 GEMMs need CUDA")
            q = _fp4_weight_of(idx, scale, tscale, grid, triton_on)
            return fg.fp4_matmul(x.contiguous(), q, "triton_bf16" if gemm == "triton_bf16" else "torch_scaled_mm", x.dtype)
        except fg.Fp4Unsupported:
            FP4_FALLBACKS["n"] += 1
    return x @ _decode_dense(idx, scale, tscale, grid, triton_on, x.dtype).t()


def _op_dx(dy, idx, scale, tscale, grid, gemm, dx_gemm, triton_on):
    if gemm != "cublas_bf16" and dy.is_cuda:
        from . import fp4_gemm as fg
        try:
            if gemm == "triton_bf16":
                return fg.fp4_matmul_dx(dy.contiguous(), _fp4_weight_of(idx, scale, tscale, grid, triton_on),
                                        "triton_bf16", dy.dtype)
            if dx_gemm == "nvfp4":
                return fg.fp4_matmul_dx(dy.contiguous(), _fp4_weight_of(idx, scale, tscale, grid, triton_on),
                                        "torch_scaled_mm", dy.dtype)
        except fg.Fp4Unsupported:
            FP4_FALLBACKS["n"] += 1
    return dy @ _decode_dense(idx, scale, tscale, grid, triton_on, dy.dtype)


def _draw_seed() -> int:
    """31-bit Philox seed from the global CPU RNG. Called from the (opaque) backward-op body, i.e. at RUN time on every
    execution of the backward: torch.compile never sees or bakes it, so each step gets fresh noise (no host-device sync:
    the CPU generator is read, not the GPU). NOT safe under CUDA-graph replay (mode="reduce-overhead"): the seed would
    be frozen into the captured launch."""
    return int(torch.randint(0, 2 ** 31 - 1, (1,)).item())


def _op_update_(grad_w, idx, scale, tscale, v_rms, clip_buf, lr_t, wd_t, grid, triton_on, umode, beta, eps, has_wd,
                track_clip, momentum=None):
    """In-place state update with global-RNG randomness (module.generator is None on the ops path). lr / wd arrive as
    0-dim fp32 device tensors (module buffers updated in place by `set_counter_lr`), so a scheduled lr is DATA, not a
    schema constant: no recompiles, no host sync."""
    if triton_on:
        seed = _draw_seed()
        clip = lg_update_(grad_w, idx, scale, tscale, v_rms, lr=lr_t, beta=beta, eps=eps, weight_decay=wd_t, m=grid.m,
                          seed=seed, track_clip=track_clip, has_wd=has_wd, momentum=momentum)
        if clip is not None:
            clip_buf.copy_(clip.sum(dtype=torch.float32) / idx.numel())
        return
    v_new, g_hat = mx_update.rms_update(grad_w.detach().float(), v_rms, beta, eps)
    v_rms.copy_(v_new)
    if has_wd:
        g_hat = g_hat + wd_t * grid.tables(idx.device).values[idx.long()]
    delta = (-lr_t * g_hat) / grid.na.expand_scale(scale, tscale)
    new, over = grid.move(idx, delta, None, umode, return_clip=True)
    clip_buf.copy_(over.float().mean())
    idx.copy_(new.to(idx.dtype))


def _op_bwd_impl(dy, x, idx, scale, tscale, v_rms, clip_buf, lr_t, wd_t, needs_dx, k, m, gemm, dx_gemm, triton_on, umode,
                 beta, eps, has_wd, track_clip):
    grid = get_grid(k, m)
    dy = dy.to(x.dtype)
    dx = _op_dx(dy, idx, scale, tscale, grid, gemm, dx_gemm, triton_on) if needs_dx else x.new_empty(0)
    grad_w = dy.t() @ x
    _op_update_(grad_w, idx, scale, tscale, v_rms, clip_buf, lr_t, wd_t, grid, triton_on, umode, beta, eps, has_wd,
                track_clip)
    return dx


_LG_LIB = None
HAS_LG_OPS = False


def _register_lg_ops() -> bool:
    global _LG_LIB
    if not (hasattr(torch.library, "register_autograd") and hasattr(torch.library, "register_fake")):
        return False
    try:
        if hasattr(torch.ops.mx6, "lg_linear") and hasattr(torch.ops.mx6, "lg_linear_bwd"):
            return True                                   # already registered by an earlier import of this module
    except (AttributeError, RuntimeError):
        pass
    try:
        lib = torch.library.Library("mx6", "FRAGMENT")
        lib.define("lg_linear(Tensor x, Tensor idx, Tensor scale, Tensor tscale, Tensor v_rms, Tensor clip_buf, "
                   "Tensor token, Tensor lr_t, Tensor wd_t, " + _CFG_SCHEMA + ") -> Tensor")
        lib.define("lg_linear_bwd(Tensor dy, Tensor x, Tensor(a!) idx, Tensor scale, Tensor tscale, Tensor(b!) v_rms, "
                   "Tensor(c!) clip_buf, Tensor lr_t, Tensor wd_t, bool needs_dx, " + _CFG_SCHEMA + ") -> Tensor")
        lib.impl("lg_linear", _op_fwd_impl, "CompositeExplicitAutograd")
        lib.impl("lg_linear_bwd", _op_bwd_impl, "CompositeExplicitAutograd")

        @torch.library.register_fake("mx6::lg_linear")
        def _fake_fwd(x, idx, scale, tscale, v_rms, clip_buf, token, lr_t, wd_t, k, m, gemm, dx_gemm, triton_on, umode,
                      beta, eps, has_wd, track_clip):
            return x.new_empty(x.shape[0], idx.shape[0])

        @torch.library.register_fake("mx6::lg_linear_bwd")
        def _fake_bwd(dy, x, idx, scale, tscale, v_rms, clip_buf, lr_t, wd_t, needs_dx, k, m, gemm, dx_gemm, triton_on,
                      umode, beta, eps, has_wd, track_clip):
            return x.new_empty(x.shape) if needs_dx else x.new_empty(0)

        def _setup(ctx, inputs, output):
            x, idx, scale, tscale, v_rms, clip_buf = inputs[:6]
            lr_t, wd_t = inputs[7:9]
            ctx.save_for_backward(x, idx, scale, tscale, v_rms, clip_buf, lr_t, wd_t)
            ctx.cfg = tuple(inputs[9:])
            ctx.needs_dx = bool(ctx.needs_input_grad[0])

        def _backward(ctx, dy):
            x, idx, scale, tscale, v_rms, clip_buf, lr_t, wd_t = ctx.saved_tensors
            dx = torch.ops.mx6.lg_linear_bwd(dy, x, idx, scale, tscale, v_rms, clip_buf, lr_t, wd_t, ctx.needs_dx,
                                             *ctx.cfg)
            return (dx if ctx.needs_dx else None,) + (None,) * (8 + len(ctx.cfg))

        torch.library.register_autograd("mx6::lg_linear", _backward, setup_context=_setup)
        _LG_LIB = lib
        return True
    except Exception:                                     # noqa: BLE001 - old torch / odd build: ops path unavailable
        return False


HAS_LG_OPS = _register_lg_ops()


# ── gradient-accumulation twins of the two ops (opt-in; the ops above are untouched) ────────────────────────────────────
#
# `lg_linear_acc` / `lg_linear_acc_bwd` = the same arithmetic plus the per-layer accumulation buffer (acc_buf, acc_scale:
# declared MUTATED in the backward) and `int acc_id` (a constant: the id of the shared lg_accum.AccumState). The micro-batch
# index, n, mode, kind and the all-reduce hook are read from that state at RUN time inside the opaque backward body (never
# traced): changing the micro-batch never changes the compiled graph. lr scaling (latent mode: lr / n) is a device-side
# multiply of the 0-dim lr tensor (no host sync); beta' arrives as the (constant) `beta` argument.

_ACC_SCHEMA = _CFG_SCHEMA + ", int acc_id"


def _op_acc_fwd_impl(x, idx, scale, tscale, v_rms, clip_buf, token, lr_t, wd_t, acc_buf, acc_scale, k, m, gemm, dx_gemm,
                     triton_on, umode, beta, eps, has_wd, track_clip, acc_id):
    return _op_fwd_impl(x, idx, scale, tscale, v_rms, clip_buf, token, lr_t, wd_t, k, m, gemm, dx_gemm, triton_on, umode,
                        beta, eps, has_wd, track_clip)


def _op_acc_bwd_impl(dy, x, idx, scale, tscale, v_rms, clip_buf, lr_t, wd_t, acc_buf, acc_scale, needs_dx, k, m, gemm,
                     dx_gemm, triton_on, umode, beta, eps, has_wd, track_clip, acc_id):
    grid = get_grid(k, m)
    dy = dy.to(x.dtype)
    dx = _op_dx(dy, idx, scale, tscale, grid, gemm, dx_gemm, triton_on) if needs_dx else x.new_empty(0)
    grad_w = dy.t() @ x
    st = lg_accum.get_state(acc_id)
    if lg_accum.fused_int4_ok(st, idx.shape[1], triton_on):      # int4 momentum EMA inside the update kernel
        st.n_updates += 1
        _op_update_(grad_w, idx, scale, tscale, v_rms, clip_buf, lr_t, wd_t, grid, triton_on, umode, beta, eps, has_wd,
                    track_clip, momentum=(acc_buf, acc_scale, st.mu, _draw_seed()))
        return dx
    finish = key = None
    if st.sparse:                                    # the update is deferred to the exit of lg_accumulate
        def finish(g):
            _op_update_(g, idx, scale, tscale, v_rms, clip_buf, lr_t, wd_t, grid, triton_on, umode, beta, eps, has_wd,
                        track_clip)
        key = acc_buf.data_ptr()
    g = lg_accum.accum_grad_(st, grad_w, acc_buf if acc_buf.numel() else None, acc_scale if acc_scale.numel() else None,
                             finish, key)
    if g is not None:
        lr_eff = lr_t * st.lr_mult if st.lr_mult != 1.0 else lr_t
        _op_update_(g, idx, scale, tscale, v_rms, clip_buf, lr_eff, wd_t, grid, triton_on, umode, beta, eps, has_wd,
                    track_clip)
    return dx


_LG_ACC_LIB = None
HAS_LG_ACC_OPS = False


def _register_lg_acc_ops() -> bool:
    global _LG_ACC_LIB
    if not HAS_LG_OPS:
        return False
    try:
        if hasattr(torch.ops.mx6, "lg_linear_acc") and hasattr(torch.ops.mx6, "lg_linear_acc_bwd"):
            return True
    except (AttributeError, RuntimeError):
        pass
    try:
        lib = torch.library.Library("mx6", "FRAGMENT")
        lib.define("lg_linear_acc(Tensor x, Tensor idx, Tensor scale, Tensor tscale, Tensor v_rms, Tensor clip_buf, "
                   "Tensor token, Tensor lr_t, Tensor wd_t, Tensor acc_buf, Tensor acc_scale, " + _ACC_SCHEMA + ") -> Tensor")
        lib.define("lg_linear_acc_bwd(Tensor dy, Tensor x, Tensor(a!) idx, Tensor scale, Tensor tscale, Tensor(b!) v_rms, "
                   "Tensor(c!) clip_buf, Tensor lr_t, Tensor wd_t, Tensor(d!) acc_buf, Tensor(e!) acc_scale, bool needs_dx, "
                   + _ACC_SCHEMA + ") -> Tensor")
        lib.impl("lg_linear_acc", _op_acc_fwd_impl, "CompositeExplicitAutograd")
        lib.impl("lg_linear_acc_bwd", _op_acc_bwd_impl, "CompositeExplicitAutograd")

        @torch.library.register_fake("mx6::lg_linear_acc")
        def _fake_acc_fwd(x, idx, scale, tscale, v_rms, clip_buf, token, lr_t, wd_t, acc_buf, acc_scale, k, m, gemm,
                          dx_gemm, triton_on, umode, beta, eps, has_wd, track_clip, acc_id):
            return x.new_empty(x.shape[0], idx.shape[0])

        @torch.library.register_fake("mx6::lg_linear_acc_bwd")
        def _fake_acc_bwd(dy, x, idx, scale, tscale, v_rms, clip_buf, lr_t, wd_t, acc_buf, acc_scale, needs_dx, k, m, gemm,
                          dx_gemm, triton_on, umode, beta, eps, has_wd, track_clip, acc_id):
            return x.new_empty(x.shape) if needs_dx else x.new_empty(0)

        def _setup(ctx, inputs, output):
            x, idx, scale, tscale, v_rms, clip_buf = inputs[:6]
            lr_t, wd_t, acc_buf, acc_scale = inputs[7:11]
            ctx.save_for_backward(x, idx, scale, tscale, v_rms, clip_buf, lr_t, wd_t, acc_buf, acc_scale)
            ctx.cfg = tuple(inputs[11:])
            ctx.needs_dx = bool(ctx.needs_input_grad[0])

        def _backward(ctx, dy):
            x, idx, scale, tscale, v_rms, clip_buf, lr_t, wd_t, acc_buf, acc_scale = ctx.saved_tensors
            dx = torch.ops.mx6.lg_linear_acc_bwd(dy, x, idx, scale, tscale, v_rms, clip_buf, lr_t, wd_t, acc_buf,
                                                 acc_scale, ctx.needs_dx, *ctx.cfg)
            return (dx if ctx.needs_dx else None,) + (None,) * (10 + len(ctx.cfg))

        torch.library.register_autograd("mx6::lg_linear_acc", _backward, setup_context=_setup)
        _LG_ACC_LIB = lib
        return True
    except Exception:                                     # noqa: BLE001
        return False


HAS_LG_ACC_OPS = _register_lg_acc_ops()


def lg_layers(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, LatentGridLinear)]


def prepare_lg_for_compile(model: nn.Module, mode: str = "op") -> dict:
    """Make the LatentGridLinear layers of `model` safe for torch.compile(model) (regional compile).
    mode "op"     : forward = torch.ops.mx6.lg_linear (opaque node, declared mutation in the backward op): NO graph breaks.
    mode "disable": forward wrapped in torch.compiler.disable (eager island; ~2 graph breaks per layer, always works).
    Returns {'mode', 'layers'}; raises if mode "op" is requested but the ops are unavailable."""
    if mode not in ("op", "disable"):
        raise ValueError("mode must be 'op' or 'disable'")
    ls = lg_layers(model)
    if mode == "op" and not HAS_LG_OPS:
        raise RuntimeError("custom ops unavailable in this torch (need torch.library.register_autograd / register_fake)")
    for m in ls:
        m.use_custom_op = mode == "op"
        if mode == "disable":
            m.forward = torch.compiler.disable(type(m).forward.__get__(m))
    return {"mode": mode, "layers": len(ls)}


class LatentGridLinear(nn.Module):
    """Bias-free linear whose weight is ONE k-bit latent index per element (see the module docstring).

    Args mirror MXNLinear where they exist: lr, rms_beta, rms_eps, weight_decay (restoring force
    g_hat += lambda * G[idx], the latent value in ladder units; 0 = off), headroom h (NVFP4 target amax/s = 6/h at
    init, h=1: the block max sits on the outer rung; the scale is frozen after init), update_mode ("exact" | "dir" |
    "min"), init ("fine": nearest fine-grid point of the fp32 init | "rung": the E2M1 rung RTN(init), the same
    deployed weights as MXNLinear's init for the same seed).
    """

    def __init__(self, in_features: int, out_features: int, k: int = 8, lr: float = 0.01,
                 rms_beta: float = 0.9, rms_eps: float = 1e-3, headroom: float = 1.0,
                 weight_decay: float = 0.0, update_mode: str = "exact", init: str = "fine",
                 grid: LatentGrid | None = None, backend: str = "auto", cache_weight: bool | None = None,
                 track_clip: bool = True, native="auto", gemm: str = "cublas_bf16", dx_gemm: str = "bf16",
                 save_weight: bool = False):
        super().__init__()
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        if native not in NATIVE_MODES:
            raise ValueError(f"native must be one of {NATIVE_MODES}, got {native!r}")
        if gemm not in GEMMS:
            raise ValueError(f"gemm must be one of {GEMMS}, got {gemm!r}")
        if dx_gemm not in ("bf16", "nvfp4"):
            raise ValueError("dx_gemm must be 'bf16' or 'nvfp4'")
        #: native: "auto" = the custom low-precision path whenever the triton backend is active (CUDA), True = force it
        #: (also on the torch backend / CPU, for tests), False = the legacy path (fp32 weight on the torch backend,
        #: x.dtype decode on triton) that needs an fp32 island under autocast.
        self.native = native
        #: forward GEMM: cublas_bf16 (exact, default) | triton_bf16 (fp4 nibbles -> bf16 tensor cores, exact) |
        #: nvfp4_scaled_mm (cuBLASLt NVFP4; x is quantized to NVFP4 on the fly: APPROXIMATE activations).
        self.gemm = gemm
        #: backward dx GEMM: "bf16" exact (default) | "nvfp4" (only with gemm=nvfp4_scaled_mm; dy quantized and W^T
        #: RE-quantized: doubly approximate, slow because W^T is rebuilt every step).
        self.dx_gemm = dx_gemm
        #: keep the decoded bf16 weight from forward to backward (+2 B/param transient per layer for the whole forward
        #: pass) instead of re-decoding in backward (default: re-decode).
        self.save_weight = bool(save_weight)
        self.fp4_fallbacks = 0                                   # times an fp4 GEMM was unsupported and bf16 ran instead
        if backend == "triton" and update_mode != "exact":
            raise ValueError("the triton backend implements update_mode='exact' only")
        self.backend = backend
        #: triton path only: keep ONE bf16/x.dtype deployed weight between updates (+2 B/param resident). Default off.
        self.cache_weight = bool(cache_weight) if cache_weight is not None else False
        self.track_clip = bool(track_clip)
        self._step = 0
        if in_features % NVFP4_BLOCK:
            raise ValueError(f"in_features must be a multiple of {NVFP4_BLOCK}")
        if update_mode not in UPDATE_MODES:
            raise ValueError(f"update_mode must be one of {UPDATE_MODES}")
        if init not in ("fine", "rung"):
            raise ValueError("init must be 'fine' or 'rung'")
        if weight_decay < 0.0 or headroom <= 0.0:
            raise ValueError("weight_decay must be >= 0 and headroom > 0")
        self.in_features, self.out_features = in_features, out_features
        self.lr, self.rms_beta, self.rms_eps = lr, rms_beta, rms_eps
        self.headroom, self.weight_decay = float(headroom), float(weight_decay)
        #: device-side copies of lr / weight_decay for the custom-op path (see the ops section); kept in sync by the
        #: `lr` / `weight_decay` setters, moved by .to(device); never saved (set from the python floats).
        self.register_buffer("_lr_t", torch.tensor(float(lr), dtype=torch.float32), persistent=False)
        self.register_buffer("_wd_t", torch.tensor(float(weight_decay), dtype=torch.float32), persistent=False)
        self.__dict__["_lr_dev"], self.__dict__["_wd_dev"] = float(lr), float(weight_decay)
        self.update_mode, self.init_mode = update_mode, init
        self.fmt = grid if grid is not None else get_grid(k)
        self.generator: torch.Generator | None = None           # None = global RNG (harness seeding)
        self._lcf: torch.Tensor | None = None                    # backing of `last_clip_frac` (legacy / Function path)
        #: forward through torch.ops.mx6.lg_linear (opaque to torch.compile, see prepare_lg_for_compile). Default off:
        #: the autograd.Function path is the reference. Needs native_enabled(), generator=None and HAS_LG_OPS.
        self.use_custom_op = False
        bound = 1.0 / in_features ** 0.5
        init_w = torch.empty(out_features, in_features).uniform_(-bound, bound)   # same RNG call as MXNLinear
        sb, ts, idx = self._fit(init_w)
        self.register_buffer("idx", idx)
        self.register_buffer("_scale", sb)                       # uint8 [out, in/16] E4M3 bytes, frozen
        self.register_buffer("_tscale", ts)                      # fp32 0-dim, frozen
        self.register_buffer("v_rms", torch.full((out_features, 1), rms_eps, dtype=torch.float32))
        self.register_buffer("_clip_buf", torch.zeros(()), persistent=False)   # ops path: fraction beyond the grid
        self._token = torch.zeros(0, requires_grad=True)
        #: opt-in gradient accumulation (mx6/lg_accum.py, set via lg_accum.set_accum): shared AccumState, its id (-1 = off;
        #: a python int, so a guard-constant for torch.compile), the beta exponent (1/n in latent mode) and the buffers
        #: (non-persistent: never in state_dict; empty tensors when the mode needs none).
        self._acc = None
        self._acc_id = -1
        self._acc_pow = 1.0
        self.register_buffer("_acc_buf", None, persistent=False)
        self.register_buffer("_acc_scale", None, persistent=False)
        self._pending = False
        self._weight_cache: torch.Tensor | None = None
        self._scale_cache: torch.Tensor | None = None

    @property
    def lr(self) -> float:
        return self._lr

    @lr.setter
    def lr(self, v) -> None:
        v = float(v)
        self.__dict__["_lr"] = v
        b = self._buffers.get("_lr_t")
        if b is not None and self.__dict__.get("_lr_dev") != v:
            b.fill_(v)                       # in place on the same tensor object (a compiled graph input), no host sync
            self.__dict__["_lr_dev"] = v

    @property
    def weight_decay(self) -> float:
        return self._wd

    @weight_decay.setter
    def weight_decay(self, v) -> None:
        v = float(v)
        self.__dict__["_wd"] = v
        self.__dict__["_has_wd"] = v > 0.0
        b = self._buffers.get("_wd_t")
        if b is not None and self.__dict__.get("_wd_dev") != v:
            b.fill_(v)
            self.__dict__["_wd_dev"] = v

    @property
    def last_clip_frac(self) -> torch.Tensor | None:
        """0-dim: fraction of update requests beyond the grid at the last update (None before it / track_clip off)."""
        if self._lcf is None and self.use_custom_op and self.track_clip:
            return self._clip_buf
        return self._lcf

    @last_clip_frac.setter
    def last_clip_frac(self, v) -> None:
        self._lcf = v

    def _fit(self, w: torch.Tensor):
        na = self.fmt.na
        sb, ts = na.nvfp4_fit(w, self.fmt.ladder_max / self.headroom)
        a = w.float() / na.expand_scale(sb, ts)
        if self.init_mode == "fine":
            idx = self.fmt.nearest_index(a)
        else:
            idx = na.round_idx(a) * self.fmt.m                      # rung r sits at index r*m
        return sb, ts, idx.to(self.fmt.dtype)

    @classmethod
    def from_dense(cls, weight: torch.Tensor, k: int = 8, **kwargs) -> "LatentGridLinear":
        layer = cls(weight.shape[1], weight.shape[0], k=k, **kwargs)
        with torch.no_grad():
            sb, ts, idx = layer._fit(weight.detach().float())
            layer._scale.copy_(sb)
            layer._tscale.copy_(ts)
            layer.idx.copy_(idx)
            layer.invalidate()
        return layer

    # ── caches ────────────────────────────────────────────────────────────────

    def invalidate(self) -> None:
        self._weight_cache = None
        self._scale_cache = None

    def _apply(self, fn, *args, **kwargs):
        out = super()._apply(fn, *args, **kwargs)
        self.invalidate()
        return out

    def _load_from_state_dict(self, *args, **kwargs):
        out = super()._load_from_state_dict(*args, **kwargs)
        self.invalidate()
        return out

    def set_index_(self, idx: torch.Tensor) -> None:
        self.idx.copy_(idx.to(self.idx.dtype))
        self.invalidate()

    # ── views of the state ────────────────────────────────────────────────────

    def _use_triton(self) -> bool:
        if self.backend == "torch":
            return False
        ok = self.idx.is_cuda and HAS_TRITON and self.update_mode == "exact"
        if self.backend == "triton" and not ok:
            raise RuntimeError("backend='triton' needs a CUDA module, Triton installed and update_mode='exact'")
        return ok

    def _scale_full(self, cache: bool = True) -> torch.Tensor:
        if self._scale_cache is not None:
            return self._scale_cache
        full = self.fmt.na.expand_scale(self._scale, self._tscale)
        if cache:
            self._scale_cache = full
        return full

    def weight(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """The deployed weight: RTN_E2M1(G[idx]) * s (a function of idx only).

        torch path: fp32, cached until the next update (resident +8 B/param with the fp32 scale cache).
        triton path: decoded on the fly by one kernel into `dtype` (default bf16; forward passes x.dtype), NO fp32
        scale cache; kept between updates only if cache_weight=True (+2 B/param for bf16)."""
        if not self._use_triton():
            if self._weight_cache is None:
                T = self.fmt.tables(self.idx.device)
                self._weight_cache = T.deploy[self.idx.long()] * self._scale_full()
            return self._weight_cache
        dt = dtype if dtype in (torch.bfloat16, torch.float16, torch.float32) else torch.bfloat16
        c = self._weight_cache
        if c is not None and c.dtype == dt:
            return c
        w = lg_decode(self.idx, self._scale, self._tscale, self.fmt.m, dt)
        self._weight_cache = w if self.cache_weight else None
        return w

    def latent_weight(self) -> torch.Tensor:
        """G[idx] * s, for analysis only (the forward never sees it)."""
        return self.fmt.tables(self.idx.device).values[self.idx.long()] * self._scale_full(cache=not self._use_triton())

    def resident_bytes(self) -> int:
        """Bytes actually held between steps (state + whatever caches exist), excluding transients."""
        n = sum(t.numel() * t.element_size() for t in (self.idx, self._scale, self._tscale, self.v_rms))
        for c in (self._weight_cache, self._scale_cache):
            if c is not None:
                n += c.numel() * c.element_size()
        return n

    def rung_index(self) -> torch.Tensor:
        """Deployed E2M1 rung (0..14, 7 = zero) of every weight, int64."""
        return self.fmt.tables(self.idx.device).rung[self.idx.long()]

    def pack_state(self) -> torch.Tensor:
        """[out, in*k/8] uint8: the bit-packed k-bit index plane."""
        return pack_bits(self.idx, self.fmt.k)

    def load_state(self, packed: torch.Tensor) -> None:
        self.set_index_(unpack_bits(packed, self.fmt.k, self.in_features))

    def persistent_bytes(self) -> int:
        """Bit-packed index + E4M3 block scales + fp32 RMS per row + fp32 tensor scale."""
        p = self.in_features * self.out_features
        return math.ceil(p * self.fmt.bits / 8) + self._scale.numel() + self.v_rms.numel() * 4 + 4

    def bits_per_param(self) -> float:
        return self.persistent_bytes() * 8.0 / (self.in_features * self.out_features)

    # ── forward / update ──────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected {self.in_features}, got {x.shape[-1]}")
        shape = x.shape
        if self.native_enabled():
            cdt = self._compute_dtype(x)
            x2 = x.reshape(-1, shape[-1]).to(cdt)
            if (self.use_custom_op and HAS_LG_OPS and self.generator is None
                    and torch.is_grad_enabled() and self.training):
                if self._acc_id >= 0:
                    if not HAS_LG_ACC_OPS:
                        raise RuntimeError("gradient accumulation on the op path needs the mx6::lg_linear_acc ops")
                    out = torch.ops.mx6.lg_linear_acc(
                        x2, self.idx, self._scale, self._tscale, self.v_rms, self._clip_buf, self._token,
                        self._lr_t, self._wd_t, self._acc_buf, self._acc_scale, self.fmt.k, self.fmt.m, self.gemm,
                        self.dx_gemm, self._use_triton(), self.update_mode, float(self.rms_beta) ** self._acc_pow,
                        float(self.rms_eps), self._has_wd, self.track_clip, self._acc_id)
                    return out.reshape(*shape[:-1], self.out_features)
                out = torch.ops.mx6.lg_linear(
                    x2, self.idx, self._scale, self._tscale, self.v_rms, self._clip_buf, self._token,
                    self._lr_t, self._wd_t, self.fmt.k, self.fmt.m, self.gemm, self.dx_gemm, self._use_triton(),
                    self.update_mode, float(self.rms_beta), float(self.rms_eps), self._has_wd, self.track_clip)
                return out.reshape(*shape[:-1], self.out_features)
            if torch.is_grad_enabled() and self.training:
                if self._pending:
                    raise RuntimeError("LatentGridLinear forwarded twice before backward; exactly one forward per "
                                       "backward is required (wrap measurement passes in torch.no_grad() / .eval())")
                self._pending = True
                out = _LGNativeFunction.apply(x2, self, self._token)
            else:
                out = self._gemm_fwd(x2)[0]
            return out.reshape(*shape[:-1], self.out_features)
        if torch.is_grad_enabled() and self.training:
            if self._pending:
                raise RuntimeError("LatentGridLinear forwarded twice before backward; exactly one forward per "
                                   "backward is required (wrap measurement passes in torch.no_grad() / .eval())")
            self._pending = True
            out = _LGFunction.apply(x.reshape(-1, shape[-1]), self, self._token)
            return out.reshape(*shape[:-1], self.out_features)
        return x @ self.decode_weight(x.dtype).t()

    # ── native low-precision path ─────────────────────────────────────────────

    def native_enabled(self) -> bool:
        """True if forward/backward run on the native path (`_LGNativeFunction`): `native=True`, or `native="auto"` with
        the triton backend active. e1_quality.fp32_islands skips such layers (no fp32 island needed under autocast)."""
        if self.native is True:
            return True
        if self.native is False:
            return False
        return self.backend != "torch" and self.idx.is_cuda and HAS_TRITON and self.update_mode == "exact"

    def _compute_dtype(self, x: torch.Tensor) -> torch.dtype:
        """Autocast dtype if autocast is on for x's device, else x.dtype (fp32 stays fp32, bf16 stays bf16)."""
        dev = x.device.type
        try:
            if torch.is_autocast_enabled(dev):
                return torch.get_autocast_dtype(dev)
        except (TypeError, AttributeError):                       # older torch: cuda/cpu-specific getters
            if dev == "cuda" and torch.is_autocast_enabled():
                return torch.get_autocast_gpu_dtype()
            if dev == "cpu" and torch.is_autocast_cpu_enabled():
                return torch.get_autocast_cpu_dtype()
        return x.dtype

    def decode_weight(self, dtype: torch.dtype) -> torch.Tensor:
        """Deployed weight as `dtype` [out, in]. Triton: one kernel, bf16/fp16/fp32 straight from the uint8 index (no
        fp32 intermediate). Torch backend: the (fp32, cached) reference weight cast to dtype."""
        if self._use_triton():
            return self.weight(dtype)
        w = self.weight()
        return w if w.dtype == dtype else w.to(dtype)

    def fp4_weight(self):
        """The deployed weight as an `Fp4Weight` (nibbles from the index by one kernel, the frozen E4M3 block scales and
        tensor scale). LOSSLESS: the deployed weight is exactly E2M1 * scale."""
        from . import fp4_gemm as fg
        from .lg_update_kernel import lg_pack_nibbles, pack_nibbles_cf
        nib = lg_pack_nibbles(self.idx, self.fmt.m) if self._use_triton() else pack_nibbles_cf(self.idx, self.fmt.m)
        return fg.Fp4Weight(nib, self._scale, self._tscale)

    def _gemm_fwd(self, x: torch.Tensor):
        """y = x W^T for x [N, in] in the compute dtype -> (y, W or None). W is the decoded weight (None for fp4 GEMMs)."""
        dt = x.dtype
        if self.gemm != "cublas_bf16":
            from . import fp4_gemm as fg
            try:
                if not x.is_cuda:
                    raise fg.Fp4Unsupported("fp4 GEMMs need CUDA")
                q = self.fp4_weight()
                be = "triton_bf16" if self.gemm == "triton_bf16" else "torch_scaled_mm"
                return fg.fp4_matmul(x.contiguous(), q, be, dt), None
            except fg.Fp4Unsupported:
                self.fp4_fallbacks += 1
        w = self.decode_weight(dt)
        return x @ w.t(), w

    def _gemm_dx(self, dy: torch.Tensor, w):
        """dx = dy W. Exact bf16 by default; triton_bf16 forward -> the exact Triton dx kernel on the nibbles; the NVFP4
        dx (dx_gemm="nvfp4", approximate) only when gemm=nvfp4_scaled_mm."""
        if self.gemm != "cublas_bf16" and dy.is_cuda:
            from . import fp4_gemm as fg
            try:
                if self.gemm == "triton_bf16":
                    return fg.fp4_matmul_dx(dy.contiguous(), self.fp4_weight(), "triton_bf16", dy.dtype)
                if self.dx_gemm == "nvfp4":
                    return fg.fp4_matmul_dx(dy.contiguous(), self.fp4_weight(), "torch_scaled_mm", dy.dtype)
            except fg.Fp4Unsupported:
                self.fp4_fallbacks += 1
        if w is None:
            w = self.decode_weight(dy.dtype)
        return dy @ w

    # ── opt-in gradient accumulation (mx6/lg_accum.py) ─────────────────────────────────────────────────────

    def enable_accum_(self, st) -> None:
        """Attach the shared AccumState and allocate this layer's buffer (buffer mode): bf16/fp32 [out,in], or
        float8_e4m3fn [out,in] + fp32 [out,1] row scale. Latent mode / unused parts are empty tensors (the op needs tensors)."""
        dev = self.idx.device
        empty = lambda dt: torch.empty(0, dtype=dt, device=dev)           # noqa: E731
        shape = tuple(self.idx.shape)
        if st.mode == "ema" and st.kind == "int4":
            bshape, sshape = lg_accum.q4_shapes(*shape)
            self._acc_buf = torch.full(bshape, 0x88, dtype=torch.uint8, device=dev)       # q = 0 in both nibbles
            self._acc_scale = torch.ones(sshape, dtype=torch.bfloat16, device=dev)
        elif st.mode in ("buffer", "ema"):
            dt = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp8": torch.float8_e4m3fn}[st.kind]
            self._acc_buf = torch.zeros(shape, dtype=dt, device=dev)
            self._acc_scale = (torch.zeros(shape[0], 1, dtype=torch.float32, device=dev) if st.kind == "fp8"
                               else empty(torch.float32))
        else:
            self._acc_buf, self._acc_scale = empty(torch.bfloat16), empty(torch.float32)
        self._acc, self._acc_id = st, st.id
        self._acc_pow = 1.0 / st.n if st.mode == "latent" else 1.0

    def disable_accum_(self) -> None:
        self._acc, self._acc_id, self._acc_pow = None, -1, 1.0
        self._acc_buf = self._acc_scale = None

    def zero_accum_(self) -> None:
        for t in (self._acc_buf, self._acc_scale):
            if t is not None and t.numel():
                t.fill_(0x88) if t.dtype == torch.uint8 else t.zero_()     # int4 buffer: q = 0 is nibble 8

    def _backward_update_(self, grad_w: torch.Tensor) -> None:
        """What backward calls: the plain update, or (accumulation configured) the accumulation step + update."""
        st = self._acc
        if st is None:
            self.apply_update_(grad_w)
            return
        if lg_accum.fused_int4_ok(st, self.in_features, self._use_triton()):   # int4 momentum EMA inside the kernel
            st.n_updates += 1
            self._apply_update_triton(grad_w, momentum=(self._acc_buf, self._acc_scale, st.mu, self._next_seed()))
            return
        finish = None
        if st.sparse:                                # the update is deferred to the exit of lg_accumulate
            def finish(g):
                self.apply_update_(g, lr_scale=st.lr_mult, beta=st.beta_eff(self.rms_beta))
        g = lg_accum.accum_grad_(st, grad_w, self._acc_buf if self._acc_buf.numel() else None,
                                 self._acc_scale if self._acc_scale.numel() else None, finish, id(self))
        if g is not None:
            self.apply_update_(g, lr_scale=st.lr_mult, beta=st.beta_eff(self.rms_beta))

    def apply_update_(self, grad_w: torch.Tensor, lr_scale: float = 1.0, beta: float | None = None) -> None:
        """In-place update of the index, called from backward. No optimizer, no momentum, no counter.
        lr_scale / beta (defaults: 1.0 / self.rms_beta = unchanged behaviour) are used by the latent accumulation mode."""
        if grad_w.shape != self.idx.shape:
            raise ValueError(f"grad shape {tuple(grad_w.shape)} != state {tuple(self.idx.shape)}")
        if self._use_triton():
            self._apply_update_triton(grad_w, lr_scale, beta)
            return
        lr = self.lr if lr_scale == 1.0 else self.lr * lr_scale
        self.v_rms, g_hat = mx_update.rms_update(grad_w.detach().float(), self.v_rms,
                                                 self.rms_beta if beta is None else beta, self.rms_eps)
        if self.weight_decay > 0.0:
            g_hat = g_hat + self.weight_decay * self.fmt.tables(self.idx.device).values[self.idx.long()]
        delta = (-lr * g_hat) / self._scale_full()              # value step in block-scale units
        new, over = self.fmt.move(self.idx, delta, self.generator, self.update_mode, return_clip=True)
        self.last_clip_frac = over.float().mean()
        self.idx.copy_(new.to(self.idx.dtype))
        self._weight_cache = None

    def _next_seed(self) -> int:
        """31-bit Philox seed for this update: drawn from `generator` (or the global CPU RNG, so torch.manual_seed
        makes runs reproducible). The per-element counter is row*cols+col, so (seed, step order) fixes the noise."""
        gen = self.generator
        dev = gen.device if gen is not None else "cpu"
        return int(torch.randint(0, 2 ** 31 - 1, (1,), generator=gen, device=dev).item())

    def _apply_update_triton(self, grad_w: torch.Tensor, lr_scale: float = 1.0, beta: float | None = None,
                             momentum=None) -> None:
        clip = lg_update_(grad_w, self.idx, self._scale, self._tscale, self.v_rms, momentum=momentum,
                          lr=self.lr if lr_scale == 1.0 else self.lr * lr_scale,
                          beta=self.rms_beta if beta is None else beta,
                          eps=self.rms_eps, weight_decay=self.weight_decay, m=self.fmt.m, seed=self._next_seed(),
                          track_clip=self.track_clip)
        self.last_clip_frac = (clip.sum(dtype=torch.float32) / self.idx.numel()) if clip is not None else None
        self._weight_cache = None

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, lr={self.lr}, "
                f"k={self.fmt.k}, m={self.fmt.m}, headroom={self.headroom}, update={self.update_mode}")
