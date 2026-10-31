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
Eager torch only. SR draws torch.rand once per update from `generator` (None = global RNG: the harness'
torch.manual_seed(seed) fixes init and update noise).
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from torch import nn

from . import update as mx_update
from .packed_na import NVFP4_BLOCK, PackedNA

E2M1_MAGS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1_SIGNED = [-v for v in reversed(E2M1_MAGS[1:])] + E2M1_MAGS      # 15 signed values
N_RUNGS = len(E2M1_SIGNED)                                           # 15
N_GAPS = N_RUNGS - 1                                                 # 14
UPDATE_MODES = ("exact", "dir", "min")


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
        weight = module.weight()
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
        grad_w = grad_out.t() @ x
        with torch.no_grad():
            module.apply_update_(grad_w)
        module._pending = False
        return grad_x, None, None


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
                 grid: LatentGrid | None = None):
        super().__init__()
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
        self.update_mode, self.init_mode = update_mode, init
        self.fmt = grid if grid is not None else get_grid(k)
        self.generator: torch.Generator | None = None           # None = global RNG (harness seeding)
        self.last_clip_frac: torch.Tensor | None = None          # 0-dim: fraction of requests beyond the grid
        bound = 1.0 / in_features ** 0.5
        init_w = torch.empty(out_features, in_features).uniform_(-bound, bound)   # same RNG call as MXNLinear
        sb, ts, idx = self._fit(init_w)
        self.register_buffer("idx", idx)
        self.register_buffer("_scale", sb)                       # uint8 [out, in/16] E4M3 bytes, frozen
        self.register_buffer("_tscale", ts)                      # fp32 0-dim, frozen
        self.register_buffer("v_rms", torch.full((out_features, 1), rms_eps, dtype=torch.float32))
        self._token = torch.zeros(0, requires_grad=True)
        self._pending = False
        self._weight_cache: torch.Tensor | None = None
        self._scale_cache: torch.Tensor | None = None

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

    def _scale_full(self) -> torch.Tensor:
        if self._scale_cache is None:
            self._scale_cache = self.fmt.na.expand_scale(self._scale, self._tscale)
        return self._scale_cache

    def weight(self) -> torch.Tensor:
        """The deployed weight: RTN_E2M1(G[idx]) * s (a function of idx only; cached until the next update)."""
        if self._weight_cache is None:
            T = self.fmt.tables(self.idx.device)
            self._weight_cache = T.deploy[self.idx.long()] * self._scale_full()
        return self._weight_cache

    def latent_weight(self) -> torch.Tensor:
        """G[idx] * s, for analysis only (the forward never sees it)."""
        return self.fmt.tables(self.idx.device).values[self.idx.long()] * self._scale_full()

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
        if torch.is_grad_enabled() and self.training:
            if self._pending:
                raise RuntimeError("LatentGridLinear forwarded twice before backward; exactly one forward per "
                                   "backward is required (wrap measurement passes in torch.no_grad() / .eval())")
            self._pending = True
            out = _LGFunction.apply(x.reshape(-1, shape[-1]), self, self._token)
            return out.reshape(*shape[:-1], self.out_features)
        return x @ self.weight().t()

    def apply_update_(self, grad_w: torch.Tensor) -> None:
        """In-place update of the index, called from backward. No optimizer, no momentum, no counter."""
        if grad_w.shape != self.idx.shape:
            raise ValueError(f"grad shape {tuple(grad_w.shape)} != state {tuple(self.idx.shape)}")
        self.v_rms, g_hat = mx_update.rms_update(grad_w.detach().float(), self.v_rms, self.rms_beta, self.rms_eps)
        if self.weight_decay > 0.0:
            g_hat = g_hat + self.weight_decay * self.fmt.tables(self.idx.device).values[self.idx.long()]
        delta = (-self.lr * g_hat) / self._scale_full()              # value step in block-scale units
        new, over = self.fmt.move(self.idx, delta, self.generator, self.update_mode, return_clip=True)
        self.last_clip_frac = over.float().mean()
        self.idx.copy_(new.to(self.idx.dtype))
        self._weight_cache = None

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, lr={self.lr}, "
                f"k={self.fmt.k}, m={self.fmt.m}, headroom={self.headroom}, update={self.update_mode}")
