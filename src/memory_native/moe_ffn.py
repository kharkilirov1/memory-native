"""CounterMoEFFN -- the architecture lever (verification-plan M4): a counter-state Mixture-of-Experts.

Replace a dense FFN (two d x 4d GEMMs touched by every token) with a *sparse* MoE: a small fp
router picks top_k of E experts per token; each expert is a counter-MLP (RMSCounterLinear d->h,
gelu, RMSCounterLinear h->d). Only the router is an fp Parameter (AdamW owns it); the experts are
counter-state (0.75 byte/weight visible) and self-update in their own backward.

Why this fits the counter method exactly (no approximation, no bias):
    A token NOT routed to an expert contributes nothing to that expert's output, so its gradient
    w.r.t. that expert is EXACTLY zero. We gather each expert's routed tokens into one contiguous
    batch and run that expert's two counter layers on exactly those tokens, so each counter layer
    sees only -- and all of -- its tokens. The fused counter update over that batch is therefore the
    exact per-expert gradient, not a sparse approximation. (Gather-per-expert also satisfies the
    RMSCounterLinear "one forward per backward" contract: every expert layer is called at most once.)

EQUAL ACTIVE COMPUTE (the gate's pivot):
    A dense FFN does 2*d*(4d) active MACs/token. We size each expert to hidden h so that the top_k
    experts a token actually visits cost ~ the dense active MACs: top_k * (2*d*h) ~ 2*d*4d, i.e.
    h ~ 4d / top_k. E (the expert count) grows TOTAL capacity / persistent bytes without touching the
    per-token active compute (a token still only visits top_k experts).

Load balancing:
    Top-k routing collapses (a few experts hog every token) unless penalized. We add the standard
    switch-transformer auxiliary loss  aux = E * sum_e (frac_tokens_e * mean_router_prob_e), exposed
    via aux_loss_weight; the last forward's value is stashed on .last_aux_loss so the training loop
    can add aux_loss_weight * ffn.last_aux_loss to the task loss.

Pure PyTorch, CPU/CUDA.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .counter import RMSCounterLinear, decode_state, encode_state, stochastic_round
from .packed import PackedRMSCounterLinear

__all__ = ["CounterMoEFFN"]


def _gelu_grad(x: torch.Tensor) -> torch.Tensor:
    """d/dx of the exact (erf) GELU = Phi(x) + x*phi(x)."""
    cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    pdf = torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    return cdf + x * pdf


@torch.no_grad()
def _batched_rms_update(state, scale, v, grad_w, active, *, C, lr, lr_scale, beta, eps):
    """RMS + stochastic-rounding counter update over a STACK of experts at once: state/grad_w are
    [E,out,in], scale/v are [E,out,1]. Identical math to RMSCounterLinear._update_tile (exact/eager,
    pulse=direct), just vectorized over the E axis -- so all experts update in ONE set of ops, no
    per-expert decode/SR/encode loop. `active` [E] masks experts that got no tokens this step (they
    keep their state/scale/v unchanged, matching the loop path which never calls them). Bit-identical
    to looping the same update per expert with the same SR draw order (verified)."""
    t, c = decode_state(state, C); t = t.float(); c = c.float()
    grad_w = grad_w.float()                                   # the tick math is always fp32
    g_sq = grad_w.pow(2).mean(dim=-1, keepdim=True)
    v_new = beta * v + (1.0 - beta) * g_sq
    denom = v_new.sqrt().clamp_min(eps)
    grad_eff = grad_w / denom
    grad_s = (grad_w * t).sum(dim=-1, keepdim=True) / math.sqrt(grad_w.shape[-1])
    s_new = (scale - lr_scale * grad_s).clamp(1e-5, 10.0)
    cc = stochastic_round(c * (scale / s_new) + (-lr * grad_eff) * (C / s_new))
    carry = torch.trunc(cc / C)
    rem = cc - carry * C
    nt = (t + carry).clamp(-1, 1)
    rem = torch.where((t + carry) != nt, torch.sign(cc) * (C - 1), rem).clamp(-(C - 1), C - 1)
    new_state = encode_state(nt, rem, C)
    m = active.view(-1, *([1] * (state.dim() - 1)))           # [E,1,1] broadcast mask
    state.copy_(torch.where(m, new_state, state))
    scale.copy_(torch.where(m, s_new, scale))
    v.copy_(torch.where(m, v_new, v))


def _grouped_grad_w(go_sorted, x_sorted, offs, E):
    """Per-expert weight-gradient gw[e] = go[seg_e]^T @ x[seg_e], stacked to [E, Nout, Nin], with NO
    python loop over experts: scatter each expert's sorted rows into a padded [E, cap, ·] batch and
    one torch.bmm. Zero-padding contributes zero, so it is bit-exact vs the per-expert matmul loop
    (up to fp reduction order). Returns (gw, active[E]).

    MEMORY GUARD: the pad allocates E*cap*(Nout+Nin) with cap = MAX segment. Under skewed routing
    (early training / collapse) cap -> M, so the pad blows up to ~E x the real work. When cap exceeds
    a small multiple of the balanced size we fall back to the per-expert loop (memory-safe, each
    segment at its real size) -- the loop-free bmm only fires when routing is balanced enough that the
    pad is bounded. torch._grouped_mm's 2d×2d→3d mode does NOT compute this per-segment product."""
    M = go_sorted.shape[0]
    starts = torch.cat([offs.new_zeros(1), offs[:-1]])
    counts = offs - starts
    Nout, Nin = go_sorted.shape[1], x_sorted.shape[1]
    cap = max(int(counts.max().item()) if M > 0 else 1, 1)
    mean = max(M // E, 1)
    if cap > 4 * mean:                                                     # too skewed -> pad blows up
        gw = go_sorted.new_zeros(E, Nout, Nin)
        b = [0] + offs.tolist()
        for e in range(E):
            s, t = b[e], b[e + 1]
            if t > s:
                gw[e] = go_sorted[s:t].t() @ x_sorted[s:t]
        return gw, counts > 0
    row = torch.arange(M, device=go_sorted.device)
    eid = torch.searchsorted(offs, row, right=True).clamp_max(E - 1)       # expert id per sorted row
    slot = eid * cap + (row - starts[eid])                                 # slot within the padded batch

    def pad(t):
        p = t.new_zeros(E * cap, t.shape[1]); p[slot] = t
        return p.view(E, cap, t.shape[1])

    gw = torch.bmm(pad(go_sorted).transpose(1, 2), pad(x_sorted))          # [E, Nout, Nin]
    return gw, counts > 0



def _stacked_dispatch(mats, active, seed_holder):
    """Fire the fused Triton stacked update on CUDA (one launch per matrix, hash-SR, accepts bf16
    grad directly); fall back to the torch batched update elsewhere. mats = [(state, scale, v,
    grad_w, kw), ...]. Same in-family SR switch PackedRMSCounterLinear makes for its fused kernel."""
    from .stacked_update import HAS_TRITON, triton_stacked_update
    if HAS_TRITON and mats and mats[0][0].is_cuda:
        for state, scale, v, gw, kw in mats:
            seed = seed_holder[0]; seed_holder[0] += 1
            triton_stacked_update(state, scale, v, gw, active,
                                  C=kw["C"], lr=kw["lr"], lr_scale=kw["lr_scale"],
                                  rms_beta=kw["beta"], rms_eps=kw["eps"], seed=seed)
        return
    for state, scale, v, gw, kw in mats:
        _batched_rms_update(state, scale, v, gw, active, **kw)


class StackedCounterExperts(nn.Module):
    """All E experts' two counter MLPs in STACKED buffers ([E,out,in]) instead of a ModuleList, so
    the weight decode (forward) and the RMS+SR update (backward) are each ONE vectorized op over the
    whole expert axis -- no python per-expert loop. Combined with the grouped GEMMs, the only
    remaining per-expert step is forming each segment's weight-gradient. Same counter math as
    RMSCounterLinear (exact/eager); state is 1 byte/weight here (unpacked codes, the stack is small)."""

    def __init__(self, E, d, h, *, C, lr, lr_scale, init_gain=1.0, rms_beta=0.9, rms_eps=1e-3,
                 compute_dtype: str = "fp32"):
        super().__init__()
        self.E, self.d, self.h, self.C = E, d, h, C
        self.lr, self.lr_scale = lr, lr_scale
        self.beta, self.eps = rms_beta, rms_eps
        self.compute_dtype = compute_dtype
        self._sr = [0]                                  # per-call seed for the fused hash-SR kernel
        for name, out, in_ in (("1", h, d), ("2", d, h)):
            t0 = torch.randint(-1, 2, (E, out, in_), dtype=torch.int16)
            self.register_buffer(f"s{name}", encode_state(t0, torch.zeros_like(t0), C))
            s0 = init_gain * math.sqrt(3.0 / (2.0 * in_))
            self.register_buffer(f"sc{name}", torch.full((E, out, 1), s0, dtype=torch.float32))
            self.register_buffer(f"v{name}", torch.zeros((E, out, 1), dtype=torch.float32))

    def _eff_dtype(self, ref: torch.Tensor) -> torch.dtype:
        # bf16 GEMM operands ONLY on CUDA with 16-byte-aligned last dims (grouped_mm constraint);
        # the counter update itself always stays fp32 (grad_w is cast back before ticking).
        if self.compute_dtype == "bf16" and ref.is_cuda and self.d % 8 == 0 and self.h % 8 == 0:
            return torch.bfloat16
        return torch.float32

    def weights(self):
        """Decode both stacked states to usable ternary*scale weights, in ONE decode each. Ternary
        t in {-1,0,1} is EXACT in bf16; only the per-row scale rounds -- the profiled fp32-SIMT GEMMs
        move to the bf16 Tensor Cores when compute_dtype='bf16' (parity-gated, GEMM operands only)."""
        dt = self._eff_dtype(self.s1)
        W1 = (self.sc1 * decode_state(self.s1, self.C)[0].float()).to(dt)   # [E,h,d]
        W2 = (self.sc2 * decode_state(self.s2, self.C)[0].float()).to(dt)   # [E,d,h]
        return W1, W2

    @torch.no_grad()
    def update(self, grad_w1, grad_w2, active):
        kw = dict(C=self.C, lr=self.lr, lr_scale=self.lr_scale, beta=self.beta, eps=self.eps)
        _stacked_dispatch([(self.s1, self.sc1, self.v1, grad_w1, kw),
                           (self.s2, self.sc2, self.v2, grad_w2, kw)], active, self._sr)

    def persistent_bytes(self):
        return (self.s1.numel() + self.s2.numel()
                + (self.sc1.numel() + self.sc2.numel() + self.v1.numel() + self.v2.numel()) * 4)


def _silu_grad(x: torch.Tensor) -> torch.Tensor:
    """d/dx of SiLU(x)=x*sigmoid(x) = sig*(1 + x*(1-sig))."""
    s = torch.sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


class StackedSwiGLUExperts(nn.Module):
    """Stacked SwiGLU experts: gate/up (d->h) and down (h->d), all in shared [E,·] buffers so the
    decode (forward) and the RMS+SR counter update (backward) are each one vectorized op over E.
    SwiGLU FFN = down( SiLU(gate(x)) * up(x) ) -- the GLM/Llama expert MLP (3 matrices vs gelu's 2)."""

    def __init__(self, E, d, h, *, C, lr, lr_scale, init_gain=1.0, rms_beta=0.9, rms_eps=1e-3,
                 compute_dtype: str = "fp32"):
        super().__init__()
        self.E, self.d, self.h, self.C = E, d, h, C
        self.lr, self.lr_scale = lr, lr_scale
        self.beta, self.eps = rms_beta, rms_eps
        self.compute_dtype = compute_dtype
        self._sr = [0]                                  # per-call seed for the fused hash-SR kernel
        for name, out, in_ in (("g", h, d), ("u", h, d), ("d", d, h)):    # gate, up, down
            t0 = torch.randint(-1, 2, (E, out, in_), dtype=torch.int16)
            self.register_buffer(f"s{name}", encode_state(t0, torch.zeros_like(t0), C))
            s0 = init_gain * math.sqrt(3.0 / (2.0 * in_))
            self.register_buffer(f"sc{name}", torch.full((E, out, 1), s0, dtype=torch.float32))
            self.register_buffer(f"v{name}", torch.zeros((E, out, 1), dtype=torch.float32))

    _eff_dtype = StackedCounterExperts._eff_dtype

    def weights(self):
        dt = self._eff_dtype(self.sg)
        Wg = (self.scg * decode_state(self.sg, self.C)[0].float()).to(dt)   # [E,h,d]
        Wu = (self.scu * decode_state(self.su, self.C)[0].float()).to(dt)   # [E,h,d]
        Wd = (self.scd * decode_state(self.sd, self.C)[0].float()).to(dt)   # [E,d,h]
        return Wg, Wu, Wd

    @torch.no_grad()
    def update(self, gw_g, gw_u, gw_d, active):
        kw = dict(C=self.C, lr=self.lr, lr_scale=self.lr_scale, beta=self.beta, eps=self.eps)
        _stacked_dispatch([(self.sg, self.scg, self.vg, gw_g, kw),
                           (self.su, self.scu, self.vu, gw_u, kw),
                           (self.sd, self.scd, self.vd, gw_d, kw)], active, self._sr)

    def persistent_bytes(self):
        b = self.sg.numel() + self.su.numel() + self.sd.numel()
        for t in (self.scg, self.scu, self.scd, self.vg, self.vu, self.vd):
            b += t.numel() * 4
        return b


class _StackedSwiGLUFn(torch.autograd.Function):
    """Grouped SwiGLU experts: gate/up/down via torch._grouped_mm + one batched counter update."""

    @staticmethod
    def forward(ctx, x_sorted, offs, tap, stk):
        Wg, Wu, Wd = stk.weights()
        xs = x_sorted.to(Wg.dtype)            # bf16 on CUDA when compute_dtype='bf16', else fp32
        yg = torch._grouped_mm(xs, Wg.transpose(1, 2).contiguous(), offs=offs)   # [M,h]
        yu = torch._grouped_mm(xs, Wu.transpose(1, 2).contiguous(), offs=offs)   # [M,h]
        sg = F.silu(yg)
        a = sg * yu                                                              # [M,h]
        y = torch._grouped_mm(a, Wd.transpose(1, 2).contiguous(), offs=offs)     # [M,d]
        ctx.stk = stk; ctx.offs = offs
        ctx.save_for_backward(xs, yg, yu, a, Wg, Wu, Wd)
        return y.to(x_sorted.dtype)

    @staticmethod
    def backward(ctx, grad_y):
        xs, yg, yu, a, Wg, Wu, Wd = ctx.saved_tensors
        offs = ctx.offs; stk = ctx.stk; E = stk.E
        gy = grad_y.to(Wd.dtype)
        grad_a = torch._grouped_mm(gy, Wd.contiguous(), offs=offs)               # [M,h]
        grad_yu = grad_a * F.silu(yg)
        grad_yg = grad_a * yu * _silu_grad(yg)
        # grad_x flows through BOTH gate and up branches
        grad_x = (torch._grouped_mm(grad_yg, Wg.contiguous(), offs=offs)
                  + torch._grouped_mm(grad_yu, Wu.contiguous(), offs=offs))      # [M,d]
        # per-expert weight-gradients, loop-free (pad + bmm), then ONE batched update over all 3.
        gw_g, active = _grouped_grad_w(grad_yg, xs, offs, E)                     # [E,h,d]
        gw_u, _ = _grouped_grad_w(grad_yu, xs, offs, E)                          # [E,h,d]
        gw_d, _ = _grouped_grad_w(gy, a, offs, E)                               # [E,d,h]
        stk.update(gw_g, gw_u, gw_d, active)   # fp32 cast happens in-kernel / in the fallback
        return grad_x.to(grad_y.dtype), None, None, None


class _StackedGroupedFn(torch.autograd.Function):
    """Grouped forward + grad_x via torch._grouped_mm over the stacked experts, then ONE batched
    counter update. The only per-expert step left is the segment weight-gradient matmul (cheap GEMM);
    the heavy decode/SR/encode is vectorized over E. tap forces backward so experts always update."""

    @staticmethod
    def forward(ctx, x_sorted, offs, tap, stk):
        W1, W2 = stk.weights()                                          # [E,h,d], [E,d,h]
        xs = x_sorted.to(W1.dtype)
        y1 = torch._grouped_mm(xs, W1.transpose(1, 2).contiguous(), offs=offs)   # [M,h]
        a = F.gelu(y1)
        y2 = torch._grouped_mm(a, W2.transpose(1, 2).contiguous(), offs=offs)    # [M,d]
        ctx.stk = stk; ctx.offs = offs
        ctx.save_for_backward(xs, y1, a, W1, W2)
        return y2.to(x_sorted.dtype)

    @staticmethod
    def backward(ctx, grad_y2):
        xs, y1, a, W1, W2 = ctx.saved_tensors
        offs = ctx.offs; stk = ctx.stk; E = stk.E
        gy2 = grad_y2.to(W2.dtype)
        grad_a = torch._grouped_mm(gy2, W2.contiguous(), offs=offs)               # [M,h]
        grad_y1 = grad_a * _gelu_grad(y1)
        grad_x = torch._grouped_mm(grad_y1, W1.contiguous(), offs=offs)           # [M,d]
        # per-expert weight-gradients, loop-free (pad + bmm), then ONE batched update.
        gw1, active = _grouped_grad_w(grad_y1, xs, offs, E)                       # [E,h,d]
        gw2, _ = _grouped_grad_w(gy2, a, offs, E)                                 # [E,d,h]
        stk.update(gw1, gw2, active)           # fp32 cast happens in-kernel / in the fallback                                              # ONE batched update
        return grad_x.to(grad_y2.dtype), None, None, None


def _expert_linear(fin: int, fout: int, *, C: int, lr: float, lr_scale: float, packed: bool):
    """Pick the expert's counter linear. packed=True -> PackedRMSCounterLinear, which on CUDA fires
    the ONE-launch fused Triton update (vs ~15 torch ops) and stores 0.75 B/weight; it needs
    in_features % 4 == 0, so fall back to RMSCounterLinear when the width isn't divisible by 4. The
    learning dynamics are identical on CPU (packed only changes storage + which update path runs)."""
    if packed and fin % 4 == 0:
        return PackedRMSCounterLinear(fin, fout, C=C, lr=lr, lr_scale=lr_scale)
    return RMSCounterLinear(fin, fout, C=C, lr=lr, lr_scale=lr_scale)


class _CounterExpert(nn.Module):
    """A single counter-MLP expert: counter-linear(d->h) -> gelu -> counter-linear(h->d). Holds no
    fp Parameters; both linears are counter-state and self-update in backward. With packed=True the
    two linears are PackedRMSCounterLinear so the per-expert update runs as one fused kernel."""

    def __init__(self, dim: int, hidden: int, *, C: int, lr: float, lr_scale: float,
                 packed: bool = True) -> None:
        super().__init__()
        self.fc1 = _expert_linear(dim, hidden, C=C, lr=lr, lr_scale=lr_scale, packed=packed)
        self.fc2 = _expert_linear(hidden, dim, C=C, lr=lr, lr_scale=lr_scale, packed=packed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _SwiGLUExpert(nn.Module):
    """A SwiGLU counter-MLP expert (GLM/Llama style): down( SiLU(gate(x)) * up(x) ). Three counter
    linears (gate d->h, up d->h, down h->d), all counter-state, self-updating in backward."""

    def __init__(self, dim: int, hidden: int, *, C: int, lr: float, lr_scale: float,
                 packed: bool = True) -> None:
        super().__init__()
        self.gate = _expert_linear(dim, hidden, C=C, lr=lr, lr_scale=lr_scale, packed=packed)
        self.up = _expert_linear(dim, hidden, C=C, lr=lr, lr_scale=lr_scale, packed=packed)
        self.down = _expert_linear(hidden, dim, C=C, lr=lr, lr_scale=lr_scale, packed=packed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class CounterMoEFFN(nn.Module):
    """Sparse Mixture-of-counter-Experts FFN drop-in.

    dim          : model width d.
    n_experts    : E -- total expert count (grows capacity / persistent bytes, not active compute).
    top_k        : experts visited per token (active compute scales with top_k, not E).
    expert_hidden: per-expert hidden width h. Default 4*dim // top_k -> top_k experts ~ dense active
                   MACs (equal-active-compute gate). Pass an explicit value to deviate.
    aux_loss_weight: caller-side weight for the load-balance aux loss (see module docstring). The
                   FFN itself only COMPUTES the aux loss (.last_aux_loss); the training loop scales
                   and adds it. Stored here so the arm carries its own recommended weight.
    """

    def __init__(self, dim: int, n_experts: int = 8, top_k: int = 2,
                 expert_hidden: int | None = None, *, C: int = 11, lr: float = 0.04,
                 lr_scale: float = 2e-4, aux_loss_weight: float = 1e-2,
                 packed_experts: bool = True, grouped: bool = False, swiglu: bool = False,
                 compute_dtype: str = "fp32") -> None:
        super().__init__()
        self.d = int(dim)
        self.E = int(n_experts)
        self.k = int(top_k)
        self.swiglu = bool(swiglu)
        if self.k > self.E:
            raise ValueError(f"top_k ({self.k}) cannot exceed n_experts ({self.E})")
        # Equal-active-compute hidden: gelu (2 matmuls) -> h=4d/top_k; SwiGLU (3 matmuls, gate/up/down)
        # -> h=8d/(3*top_k) so top_k experts still ~ the dense FFN active MACs. Round to a multiple of
        # 8: torch._grouped_mm needs the operand's last dim (h) stride to be a multiple of 16 bytes.
        def _r8(x):
            return max(8, ((int(x) + 7) // 8) * 8)
        if expert_hidden is not None:
            self.h = int(expert_hidden)
        elif self.swiglu:
            self.h = _r8((8 * self.d) / (3 * self.k))
        else:
            self.h = _r8((4 * self.d) / self.k)
        self.aux_loss_weight = float(aux_loss_weight)
        # grouped=True replaces the python per-expert loop with grouped GEMMs (torch._grouped_mm) +
        # stacked experts: all experts' fc1/fc2 run in one launch each AND the counter update is one
        # vectorized op over the expert axis (StackedCounterExperts). Same math as the loop (fp32, up
        # to reduction order + batched-vs-per-expert SR ordering). Needs torch._grouped_mm.
        self.grouped = bool(grouped) and hasattr(torch, "_grouped_mm")

        # The ONLY fp Parameter: the router (a tiny d->E linear). AdamW owns it.
        self.router = nn.Linear(self.d, self.E, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)

        # grouped -> stacked storage (one shared buffer, vectorized decode+update); else a ModuleList
        # of per-expert counter MLPs. packed_experts=True: PackedRMSCounterLinear (fused update + 0.75
        # B/weight) on the loop path. Identical dynamics on CPU.
        if self.grouped:
            Stacked = StackedSwiGLUExperts if self.swiglu else StackedCounterExperts
            self.stacked = Stacked(self.E, self.d, self.h, C=C, lr=lr, lr_scale=lr_scale,
                                   compute_dtype=compute_dtype)
            self.experts = nn.ModuleList()
        else:
            Expert = _SwiGLUExpert if self.swiglu else _CounterExpert
            self.experts = nn.ModuleList(
                Expert(self.d, self.h, C=C, lr=lr, lr_scale=lr_scale, packed=packed_experts)
                for _ in range(self.E)
            )

        # Diagnostics (not optimizer state): O(E) scalars.
        self.register_buffer("token_count", torch.zeros(self.E, dtype=torch.float64),
                             persistent=False)
        # The last forward's load-balance aux loss, for the training loop to add (scaled).
        self.last_aux_loss: torch.Tensor = torch.zeros(())
        # The last forward's per-expert token fraction, for the routing-collapse check.
        self.last_token_fraction: torch.Tensor = torch.zeros(self.E)

    def _aux_loss(self, probs: torch.Tensor, top_idx: torch.Tensor) -> torch.Tensor:
        """Switch-transformer load-balance loss: E * sum_e f_e * P_e, where
        f_e = fraction of tokens routed to e (hard, from the top-k assignment) and
        P_e = mean router probability mass on e (soft). Minimized when both are uniform (1/E)."""
        N = probs.shape[0]
        # P_e: soft mean probability per expert.
        mean_prob = probs.mean(dim=0)                                      # [E]
        # f_e: hard fraction of (token, slot) assignments that picked e.
        one_hot = F.one_hot(top_idx.reshape(-1), num_classes=self.E).to(probs.dtype)  # [N*k, E]
        frac = one_hot.sum(dim=0) / max(N * self.k, 1)                     # [E]
        return self.E * (frac * mean_prob).sum()

    def _route(self, h: torch.Tensor):
        """Shared routing: router -> softmax -> top_k -> renormalized weights, aux loss, diagnostics.
        Returns (flat_tok, flat_exp, flat_w) over the N*k (token, slot) pairs."""
        N = h.shape[0]
        logits = self.router(h)                                           # [N, E]
        probs = torch.softmax(logits, dim=-1)                             # [N, E]
        top_w, top_idx = probs.topk(self.k, dim=-1)                       # [N, k], [N, k]
        top_w = top_w / top_w.sum(dim=-1, keepdim=True).clamp_min(1e-9)   # convex combination
        self.last_aux_loss = self._aux_loss(probs, top_idx)
        with torch.no_grad():
            counts = F.one_hot(top_idx.reshape(-1), num_classes=self.E).sum(dim=0).double()
            self.token_count += counts
            self.last_token_fraction = (counts / max(N * self.k, 1)).float()
        flat_tok = torch.arange(N, device=h.device).repeat_interleave(self.k)  # [N*k] token id
        return flat_tok, top_idx.reshape(-1), top_w.reshape(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.grouped:
            return self._forward_grouped(x)
        sh = x.shape
        h = x.reshape(-1, self.d)                                          # [N, d]
        flat_tok, flat_exp, flat_w = self._route(h)

        # Gather each expert's routed tokens into ONE batch so its two counter layers each see
        # exactly -- and only -- their tokens (exact update, one forward per counter layer per bwd).
        y = torch.zeros_like(h)                                                 # [N, d]
        for e, expert in enumerate(self.experts):
            sel = (flat_exp == e).nonzero(as_tuple=True)[0]                     # slots routed to e
            if sel.numel() == 0:
                continue
            tok_ids = flat_tok[sel]
            out_e = expert(h[tok_ids])                                          # [n_e, d]
            y.index_add_(0, tok_ids, flat_w[sel].unsqueeze(-1) * out_e)
        return y.reshape(sh)

    def _forward_grouped(self, x: torch.Tensor) -> torch.Tensor:
        """Loop-free path: sort the (token, slot) pairs by expert, run all experts as grouped GEMMs
        (one launch each), then the weighted scatter. Routing + weighting + scatter are ordinary
        autograd (router gets its grad); _GroupedExperts owns the grouped matmuls + counter update."""
        sh = x.shape
        h = x.reshape(-1, self.d)                                          # [N, d]
        flat_tok, flat_exp, flat_w = self._route(h)
        order = torch.argsort(flat_exp)                                    # sort pairs by expert
        sorted_tok = flat_tok[order]
        sorted_w = flat_w[order]
        offs = torch.bincount(flat_exp, minlength=self.E).cumsum(0).to(torch.int32)  # per-expert ends
        x_sorted = h[sorted_tok]                                          # [M, d] (autograd-tracked)
        tap = (torch.zeros((), device=h.device, dtype=h.dtype, requires_grad=True)
               if torch.is_grad_enabled() else h.new_zeros(()))           # forces experts to update
        fn = _StackedSwiGLUFn if self.swiglu else _StackedGroupedFn
        y2 = fn.apply(x_sorted, offs, tap, self.stacked)                 # [M, d] unweighted outputs
        weighted = y2 * sorted_w.unsqueeze(-1)                            # router grad flows here
        y = torch.zeros_like(h).index_add(0, sorted_tok, weighted)
        return y.reshape(sh)

    # --- accounting (for the equal-active-compute comparison) ----------------------
    def active_macs_per_token(self) -> int:
        """MACs a single token touches: router (d*E) + its top_k experts. Each gelu expert is 2*d*h
        (fc1+fc2); each SwiGLU expert is 3*d*h (gate+up+down)."""
        per_expert = (3 if self.swiglu else 2) * self.d * self.h
        return self.d * self.E + self.k * per_expert

    def coeff_count(self) -> int:
        """TOTAL counter weights across all E experts (gelu: 2*d*h; SwiGLU: 3*d*h per expert). These
        live in the stacked buffers (grouped) or the ModuleList (loop) -- NOT as CompactCounterLinear
        instances in the grouped path, so a `counter_layers()` scan misses them; count them here."""
        return self.E * (3 if self.swiglu else 2) * self.d * self.h

    def persistent_bytes(self) -> int:
        """Counter experts (the bulk: ~0.75 B/weight visible + per-row scale/v) + fp router."""
        if self.grouped:
            b = self.stacked.persistent_bytes()
        else:
            b = 0
            for expert in self.experts:
                mats = (expert.gate, expert.up, expert.down) if self.swiglu else (expert.fc1, expert.fc2)
                for m in mats:
                    b += m.state.numel() + m.scale.numel() * 4 + m.v.numel() * 4
        b += self.router.weight.numel() * 4                                # fp router
        return b

    @torch.no_grad()
    def routing_report(self) -> dict[str, float]:
        """Cumulative per-expert token fractions + collapse flags (for the witness)."""
        total = float(self.token_count.sum().clamp_min(1))
        frac = (self.token_count / total).tolist()
        return {
            "fractions": frac,
            "min_frac": min(frac),
            "max_frac": max(frac),
            "starved": any(f < 0.01 for f in frac),    # an expert getting < 1% of tokens
            "dominant": any(f > 0.90 for f in frac),   # one expert hogging > 90% of tokens
        }
