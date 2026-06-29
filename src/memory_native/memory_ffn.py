"""CounterMemoryFFN -- the architecture lever (verification-plan M1).

Replace a dense FFN (two d x 4d GEMMs, every weight touched by every token) with a *retrieval*
memory: a token forms a query, retrieves its top-k cells out of E via a product-key index, and
reads out a weighted sum of those k value rows. The value table is counter-state (0.75 byte/weight),
so the table can be made huge (capacity) WITHOUT growing the per-token active compute -- the index
is product-key, so retrieval costs O(sqrt(E)) dot-products, not O(E).

Why this fits the counter method exactly (no approximation, no bias):
    A cell that a token did NOT retrieve contributed nothing to that token's output, so its gradient
    is EXACTLY zero. Updating only the retrieved rows is therefore the exact gradient, not a sparse
    approximation. The counter optimizer lives in the value table; only retrieved rows tick.

Layout (single memory head):
    query q = W_q ln(x)              -> split into two half-queries q1,q2 in R^{dk}
    sub-keys K1,K2 in R^{m x dk}     (m = sqrt(E) per half; the full key set is the m x m product)
    per half: top-k over m sub-keys; combine the two half-scorelists -> top-k over the m^2 cells
    weights = softmax(top-k scores); y = sum_j weights_j * Value[cell_j]

The router (W_q, K1, K2) are small fp Parameters trained by the outer optimizer (negligible:
O(d*dk + sqrt(E)*dk)). The value table is the bulk and is counter-state, self-updating in backward.
Pure PyTorch, CPU/CUDA.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .counter import decode_state, encode_state, stochastic_round

__all__ = ["CounterMemoryFFN", "CounterValueMemory"]


class _CounterValueRead(torch.autograd.Function):
    """Reads counter value rows for the given flat cell ids and, on backward, applies the counter
    RMS+SR update to exactly those rows (the exact-for-active gradient). Returns no grad to its
    inputs: the value table is counter-state, not an fp leaf -- it self-updates here."""

    @staticmethod
    def forward(ctx, mem, flat_ids, tap):
        # mem: CounterValueMemory; flat_ids: [P] int64 of cells to read (P = M*k, with repeats).
        rows = mem._decode_rows(flat_ids)          # [P, d] fp visible value rows
        ctx.mem = mem
        ctx.save_for_backward(flat_ids)
        return rows

    @staticmethod
    def backward(ctx, grad_rows):
        (flat_ids,) = ctx.saved_tensors
        if ctx.mem.update_enabled:
            ctx.mem._update_active(flat_ids, grad_rows)
        return None, None, None


class CounterValueMemory(nn.Module):
    """E value rows of width d, stored as counter state (ternary t + counter c, per-row scale).
    Retrieval gathers rows by id; the update ticks ONLY the rows that were read (exact-for-active).
    The RMS+SR tick mirrors RMSCounterLinear, but over a scattered row subset instead of a tile."""

    def __init__(self, n_cells: int, dim: int, *, C: int = 11, lr: float = 0.04,
                 lr_scale: float = 2e-4, init_gain: float = 1.0, rms_beta: float = 0.9,
                 rms_eps: float = 1e-3, value_compute: str = "counter") -> None:
        super().__init__()
        self.E = int(n_cells)
        self.d = int(dim)
        self.C = int(C)
        self.lr = float(lr)
        self.lr_scale = float(lr_scale)
        self.rms_beta = float(rms_beta)
        self.rms_eps = float(rms_eps)
        self.update_enabled = True
        if value_compute not in {"counter", "fp"}:
            raise ValueError("value_compute must be 'counter' or 'fp'")
        self.value_compute = value_compute

        t0 = torch.randint(-1, 2, (self.E, self.d), dtype=torch.int16)
        c0 = torch.zeros_like(t0)
        self.register_buffer("state", encode_state(t0, c0, C))
        s0 = init_gain * math.sqrt(3.0 / (2.0 * self.d))
        self.register_buffer("scale", torch.full((self.E, 1), s0, dtype=torch.float32))
        self.register_buffer("v", torch.zeros((self.E, 1), dtype=torch.float32))
        if value_compute == "fp":   # diagnostic arm: fp values (isolates architecture from quant)
            self.fp_values = nn.Parameter(torch.randn(self.E, self.d) * s0)
        self.register_buffer("rows_touched", torch.zeros((), dtype=torch.int64), persistent=False)
        # cell-utilization diagnostics (the M2-style 'ever visible' for the memory): which cells are
        # EVER retrieved, and how often. Reveals router starvation (dead cells) at large E.
        self.register_buffer("ever_retrieved", torch.zeros(self.E, dtype=torch.bool), persistent=False)
        self.register_buffer("retrieval_count", torch.zeros(self.E, dtype=torch.int64), persistent=False)

    @torch.no_grad()
    def _note_use(self, flat_ids: torch.Tensor) -> None:
        self.ever_retrieved[flat_ids] = True
        self.retrieval_count.index_add_(0, flat_ids, torch.ones_like(flat_ids))

    def live_fraction(self) -> float:
        """Fraction of cells ever retrieved (1.0 = all used; low = router starves most cells)."""
        return float(self.ever_retrieved.float().mean())

    @torch.no_grad()
    def _decode_rows(self, flat_ids: torch.Tensor) -> torch.Tensor:
        if self.value_compute == "fp":
            return self.fp_values[flat_ids]
        t, _ = decode_state(self.state[flat_ids], self.C)
        return self.scale[flat_ids] * t.to(self.scale.dtype)

    def read(self, flat_ids: torch.Tensor, tap: torch.Tensor) -> torch.Tensor:
        self._note_use(flat_ids)
        if self.value_compute == "fp":
            return self.fp_values[flat_ids]   # plain autograd updates fp values
        return _CounterValueRead.apply(self, flat_ids, tap)

    @torch.no_grad()
    def _update_active(self, flat_ids: torch.Tensor, grad_rows: torch.Tensor) -> None:
        """Aggregate per-(read) gradients into per-UNIQUE-cell gradients, then RMS+SR-tick each
        touched cell once. A cell read by several tokens accumulates their grads first (the true
        gradient of a shared row), so it ticks once with the summed signal -- not once per read."""
        uniq, inv = torch.unique(flat_ids, return_inverse=True)            # [U], [P]
        g = torch.zeros((uniq.shape[0], self.d), dtype=grad_rows.dtype, device=grad_rows.device)
        g.index_add_(0, inv, grad_rows)                                    # [U, d] summed gradient
        self.rows_touched += uniq.numel()

        t, c = decode_state(self.state[uniq], self.C)
        t = t.to(torch.float32); c = c.to(torch.float32)
        s_i = self.scale[uniq]                                            # [U,1]

        g_sq = g.pow(2).mean(dim=1, keepdim=True)
        self.v[uniq] = self.rms_beta * self.v[uniq] + (1.0 - self.rms_beta) * g_sq
        denom = self.v[uniq].sqrt().clamp_min(self.rms_eps)
        grad_eff = g / denom

        grad_s = (g * t).sum(dim=1, keepdim=True) / math.sqrt(self.d)
        s_new = (s_i - self.lr_scale * grad_s).clamp_(1e-5, 10.0)

        c_rebased = c * (s_i / s_new)
        ticks = (-self.lr * grad_eff) * (self.C / s_new)
        cc = stochastic_round(c_rebased + ticks)
        carry = torch.trunc(cc / self.C)
        remainder = cc - carry * self.C
        proposed_t = t + carry
        new_t = proposed_t.clamp_(-1, 1)
        blocked = proposed_t != new_t
        remainder = torch.where(blocked, torch.sign(cc) * (self.C - 1), remainder
                                ).clamp_(-(self.C - 1), self.C - 1)
        self.state[uniq] = encode_state(new_t.to(torch.int16), remainder.to(torch.int16), self.C)
        self.scale[uniq] = s_new

    def persistent_bytes(self) -> int:
        if self.value_compute == "fp":
            return self.fp_values.numel() * 4
        return self.state.numel() + self.scale.numel() * 4 + self.v.numel() * 4


class CounterMemoryFFN(nn.Module):
    """Product-key memory used as an FFN replacement: ln(x) -> retrieve top-k of E cells -> readout.

    n_cells must be a perfect square (E = m^2; m sub-keys per half). active compute per token is
    O(d*dk + sqrt(E)*dk + k*d) -- sublinear in E, so capacity scales with E at fixed active FLOPs.
    """

    def __init__(self, dim: int, n_cells: int = 16384, k: int = 8, key_dim: int = 32, *,
                 C: int = 11, lr: float = 0.04, lr_scale: float = 2e-4,
                 value_compute: str = "counter") -> None:
        super().__init__()
        m = int(round(math.sqrt(n_cells)))
        if m * m != n_cells:
            raise ValueError(f"n_cells must be a perfect square, got {n_cells}")
        self.d = int(dim)
        self.m = m
        self.E = n_cells
        self.k = int(k)
        self.dk = int(key_dim)
        self.query = nn.Linear(dim, 2 * self.dk, bias=False)
        self.k1 = nn.Parameter(torch.randn(m, self.dk) / math.sqrt(self.dk))
        self.k2 = nn.Parameter(torch.randn(m, self.dk) / math.sqrt(self.dk))
        self.values = CounterValueMemory(n_cells, dim, C=C, lr=lr, lr_scale=lr_scale,
                                         value_compute=value_compute)

    def _retrieve(self, q: torch.Tensor):
        """q: [N, 2*dk] -> (weights [N,k], flat_ids [N,k]) via product-key top-k."""
        q1, q2 = q[:, :self.dk], q[:, self.dk:]
        s1 = q1 @ self.k1.t()                              # [N, m]
        s2 = q2 @ self.k2.t()
        ks = min(self.k, self.m)
        v1, i1 = s1.topk(ks, dim=1)                        # [N, ks]
        v2, i2 = s2.topk(ks, dim=1)
        # combine the two half-scorelists -> scores over the ks*ks candidate cells, take top-k
        cand = v1[:, :, None] + v2[:, None, :]            # [N, ks, ks]
        cand_id = i1[:, :, None] * self.m + i2[:, None, :]  # flat cell id within E
        cand = cand.reshape(cand.shape[0], -1)
        cand_id = cand_id.reshape(cand_id.shape[0], -1)
        sc, sel = cand.topk(self.k, dim=1)                # [N, k]
        flat_ids = torch.gather(cand_id, 1, sel)          # [N, k]
        weights = torch.softmax(sc, dim=1)                # [N, k]
        return weights, flat_ids

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sh = x.shape
        h = x.reshape(-1, self.d)                          # [N, d]
        q = self.query(h)                                  # [N, 2dk]
        weights, flat_ids = self._retrieve(q)              # [N,k], [N,k]
        tap = (torch.zeros((), device=x.device, dtype=x.dtype, requires_grad=True)
               if torch.is_grad_enabled() else x.new_zeros(()))
        rows = self.values.read(flat_ids.reshape(-1), tap).reshape(h.shape[0], self.k, self.d)
        y = (weights.unsqueeze(-1) * rows).sum(dim=1)      # [N, d]
        return y.reshape(sh)

    def active_macs_per_token(self) -> int:
        """Multiply-accumulates touched per token (for the equal-active-compute comparison)."""
        ks = min(self.k, self.m)
        return (self.d * 2 * self.dk          # query projection
                + 2 * self.m * self.dk        # sub-key scores (the sqrt(E) term)
                + ks * ks                      # candidate combine
                + self.k * self.d)            # value readout

    def live_fraction(self) -> float:
        """Fraction of cells ever retrieved -- router-starvation diagnostic (low = many dead cells)."""
        return self.values.live_fraction()

    def persistent_bytes(self) -> int:
        """Counter value table (the bulk) + the small fp router (query proj + the two sub-key sets)."""
        router = (self.query.weight.numel() + self.k1.numel() + self.k2.numel()) * 4
        return self.values.persistent_bytes() + router
