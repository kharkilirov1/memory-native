"""Transparent training-memory budget (the deep-v2 accounting model, engine-independent).

Not an allocator profiler -- a symbolic budget separating the four pools (persistent
weights/state, gradients, optimizer state, saved activations) so you can see where the next
memory wall is. Reproduces the deep-v2 numbers: for L=24, d=2048, V=50257, seq=1024 it gives
BF16-AdamW ~21.10 GiB vs finite-state(6b)+reversible(4b acts) ~2.06 GiB vs ternary LB ~1.44 GiB.
"""
from __future__ import annotations

from dataclasses import dataclass

GIB = 1024 ** 3
TERNARY_BITS = 1.5849625007211563  # log2(3)


@dataclass
class BudgetRow:
    name: str
    persistent_gib: float
    grad_gib: float
    optim_gib: float
    acts_gib: float
    misc_gib: float

    @property
    def total_gib(self) -> float:
        return self.persistent_gib + self.grad_gib + self.optim_gib + self.acts_gib + self.misc_gib


def _act_bytes(layers, batch, seq, d, bytes_per_elem, multiplier, policy, anchor_every):
    token = batch * seq * d * bytes_per_elem * multiplier
    if policy == "store_all":
        return layers * token
    if policy == "selective_half":
        return 0.5 * layers * token
    if policy == "reversible":
        return (1 + layers // max(anchor_every, 1)) * token
    if policy == "none_lower_bound":
        return 0.0
    raise ValueError(f"unknown activation policy {policy!r}")


def training_budget(*, layers=24, d_model=2048, vocab=50257, linear_param_factor=12.0,
                    tied_lm_head=True, batch=1, seq=1024, state_bits=6.0, embedding_bits=16.0,
                    counter_act_bits=4.0, row_state_bytes=8.0, baseline_act_bytes=2.0,
                    activation_multiplier=6.0, baseline_policy="store_all",
                    counter_policy="reversible", anchor_every=8, misc_gib=1.0):
    """Return [BF16 AdamW, finite-state, ternary lower-bound] BudgetRows for the given shape."""
    n_lin = int(round(linear_param_factor * layers * d_model * d_model))
    n_emb = vocab * d_model * (1 if tied_lm_head else 2)
    n = n_lin + n_emb
    rows = int(round(linear_param_factor * layers * d_model))

    base_acts = _act_bytes(layers, batch, seq, d_model, baseline_act_bytes,
                           activation_multiplier, baseline_policy, anchor_every)
    counter_acts = _act_bytes(layers, batch, seq, d_model, counter_act_bits / 8.0,
                              activation_multiplier, counter_policy, anchor_every)

    baseline = BudgetRow("BF16 AdamW-style", 2 * n / GIB, 2 * n / GIB, 12 * n / GIB,
                         base_acts / GIB, misc_gib)
    cstate = (state_bits / 8.0) * n_lin + row_state_bytes * rows + embedding_bits / 8.0 * n_emb
    counter = BudgetRow(f"finite-state {state_bits:g}b + {counter_act_bits:g}b acts",
                        cstate / GIB, 0.0, 0.0, counter_acts / GIB, misc_gib)
    lb_state = (TERNARY_BITS / 8.0) * n_lin + embedding_bits / 8.0 * n_emb
    lower = BudgetRow("visible ternary entropy lower-bound", lb_state / GIB, 0.0, 0.0,
                      counter_acts / GIB, misc_gib)
    return [baseline, counter, lower]


def format_budget(rows) -> str:
    lines = ["| regime | persistent | grad | optim | acts | misc | total (GiB) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r.name} | {r.persistent_gib:.3f} | {r.grad_gib:.3f} | {r.optim_gib:.3f} | "
                     f"{r.acts_gib:.3f} | {r.misc_gib:.3f} | {r.total_gib:.3f} |")
    return "\n".join(lines)
