"""Fused MoE experts (HF Gemma 4 / Qwen3-MoE layout) -> per-expert LatentGridLinear.

HF stores the routed experts of one layer as two 3-D parameters (~91 % of gemma-4-26B-A4B's weights, not nn.Linear):
    gate_up_proj [E, 2*I, H]   rows 0..I = gate, I..2I = up (chunk(2, -1) of the output)
    down_proj    [E, H, I]
and `experts(hidden[T,H], top_k_index[T,K], top_k_weights[T,K]) -> [T,H]` loops over the experts that got tokens:
    gate, up = x[tok] @ gate_up[e].T .chunk(2); y = (act(gate) * up) @ down[e].T * top_k_weights[tok, pos]; out.index_add_(tok, y)
`LGExperts` computes exactly that with one LatentGridLinear `gate_up` and one `down` per expert (the router, the shared
expert `mlp`, norms and attention stay outside; use convert_to_lg for the Linear ones). Each expert is called at most ONCE per
forward with all its tokens batched, so it gets at most one backward per micro-batch (needed by the sparse accumulation:
lg_accum.set_momentum(..., sparse=True)).

Use `moe_skip()` as `skip` of convert_to_lg (it must not convert the router projection or non-text towers).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

_MX6 = Path(__file__).resolve().parents[3] / "research" / "mxfp4-native-no-state" / "mxfp4-native"
if str(_MX6) not in sys.path:
    sys.path.insert(0, str(_MX6))

from mx6.latent_grid import LatentGridLinear  # noqa: E402


def moe_skip(extra: tuple[str, ...] = ()) -> tuple[str, ...]:
    """`skip` for convert_to_lg on an MoE model: keep embeddings / lm_head / the router (Gemma4TextRouter.proj) / vision+audio."""
    return ("lm_head", "embed", "router", "vision", "audio") + tuple(extra)


def is_fused_experts(mod: nn.Module) -> bool:
    gu, dn = getattr(mod, "gate_up_proj", None), getattr(mod, "down_proj", None)
    return (isinstance(gu, nn.Parameter) and isinstance(dn, nn.Parameter) and gu.dim() == 3 and dn.dim() == 3
            and hasattr(mod, "act_fn"))


class LGExperts(nn.Module):
    """Drop-in replacement of Gemma4TextExperts / Qwen3MoeExperts with per-expert LatentGridLinear layers."""

    def __init__(self, gate_up: list[LatentGridLinear], down: list[LatentGridLinear], act_fn, intermediate_dim: int):
        super().__init__()
        self.num_experts = len(gate_up)
        self.hidden_dim = gate_up[0].in_features
        self.intermediate_dim = intermediate_dim
        self.gate_up = nn.ModuleList(gate_up)
        self.down = nn.ModuleList(down)
        self.act_fn = act_fn

    @classmethod
    def from_fused(cls, experts: nn.Module, k: int = 6, headroom: float = 2.0, lr: float = 0.001,
                   backend: str = "torch") -> "LGExperts":
        kw = dict(k=k, headroom=headroom, lr=lr, backend=backend, update_mode="exact", rms_beta=0.9, rms_eps=1e-3)
        gu, dn = experts.gate_up_proj.detach(), experts.down_proj.detach()
        E = gu.shape[0]
        gate_up = [LatentGridLinear.from_dense(gu[e], **kw) for e in range(E)]
        down = [LatentGridLinear.from_dense(dn[e], **kw) for e in range(E)]
        return cls(gate_up, down, experts.act_fn, dn.shape[2])

    def expert_dense(self, e: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(gate_up [2I,H], down [H,I]) as deployed (decoded from the latent index), fp32."""
        return (self.gate_up[e].decode_weight(torch.float32), self.down[e].decode_weight(torch.float32))

    def to_fused(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Decoded [E,2I,H], [E,H,I] tensors (for tests / export)."""
        gu, dn = zip(*(self.expert_dense(e) for e in range(self.num_experts)))
        return torch.stack(gu), torch.stack(dn)

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        seen = set()
        for expert_idx in expert_hit:
            e = int(expert_idx[0])
            if e == self.num_experts:
                continue
            assert e not in seen, "an expert must be called once per forward (all its tokens batched)"
            seen.add(e)
            top_k_pos, token_idx = torch.where(expert_mask[e])
            gate, up = self.gate_up[e](hidden_states[token_idx]).chunk(2, dim=-1)
            y = self.down[e](self.act_fn(gate) * up)
            y = y * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, y.to(final_hidden_states.dtype))
        return final_hidden_states


def convert_moe_experts_to_lg(model: nn.Module, k: int = 6, headroom: float = 2.0, lr: float = 0.001,
                              backend: str = "torch") -> dict:
    """Replace every fused experts module of `model` by an LGExperts. Run BEFORE convert_to_lg / set_momentum.
    Returns {n_modules, n_experts, n_layers, params_in_lg, report_lines}."""
    targets = [(n, m) for n, m in model.named_modules() if is_fused_experts(m)]
    n_exp = n_par = 0
    lines = []
    for name, mod in targets:
        parent = model
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        new = LGExperts.from_fused(mod, k=k, headroom=headroom, lr=lr, backend=backend)
        setattr(parent, parts[-1], new)
        n_exp += new.num_experts
        n_par += mod.gate_up_proj.numel() + mod.down_proj.numel()
        lines.append(f"  {name}: {new.num_experts} experts (gate_up {tuple(mod.gate_up_proj.shape[1:])}, "
                     f"down {tuple(mod.down_proj.shape[1:])}) -> LG k={k}")
    return {"n_modules": len(targets), "n_experts": n_exp, "n_layers": 2 * n_exp, "params_in_lg": n_par,
            "report_lines": lines}
