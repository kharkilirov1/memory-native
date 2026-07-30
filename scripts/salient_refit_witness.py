"""L1 witness: salient VALUES as a solved corrector instead of exact copies (CPU, minutes).

Today the salient channel stores EXACT COPIES of the original weights at positions chosen
BEFORE the sweep by |w|*sqrt(diag H) (ptq.py `qsal = W0[...]`, pinned comment at ~315).
After the sweep those k% coordinates are the only continuous freedom the format has left —
so their optimal use is to absorb the RESIDUAL layer error, not to replicate w:

    minimize over q_S:  (w - q)^T H (w - q),  q_B fixed (ternary+scales)
    =>  e_S = -H_SS^{-1} H_SB e_B   (per row; S = the row's salient coords)

This is exact linear algebra on tiny per-row SPD systems (~k_o x k_o), zero format change,
zero extra bytes. The witness follows the two-blob gate protocol (the 2048-token layerwise
gate overfits — CLAUDE.md item 7): H_train drives the solve/refit, H_eval judges.

Arms per layer: (a) v3-lite baseline (salient copies), (b) + value refit. Same sweep, same
positions, same scales — the refit effect in isolation.

Run: PYTHONPATH=src python scripts/salient_refit_witness.py
env: MODEL, SALIENT (0.02), GROUP (128), CALIB_BATCHES/CALIB_SEQ (16x256 train, 8x256 eval).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
import torch.nn as nn

from memory_native.donor.ptq import collect_hessians, gptq_group_ternary

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B")
SALIENT = float(os.environ.get("SALIENT", "0.02"))
GROUP = int(os.environ.get("GROUP", "128"))
CALIB_BATCHES = int(os.environ.get("CALIB_BATCHES", "16"))
EVAL_BATCHES = int(os.environ.get("EVAL_BATCHES", "8"))
CALIB_SEQ = int(os.environ.get("CALIB_SEQ", "256"))
PERCDAMP = 0.01
TARGETS = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.mlp.up_proj",
    "model.layers.0.mlp.down_proj",
]


def h_err(W: torch.Tensor, Q: torch.Tensor, H: torch.Tensor) -> float:
    E = (W - Q).double()
    return float(torch.einsum("oi,ij,oj->", E, H.double(), E))


@torch.no_grad()
def refit_salient(W: torch.Tensor, Q: torch.Tensor, H: torch.Tensor,
                  salient_idx: torch.Tensor) -> torch.Tensor:
    """Optimal salient values under H, ternary part fixed. Returns the refit Q."""
    out, cols = W.shape
    damp = PERCDAMP * H.diag().mean()
    Hd = H + damp * torch.eye(cols, dtype=H.dtype)
    Qr = Q.clone()
    rows = salient_idx.long() // cols
    js = salient_idx.long() % cols
    for o in rows.unique().tolist():
        S = js[rows == o]
        if S.numel() == 0:
            continue
        mask = torch.zeros(cols, dtype=torch.bool)
        mask[S] = True
        e = (W[o] - Q[o]).double()
        H_SS = Hd[S][:, S].double()
        H_SB = Hd[S][:, ~mask].double()
        rhs = -(H_SB @ e[~mask])
        try:
            e_S = torch.linalg.solve(H_SS, rhs)
        except RuntimeError:
            e_S = torch.linalg.lstsq(H_SS, rhs).solution
        Qr[o, S] = (W[o, S].double() - e_S).to(Q.dtype)
    return Qr


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.model.layers = nn.ModuleList(list(model.model.layers)[:1])
    model.config.num_hidden_layers = 1
    model.eval()

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]

    def windows(offset, n):
        return [ids[offset + i * CALIB_SEQ: offset + (i + 1) * CALIB_SEQ].unsqueeze(0)
                for i in range(n)]

    train_b = windows(0, CALIB_BATCHES)
    eval_b = windows(CALIB_BATCHES * CALIB_SEQ + 50_000, EVAL_BATCHES)

    H_train = collect_hessians(model, TARGETS, train_b)
    H_eval = collect_hessians(model, TARGETS, eval_b)

    print(f"model={MODEL} salient={SALIENT} group={GROUP} "
          f"train={CALIB_BATCHES}x{CALIB_SEQ} eval={EVAL_BATCHES}x{CALIB_SEQ} (+50k offset)")
    print(f"{'layer':<34} {'arm':<12} {'rel err (train H)':>18} {'rel err (EVAL H)':>18}")
    for path in TARGETS:
        mod = model.get_submodule(path)
        W = mod.weight.detach().float()
        Ht, He = H_train[path].float(), H_eval[path].float()
        base = float(torch.einsum("oi,ij,oj->", W.double(), Ht.double(), W.double()))
        base_e = float(torch.einsum("oi,ij,oj->", W.double(), He.double(), W.double()))

        Q, S_out, T, perm, Wadj, (sal_idx, sal_val) = gptq_group_ternary(
            W, Ht, group=GROUP, act_order=True, refine_iters=2, scale_refit="align",
            grid="itf", salient_first=SALIENT, salient_scope="layer",
            return_perm=True, return_salient=True,
        )
        Qr = refit_salient(W, Q, Ht, sal_idx)
        for arm, q in (("copy", Q), ("refit", Qr)):
            print(f"{path:<34} {arm:<12} {h_err(W, q, Ht) / base:>18.6f} "
                  f"{h_err(W, q, He) / base_e:>18.6f}")
        moved = (Qr.reshape(-1)[sal_idx.long()] - Q.reshape(-1)[sal_idx.long()])
        print(f"{'':<34} {'delta':<12} salient values moved: mean|d|="
              f"{moved.abs().mean():.4e} max|d|={moved.abs().max():.4e}")


if __name__ == "__main__":
    main()
