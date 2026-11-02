"""MoE mode: fused experts -> per-expert lg layers (rl/convert_moe.py), sparse momentum accumulation (mx6/lg_accum), and one
GRPO update on a tiny random Gemma 4 MoE (the real transformers config/model classes; 8 experts, top-2)."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

_MX6 = Path(__file__).resolve().parents[3] / "research" / "mxfp4-native-no-state" / "mxfp4-native"
if str(_MX6) not in sys.path:
    sys.path.insert(0, str(_MX6))

pytest.importorskip("transformers")
from transformers import Gemma4ForCausalLM, Gemma4TextConfig  # noqa: E402

from mx6 import lg_accum  # noqa: E402
from mx6.latent_grid import LatentGridLinear, lg_layers  # noqa: E402
from mx6.lg_accum import lg_accumulate, set_momentum  # noqa: E402
from rl.convert import convert_to_lg  # noqa: E402
from rl.convert_moe import LGExperts, convert_moe_experts_to_lg, is_fused_experts, moe_skip  # noqa: E402

E, K, H, I = 8, 2, 64, 32


def tiny_moe(seed: int = 0) -> Gemma4ForCausalLM:
    torch.manual_seed(seed)
    cfg = Gemma4TextConfig(
        vocab_size=128, hidden_size=H, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, vocab_size_per_layer_input=128, hidden_size_per_layer_input=16,
        enable_moe_block=True, num_experts=E, top_k_experts=K, moe_intermediate_size=I, sliding_window=16,
        max_position_embeddings=64, layer_types=["sliding_attention", "full_attention"])
    m = Gemma4ForCausalLM(cfg)
    for n, p in m.named_parameters():
        if p.dim() == 3:                                       # fused experts: make sure they are not uninitialised
            torch.nn.init.normal_(p, std=0.08)
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval()


def experts_of(model):
    return [l.experts for l in model.model.layers]


def test_experts_are_fused_3d_params():
    m = tiny_moe()
    ex = experts_of(m)[0]
    assert is_fused_experts(ex)
    assert tuple(ex.gate_up_proj.shape) == (E, 2 * I, H) and tuple(ex.down_proj.shape) == (E, H, I)


def test_converted_forward_is_exact():
    """The re-implemented forward computes the same function: logits == original model whose fused tensors are replaced by the
    DECODED lg weights (so only the k=6 quantisation differs from the dense model, not the routing / activation / combine)."""
    m = tiny_moe()
    ids = torch.randint(0, 128, (3, 12))
    with torch.no_grad():
        ref_dense = m(input_ids=ids).logits
    lg_model = copy.deepcopy(m)
    rep = convert_moe_experts_to_lg(lg_model, k=6, headroom=2.0, lr=0.01, backend="torch")
    assert rep["n_modules"] == 2 and rep["n_experts"] == 2 * E and rep["n_layers"] == 4 * E
    assert all(isinstance(e, LGExperts) for e in experts_of(lg_model))
    assert not any(is_fused_experts(mod) for mod in lg_model.modules())
    quant_ref = copy.deepcopy(m)                               # same model, quantisation only
    for e_ref, e_lg in zip(experts_of(quant_ref), experts_of(lg_model)):
        gu, dn = e_lg.to_fused()
        e_ref.gate_up_proj.data.copy_(gu)
        e_ref.down_proj.data.copy_(dn)
    with torch.no_grad():
        out_lg = lg_model(input_ids=ids).logits
        out_q = quant_ref(input_ids=ids).logits
    assert (out_q - out_lg).abs().max().item() < 1e-4
    # sanity: the quantisation itself is a visible, modest perturbation of the dense model (so the check above is not vacuous)
    d = (ref_dense - out_q).abs().max().item()
    assert 1e-6 < d < 0.5 * ref_dense.abs().max().item()


def test_full_conversion_skips_router_and_converts_shared_expert():
    m = tiny_moe()
    convert_moe_experts_to_lg(m, lr=0.01)
    rep = convert_to_lg(m, lr=0.01, momentum=0.0, backend="torch", skip=moe_skip())
    layer = m.model.layers[0]
    assert isinstance(layer.router.proj, torch.nn.Linear)       # router stays dense
    assert isinstance(layer.mlp.gate_proj, LatentGridLinear)    # shared expert converted
    assert isinstance(layer.self_attn.q_proj, LatentGridLinear)
    assert rep["n_layers"] > 0 and len(lg_layers(m)) == rep["n_layers"] + 4 * E


# ── sparse accumulation ────────────────────────────────────────────────────────────────────────────────────

def micro_inputs(hits_per_micro, T=6, seed=0):
    """For each micro-batch: (hidden[T,H], top_k_index[T,K], top_k_weights[T,K]) where exactly the experts in `hits` are hit."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for hits in hits_per_micro:
        hits = list(hits)
        idx = torch.tensor([[hits[(2 * t) % len(hits)], hits[(2 * t + 1) % len(hits)]] for t in range(max(T, len(hits)))])
        idx = torch.where(idx[:, :1] == idx[:, 1:], (idx + 1) % E, idx) if len(hits) == 1 else idx
        x = torch.randn(idx.shape[0], H, generator=g)
        w = torch.rand(idx.shape, generator=g) + 0.5
        out.append((x, idx, w))
    return out


def hit_set(idx):
    return set(idx.flatten().tolist())


def make_experts(seed=0, kind="fp32", sparse=True, n=3, mu=0.9, lr=0.05):
    torch.manual_seed(seed)
    ex = tiny_moe(seed)
    convert_moe_experts_to_lg(ex, lr=lr)
    lg = experts_of(ex)[0]
    st = set_momentum(lg, mu, kind, n=n, sparse=sparse)
    return lg, st


def run_step(lg, micros, n):
    with lg_accumulate(lg, n) as acc:
        for i in acc.micro_batches():
            x, idx, w = micros[i]
            (lg(x, idx, w).pow(2).mean() / n).backward()


def snapshot(lg):
    return [(m.idx.clone(), m._acc_buf.clone(), None if not m._acc_scale.numel() else m._acc_scale.clone(), m.v_rms.clone())
            for m in lg_layers(lg)]


def expert_layers(lg, e):
    return lg.gate_up[e], lg.down[e]


def test_sparse_accumulate_hits_0_1_2_3():
    n = 3
    lg, st = make_experts(n=n)
    plan = [{0, 1, 2}, {0, 1, 3}, {0, 4}]                       # expert hits: e0:3, e1:2, e2:1, e3:1, e4:1, e5..e7:0
    micros = micro_inputs(plan)
    for (x, idx, w), p in zip(micros, plan):
        assert hit_set(idx) == p
    run_step(lg, micros, n)                                     # step 1 (primes buffers; no exception)
    before = {id(m): s for m, s in zip(lg_layers(lg), snapshot(lg))}
    n_upd0 = st.n_updates
    micros2 = micro_inputs(plan, seed=1)
    run_step(lg, micros2, n)                                    # step 2: untouched experts must stay bit-identical
    after = {id(m): s for m, s in zip(lg_layers(lg), snapshot(lg))}
    hits = {0: 3, 1: 2, 2: 1, 3: 1, 4: 1}
    for e in range(E):
        for m in expert_layers(lg, e):
            b, a = before[id(m)], after[id(m)]
            if e in hits:
                assert not torch.equal(b[1], a[1]), f"expert {e}: momentum buffer must change"
                assert (b[0] != a[0]).any(), f"expert {e}: idx must be updated"
            else:
                assert torch.equal(b[0], a[0]) and torch.equal(b[1], a[1]) and torch.equal(b[3], a[3]), \
                    f"expert {e} (0 hits) must be untouched: no decay, no update, no RMS-EMA step"
    assert st.n_updates - n_upd0 == 2 * len(hits)               # one update per touched layer


def test_sparse_matches_dense_when_everything_is_hit():
    n = 3
    plan = [set(range(E))] * n
    micros = micro_inputs(plan)
    lg_d, st_d = make_experts(sparse=False, n=n)
    lg_s, st_s = make_experts(sparse=True, n=n)
    for step in range(2):
        micros = micro_inputs(plan, seed=step)
        torch.manual_seed(100 + step)
        run_step(lg_d, micros, n)
        torch.manual_seed(100 + step)
        run_step(lg_s, micros, n)
    for md, ms in zip(lg_layers(lg_d), lg_layers(lg_s)):
        assert torch.allclose(md._acc_buf, ms._acc_buf, atol=1e-6, rtol=1e-4)     # same momentum
        assert torch.allclose(md.v_rms, ms.v_rms, atol=1e-7, rtol=1e-5)
        diff = (md.idx != ms.idx).float().mean().item()
        assert diff < 1e-3, f"sparse and dense updates should coincide (SR draws may differ in order): {diff}"


def test_dense_mode_still_raises_on_missing_backward():
    n = 3
    lg, _ = make_experts(sparse=False, n=n)
    with pytest.raises(RuntimeError, match="lg backward calls"):
        run_step(lg, micro_inputs([{0, 1}, {0, 2}, {0, 3}]), n)


def test_sparse_error_path_resets_touched_buffers_only():
    n = 3
    lg, _ = make_experts(n=n)
    run_step(lg, micro_inputs([{0, 1}, {2, 3}, {4, 5}]), n)
    snap = {id(m): m._acc_buf.clone() for m in lg_layers(lg)}
    with pytest.raises(ZeroDivisionError):
        with lg_accumulate(lg, n) as acc:
            for i in acc.micro_batches():
                x, idx, w = micro_inputs([{0, 6}] * 3)[i]
                (lg(x, idx, w).pow(2).mean()).backward()
                1 / 0
    assert torch.count_nonzero(lg.gate_up[0]._acc_buf) == 0      # touched: reset
    assert torch.equal(lg.gate_up[2]._acc_buf, snap[id(lg.gate_up[2])])   # untouched: kept


def test_two_backwards_of_a_layer_in_one_micro_batch_raise():
    lay = LatentGridLinear(64, 32, k=6, lr=0.01)
    set_momentum(lay, 0.9, "fp32", n=2, sparse=True)
    x = torch.randn(8, 64)
    with pytest.raises(RuntimeError, match="same micro-batch"):
        with lg_accumulate(lay, 2) as acc:
            for i in acc.micro_batches():
                lay(x).pow(2).mean().backward()
                if i == 0:
                    lay(x).pow(2).mean().backward()


def test_sparse_guards():
    lay = LatentGridLinear(64, 32, k=6)
    with pytest.raises(ValueError):
        set_momentum(lay, 0.9, nesterov=True, n=2, sparse=True)
    with pytest.raises(ValueError):
        set_momentum(lay, 0.9, n=2, sparse=True, grad_allreduce=lambda g: g)
    set_momentum(lay, 0.9, "fp32", n=2, sparse=True)
    with pytest.raises(RuntimeError, match="lg_accumulate"):     # outside the context: refused
        lay(torch.randn(4, 64)).pow(2).mean().backward()


def test_sparse_custom_op_path_matches_module_path():
    """The mx6::lg_linear_acc op path (what torch.compile uses) defers the update the same way."""
    from mx6 import latent_grid
    if not (latent_grid.HAS_LG_OPS and latent_grid.HAS_LG_ACC_OPS):
        pytest.skip("custom ops unavailable")
    torch.manual_seed(0)
    a = LatentGridLinear(64, 32, k=6, lr=0.05, native=True)
    b = copy.deepcopy(a)
    b.use_custom_op = True
    for lay in (a, b):
        set_momentum(lay, 0.9, "fp32", n=2, sparse=True)
    x = torch.randn(16, 64)
    for i_step in range(2):
        for lay in (a, b):
            torch.manual_seed(7 + i_step)
            with lg_accumulate(lay, 2) as acc:
                for i in acc.micro_batches():
                    (lay(x[i * 8:(i + 1) * 8]).pow(2).mean() / 2).backward()
    assert torch.allclose(a._acc_buf, b._acc_buf, atol=1e-6) and torch.equal(a.idx, b.idx)


# ── end to end: one GRPO update on the tiny MoE ───────────────────────────────────────────────────────────

def fake_turns(n=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [{"prompt_ids": torch.randint(3, 128, (6 + t,), generator=g), "completion_ids": torch.randint(3, 128, (4 + t,), generator=g),
             "advantage": (1.0 if t % 2 == 0 else -0.7)} for t in range(n)]


@pytest.mark.parametrize("n_micro,n_turns", [(4, 5), (4, 2)])   # 2 turns: two empty micro-batches are skipped
def test_grpo_update_end_to_end(n_micro, n_turns):
    from rl.train_grpo import update
    m = tiny_moe()
    convert_moe_experts_to_lg(m, lr=0.05)
    convert_to_lg(m, lr=0.05, momentum=0.0, backend="torch", skip=moe_skip())
    st = set_momentum(m, 0.9, "fp8", n=n_micro, sparse=True)
    assert st.sparse
    m.train()
    idx0 = {name: mod.idx.clone() for name, mod in m.named_modules() if isinstance(mod, LatentGridLinear)}
    loss = update(m, fake_turns(n_turns), n_micro, "lg", None, 0, "cpu")
    assert loss == loss and loss != 0.0
    changed = {name for name, mod in m.named_modules() if isinstance(mod, LatentGridLinear) and (mod.idx != idx0[name]).any()}
    exp_changed = {n_ for n_ in changed if ".experts." in n_}
    assert exp_changed, "some expert weights must change"
    assert any(".experts." not in n_ for n_ in changed)           # attention / shared expert also train
    router_w = m.model.layers[0].router.proj.weight
    assert not router_w.requires_grad
