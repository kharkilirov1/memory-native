"""Calibrated PTQ warm-start witnesses.

The claim chain: optimal_ternary is the EXACT per-row L2 minimizer (brute-force checked),
it never loses to the TWN heuristic in weight error; gptq_ternary additionally minimizes what
the layer OUTPUT sees (activation error under a calibration Hessian) and must beat optimal
there; and the whole path loads into counter layers and lowers end-to-end logit error vs the
naive TWN warm-start on a real (tiny, overfit) model."""
import copy

import pytest

torch = pytest.importorskip("torch")

from memory_native.counter import (C_DEFAULT, RMSCounterLinear, decode_state,
                                   weight_to_counter_state)
from memory_native.donor.ptq import gptq_ternary, optimal_ternary, residual_counter
from memory_native.packed import PackedRMSCounterLinear


def _werr(w, s, t):
    return float(((w - s * t) ** 2).sum())


def test_optimal_ternary_matches_bruteforce():
    torch.manual_seed(0)
    w = torch.randn(5, 12)
    s, t = optimal_ternary(w)
    for r in range(w.shape[0]):
        row = w[r]
        absr = row.abs()
        vals, _ = absr.sort(descending=True)
        best = None
        for k in range(1, 13):
            thr = vals[k - 1]
            tt = torch.sign(row) * (absr >= thr).float()
            ss = absr[absr >= thr].mean()
            err = float(((row - ss * tt) ** 2).sum())
            best = min(best, err) if best is not None else err
        got = float(((row - s[r] * t[r]) ** 2).sum())
        assert got <= best + 1e-6, (r, got, best)


def test_optimal_ternary_never_worse_than_twn():
    torch.manual_seed(1)
    w = torch.randn(64, 256) * 0.05
    s_opt, t_opt = optimal_ternary(w)
    e_opt = _werr(w, s_opt, t_opt)
    for thr in (0.5, 0.7):
        s, t, _ = weight_to_counter_state(w, C_DEFAULT, thr)
        assert e_opt <= _werr(w, s, t.float()) + 1e-5


def test_gptq_beats_optimal_on_activation_error():
    torch.manual_seed(2)
    n, in_f, out_f = 1024, 128, 48
    X = torch.randn(n, in_f) @ torch.randn(in_f, in_f) * 0.1   # correlated features
    H = X.t() @ X
    w = torch.randn(out_f, in_f) * 0.05

    s_o, t_o = optimal_ternary(w)
    s_g, t_g, c_g = gptq_ternary(w, H)

    def act_err(s, t):
        return float(((X @ (w - s * t.float()).t()) ** 2).sum())

    e_o, e_g = act_err(s_o, t_o), act_err(s_g, t_g)
    assert e_g < e_o * 0.9, (e_g, e_o)          # calibration must buy a real margin
    assert c_g.abs().max() <= C_DEFAULT - 1
    assert set(t_g.unique().tolist()) <= {-1, 0, 1}


def test_group_ternary_beats_rowwise_gptq_on_activation_error():
    torch.manual_seed(4)
    n, in_f, out_f = 1024, 256, 32
    X = torch.randn(n, in_f) @ torch.randn(in_f, in_f) * 0.1
    H = X.t() @ X
    w = torch.randn(out_f, in_f) * 0.05

    from memory_native.donor.ptq import gptq_group_ternary
    s_r, t_r, _ = gptq_ternary(w, H)                       # per-row scale
    w_row = s_r * t_r.float()
    w_grp, S, t_g = gptq_group_ternary(w, H, group=64)     # per-(row,group-64) scale

    def act_err(w_hat):
        return float(((X @ (w - w_hat).t()) ** 2).sum())

    e_row, e_grp = act_err(w_row), act_err(w_grp)
    # On outlier-free gaussian synthetic weights the granularity win is small (measured ~2-7%);
    # the big group-scale gains appear on real LLM weights with outliers (the 1.5B witness).
    # Here we only pin the ORDERING: finer scales must not lose, even to act-ordered row-wise.
    assert e_grp < e_row, (e_grp, e_row)
    assert S.shape == (out_f, in_f // 64)
    assert set(t_g.unique().tolist()) <= {-1, 0, 1}
    # reconstruction consistency: w_hat == s_g * t group-wise
    rec = torch.zeros_like(w_grp)
    for g in range(in_f // 64):
        rec[:, g*64:(g+1)*64] = S[:, g:g+1] * t_g[:, g*64:(g+1)*64].float()
    assert torch.allclose(rec, w_grp, atol=1e-5)


def test_load_counter_state_roundtrip():
    for cls in (RMSCounterLinear, PackedRMSCounterLinear):
        torch.manual_seed(3)
        layer = cls(64, 16)
        w = torch.randn(16, 64) * 0.1
        s, t = optimal_ternary(w)
        c = residual_counter(w, s, t)
        layer.v.fill_(0.5)                       # must be reset by the import
        layer.load_counter_state(s, t.to(torch.int16), c)
        td, cd = layer._decode_rows(0, 16)
        assert torch.equal(td.to(torch.int16), t.to(torch.int16))
        assert torch.equal(cd.to(torch.int16), c)
        assert torch.allclose(layer.scale, s, atol=1e-6)
        assert float(layer.v.abs().sum()) == 0.0
        assert torch.allclose(layer.s_base, layer.scale)


def test_ptq_warm_start_lowers_logit_error_end_to_end():
    pytest.importorskip("transformers")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from memory_native.donor.ptq import ptq_warm_start
    from memory_native.donor.qwen import qwen_to_counter

    def tiny():
        torch.manual_seed(0)
        cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          max_position_embeddings=64, tie_word_embeddings=True)
        cfg._attn_implementation = "sdpa"
        return Qwen2ForCausalLM(cfg)

    fp = tiny()
    torch.manual_seed(1)
    batch = torch.randint(0, 64, (2, 16))
    opt = torch.optim.AdamW(fp.parameters(), lr=1e-2)   # sharpen: make ternarization hurt
    fp.train()
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        fp(batch, labels=batch).loss.backward()
        opt.step()
    fp.eval()
    with torch.no_grad():
        ref = fp(batch).logits

    def logit_mse(student):
        student.eval()
        with torch.no_grad():
            return float(((student(batch).logits - ref) ** 2).mean())

    naive = copy.deepcopy(fp)
    qwen_to_counter(naive, threshold_ratio=0.5)
    m_naive = logit_mse(naive)

    optm = copy.deepcopy(fp)
    ptq_warm_start(optm, [], mode="optimal")
    m_opt = logit_mse(optm)

    gptq = copy.deepcopy(fp)
    ptq_warm_start(gptq, [batch], mode="gptq")
    m_gptq = logit_mse(gptq)

    print(f"\n[ptq witness] logit MSE vs fp: naive={m_naive:.5f} "
          f"optimal={m_opt:.5f} gptq={m_gptq:.5f}")
    assert m_gptq < m_naive * 0.9               # calibrated start must clearly beat naive TWN
    assert m_opt < m_naive * 1.05               # exact optimum at least matches the heuristic
