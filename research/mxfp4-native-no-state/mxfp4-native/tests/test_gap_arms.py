"""probes/e1_quality.py gap-decomposition arms (dense_nobias, dense_rms, qat_*) and the opt-in mx6 knobs
(MXNLinear weight_decay = --counter-wd, scale_lr = --scale-lr). CPU, tiny, mechanism-level; every test that
checks an effect has a null twin (the knob off / the mechanism removed gives zero effect).
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("e1_quality", ROOT / "probes" / "e1_quality.py")
e1 = importlib.util.module_from_spec(_spec)
sys.modules["e1_quality"] = e1
_spec.loader.exec_module(e1)

from mx6 import update                                        # noqa: E402
from mx6.layer import MXNLinear                               # noqa: E402
from mx6.packed_na import PackedNA, e4m3_decode, e4m3_ln_gaps  # noqa: E402

VOCAB, D = 24, 64
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
LAD = [0.0, 0.5, 1.0, 6.0]


def nvna(c=5):
    return PackedNA(LAD, C=c, scale_format="nvfp4")


def _batch():
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, VOCAB, (8, 32), generator=g)
    return x, torch.roll(x, -1, dims=1)


def _block_linears(model):
    return [lin for blk in model.blocks for lin in (blk.qkv, blk.proj, blk.fc, blk.out)]


# ── dense_nobias ──────────────────────────────────────────────────────────────

def test_dense_nobias_has_no_bias_params_and_dense_bf16_does():
    nb = e1.CharLM("dense_nobias", VOCAB, d=D, blocks=2)
    lins = _block_linears(nb)
    assert len(lins) == 8 and all(isinstance(m, torch.nn.Linear) and m.bias is None for m in lins)
    assert not any(n.endswith(("qkv.bias", "proj.bias", "fc.bias", "out.bias"))
                   for n, _ in nb.named_parameters())
    twin = e1.CharLM("dense_bf16", VOCAB, d=D, blocks=2)          # null twin: the bias is the only difference
    assert all(m.bias is not None for m in _block_linears(twin))
    n_bias = sum(m.out_features for m in _block_linears(twin))
    assert sum(p.numel() for p in twin.parameters()) - sum(p.numel() for p in nb.parameters()) == n_bias
    # every new arm's linears are bias-free
    for arm in ("dense_rms", "qat_tern", "qat_fp4", "qat_e2m1lad"):
        for m in _block_linears(e1.CharLM(arm, VOCAB, d=D, blocks=1)):
            assert getattr(m, "bias", None) is None


def test_dense_nobias_bits_and_same_adamw_as_dense_bf16():
    m = e1.CharLM("dense_nobias", VOCAB, d=D, blocks=1)
    assert e1.measured_bits(m, "dense_nobias") == e1.expected_bits(m, "dense_nobias") == 96.0
    opt_params = []
    orig = torch.optim.AdamW

    class Spy(orig):
        def __init__(self, params, **kw):
            super().__init__(params, **kw)
            opt_params.append(self)

    ids = torch.randint(0, VOCAB, (4000,))
    torch.optim.AdamW = Spy
    try:
        e1.run_arm("dense_nobias", 0, D, 2, 4, 32, ids, VOCAB, lr=0.01, adam_lr=3e-3, eval_every=2)
    finally:
        torch.optim.AdamW = orig
    (opt,) = opt_params
    assert len(opt.param_groups) == 1 and opt.param_groups[0]["weight_decay"] == 0.01   # AdamW default decay
    names = {id(p) for p in opt.param_groups[0]["params"]}
    model = e1.CharLM("dense_nobias", VOCAB, d=D, blocks=2)
    assert len(names) == len(list(model.parameters()))                   # ALL params incl. the linears


# ── dense_rms ─────────────────────────────────────────────────────────────────

def test_dense_rms_update_equals_formula_hand_case():
    lin = e1.DenseRMSLinear(4, 3, lr=0.1, rms_beta=0.9, rms_eps=1e-3)
    w0 = torch.tensor([[0.1, -0.2, 0.3, -0.4], [0.5, 0.5, -0.5, -0.5], [1.0, 2.0, 3.0, 4.0]])
    lin.weight.copy_(w0)
    assert torch.equal(lin.v_rms, torch.full((3, 1), 1e-3))              # init v = rms_eps
    g1 = torch.tensor([[1.0, -1.0, 1.0, -1.0], [0.0] * 4, [0.002, 0.0, -0.002, 0.0]])
    g2 = torch.tensor([[0.5, 0.5, 0.5, 0.5], [1.0, 0.0, 0.0, 0.0], [0.0] * 4])
    w = w0.double().clone()
    v = torch.full((3, 1), 1e-3, dtype=torch.float64)
    for g in (g1, g2):                                                   # two steps: EMA is exercised
        lin.apply_update_(g)
        g = g.double()
        v = 0.9 * v + 0.1 * g.square().mean(dim=1, keepdim=True)         # per-ROW mean square, beta 0.9
        gh = g / v.sqrt().clamp_min(1e-3)                                # g / max(sqrt(v), eps)
        w = w - 0.1 * gh                                                 # w <- w - lr * g_hat
        assert torch.allclose(lin.weight.double(), w, rtol=1e-6, atol=1e-7)
        assert torch.allclose(lin.v_rms.double(), v, rtol=1e-6)
    # row 1 got zero gradient in step 1 -> unchanged then (null twin of the update)
    lin2 = e1.DenseRMSLinear(4, 3, lr=0.1)
    lin2.weight.copy_(w0)
    lin2.apply_update_(g1)
    assert torch.equal(lin2.weight[1], w0[1])
    # eps clamp active: v init = 2 -> v1 = 1.825, sqrt = 1.35 < eps = 2 -> denominator is eps
    big = e1.DenseRMSLinear(2, 1, lr=0.1, rms_eps=2.0)
    big.weight.copy_(torch.zeros(1, 2))
    big.apply_update_(torch.tensor([[0.5, 0.5]]))
    assert torch.allclose(big.weight, torch.full((1, 2), -0.1 * 0.5 / 2.0))


def test_dense_rms_normalisation_is_mx6s_and_step_is_lr_ghat():
    torch.manual_seed(0)
    mx = MXNLinear(64, 8, nvna(), lr=0.01)
    dr = e1.DenseRMSLinear(64, 8, lr=0.01)
    assert torch.equal(mx.v_rms, dr.v_rms) and (mx.rms_beta, mx.rms_eps) == (dr.rms_beta, dr.rms_eps)
    for k in range(4):
        g = torch.randn(8, 64, generator=torch.Generator().manual_seed(k)) * 10.0 ** (-k)
        w_before = dr.weight.clone()
        mx.apply_update_(g)
        dr.apply_update_(g)
        assert torch.equal(mx.v_rms, dr.v_rms)                           # the SAME second moment, bit for bit
        assert torch.allclose(w_before - dr.weight, 0.01 * (g / dr.v_rms.sqrt().clamp_min(1e-3)),
                              rtol=1e-4, atol=2e-7)                      # fp32 w - w' cancellation


def test_dense_rms_not_in_adamw_trains_and_guards():
    torch.manual_seed(0)
    model = e1.CharLM("dense_rms", VOCAB, d=D, blocks=1, lr=0.01)
    names = dict(model.named_parameters())
    assert not any(".qkv." in n or ".proj." in n or ".fc." in n or ".out." in n for n in names)  # buffers
    assert "blocks.0.qkv.weight" in model.state_dict()
    w0 = [m.weight.clone() for m in _block_linears(model)]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    x, y = _batch()
    losses = []
    for _ in range(40):
        loss = torch.nn.functional.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert all(not torch.equal(a, m.weight) for a, m in zip(w0, _block_linears(model)))
    assert losses[-1] < losses[0] - 0.5
    model(x)                                                              # forward without backward ...
    with pytest.raises(RuntimeError):                                     # ... the one-forward guard is armed
        model(x)
    # lr schedule reaches the layer like it reaches counters
    m2 = e1.CharLM("dense_rms", VOCAB, d=D, blocks=1, lr=0.01)
    e1.set_counter_lr(m2, 0.123)
    assert all(m.lr == 0.123 for m in _block_linears(m2))


def test_counter_expected_step_equals_dense_rms_step():
    """The docstring claim: the counter's latent weight (value + acc/C*gap*s) moves by -lr*g_hat IN EXPECTATION
    (SR is unbiased; gap, s, C cancel) -> dense_rms's absolute step lr*g_hat is the counter's expected step."""
    torch.manual_seed(0)
    fmt = nvna(5)
    w = torch.randn(8, 64) * 0.1
    lr = 2e-4
    layer = MXNLinear.from_dense(w, fmt, lr=lr)
    g = torch.randn(8, 64, generator=torch.Generator().manual_seed(1))
    _, gh = update.rms_update(g, layer.v_rms.clone(), layer.rms_beta, layer.rms_eps)
    expected = -lr * gh

    def latent():
        c = layer.codes.to(torch.long)
        i = (c // fmt.acc_levels).clamp(0, fmt.n_vals - 1)
        acc = ((c % fmt.acc_levels) - (fmt.C - 1)).float()
        s = fmt.expand_scale(layer._scale, layer._tscale)
        gap = torch.where(acc >= 0, fmt._gaps_up[i], fmt._gaps_down[i])
        return fmt.signed[i] * s + acc / fmt.C * gap * s

    lat0, st0, v0 = latent(), layer._state.clone(), layer.v_rms.clone()
    i0 = (layer.codes.to(torch.long) // fmt.acc_levels).clamp(0, fmt.n_vals - 1)
    s = fmt.expand_scale(layer._scale, layer._tscale)
    tick = torch.where(expected >= 0, fmt._gaps_up[i0], fmt._gaps_down[i0]) * s / fmt.C
    frac = expected.abs() / tick                                          # pre-SR ticks (acc starts at 0)
    assert float(frac.max()) < 1.0                                        # no carry in one step -> pure SR
    K, acc_delta = 600, torch.zeros_like(w)
    for k in range(K):
        layer._state.copy_(st0)
        layer.v_rms.copy_(v0)
        layer._codes_cache = layer._weight_cache = None
        torch.manual_seed(1000 + k)
        layer.apply_update_(g)
        acc_delta += latent() - lat0
    mean = acc_delta / K
    se = tick * torch.sqrt((frac * (1 - frac)).clamp_min(1e-12) / K)      # binomial SE of the tick count
    assert bool(((mean - expected).abs() <= 6.0 * se + 1e-9).all())
    # the mean step is NOT zero: the test can fail (null twin: the opposite sign is far outside the SE)
    assert float(((mean + expected).abs() / (se + 1e-12)).median()) > 6.0


# ── qat_* ─────────────────────────────────────────────────────────────────────

def _qat(mode, ste=True, i=64, o=8, seed=0):
    torch.manual_seed(seed)
    return e1.QATLinear(i, o, mode, ste=ste)


def test_fq_tern_hand_case_and_values():
    w = torch.tensor([[1.0, -2.0, 0.1, 0.4]])
    gamma = (1.0 + 2.0 + 0.1 + 0.4) / 4                                   # per-row absmean = 0.875
    assert torch.allclose(e1.fq_tern(w), torch.tensor([[gamma, -gamma, 0.0, 0.0]]))
    torch.manual_seed(0)
    big = torch.randn(16, 128)
    q = e1.fq_tern(big)
    g = big.abs().mean(dim=1, keepdim=True)
    assert set(torch.unique((q / g).round()).tolist()) <= {-1.0, 0.0, 1.0}
    assert torch.allclose(q / g, (q / g).round(), atol=1e-6)
    assert torch.equal(e1.fq_tern(-big), -q)                              # odd map


def _ref_nvfp4(w, mags):
    """Independent reference: torch's fp8 cast for the E4M3 scale, brute-force nearest codebook value."""
    vals = torch.tensor(sorted({-m for m in mags} | set(mags)))
    top = max(mags)
    out, inn = w.shape
    blocks = w.reshape(out, inn // 16, 16)
    ts = w.abs().amax() / (top * 448.0)
    s = (blocks.abs().amax(-1) / (top * ts)).to(torch.float8_e4m3fn).to(torch.float32) * ts
    a = blocks / s[..., None]
    idx = (a[..., None] - vals).abs().argmin(-1)
    return (vals[idx] * s[..., None]).reshape(out, inn)


@pytest.mark.parametrize("mode,mags", [("fp4", E2M1), ("e2m1lad", LAD)])
def test_fq_nvfp4_matches_independent_reference_and_codebook(mode, mags):
    torch.manual_seed(0)
    w = torch.randn(32, 128)
    q = e1.QAT_FQ[mode](w)
    assert torch.allclose(q, _ref_nvfp4(w, mags), rtol=1e-6, atol=0)
    fmt = e1._QAT_FMT[mode]
    sb, ts = fmt.nvfp4_fit(w)
    r = (q / fmt.expand_scale(sb, ts)).abs()
    allowed = torch.tensor(mags)
    assert float((r[..., None] - allowed).abs().min(-1).values.max()) < 1e-5     # values ⊂ codebook
    assert torch.equal(e1.QAT_FQ[mode](-w), -q)                                  # odd map (ties away from 0)
    used = set(torch.unique(r.round(decimals=4)).tolist())
    if mode == "fp4":
        assert {0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} <= used                      # the FULL E2M1 set is used
    else:
        assert used <= {0.0, 0.5, 1.0, 6.0} and not ({1.5, 2.0, 3.0, 4.0} & used)   # only the mx6 ladder


def test_ladder_subset_is_coarser_than_full_e2m1():
    torch.manual_seed(0)
    w = torch.randn(64, 128)
    mse = lambda mode: float((e1.QAT_FQ[mode](w) - w).square().mean())
    assert mse("fp4") < mse("e2m1lad")                                    # dropping rungs costs error


@pytest.mark.parametrize("mode", ["tern", "fp4", "e2m1lad"])
def test_qat_forward_is_quantize_exactly_and_backward_is_ste(mode):
    lin = _qat(mode)
    x, c = torch.randn(5, 64), torch.randn(5, 8)
    y = lin(x)
    q = e1.QAT_FQ[mode](lin.weight.detach())
    assert torch.equal(lin.quantized_weight().detach(), q)               # forward weight == quantize(w), exactly
    assert torch.equal(y, torch.nn.functional.linear(x, q))
    assert not torch.equal(q, lin.weight.detach())                        # (and it really quantises)
    (y * c).sum().backward()
    wq = q.clone().requires_grad_(True)                                   # dL/dw_q of the same loss ...
    (torch.nn.functional.linear(x, wq) * c).sum().backward()
    assert float(lin.weight.grad.abs().sum()) > 0
    assert torch.equal(lin.weight.grad, wq.grad)                          # ... is passed straight through (STE)
    # eval path (no_grad) also uses the quantised weight
    with torch.no_grad():
        assert torch.equal(lin(x), torch.nn.functional.linear(x, q))


@pytest.mark.parametrize("mode", ["tern", "fp4", "e2m1lad"])
def test_null_twin_rounding_without_ste_gives_zero_gradient(mode):
    lin = _qat(mode, ste=False)
    x = torch.randn(5, 64)
    y = lin(x)
    assert torch.equal(y, torch.nn.functional.linear(x, e1.QAT_FQ[mode](lin.weight.detach())))  # same forward
    assert not y.requires_grad                                            # no path to the latent weight at all
    w = torch.randn(8, 64, requires_grad=True)                            # and autograd's own derivative of round:
    gamma = w.detach().abs().mean(1, keepdim=True)
    (gamma * torch.clamp(torch.round(w / gamma), -1, 1)).sum().backward()
    assert float(w.grad.abs().sum()) == 0.0


def test_qat_params_groups_lr_and_schedule(monkeypatch):
    made = []
    orig = torch.optim.AdamW

    class Spy(orig):
        def __init__(self, params, **kw):
            super().__init__(params, **kw)
            made.append(self)

    monkeypatch.setattr(torch.optim, "AdamW", Spy)
    ids = torch.randint(0, VOCAB, (4000,))
    r = e1.run_arm("qat_fp4", 0, D, 4, 4, 32, ids, VOCAB, lr=1e-2, adam_lr=3e-3, sched="cosine", eval_every=2)
    (opt,) = made
    assert [g["base_lr"] for g in opt.param_groups] == [3e-3, 1e-2]       # FP params @adam_lr, latent @qat_lr
    for g in opt.param_groups:                                            # both follow --sched
        assert g["lr"] == pytest.approx(e1.sched_lr("cosine", g["base_lr"], 3, 4)) and g["lr"] < g["base_lr"]
        assert g["weight_decay"] == 0.01                                  # AdamW default, same as dense_nobias
    latent = {id(p) for p in opt.param_groups[1]["params"]}
    model = e1.CharLM("qat_fp4", VOCAB, d=D)
    assert len(latent) == 8 == sum(isinstance(m, e1.QATLinear) for m in model.modules())
    assert math.isfinite(r["val"]) and r["bits"] == 96.0 and math.isnan(r["edge"])


@pytest.mark.parametrize("arm", ["dense_nobias", "dense_rms", "qat_tern", "qat_fp4", "qat_e2m1lad"])
def test_new_arms_run_end_to_end_with_weight_stats(arm):
    ids = torch.randint(0, VOCAB, (4000,))
    r = e1.run_arm(arm, 0, D, 4, 4, 32, ids, VOCAB, lr=0.01, adam_lr=3e-3, sched="cosine", eval_every=2)
    assert math.isfinite(r["val"]) and [c[0] for c in r["curve"]] == [2, 4]
    ws = r["wstats"]
    assert ws["kurt"] > 1.0 and 0.0 <= ws["tail3"] < 0.1 and 0.0 <= ws["zero"] <= 1.0


def test_dense_rms_bits_measured_equals_expected():
    m = e1.CharLM("dense_rms", VOCAB, d=D, blocks=1)
    mb, eb = e1.measured_bits(m, "dense_rms"), e1.expected_bits(m, "dense_rms")
    assert abs(mb - eb) < 0.02 and 32.0 < mb < 33.0                       # fp32 weight + fp32 v per row


# ── deployed-weight distribution summary ──────────────────────────────────────

def test_weight_stats_hand_cases():
    model = e1.CharLM("dense_nobias", VOCAB, d=D, blocks=1)
    lins = _block_linears(model)
    for m in lins:                                                        # one weight = 1, rest 0, per row
        m.weight.data.zero_()
        m.weight.data[:, 0] = 1.0
    ws = e1.weight_stats(model)
    rows = [m.out_features for m in lins]
    ns = [m.in_features for m in lins]                                    # kurt of (1,0,..0) is N
    assert ws["kurt"] == pytest.approx(sum(r * n for r, n in zip(rows, ns)) / sum(rows))
    tot = sum(r * n for r, n in zip(rows, ns))
    assert ws["tail3"] == pytest.approx(sum(rows) / tot) and ws["zero"] == pytest.approx(1 - sum(rows) / tot)
    for m in lins:                                                        # two-point +-1: kurt 1, no tails
        m.weight.data = torch.where(torch.rand_like(m.weight) < 0.5, -1.0, 1.0)
    ws = e1.weight_stats(model)
    assert ws["kurt"] == pytest.approx(1.0) and ws["tail3"] == 0.0 and ws["zero"] == 0.0
    torch.manual_seed(0)                                                  # Gaussian rows: kurt ~ 3, tail ~ 0.27 %
    for m in lins:
        m.weight.data = torch.randn(m.out_features, m.in_features)
    ws = e1.weight_stats(model)
    assert 2.6 < ws["kurt"] < 3.4 and 0.0005 < ws["tail3"] < 0.006


def test_weight_stats_reads_the_deployed_quantised_weight():
    torch.manual_seed(0)
    model = e1.CharLM("qat_tern", VOCAB, d=D, blocks=1)
    ws = e1.weight_stats(model)
    assert ws["zero"] > 0.1                                               # ternary: real zeros in the DEPLOYED weight
    latent_zero = sum(float((m.weight == 0).sum()) for m in _block_linears(model))
    assert latent_zero == 0.0                                             # the latent fp32 weight has none
    mx = e1.CharLM("na_e2m1_c5", VOCAB, d=D, blocks=1)
    ws = e1.weight_stats(mx)
    assert ws["zero"] > 0.0 and ws["kurt"] > 1.0


# ── --counter-wd (MXNLinear weight_decay) ─────────────────────────────────────

def _fresh(wd=0.0, scale="nvfp4", **kw):
    torch.manual_seed(3)
    fmt = PackedNA(LAD, C=5, scale_format=scale)
    return MXNLinear(64, 8, fmt, lr=0.01, weight_decay=wd, **kw)


def _old_apply(layer, grad_w):
    """The update exactly as it was BEFORE weight_decay / scale_lr existed (non-growth path)."""
    layer.v_rms, gh = update.rms_update(grad_w.detach().float(), layer.v_rms, layer.rms_beta, layer.rms_eps)
    new = layer.fmt.step(layer.codes, layer._scale, gh, layer.lr, tscale=layer._tscale)
    layer._state.copy_(layer.fmt.pack_state(new))
    layer._codes_cache = layer._weight_cache = None


@pytest.mark.parametrize("scale", ["e8m0", "nvfp4"])
def test_counter_wd_zero_is_bit_identical_to_before(scale):
    a, b, ref = _fresh(scale=scale), _fresh(0.0, scale=scale, scale_lr=0.0), _fresh(scale=scale)
    for k in range(6):
        g = torch.randn(8, 64, generator=torch.Generator().manual_seed(k))
        for layer, apply in ((a, lambda l, g: l.apply_update_(g)), (b, lambda l, g: l.apply_update_(g)),
                             (ref, _old_apply)):
            torch.manual_seed(100 + k)
            apply(layer, g)
    for other in (b, ref):
        assert torch.equal(a._state, other._state) and torch.equal(a._scale, other._scale)
        assert torch.equal(a.v_rms, other.v_rms)
    assert a.v_scale is None and a.weight_decay == 0.0 and "v_scale" not in a.state_dict()
    assert a.persistent_bytes() == a.fmt.persistent_bytes(8, 64)


def test_counter_wd_pulls_toward_zero_under_zero_gradient_and_not_without():
    torch.manual_seed(0)
    w = torch.randn(8, 64) * 0.1
    zero = torch.zeros(8, 64)

    def run(lam):
        fmt = nvna(5)
        # lr*lam small enough that one step is < C ticks: a single carry per step, so the pull can never
        # overshoot across 0 (a multi-rung carry from a big step could; that is a step-size artefact)
        layer = MXNLinear.from_dense(w, fmt, lr=0.01, weight_decay=lam)
        st0 = layer._state.clone()
        mags = [fmt.signed[(layer.codes.to(torch.long) // fmt.acc_levels)].abs()]
        for k in range(25):
            torch.manual_seed(7 + k)
            layer.apply_update_(zero)                                     # deterministic: no gradient, no noise source
            mags.append(fmt.signed[(layer.codes.to(torch.long) // fmt.acc_levels)].abs())
        return layer, st0, mags

    layer0, st0, mags0 = run(0.0)
    assert torch.equal(layer0._state, st0)                                # λ = 0: nothing moves (null twin)
    layer1, _, mags1 = run(1.0)
    assert float(mags1[-1].mean()) < 0.2 * float(mags1[0].mean())         # λ > 0: collapses toward 0 ...
    for a, b in zip(mags1[:-1], mags1[1:]):
        assert bool((b <= a).all())                                       # ... monotonically, never away from / across 0
    assert float((mags1[-1] == 0).float().mean()) > 0.9
    # a weaker λ drifts less in the same number of steps (dose-response)
    _, _, mags_weak = run(0.2)
    assert float(mags1[-1].mean()) < float(mags_weak[-1].mean()) < float(mags0[-1].mean())


def test_counter_wd_validation():
    with pytest.raises(ValueError):
        _fresh(-0.1)


# ── --scale-lr (MXNLinear scale_lr) ───────────────────────────────────────────

def test_e4m3_ln_gaps_match_decode():
    up, dn = e4m3_ln_gaps(torch.device("cpu"))
    b = torch.arange(2, 126)
    v = lambda x: e4m3_decode(x.to(torch.uint8))
    assert torch.allclose(up[b], torch.log(v(b + 1) / v(b)), atol=1e-6)
    assert torch.allclose(dn[b], torch.log(v(b) / v(b - 1)), atol=1e-6)
    assert float(dn[1]) == float(up[1]) == pytest.approx(math.log(2.0))   # 1 -> 2 subnormal; no code below
    assert float(up[126]) == float(dn[126]) and 0.05 < float(up[126]) < 0.1   # 448: reuse the gap below
    assert 0.05 < float(up[70]) < 0.13                                    # normals: 6-12 %


def test_scale_code_step_unbiased_in_log_space_and_null_twin():
    fmt = nvna()
    out, nb = 200, 100
    sb = torch.full((out, nb), 70, dtype=torch.uint8)
    codes = torch.full((out, nb * 16), fmt.encode(torch.tensor(3), torch.tensor(0)).item(), dtype=torch.uint8)
    ts = torch.tensor(1e-3)
    ln = lambda b: torch.log(e4m3_decode(b))
    for dlog in (0.02, -0.02):
        torch.manual_seed(0)
        new_codes, new_sb, changed = fmt.scale_code_step(codes, sb, torch.full((out, nb), dlog), ts)
        d = ln(new_sb) - ln(sb)
        n = out * nb
        up, dn = e4m3_ln_gaps(torch.device("cpu"))
        gap = float(up[70] if dlog > 0 else dn[70])
        se = math.sqrt(abs(dlog) * gap / n)                               # Bernoulli(p = |dlog|/gap) * gap
        assert abs(float(d.mean()) - dlog) < 5 * se                       # E[delta ln s] = dlog
        assert bool(changed.any())
        assert int((new_sb.to(torch.int32) - sb.to(torch.int32)).abs().max()) == 1    # one code step at most here
        assert float(d.abs().max()) == pytest.approx(gap, rel=1e-5)                   # ... of exactly the ln-gap
        # blocks whose byte did not move are bit-identical
        keep = (~changed).repeat_interleave(16, dim=1)
        assert torch.equal(new_codes[keep], codes[keep])
    c2, s2, ch2 = fmt.scale_code_step(codes, sb, torch.zeros(out, nb), ts)   # null twin: dlog = 0
    assert torch.equal(s2, sb) and torch.equal(c2, codes) and not bool(ch2.any())


def test_scale_code_step_clamps_to_valid_e4m3_codes():
    fmt = nvna()
    ts = torch.tensor(1e-3)
    codes = torch.full((2, 32), 30, dtype=torch.uint8)
    top, bot = torch.full((2, 2), 126, dtype=torch.uint8), torch.full((2, 2), 1, dtype=torch.uint8)
    _, s, ch = fmt.scale_code_step(codes, top, torch.full((2, 2), 5.0), ts)
    assert torch.equal(s, top) and not bool(ch.any())                    # 448 cannot grow
    _, s, ch = fmt.scale_code_step(codes, bot, torch.full((2, 2), -5.0), ts)
    assert torch.equal(s, bot) and not bool(ch.any())                    # 2^-9 cannot shrink (and never hits byte 0)
    mid = torch.full((2, 2), 70, dtype=torch.uint8)
    _, s, _ = fmt.scale_code_step(codes, mid, torch.full((2, 2), 0.5), ts)
    assert bool((s > 70).all()) and bool((s <= 126).all())               # a big dlog = several code steps, still valid


def test_scale_lr_validation_and_accounting():
    with pytest.raises(ValueError):
        MXNLinear(64, 8, PackedNA(LAD, C=5), scale_lr=0.01)              # e8m0: no
    with pytest.raises(ValueError):
        MXNLinear(64, 8, nvna(), scale_lr=0.01, scale_growth="pressure")
    with pytest.raises(ValueError):
        MXNLinear(64, 8, nvna(), scale_lr=0.01, rescale_every=10)
    off, on = _fresh(), _fresh(scale_lr=0.01)
    assert off.v_scale is None and on.v_scale.shape == (8, 1)
    assert on.persistent_bytes() == off.persistent_bytes() + 8 * 4       # one fp32 per row, counted
    assert on.bits_per_param() > off.bits_per_param()
    m = e1.CharLM("na_e2m1_c5", VOCAB, d=D, blocks=1)
    assert m.blocks[0].qkv.v_scale is None


def test_scale_lr_moves_scale_in_the_gradient_direction_and_keeps_rungs():
    fmt = nvna()
    torch.manual_seed(0)
    w = torch.randn(8, 128) * 0.1
    results = {}
    for name, sign, slr in (("shrink", +1.0, 0.02), ("grow", -1.0, 0.02), ("null", +1.0, 0.0)):
        layer = MXNLinear.from_dense(w, fmt, lr=0.0, scale_lr=slr)       # lr=0: weights never move on their own
        sb0, ts0 = layer._scale.clone(), layer._tscale.clone()
        rung0 = layer.codes.to(torch.long) // fmt.acc_levels
        for k in range(25):
            torch.manual_seed(k)
            gw = sign * layer.weight()                                   # dL/dln(s) = sum(grad * w) = sign * sum(w^2)
            layer.apply_update_(gw)
        # the EFFECTIVE scale (E4M3 byte x per-tensor scale): growth past 448 rebases the tensor scale and
        # lowers every byte by 8, so the bytes alone no longer track the scale
        s0 = fmt.decode_scale(sb0, ts0).mean()
        s1 = fmt.decode_scale(layer._scale, layer._tscale).mean()
        results[name] = (float(s1 / s0), layer, rung0)
    assert results["shrink"][0] < 0.9           # positive scale-gradient -> scale falls
    assert results["grow"][0] > 1.05            # negative -> scale rises (past 448: tensor-scale rebase)
    assert results["grow"][1].total_rebased > 0 and results["shrink"][1].total_rebased == 0
    assert results["null"][0] == 1.0 and results["null"][1].total_scale_moves == 0   # scale_lr = 0: frozen
    for name in ("shrink", "grow"):
        _, layer, rung0 = results[name]
        assert layer.total_scale_moves > 0
        rung1 = layer.codes.to(torch.long) // fmt.acc_levels
        assert float((rung1 != rung0).float().mean()) < 0.05            # a value-preserving re-encode: rungs stay
    # weights scale with their block: w_new == rung * s_new (the readout follows the learned scale)
    _, layer, _ = results["shrink"]
    assert torch.allclose(layer.weight(), fmt.signed[layer.codes.to(torch.long) // fmt.acc_levels]
                          * fmt.expand_scale(layer._scale, layer._tscale))


def test_scale_lr_gradient_is_dloss_dlogscale():
    """g_u = sum_block(grad_w * w) is the exact derivative wrt u = ln(s) of any loss linear in w (w = v*s*ts)."""
    torch.manual_seed(0)
    v = torch.tensor([0.5, -1.0, 6.0, 0.0, 1.0, -0.5, 0.5, 6.0, -6.0, 1.0, 0.0, 0.5, -1.0, 0.5, 1.0, -0.5])
    u = torch.tensor([-1.2], requires_grad=True)
    ts = 0.01
    w = v * torch.exp(u) * ts
    c = torch.randn(16)
    (c * w).sum().backward()
    assert torch.allclose(u.grad, (c * w.detach()).sum().reshape(1), rtol=1e-5)


def test_scale_lr_follows_schedule_via_scale_frac(monkeypatch):
    monkeypatch.setattr(e1, "SCALE_FORMAT", "nvfp4")
    monkeypatch.setattr(e1, "SCALE_LR", 0.02)
    model = e1.CharLM("na_e2m1_c5", VOCAB, d=D, blocks=1, lr=0.01)
    layers = _block_linears(model)
    assert all(m.scale_lr == m.scale_lr_base == 0.02 for m in layers)
    e1.set_counter_lr(model, 0.005, scale_frac=0.5)
    assert all(m.lr == 0.005 and m.scale_lr == pytest.approx(0.01) for m in layers)
    e1.set_counter_lr(model, 0.001)                                       # no scale_frac: scale_lr untouched
    assert all(m.scale_lr == pytest.approx(0.01) for m in layers)
    assert abs(e1.measured_bits(model, "na_e2m1_c5") - e1.expected_bits(model, "na_e2m1_c5")) < 0.02


def test_na_arm_runs_end_to_end_with_counter_wd_and_scale_lr(monkeypatch):
    monkeypatch.setattr(e1, "SCALE_FORMAT", "nvfp4")
    monkeypatch.setattr(e1, "COUNTER_WD", 0.1)
    monkeypatch.setattr(e1, "SCALE_LR", 0.01)
    ids = torch.randint(0, VOCAB, (4000,))
    r = e1.run_arm("na_e2m1_c5", 0, D, 6, 4, 32, ids, VOCAB, lr=0.01, adam_lr=3e-3, sched="cosine", eval_every=3)
    assert math.isfinite(r["val"]) and r["scale_moves"] >= 0 and "kurt" in r["wstats"]


# ── CLI plumbing ──────────────────────────────────────────────────────────────

def _fake_results(arm, lr):
    return {"arm": arm, "seed": 0, "bits": 1.0, "val": 1.0, "sec": 0.0, "hist": None,
            "clip": float("nan"), "grown": 0, "curve": [(1, 1.0, lr, 0.0, 0.0, float("nan"), 1.0)],
            "best": 1.0, "flip": (0.0, 0.0, 0.0), "edge": 0.0, "train": 1.0, "gap": 0.0,
            "wstats": {"kurt": 3.0, "tail3": 0.0027, "zero": 0.1}, "scale_moves": 0}


def _keep_globals(monkeypatch):
    for n in ("RESCALE_EVERY", "SCALE_MODE", "CLIP_FRAC", "SCALE_FORMAT", "HEADROOM_MODE", "HEADROOM",
              "SCALE_GROWTH", "GROWTH_TAU", "COUNTER_WD", "SCALE_LR"):
        monkeypatch.setattr(e1, n, getattr(e1, n))


def test_flags_route_lrs_and_set_globals(monkeypatch, tmp_path, capsys):
    _keep_globals(monkeypatch)
    data = tmp_path / "t.txt"
    data.write_text("abcdefgh" * 400, encoding="utf-8")
    seen = {}

    def fake_run_arm(arm, seed, d, steps, batch, seq, ids, vocab, lr, adam_lr, **kw):
        seen[arm] = (lr, e1.COUNTER_WD, e1.SCALE_LR)
        return _fake_results(arm, lr)

    monkeypatch.setattr(e1, "run_arm", fake_run_arm)
    base = ["--d", "32", "--steps", "2", "--seeds", "0", "--lr", "0.02", "--adam-lr", "0.004",
            "--dense-rms-lr", "0.03", "--data", str(data), "--arms",
            "dense_nobias,dense_rms,qat_tern,qat_fp4,qat_e2m1lad,na_e2m1_c5"]
    e1.main(base)
    assert seen == {"dense_nobias": (0.02, 0.0, 0.0), "dense_rms": (0.03, 0.0, 0.0),
                    "qat_tern": (0.004, 0.0, 0.0), "qat_fp4": (0.004, 0.0, 0.0),   # --qat-lr defaults to --adam-lr
                    "qat_e2m1lad": (0.004, 0.0, 0.0), "na_e2m1_c5": (0.02, 0.0, 0.0)}
    out = capsys.readouterr().out
    assert "gap arms: dense_rms_lr=0.03 qat_lr=0.004 counter_wd=0 scale_lr=0" in out
    assert "deployed weights of the 4 block linears" in out and "qat_fp4" in out
    seen.clear()
    e1.main(base + ["--qat-lr", "0.01", "--counter-wd", "0.3", "--scale-lr", "0.005", "--scale-format", "nvfp4"])
    assert seen["qat_fp4"][0] == 0.01 and seen["qat_tern"][0] == 0.01 and seen["dense_rms"][0] == 0.03
    assert seen["na_e2m1_c5"] == (0.02, 0.3, 0.005)


@pytest.mark.parametrize("extra", [["--scale-lr", "0.01"],                                 # e8m0 (default)
                                   ["--scale-lr", "0.01", "--scale-format", "nvfp4", "--scale-growth", "pressure"],
                                   ["--scale-lr", "0.01", "--scale-format", "nvfp4", "--rescale", "50"],
                                   ["--counter-wd", "-1"]])
def test_invalid_flag_combinations_are_rejected(monkeypatch, tmp_path, extra):
    _keep_globals(monkeypatch)
    data = tmp_path / "t.txt"
    data.write_text("abcdefgh" * 400, encoding="utf-8")
    with pytest.raises(SystemExit):
        e1.main(["--d", "32", "--steps", "1", "--seeds", "0", "--arms", "na_e2m1_c5",
                 "--data", str(data)] + extra)
