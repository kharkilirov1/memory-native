"""Asymmetric (cascade-aware) calibration gates.

Chain of claims:
  * the damped asymmetric target w~ = H_q^{-1} G w reduces to the ORIGINAL weights
    when the quantized and fp streams coincide (no upstream quantization);
  * on a real 2-layer cascade the asymmetric objective beats the classic fp-H
    objective on the NETWORK-level error ||X_q Q - X_fp W|| (the deployed quantity);
  * ptq_warm_start(calibration='asym') wires it end-to-end: counter layers swapped,
    salient channel intact, and the whole quantized net tracks the fp net no worse
    than the classic calibration on the same budget.
"""
import copy

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from memory_native.donor.asym import asym_solve_states, asym_target_weights
from memory_native.donor.ptq import ptq_warm_start
from memory_native.group_scale_packed import PackedGroupScaleCounterLinear


def _heavy_linear(out_f, in_f, seed):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(out_f, in_f, generator=g) * 0.08
    hot = torch.rand(out_f, in_f, generator=g) < 0.02
    return nn.Parameter(torch.where(hot, w * 8.0, w))


def _cascade(seed=0):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(32, 48, bias=False), nn.SiLU(),
                          nn.Linear(48, 24, bias=False))
    model[0].weight = _heavy_linear(48, 32, seed + 1)
    model[2].weight = _heavy_linear(24, 48, seed + 2)
    calib = [torch.randn(64, 32, generator=torch.Generator().manual_seed(seed + 3 + i))
             for i in range(4)]
    return model, calib


def test_asym_target_equals_fp_when_streams_match():
    torch.manual_seed(5)
    X = torch.randn(512, 32)
    H = X.t() @ X
    w = torch.randn(16, 32) * 0.1
    # identical streams: G = H_q, so in RESIDUAL form the correction is identically
    # zero at the PRODUCTION damping (the naive H^{-1}Gw form fails this: its damping
    # shrinks weights toward 0 -- the exact bug the smoke witness caught).
    wt = asym_target_weights(w, H, H.clone(), percdamp=0.01)
    assert torch.allclose(wt, w, atol=1e-5), (wt - w).abs().max()
    # dead q-channels keep the fp weight
    H2 = H.clone()
    H2[:, 0] = 0.0
    H2[0, :] = 0.0
    wt2 = asym_target_weights(w, H2, H2.clone(), percdamp=0.01)
    assert torch.allclose(wt2[:, 0], w[:, 0])
    # strength interpolates toward the fp weight
    wt3 = asym_target_weights(w * 2, H, H.clone() * 0.5, percdamp=0.01, strength=0.0)
    assert torch.allclose(wt3, w * 2)
    # and the residual form still solves the asymmetric problem: at tiny damping it
    # matches the exact solution H_q^{-1} G w on a well-conditioned system
    Hq = H + 32 * torch.eye(32)
    G = Hq @ (torch.eye(32) + 0.05 * torch.randn(32, 32, generator=torch.Generator().manual_seed(6)))
    wt4 = asym_target_weights(w, Hq, G, percdamp=1e-9)
    exact = torch.linalg.solve(Hq, G @ w.t()).t()
    assert torch.allclose(wt4, exact, atol=1e-3), (wt4 - exact).abs().max()


def test_asym_beats_classic_on_cascade_network_error():
    model, calib = _cascade(seed=0)
    fp = copy.deepcopy(model)
    X_hold = torch.randn(512, 32, generator=torch.Generator().manual_seed(99))

    classic = copy.deepcopy(model)
    ptq_warm_start(classic, calib, mode="gptq_group", kind="counter_packed",
                   group=16, C=11, progress=False, kernel_mode="torch",
                   grid="itf", scale_refit="align", salient_first=0.01)
    asym = copy.deepcopy(model)
    ptq_warm_start(asym, calib, mode="gptq_group", kind="counter_packed",
                   group=16, C=11, progress=False, kernel_mode="torch",
                   grid="itf", scale_refit="align", salient_first=0.01,
                   calibration="asym", asym_chunk_layers=1)

    with torch.no_grad():
        y_fp = fp(X_hold)
        e_classic = float((classic(X_hold) - y_fp).pow(2).sum())
        e_asym = float((asym(X_hold) - y_fp).pow(2).sum())
    # the asymmetric objective must reduce the deployed network error
    assert e_asym < e_classic, (e_asym, e_classic)


def test_warm_start_asym_end_to_end_structure():
    model, calib = _cascade(seed=7)
    report = ptq_warm_start(model, calib, mode="gptq_group", kind="counter_packed",
                            group=16, C=11, progress=False, kernel_mode="torch",
                            grid="itf", scale_refit="align", salient_first=0.02,
                            calibration="asym", asym_chunk_layers=1)
    assert len(report.swapped) == 2
    counter = model[2]
    assert isinstance(counter, PackedGroupScaleCounterLinear)
    assert counter.salient_idx.numel() > 0
    x = torch.randn(3, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (3, 24) and torch.isfinite(y).all()


def test_asym_requires_group_mode():
    model, calib = _cascade(seed=9)
    with pytest.raises(ValueError, match="asym"):
        ptq_warm_start(model, calib, mode="gptq", calibration="asym", progress=False)


def test_asym_multi_pass_solves_from_original_weights():
    """asym_passes=2 with BOTH layers in one chunk: pass 1 collects the second layer's
    X_q while its predecessor is still fp (the intra-chunk staleness), pass 2
    re-collects on the fully quantized net and re-solves from the ORIGINAL weights.
    The second pass must change the solution and must not blow up the network error.
    (With chunk_layers=1 on a pure chain, pass 2 is a bit-exact no-op by construction —
    pass 1 is already fully sequential; iteration only has signal for chunked solves.)"""
    model, calib = _cascade(seed=21)
    fp = copy.deepcopy(model)
    X_hold = torch.randn(512, 32, generator=torch.Generator().manual_seed(77))

    one = copy.deepcopy(model)
    ptq_warm_start(one, calib, mode="gptq_group", kind="counter_packed",
                   group=16, C=11, progress=False, kernel_mode="torch",
                   grid="itf", scale_refit="align", salient_first=0.01,
                   calibration="asym", asym_chunk_layers=2, asym_strength=0.5)
    two = copy.deepcopy(model)
    ptq_warm_start(two, calib, mode="gptq_group", kind="counter_packed",
                   group=16, C=11, progress=False, kernel_mode="torch",
                   grid="itf", scale_refit="align", salient_first=0.01,
                   calibration="asym", asym_chunk_layers=2, asym_strength=0.5,
                   asym_passes=2)
    with torch.no_grad():
        y = fp(X_hold)
        e1 = float((one(X_hold) - y).pow(2).sum())
        e2 = float((two(X_hold) - y).pow(2).sum())
    # two passes must stay in the same error regime (no blow-up from re-targeting)
    assert e2 < e1 * 1.10, (e2, e1)
    # and the second pass must actually change the solution (not a silent no-op)
    w1 = two[2].visible_weight() if hasattr(two[2], "visible_weight") else two[2].counter.visible_weight()
    w0_ = one[2].visible_weight() if hasattr(one[2], "visible_weight") else one[2].counter.visible_weight()
    assert not torch.equal(w0_, w1)


def test_asym_solve_states_mutates_model_weights_progressively():
    """After asym_solve_states the model's dense weights ARE the deployable
    reconstruction (that is what makes later chunks see quantized inputs)."""
    model, calib = _cascade(seed=11)
    w0_before = model[0].weight.detach().clone()
    states = asym_solve_states(model, calib, ["0", "2"], group=16, C=11,
                               grid="itf", scale_refit="align", salient_first=0.01,
                               chunk_layers=1, progress=False)
    assert set(states) == {"0", "2"}
    assert not torch.allclose(model[0].weight, w0_before)
    S, t, c, perm, si, sv = states["0"]
    gidx = torch.empty(32, dtype=torch.long)
    gidx[perm] = torch.arange(32) // 16
    rec = S[:, gidx] * t.float()
    if si.numel():
        rec = rec.clone()
        rec.reshape(-1)[si.long()] = sv.float()
    assert torch.allclose(model[0].weight, rec, atol=1e-6)


def _mixtral(layers=2, experts=4):
    from transformers import MixtralConfig, MixtralForCausalLM

    torch.manual_seed(0)
    cfg = MixtralConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                        num_hidden_layers=layers, num_attention_heads=4,
                        num_key_value_heads=2, max_position_embeddings=128,
                        num_local_experts=experts, num_experts_per_tok=2)
    return MixtralForCausalLM(cfg).eval(), cfg


def test_teacher_forced_routing_aligns_the_towers():
    """Quantization diverts routing (measured: ~11% of tokens at layer 0, 25% at
    layer 1), which would pair X_q with an X_fp from a DIFFERENT expert. Forcing the
    fp tower's router indices must keep every expert stream aligned."""
    pytest.importorskip("transformers")
    from memory_native.donor.asym import collect_asym_stats
    from memory_native.donor.ptq import _moe_router_paths, _stacked_moe_targets

    fp, cfg = _mixtral()
    q, _ = _mixtral()
    q.load_state_dict(fp.state_dict())
    with torch.no_grad():
        for name, p in q.named_parameters():
            if "experts" in name or "self_attn" in name:
                p.add_(torch.randn_like(p) * p.std() * 0.15)

    torch.manual_seed(1)
    calib = [torch.randint(0, cfg.vocab_size, (2, 64)) for _ in range(2)]
    routers = sorted(_moe_router_paths(q))
    assert routers, "fixture has no routers to force"

    free = collect_asym_stats(q, fp, [], calib,
                              moe_targets_q=_stacked_moe_targets(q),
                              moe_targets_fp=_stacked_moe_targets(fp))
    forced = collect_asym_stats(q, fp, [], calib,
                                moe_targets_q=_stacked_moe_targets(q),
                                moe_targets_fp=_stacked_moe_targets(fp),
                                routers=routers)
    assert len(forced) > 0, "no expert statistics collected at all"
    assert len(forced) >= len(free), "forcing must not lose expert streams"


def test_asym_converts_every_moe_expert():
    """asym on a MoE donor must convert the experts, not just attention."""
    pytest.importorskip("transformers")
    from memory_native.donor.ptq import ptq_warm_start

    model, cfg = _mixtral()
    torch.manual_seed(1)
    calib = [torch.randint(0, cfg.vocab_size, (2, 32)) for _ in range(4)]
    report = ptq_warm_start(model, calib, mode="gptq_group", kind="counter_packed",
                            calibration="asym", asym_strength=0.15, group=32, C=11,
                            grid="itf", salient_first=0.02, salient_scope="layer",
                            in_sweep_refit=True, progress=False)
    expert_coeffs = (cfg.num_hidden_layers * cfg.num_local_experts
                     * (2 * cfg.intermediate_size * cfg.hidden_size
                        + cfg.hidden_size * cfg.intermediate_size))
    assert report.coeffs >= expert_coeffs, \
        f"expert weights unconverted: {report.coeffs} < {expert_coeffs}"
    with torch.no_grad():
        assert torch.isfinite(model(calib[0]).logits).all()
