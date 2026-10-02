"""Check the controls that make the new-math pilot comparisons interpretable.

Tiny integer streams here are test fixtures, not experimental training evidence.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from memory_native.counter import RMSCounterLinear


_SPEC = importlib.util.spec_from_file_location(
    "new_math_experiments_protocol", Path(__file__).resolve().parents[1] / "scripts" / "new_math_experiments.py")
experiment = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(experiment)

BODY_NAMES = ("q", "k", "v", "proj", "fc", "fc2")


def _args(**changes):
    values = dict(dim=16, layers=1, heads=2, context=4, batch=2,
                  block_size=8, rank=2, eval_batches=1, eval_every=1)
    values.update(changes)
    return SimpleNamespace(**values)


def _shell(model):
    """Explicit FP shell, including its tied output weight only once."""
    result = {"token": model.tok.weight, "position": model.pos.weight,
              "final_norm.weight": model.lnf.weight, "final_norm.bias": model.lnf.bias}
    for index, block in enumerate(model.blocks):
        for name in ("ln1", "ln2"):
            norm = getattr(block, name)
            result[f"block{index}.{name}.weight"] = norm.weight
            result[f"block{index}.{name}.bias"] = norm.bias
    return result


def _stream_fixture(args, vocab=17):
    data = torch.arange(257, dtype=torch.long) % vocab
    validation = experiment.fixed_eval(data, args, seed=1234)
    return data, validation, vocab


def test_equal_width_variants_share_the_fp_shell_and_tied_head_at_initialization():
    args = _args()
    reference = experiment.make_model("dense", seed=41, lr=.001, args=args, vocab=17)
    reference_shell = {name: parameter.detach().clone() for name, parameter in _shell(reference).items()}
    for kind in experiment.VARIANTS:
        if kind == "dense_narrow":
            continue  # Its smaller shell is an explicitly different capacity control.
        model = experiment.make_model(kind, seed=41, lr=.009, args=args, vocab=17)
        assert model.head.weight is model.tok.weight, f"{kind} lost the embedding/head tie"
        assert model.cfg.n_embd == reference.cfg.n_embd
        shell = _shell(model)
        assert shell.keys() == reference_shell.keys()
        for name, parameter in shell.items():
            assert torch.equal(parameter, reference_shell[name]), f"{kind} changed {name} initialization"


def test_pam_arms_start_from_the_same_actual_weights_as_the_dense_control():
    args = _args(layers=2)
    dense = experiment.make_model("dense", seed=5, lr=.001, args=args, vocab=17)
    for kind in ("pam_exact", "pam_surrogate"):
        model = experiment.make_model(kind, seed=5, lr=.01, args=args, vocab=17)
        for original_block, pam_block in zip(dense.blocks, model.blocks):
            for name in BODY_NAMES:
                original, candidate = getattr(original_block, name), getattr(pam_block, name)
                assert torch.equal(candidate.weight, original.weight), f"{kind}/{name} changed starting weights"
        # Different arithmetic may change outputs; this is an initialization control.
        assert model.head.weight is model.tok.weight


def test_fixed_validation_windows_use_an_independent_generator():
    args = _args()
    data = torch.arange(257, dtype=torch.long) % 17
    torch.manual_seed(8)
    before = torch.random.get_rng_state().clone()
    first = experiment.fixed_eval(data, args, seed=24)
    assert torch.equal(torch.random.get_rng_state(), before)
    torch.rand(139)  # Global operator randomness must not move held-out windows.
    second = experiment.fixed_eval(data, args, seed=24)
    for (x1, y1), (x2, y2) in zip(first, second):
        assert torch.equal(x1, x2) and torch.equal(y1, y2)


def test_actual_training_batches_survive_evaluation_draws_and_counter_sr(monkeypatch):
    args = _args(eval_every=1)
    data, validation, vocab = _stream_fixture(args)
    real_batch_at, real_evaluate, real_rand_like = experiment.batch_at, experiment.evaluate, torch.rand_like
    starts_seen, rounding_draws = [], [0]
    eval_extra_draws = [0]

    def capture_batch(data, starts, context):
        starts_seen.append(starts.detach().clone())
        return real_batch_at(data, starts, context)

    def perturb_evaluation(model, batches):
        if eval_extra_draws[0]:
            torch.rand(eval_extra_draws[0])
        return real_evaluate(model, batches)

    def capture_rounding(*args, **kwargs):
        rounding_draws[0] += 1
        return real_rand_like(*args, **kwargs)

    monkeypatch.setattr(experiment, "batch_at", capture_batch)
    monkeypatch.setattr(experiment, "evaluate", perturb_evaluation)
    monkeypatch.setattr(torch, "rand_like", capture_rounding)
    reference = None
    global_states = {}
    for kind, extra_draws in (("dense", 0), ("dense", 227), ("counter", 0),
                              ("counter", 227), ("blast_counter", 227)):
        starts_seen.clear()
        rounding_draws[0] = 0
        eval_extra_draws[0] = extra_draws
        record = experiment.run_lm(kind, seed=11, lr=.003, steps=2, args=args,
                                   data=data, vocab=vocab, validation=validation, test=None, phase="protocol-test")
        assert record["training_tokens"] == 2 * args.batch * args.context
        assert len(starts_seen) == 2
        if reference is None:
            reference = [starts.clone() for starts in starts_seen]
        else:
            assert all(torch.equal(a, b) for a, b in zip(reference, starts_seen)), (kind, extra_draws)
        if "counter" in kind:
            assert rounding_draws[0] > 0, "test must exercise actual stochastic counter updates"
        if extra_draws == 0:
            global_states[kind] = torch.random.get_rng_state().clone()
    # Operator RNG really differed; identical windows were not accidental RNG equivalence.
    assert not torch.equal(global_states["dense"], global_states["counter"])


@pytest.mark.parametrize("kind", experiment.VARIANTS)
def test_body_lr_tuning_keeps_shell_lr_fixed_and_all_clipping_disabled(kind, monkeypatch):
    args = _args()
    data, validation, vocab = _stream_fixture(args)
    real_adamw, real_make_model = torch.optim.AdamW, experiment.make_model
    models, optimizers = [], []

    def capture_model(*positional, **keywords):
        model = real_make_model(*positional, **keywords)
        models.append(model)
        return model

    def capture_optimizer(*positional, **keywords):
        optimizer = real_adamw(*positional, **keywords)
        optimizers.append(optimizer)
        return optimizer

    def unexpected_clipping(*args, **kwargs):
        pytest.fail("global gradient clipping would change the controlled protocol")

    monkeypatch.setattr(experiment, "make_model", capture_model)
    monkeypatch.setattr(torch.optim, "AdamW", capture_optimizer)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", unexpected_clipping)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_value_", unexpected_clipping)
    for body_lr in (.0005, .007):
        record = experiment.run_lm(kind, seed=3, lr=body_lr, steps=2, args=args,
                                   data=data, vocab=vocab, validation=validation, test=None, phase="protocol-test")
        model, optimizer = models[-1], optimizers[-1]
        assert record["fp_lr"] == .003
        assert record["body_lr"] == body_lr
        assert optimizer.param_groups[0]["lr"] == .003
        assert optimizer.param_groups[1]["lr"] == body_lr
        shell_ids = {id(parameter) for parameter in _shell(model).values()}
        assert {id(parameter) for parameter in optimizer.param_groups[0]["params"]} == shell_ids
        assert {id(parameter) for parameter in optimizer.param_groups[1]["params"]} == {
            id(parameter) for parameter in model.parameters() if id(parameter) not in shell_ids}
        counters = [module for module in model.modules() if isinstance(module, RMSCounterLinear)]
        assert all(module.local_grad_clip == 0 for module in counters)
        if "counter" in kind:
            assert counters, "the protocol check must reach self-updating counter factors"
            assert optimizer.param_groups[1]["params"] == []


def test_reported_capacity_and_pair_budgets_match_the_stated_controls():
    args = _args(dim=64, layers=2, heads=4, context=32, batch=4, block_size=16, rank=8)
    # Independent counts for the published pilot shape: 4 attention matrices and
    # two 4x FFN matrices per block; tied token/head is one FP shell parameter.
    expected = {
        "dense": (98_304, 105_152, 0),
        "dense_narrow": (24_576, 28_000, 0),
        "counter": (98_304, 6_848, 98_304),
        "blast": (21_504, 28_352, 0),
        "blast_counter": (21_504, 6_848, 21_504),
        "lowrank": (20_736, 27_584, 0),
        "pam_exact": (98_304, 105_152, 0),
        "pam_surrogate": (98_304, 105_152, 0),
    }
    reports = {}
    tokens = args.batch * args.context
    for kind, (body, fp, counters) in expected.items():
        model = experiment.make_model(kind, seed=0, lr=.003, args=args, vocab=65)
        model.kind = kind
        report = experiment.parameter_report(model, tokens)
        reports[kind] = report
        assert report["body_coefficients"] == body, kind
        assert report["fp_trainable_parameters"] == fp, kind
        assert sum(parameter.numel() for parameter in model.parameters()) == fp, kind
        assert report["counter_coefficients"] == counters, kind
        assert sum(module.state.numel() for module in model.modules()
                   if isinstance(module, RMSCounterLinear)) == counters, kind
        assert report["body_forward_pair_products_per_step"] == tokens * body, kind
        assert report["body_forward_backward_pair_evaluations_per_step"] == 3 * tokens * body, kind
        if not counters:
            body_parameters = sum(parameter.numel() for block in model.blocks for name in BODY_NAMES
                                  for parameter in getattr(block, name).parameters())
            assert body_parameters == body, kind
    assert reports["blast"]["fp_trainable_parameters"] - reports["dense_narrow"]["fp_trainable_parameters"] == 352
    assert reports["blast"]["body_matrix_rank_bounds"] == [32] * 12
    assert reports["lowrank"]["body_matrix_rank_bounds"] == [9] * 12
