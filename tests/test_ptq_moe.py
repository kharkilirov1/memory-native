import copy
import warnings

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
F = torch.nn.functional
pytest.importorskip("transformers")

from transformers import MixtralConfig, MixtralForCausalLM

from memory_native.donor.ptq import optimal_ternary, ptq_warm_start
from memory_native.moe_ffn import (
    HFModuleListSwiGLUExperts,
    HFStackedSwiGLUExperts,
)
from memory_native.recovery.runtime import restore_counter_structure


def _tiny_mixtral():
    config = MixtralConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=32,
    )
    config._attn_implementation = "eager"
    return MixtralForCausalLM(config).eval()


class _UnsupportedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(4, 8, 8))
        self.w2 = nn.Parameter(torch.randn(4, 8, 8))


class _UnsupportedMoe(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _UnsupportedExperts()
        self.gate = nn.Linear(8, 4, bias=False)

    def forward(self, x):
        return x


def test_ptq_rejects_unsupported_moe_before_partial_conversion():
    with pytest.raises(
        RuntimeError,
        match="refusing partial PTQ conversion.*expert weights unconverted",
    ):
        ptq_warm_start(_UnsupportedMoe(), [], mode="optimal", progress=False)


@pytest.fixture(scope="module")
def converted_mixtral():
    torch.manual_seed(0)
    model = _tiny_mixtral()
    moe = model.model.layers[0].mlp
    moe.gate.weight.data.zero_()
    router_before = moe.gate.weight.detach().clone()
    expert_before = (
        moe.experts.gate_up_proj.detach().clone(),
        moe.experts.down_proj.detach().clone(),
    )
    batch = torch.tensor([[1, 2, 3]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = ptq_warm_start(
            model, [batch], mode="gptq", progress=False
        )
    logits = model(batch).logits.detach()
    return {
        "model": model,
        "batch": batch,
        "report": report,
        "router_before": router_before,
        "expert_before": expert_before,
        "logits": logits,
        "warnings": [str(item.message) for item in caught],
    }


def test_mixtral_converts_every_expert_and_keeps_router_fp32(converted_mixtral):
    model = converted_mixtral["model"]
    report = converted_mixtral["report"]
    moe = model.model.layers[0].mlp
    expected_attention = 4
    expected_experts = 1 * 4 * 2

    assert len(report.swapped) == expected_attention + expected_experts
    assert sum("experts." in path for path in report.swapped) == expected_experts
    assert isinstance(moe.experts, HFStackedSwiGLUExperts)
    assert isinstance(moe.gate.weight, nn.Parameter)
    assert moe.gate.weight.dtype == torch.float32
    assert torch.equal(moe.gate.weight, converted_mixtral["router_before"])
    assert torch.isfinite(converted_mixtral["logits"]).all()


def test_mixtral_dead_experts_use_data_free_optimum_and_are_reported(
    converted_mixtral,
):
    model = converted_mixtral["model"]
    report = converted_mixtral["report"]
    gate_up_before, down_before = converted_mixtral["expert_before"]
    Wg, Wu, Wd = model.model.layers[0].mlp.experts.stacked.weights()

    assert len(report.dead_experts) == 2
    for expert_path in report.dead_experts:
        expert = int(expert_path.rsplit("[", 1)[1][:-1])
        assert report.expert_token_counts[expert_path] == 0
        s_gu, t_gu = optimal_ternary(gate_up_before[expert])
        s_d, t_d = optimal_ternary(down_before[expert])
        assert torch.equal(torch.cat((Wg[expert], Wu[expert])), s_gu * t_gu)
        assert torch.equal(Wd[expert], s_d * t_d)


def test_mixtral_rank_deficient_experts_warn_and_are_reported(converted_mixtral):
    report = converted_mixtral["report"]
    messages = converted_mixtral["warnings"]

    assert sorted(report.expert_token_counts.values()) == [0, 0, 3, 3]
    assert len(report.rank_deficient_experts) == 2
    for expert_path in report.rank_deficient_experts:
        assert report.expert_token_counts[expert_path] == 3
        assert any(
            "rank-deficient MoE Hessian" in message and expert_path in message
            for message in messages
        )


def test_mixtral_save_restore_is_bit_exact(converted_mixtral):
    checkpoint = copy.deepcopy(converted_mixtral["model"].state_dict())
    restored = _tiny_mixtral()
    report = restore_counter_structure(
        restored, checkpoint, kind="counter_rms", group=8, C=8
    )
    restored.load_state_dict(checkpoint)

    assert len(report.swapped) == 12
    assert torch.equal(
        restored(converted_mixtral["batch"]).logits,
        converted_mixtral["logits"],
    )


def test_mixtral_group_solver_converts_and_restores_bit_exactly():
    torch.manual_seed(1)
    model = _tiny_mixtral()
    model.model.layers[0].mlp.gate.weight.data.zero_()
    batch = torch.tensor([[4, 5, 6]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = ptq_warm_start(
            model, [batch], mode="gptq_group", kind="group_scale",
            group=4, C=11, progress=False,
        )
    expected = model(batch).logits.detach()
    checkpoint = copy.deepcopy(model.state_dict())
    restored = _tiny_mixtral()
    restore_counter_structure(
        restored, checkpoint, kind="group_scale", group=4, C=11
    )
    restored.load_state_dict(checkpoint)

    assert len(report.swapped) == 12
    assert isinstance(model.model.layers[0].mlp.experts, HFModuleListSwiGLUExperts)
    assert torch.isfinite(expected).all()
    assert torch.equal(restored(batch).logits, expected)


def test_qwen3_moe_stacked_parameters_use_the_same_supported_path():
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=32,
    )
    config._attn_implementation = "eager"
    model = Qwen3MoeForCausalLM(config).eval()
    batch = torch.tensor([[1, 2, 3]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = ptq_warm_start(
            model, [batch], mode="optimal", progress=False
        )

    assert len(report.swapped) == 12
    assert isinstance(model.model.layers[0].mlp.experts, HFStackedSwiGLUExperts)
    assert torch.isfinite(model(batch).logits).all()


class _LegacyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(8, 4, bias=False)
        self.up_proj = nn.Linear(8, 4, bias=False)
        self.down_proj = nn.Linear(4, 8, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _LegacyMoe(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(8, 3, bias=False)
        self.experts = nn.ModuleList(_LegacyExpert() for _ in range(3))

    def forward(self, x):
        selected = self.gate(x).argmax(dim=-1)
        output = torch.zeros_like(x)
        for expert_idx, expert in enumerate(self.experts):
            token_idx = torch.where(selected == expert_idx)[0]
            if token_idx.numel():
                output.index_add_(0, token_idx, expert(x[token_idx]))
        return output


def test_legacy_modulelist_experts_convert_once_and_router_is_excluded():
    model = _LegacyMoe()
    model.gate.weight.data.zero_()
    router_before = model.gate.weight.detach().clone()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = ptq_warm_start(
            model, [torch.randn(2, 8)], mode="gptq", progress=False
        )

    assert len(report.swapped) == 3 * 3
    assert len(set(report.swapped)) == len(report.swapped)
    assert all("experts." in path for path in report.swapped)
    assert isinstance(model.gate, nn.Linear)
    assert model.gate.weight.dtype == torch.float32
    assert torch.equal(model.gate.weight, router_before)
