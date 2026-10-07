import pytest
from memory_native.recovery.strict_exposure import strict_exposure_alpha


def test_disabled_preserves_all_homotopy_values():
    assert [strict_exposure_alpha(.75, step, 0) for step in range(9)] == [.75]*9


def test_every_fourth_step_strict_only():
    assert [strict_exposure_alpha(.8, step, 4) for step in range(10)] == [.8,.8,.8,0,.8,.8,.8,0,.8,.8]


def test_already_strict_stays_strict():
    assert strict_exposure_alpha(0,12,4) == 0


@pytest.mark.parametrize("alpha,step,every", [
    (-1,0,4), (2,0,4), (float("nan"),0,4),
    (.5,-1,4), (.5,0,-3), (.5,0,True),
])
def test_invalid(alpha,step,every):
    with pytest.raises(ValueError):
        strict_exposure_alpha(alpha,step,every)


def test_cpu_counter_receives_one_update_per_step_with_exposure():
    import torch
    from memory_native.group_scale_packed import PackedGroupScaleCounterLinear

    torch.manual_seed(12)
    layer = PackedGroupScaleCounterLinear(
        16, 8, group=8, C=11, kernel_mode="torch",
        stats_scope="group", decimation=1, local_grad_clip=1.0,
    ).train()
    x = torch.randn(3, 16)
    for step in range(8):
        expected_alpha = strict_exposure_alpha(.8, step, 4)
        layer.set_residual_alpha(expected_alpha)
        # One forward/backward only. No deferred dual-graph update is attempted.
        loss = layer(x.clone().requires_grad_(True)).square().mean()
        loss.backward()
        assert layer._sr_step == step + 1
        assert int(layer.sr_step) == step + 1
        assert layer._outstanding_forward is False
        assert layer.residual_alpha == expected_alpha
