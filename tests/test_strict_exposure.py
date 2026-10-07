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
