import pytest
from cascade.executor.epiviz_runner import compute_age_steps


@pytest.mark.parametrize("min_step", [
    (1,), (5,), (0.5,), (0.1,), (10,)
])
def test_age_steps(min_step):
    steps = compute_age_steps(min_step[0])
    assert steps[0] <= 7 / 365
    assert 3.5 / 365 <= steps[0]
    assert steps[-1] < 0.75 * min_step[0]
    assert steps[-1] > 0.4 * min_step[0]


@pytest.mark.parametrize("min_step", [
    (6 / 365,), (1 / 365,), (0.000001,),
])
def test_small_small_step(min_step):
    steps = compute_age_steps(min_step[0])
    assert len(steps) == 0
