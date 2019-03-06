import numpy as np
import pytest

from cascade.model.grids import AgeTimeGrid, unique_floats


@pytest.fixture
def age_time_grid():
    return AgeTimeGrid.uniform(age_lower=0, age_upper=120, age_step=5, time_lower=1990, time_upper=2018, time_step=1)


def test_unique_floats():
    assert np.all(np.equal(unique_floats([1, 1, 2, 3, 4, 4 + 2e-16]), [1, 2, 3, 4]))
