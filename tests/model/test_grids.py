import pytest

import numpy as np

from cascade.model.grids import PriorGrid, AgeTimeGrid, _any_close, GRID_SNAP_DISTANCE, unique_floats
from cascade.model.priors import Gaussian


@pytest.fixture
def age_time_grid():
    return AgeTimeGrid.uniform(age_lower=0, age_upper=120, age_step=5, time_lower=1990, time_upper=2018, time_step=1)


def test_PriorGrid__development_target():
    # Make a grid using a standard structure. Other options might be
    # a constructor which makes a grid that's denser at extreme ages.
    grid = AgeTimeGrid.uniform(age_lower=0, age_upper=120, age_step=5, time_lower=1990, time_upper=2018, time_step=1)

    d_time = PriorGrid(grid)
    d_time[:, :].prior = Gaussian(0, 0.1)

    # There's a shock in 1995
    d_time[:, 1995].prior = Gaussian(0, 3)

    d_age = PriorGrid(grid)
    d_age[:, :].prior = Gaussian(0, 0.1)

    # Kids are different
    d_age[0:15, :].prior = Gaussian(1, 0.01)

    value = PriorGrid(grid)
    value[:, :].prior = Gaussian(20, 1)

    # The shock in 1995 effects the value too
    value[:, 1995].prior = Gaussian(200, 10)

    assert value[10, 1995].prior == Gaussian(200, 10)
    assert value[10, 1996].prior == Gaussian(20, 1)


def test_PriorGrid__point_query(age_time_grid):
    priors = PriorGrid(age_time_grid)

    priors[:, :].prior = Gaussian(0, 3)
    priors[20:40, 2000:2005].prior = Gaussian(20, 3)

    assert priors[10, 1990].prior == Gaussian(0, 3)
    assert priors[10, 2006].prior == Gaussian(0, 3)
    assert priors[50, 2006].prior == Gaussian(0, 3)

    assert priors[25, 2001].prior == Gaussian(20, 3)


def test_PriorGrid__point_query_with_extreme_values(age_time_grid):
    priors = PriorGrid(age_time_grid)

    priors[:, :].prior = Gaussian(0, 3)

    assert priors[min(age_time_grid.ages), 1990].prior == Gaussian(0, 3)
    assert priors[max(age_time_grid.ages), 1990].prior == Gaussian(0, 3)
    assert priors[15, min(age_time_grid.times)].prior == Gaussian(0, 3)
    assert priors[15, max(age_time_grid.times)].prior == Gaussian(0, 3)
    assert priors[min(age_time_grid.ages), min(age_time_grid.times)].prior == Gaussian(0, 3)
    assert priors[max(age_time_grid.ages), max(age_time_grid.times)].prior == Gaussian(0, 3)


def test_PriorGrid__bad_alignment(age_time_grid):
    priors = PriorGrid(age_time_grid)

    with pytest.raises(ValueError):
        priors[1:12, 1990.4:2001].prior = Gaussian(0, 3)


def test_PriorGrid__wrong_dimensions(age_time_grid):
    priors = PriorGrid(age_time_grid)

    with pytest.raises(ValueError):
        priors[1]

    with pytest.raises(ValueError):
        priors[1, 2, 3]


def test_any_close():
    assert _any_close(1, [1, 2, 3])
    assert not _any_close(1, [1.5, 2, 3])
    assert _any_close(1 + GRID_SNAP_DISTANCE, [1.5, 2, 1])


def test_unique_floats():
    assert np.all(np.equal(unique_floats([1, 1, 2, 3, 4, 4 + 2e-16]), [1, 2, 3, 4]))
