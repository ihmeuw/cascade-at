import pytest

from cascade.model.grids import PriorGrid, AgeTimeGrid, _any_close
from cascade.model.priors import GaussianPrior


@pytest.fixture
def age_time_grid():
    return AgeTimeGrid.uniform(age_start=0, age_end=120, age_step=5, time_start=1990, time_end=2018, time_step=1)


# def test_PriorGrid__development_target():
#    # Make a grid using a standard structure. Other options might be
#    # a constructor which makes a grid that's denser at extreme ages.
#    grid = AgeTimeGrid.uniform(age_start=0, age_end=120, age_step=1, time_start=1990, time_end=2018, time_step=1)
#
#    d_time = PriorGrid(grid)
#    d_time[:, :].prior = GaussianPrior(0, 0.1)
#
#    # There's a shock in 1995
#    d_time[:, 1995].prior = GaussianPrior(0, 3)
#
#    d_age = PriorGrid(grid)
#    d_age[:, :].prior = GaussianPrior(0, 0.1)
#
#    # Kids are different
#    d_age[0:15, :].prior = GaussianPrior(1, 0.01)
#
#    value = PriorGrid(grid)
#    value[:, :].prior = GaussianPrior(20, 1)
#
#    # The shock in 1995 effects the value too
#    value[:, 1995].prior = GaussianPrior(200, 10)
#
#    assert value[10, 1995].prior == GaussianPrior(200, 10)
#    assert value[10, 1996].prior == GaussianPrior(20, 1)


def test_PriorGrid__point_query(age_time_grid):
    priors = PriorGrid(age_time_grid)

    priors[:, :].prior = GaussianPrior(0, 3)
    priors[20:40, 2000:2005].prior = GaussianPrior(20, 3)

    assert priors[10, 1990].prior == GaussianPrior(0, 3)
    assert priors[10, 2006].prior == GaussianPrior(0, 3)
    assert priors[50, 2006].prior == GaussianPrior(0, 3)

    assert priors[25, 2001].prior == GaussianPrior(20, 3)


def test_PriorGrid__point_query_with_extreme_values(age_time_grid):
    priors = PriorGrid(age_time_grid)

    priors[:, :].prior = GaussianPrior(0, 3)

    assert priors[min(age_time_grid.ages), 1990].prior == GaussianPrior(0, 3)
    assert priors[max(age_time_grid.ages), 1990].prior == GaussianPrior(0, 3)
    assert priors[15, min(age_time_grid.times)].prior == GaussianPrior(0, 3)
    assert priors[15, max(age_time_grid.times)].prior == GaussianPrior(0, 3)
    assert priors[min(age_time_grid.ages), min(age_time_grid.times)].prior == GaussianPrior(0, 3)
    assert priors[max(age_time_grid.ages), max(age_time_grid.times)].prior == GaussianPrior(0, 3)


def test_PriorGrid__bad_alignment(age_time_grid):
    priors = PriorGrid(age_time_grid)

    with pytest.raises(ValueError):
        priors[1:12, 1990.4:2001].prior = GaussianPrior(0, 3)


def test_PriorGrid__wrong_dimensions(age_time_grid):
    priors = PriorGrid(age_time_grid)

    with pytest.raises(ValueError):
        priors[1]

    with pytest.raises(ValueError):
        priors[1, 2, 3]


def test_any_close():
    assert _any_close(1, [1, 2, 3], 0)
    assert not _any_close(1, [1.5, 2, 3], 0)
    assert _any_close(1, [1.5, 2, 3], 0.75)
