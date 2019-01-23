from numpy import isclose

import pytest

from cascade.model import SmoothGrid
from cascade.model.priors import Gaussian


def test_smooth_grid__development_target():
    """This is how we hope to work with SmoothGrid objects."""
    grid = SmoothGrid([0, 5, 10, 20], [2000, 2010])
    # Assign a default value to all points.
    grid.value[:, :] = Gaussian(mean=0.01, standard_deviation=5.0, lower=0.0, upper=10.0)
    assert grid.value[5, 2010].density == "gaussian"
    assert grid.value[5, 2010].mean == 0.01
    grid.value[10, 2000:] = Gaussian(mean=0.1, standard_deviation=5.0)

    # Assign a prior to points by region of age or time.
    grid.value[2:15, 2005:] = Gaussian(mean=0.1, standard_deviation=3.0, lower=1e-4, upper=12)
    assert grid.value[5, 2010].mean == 0.1

    for a, t in grid.age_time():
        grid.value[a, t] = Gaussian(mean=0.01, standard_deviation=0.001)

    for a, t, da, dt in grid.age_time_diff():
        grid.dage[a, t] = Gaussian(mean=0, standard_deviation=0.01 * da)

    for a, t in grid.age_time():
        grid.value[a, t] = grid.value[a, t].assign(standard_deviation=0.2)
    assert grid.value[20, 2000].standard_deviation == 0.2


def test_smooth_grid__edge_cases_as_written():
    """If we set a particular age and time, that exact one is set."""
    grid = SmoothGrid([0, 5, 10, 20], [2000, 2010])
    for a, t in grid.age_time():
        low = a * 0.01
        high = t - 1999
        grid.value[a, t] = Gaussian(mean=(low + high) / 2, standard_deviation=5.0, lower=low, upper=high)

    assert isclose(grid.value[0, 2000].lower, 0)
    assert isclose(grid.value[0, 2000].upper, 1)
    assert isclose(grid.value[5, 2010].lower, 0.05)
    assert isclose(grid.value[5, 2010].upper, 11)


def test_out_of_bounds_setitem():
    grid = SmoothGrid([0, 5, 10, 20], [2000, 2010])
    with pytest.raises(ValueError):
        grid.value[-0.5, 2000] = Gaussian(mean=0.1, standard_deviation=5.0)

    with pytest.raises(ValueError):
        grid.value[-0.5, 2015:] = Gaussian(mean=0.1, standard_deviation=5.0)

    with pytest.raises(ValueError):
        grid.value[:, 2015:] = Gaussian(mean=0.1, standard_deviation=5.0)


def test_smooth_grid_mulstd():
    grid = SmoothGrid([0, 5, 10, 20], [2000, 2010])
    grid.value.mulstd_prior = Gaussian(mean=0.1, standard_deviation=0.02)
    assert grid.value.mulstd_prior.standard_deviation == 0.02
    assert isinstance(grid.value.mulstd_prior, Gaussian)
