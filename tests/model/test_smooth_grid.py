import pytest

from cascade.model import SmoothGrid
from cascade.model.priors import Gaussian


@pytest.mark.skip()
def test_smooth_grid__development_target():
    grid = SmoothGrid(([0, 5, 10, 20], [2000, 2010]))
    # Assign a default value to all points.
    grid.value[:, :] = Gaussian(mean=0.01, standard_deviation=5.0, lower=0.0, upper=10.0)
    assert grid.value[0, 0]
    # Assign a prior to points by region of age or time.
    grid.value[2:, :] = Gaussian(mean=0.1, standard_deviation=3.0, lower=1e-4, upper=12)

    def by_size(a, t, da, dt, previous_prior):
        return previous_prior.update(standard_deviation=da * 0.1)

    # Assign a value as a function of age, time, age difference, time difference,
    # and previous prior
    grid.dage[:, :].apply(by_size)
    # Do the same over a region.
    grid.dage[:, 2000:2005].apply(by_size)
