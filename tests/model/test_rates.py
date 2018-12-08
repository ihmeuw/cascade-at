from cascade.model.rates import Rate, Smooth
from cascade.model.grids import PriorGrid, AgeTimeGrid
from cascade.model import priors


def test_positive():
    grid = AgeTimeGrid.uniform(age_lower=0, age_upper=120, age_step=5, time_lower=1990, time_upper=2018, time_step=1)

    positive_pg = PriorGrid(grid)
    positive_pg[:, :].prior = priors.Gaussian(0.1, 0.1, lower=0.01)
    assert Rate("iota", parent_smooth=Smooth(positive_pg)).positive

    zero_pg = PriorGrid(grid)
    zero_pg[:, :].prior = priors.Uniform(0, 0)
    assert not Rate("iota", parent_smooth=Smooth(zero_pg)).positive

    positive_and_negative_pg = PriorGrid(grid)
    positive_and_negative_pg[:, :].prior = priors.Gaussian(0.1, 0.1, lower=0.01)
    positive_and_negative_pg[0, 1995].prior = priors.Uniform(0, 0)
    assert not Rate("iota", parent_smooth=Smooth(positive_and_negative_pg)).positive

    assert not Rate("iota", parent_smooth=None).positive


def test_zero():
    grid = AgeTimeGrid.uniform(age_lower=0, age_upper=120, age_step=5, time_lower=1990, time_upper=2018, time_step=1)

    assert Rate("iota", parent_smooth=None).zero

    zero_pg = PriorGrid(grid)
    zero_pg[:, :].prior = priors.Uniform(0, 0)
    assert Rate("iota", parent_smooth=Smooth(zero_pg)).zero

    positive_pg = PriorGrid(grid)
    positive_pg[:, :].prior = priors.Gaussian(0.1, 0.1, lower=0.01)
    assert not Rate("iota", parent_smooth=Smooth(positive_pg)).zero

    positive_and_negative_pg = PriorGrid(grid)
    positive_and_negative_pg[:, :].prior = priors.Gaussian(0.1, 0.1, lower=0.01)
    positive_and_negative_pg[0, 1995].prior = priors.Uniform(0, 0)
    assert not Rate("iota", parent_smooth=Smooth(positive_and_negative_pg)).zero
