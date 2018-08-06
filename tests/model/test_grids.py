from cascade.model.grids import Priors, AgeTimeGrid
from cascade.model.priors import GaussianPrior


def test_Smoothing__development_target():
    # Make a grid using a standard structure. Other options might be
    # a constructor which makes a grid that's denser at extreme ages.
    grid = AgeTimeGrid.uniform(age_start=0, age_end=120, age_step=5, time_start=1990, time_end=2018, time_step=1)

    d_time = Priors(grid)
    d_time[:, :].prior = GaussianPrior(0, 0.1)

    # There's a shock in 1995
    d_time[:, 1995].prior = GaussianPrior(0, 3)

    d_age = Priors(grid)
    d_age[:, :].prior = GaussianPrior(0, 0.1)

    # Kids are different
    d_age[0:15, :].prior = GaussianPrior(1, 0.01)

    value = Priors(grid)
    value[:, :].prior = GaussianPrior(20, 1)

    # The shock in 1995 effects the value too
    value[:, 1995].prior = GaussianPrior(200, 10)

    assert value[10, 1995].prior == GaussianPrior(200, 10)
    assert value[10, 1996].prior == GaussianPrior(20, 1)

    # dm = DismodFile(...)
    # dm.add_smooth(d_age)
    # dm.add_smooth(d_time)
    # dm.add_smooth(value)
    # dm.flush()
    # Or something
