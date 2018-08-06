from cascade.model.grids import Priors, AgeTimeGrid, GaussianPrior, UniformPrior


def test_prior_equality():
    a = GaussianPrior(0, 1)
    b = GaussianPrior(0, 1)
    assert a == b

    a = GaussianPrior(0, 1, -1, 1)
    b = GaussianPrior(0, 1, -1, 1)
    assert a == b

    a = UniformPrior(10)
    b = UniformPrior(10)
    assert a == b

    a = UniformPrior(10, "test_prior")
    b = UniformPrior(10, "test_prior")
    assert a == b


def test_prior_nonequality():
    a = GaussianPrior(0, 1)
    b = GaussianPrior(1, 1)
    assert a != b

    a = UniformPrior(1)
    b = UniformPrior(-1)
    assert a != b

    a = GaussianPrior(0, 1, name="test_prior")
    b = GaussianPrior(0, 1, name="other_test_prior")
    assert a != b

    a = GaussianPrior(0, 1)
    b = UniformPrior(0)
    assert a != b


def test_prior_hashing():
    s = {GaussianPrior(0, 1), UniformPrior(1), GaussianPrior(0, 1), UniformPrior(2), UniformPrior(1)}

    assert len(s) == 3
    assert GaussianPrior(0, 1) in s
    assert UniformPrior(10) not in s


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
