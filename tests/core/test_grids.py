from cascade.model.grids import Priors, AgeTimeGrid


def test_Smoothing__development_target():
    # Make a grid using a standard structure. Another option might be
    # a constructor which makes a grid that's denser at extreme ages.
    grid = AgeTimeGrid.uniform(age_start=0, age_end=120, age_step=5, time_start=1990, time_end=2018, time_step=1)

    d_time = Priors(grid)
    d_time[:, :].density = "gaussian"
    d_time[:, :].mean = 0
    d_time[:, :].standard_deviation = 0.1

    # There's a shock in 1995
    d_time[:, 1995].density = "gaussian"
    d_time[:, 1995].mean = 0
    d_time[:, 1995].standard_deviation = 3

    d_age = Priors(grid)
    d_age[:, :].density = "gaussian"
    d_age[:, :].mean = 0
    d_age[:, :].standard_deviation = 0.1

    # Kids are different
    d_age[0:15, :].density = "gaussian"
    d_age[0:15, :].mean = 1
    d_age[0:15, :].standard_deviation = 0.01

    value = Priors(grid)
    value[:, :].density = "gaussian"
    value[:, :].mean = 20
    value[:, :].standard_deviation = 1

    # The shock in 1995 effects the value too
    value[:, :].density = "gaussian"
    value[:, :].mean = 200
    value[:, :].standard_deviation = 10
