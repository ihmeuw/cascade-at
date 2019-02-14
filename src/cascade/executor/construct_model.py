import numpy as np

from cascade.core.log import getLoggers
from cascade.model import (
    Model, Var, SmoothGrid,
    Uniform, Gaussian
)

CODELOG, MATHLOG = getLoggers(__name__)


def rectangular_data_to_var(gridded_data):
    """Using this very regular data, where every age and time is present,
    construct an initial guess as a Var object. Very regular means that there
    is a complete set of ages-cross-times."""
    initial_ages = np.sort(np.unique(0.5 * (gridded_data.age_lower + gridded_data.age_upper)))
    initial_times = np.sort(np.unique(0.5 * (gridded_data.time_lower + gridded_data.time_upper)))

    guess = Var(ages=initial_ages, times=initial_times)
    for age, time in guess.age_time():
        found = gridded_data.query(
            "(age_lower <= @age) & (@age <= age_upper) & (time_lower <= @time) & (@time <= time_upper)")
        assert len(found) == 1, f"found {found}"
        guess[age, time] = float(found.iloc[0]["mean"])
    return guess


def const_value(value):

    def at_function(age, time):
        return value

    return at_function


def construct_model(data, local_settings):
    ages = np.array(local_settings.settings.model.default_age_grid, dtype=np.float)
    times = np.array(local_settings.settings.model.default_time_grid, dtype=np.float)

    if data.age_specific_death_rate:
        initial_mtother_guess = rectangular_data_to_var(data.age_specific_death_rate)
    else:
        initial_mtother_guess = const_value(0.01)

    model = Model(nonzero_rates=["omega"],
                  parent_location=local_settings.parent_location_id,
                  child_location=[],
                  weights=None,
                  covariates=None)
    omega_grid = SmoothGrid(ages=ages, times=times)
    omega_grid.value[:, :] = Uniform(lower=0, upper=1.5, mean=0.01)
    # omega_grid.value[:, :] = Gaussian(lower=0, upper=1.5, mean=0.01, standard_deviation=value_stdev)
    # XXX This for-loop sets the mean as the initial guess because the fit command
    # needs the initial var and scale var to be on the same age-time grid, and
    # this set is not. The session could switch it to the other age-time grid.
    for age, time in omega_grid.age_time():
        omega_grid.value[age, time] = omega_grid.value[age, time].assign(mean=initial_mtother_guess(age, time))

    omega_grid.dage[:, :] = Gaussian(mean=0.0, standard_deviation=0.5)
    omega_grid.dtime[:, :] = Gaussian(mean=0.0, standard_deviation=0.5)
    model.rate["omega"] = omega_grid
    return model
