import numpy as np

from cascade.core.log import getLoggers
from cascade.model import (
    Model, Var, SmoothGrid
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
    ev_settings = local_settings.settings
    parent_location_id = local_settings.parent_location_id
    default_age_time = dict()
    default_age_time["age"] = np.linspace(0, 100, 21)
    default_age_time["time"] = np.linspace(1990, 2015, 6)
    for kind in ["age", "time"]:
        if hasattr(ev_settings.model, f"dfault_{kind}_grid"):
            default_grid = getattr(ev_settings.model, f"dfault_{kind}_grid")
            if default_grid is not None:
                default_age_time[kind] = np.sort(np.array(default_grid, dtype=np.float))

    # Use this age and time when a smooth grid doesn't depend on age and time.
    single_age = default_age_time["age"][:1]
    single_time = [default_age_time["time"][len(default_age_time["time"]) // 2]]
    single_age_time = (single_age, single_time)

    nonzero_rates = [smooth.rate for smooth in ev_settings.rate]

    model = Model(
        nonzero_rates=nonzero_rates,
        parent_location=parent_location_id,
        child_location=list(data.locations.successors(parent_location_id)),
        weights=None,
        covariates=None
    )

    construct_model_rates(default_age_time, single_age_time, ev_settings, model)
    construct_model_random_effects(default_age_time, single_age_time, ev_settings, model)

    return model


def construct_model_rates(default_age_time, single_age_time, ev_settings, model):
    for smooth in ev_settings.rate:
        ages, times = construct_grid_ages_times(default_age_time, single_age_time, smooth)

        rate_grid = SmoothGrid(ages=ages, times=times)
        for kind in ["value", "dage", "dtime"]:
            if getattr(smooth.default, kind) is not None:
                getattr(rate_grid, kind)[:, :] = getattr(smooth.default, kind).prior_object
            else:
                pass  # An unset prior should be unused (dage for one age, dtime for one time)

        model.rate[smooth.rate] = rate_grid


def construct_grid_ages_times(default_age_time, single_age_time, smooth):
    if hasattr(smooth, "age_time_specific") and smooth.age_time_specific == 0:
        return single_age_time

    ages = smooth.age_grid
    if ages is None:
        if smooth.rate == "pini":
            ages = np.array([0], dtype=np.float)
        else:
            ages = default_age_time["age"]
    else:
        ages = np.sort(np.array(ages, dtype=np.float))
    times = smooth.time_grid
    if times is None:
        times = default_age_time["time"]
    else:
        times = np.sort(np.array(times, dtype=np.float))
    return ages, times


def construct_model_random_effects(default_age_time, single_age_time, ev_settings, model):
    if not hasattr(ev_settings, "random_effect"):
        return
    if not ev_settings.random_effect:
        return

    for smooth in ev_settings.random_effect:
        ages, times = construct_grid_ages_times(default_age_time, single_age_time, smooth)

        rate_grid = SmoothGrid(ages=ages, times=times)
        for kind in ["value", "dage", "dtime"]:
            if getattr(smooth.default, kind) is not None:
                getattr(rate_grid, kind)[:, :] = getattr(smooth.default, kind).prior_object
            else:
                pass  # An unset prior should be unused (dage for one age, dtime for one time)

        if hasattr(smooth, "location") and smooth.location is not None and smooth.location != model.location_id:
            location = smooth.location
        else:
            # One smooth for all children when there isn't a child location.
            location = None
        model.random_effect[(smooth.rate, location)] = rate_grid
