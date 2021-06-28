"""
Helper functions for Model and ConstructModel
"""

import numpy as np
import pandas as pd
import itertools as it
from typing import Dict, Tuple, Union, Optional

from cascade_at.model.var import Var
from cascade_at.model.smooth_grid import SmoothGrid, _PriorGrid
from cascade_at.model.priors import Constant, prior_distribution
from cascade_at.core.log import get_loggers
from cascade_at.settings.settings_config import SmoothingPrior, Smoothing

LOG = get_loggers(__name__)


def rectangular_data_to_var(gridded_data):
    """Using this very regular data, where every age and time is present,
    construct an initial guess as a Var object. Very regular means that there
    is a complete set of ages-cross-times."""
    gridded_data = gridded_data.copy()
    try:
        gridded_data['age'] = gridded_data[['age_lower', 'age_upper']].mean(axis=1)
        gridded_data['time'] = gridded_data[['time_lower', 'time_upper']].mean(axis=1)
    except AttributeError:
        LOG.error(f"Data to make a var has columns {gridded_data.columns}")
        raise RuntimeError(
            f"Wrong columns in rectangular_data_to_var {gridded_data.columns}")
    gridded_data = gridded_data.sort_values(by=['age', 'time'])
    guess = Var(ages=sorted(gridded_data['age'].unique()), times=sorted(gridded_data['time'].unique()))
    assert guess.variable_count() == len(gridded_data), \
        "Number of age/time points exceed number of unique age/time points"
    guess[:, :] = gridded_data['mean'].values.reshape((-1, 1))
    return guess


def constraint_from_rectangular_data(rate_var, default_age_time):
    """Takes data on a complete set of ages and times, makes a constraint grid.

    Args:
        rate_var: A function of age and time to represent a rate.
        default_age_time:
    """
    omega_grid = SmoothGrid(ages=default_age_time["age"], times=default_age_time["time"])
    for age, time in omega_grid.age_time():
        omega_grid.value[age, time] = Constant(rate_var(age, time))
    return omega_grid


def smooth_grid_from_smoothing_form(default_age_time: Dict[str, np.array],
                                    single_age_time: Tuple[np.array, np.array],
                                    smooth: Smoothing) -> SmoothGrid:
    """
    Create a new SmoothGrid from the settings in EpiViz-AT at the Smoothing
    level.

    Arguments
    ---------
    default_age_time
        Two members, the ages and the time, with "age" and "time" keys
    single_age_time
        Two members, an age and a time.
    smooth
        A smoothing form from the settings.

    Returns
    -------
    SmoothGrid: A new smooth grid.
    """
    ages, times = construct_grid_ages_times(default_age_time, single_age_time, smooth)
    rate_grid = SmoothGrid(ages=ages, times=times)
    for kind in ["value", "dage", "dtime"]:
        if not smooth.is_field_unset("default") and not smooth.default.is_field_unset(kind):
            getattr(rate_grid, kind)[:, :] = getattr(smooth.default, kind).prior_object
        if not smooth.is_field_unset("mulstd") and not smooth.mulstd.is_field_unset(kind):
            getattr(rate_grid, kind).mulstd_prior = getattr(smooth.mulstd, kind).prior_object
    if not smooth.is_field_unset("detail"):
        for smoothing_prior in smooth.detail:
            for a, t in matching_knots(rate_grid, smoothing_prior):
                getattr(rate_grid, smoothing_prior.prior_type)[a, t] = smoothing_prior.prior_object
    return rate_grid


def matching_knots(rate_grid: SmoothGrid, smoothing_prior: SmoothingPrior):
    """
    Get lower and upper out of the smoothing prior. This uses the age, time,
    and "born" lower and upper bounds to return
    ages and times in the grid that are within those bounds.
    The goal is to apply a prior selectively to those knots.

    Parameters
    ----------
    rate_grid
        A grid of rates
    smoothing_prior
        A single smoothing prior

    Returns
    -------
    Iterator over (a, t) that match. Can be nothing.
    """
    extents = dict()
    for extent in ["age", "time", "born"]:
        extents[extent] = np.zeros(2, dtype=float)
        for side_idx, side, default_extent in [(0, "lower", -np.inf), (1, "upper", np.inf)]:
            name = f"{extent}_{side}"
            if smoothing_prior.is_field_unset(name):
                extents[extent][side_idx] = default_extent
            else:
                extents[extent][side_idx] = getattr(smoothing_prior, name)
    # `np.meshgrid` generates every combination of age and time as two numpy arrays.
    ages, times = np.meshgrid(rate_grid.ages, rate_grid.times)
    assert ages.shape == (len(rate_grid.times), len(rate_grid.ages))
    assert times.shape == (len(rate_grid.times), len(rate_grid.ages))
    in_age = (ages >= extents["age"][0]) & (ages <= extents["age"][1])
    in_time = (times >= extents["time"][0]) & (times <= extents["time"][1])
    in_born = (ages <= times - extents["born"][0]) & (ages >= times - extents["born"][1])
    cover = in_age & in_time & in_born
    if not np.any(cover):
        LOG.info(f"No ages and times match prior with extents {extents}.")
    yield from zip(ages[cover], times[cover])


def construct_grid_ages_times(default_age_time: Dict[str, np.ndarray],
                              single_age_time: Tuple[np.ndarray, np.ndarray],
                              smooth: Smoothing) -> Tuple[np.ndarray, np.ndarray]:

    if not smooth.is_field_unset("age_time_specific") and smooth.age_time_specific == 0:
        return single_age_time

    ages = smooth.age_grid
    if ages is None:
        # hasattr because this may be a Smoothing form or a Covariate form.
        if hasattr(smooth, "rate") and smooth.rate == "pini":
            ages = np.array([0], dtype=float)
        else:
            ages = default_age_time["age"]
    else:
        ages = np.sort(np.array(ages, dtype=float))
    times = smooth.time_grid
    if times is None:
        times = default_age_time["time"]
    else:
        times = np.sort(np.array(times, dtype=float))
    return ages, times


def expand_grid(data_dict: Dict[str, Union[int, float, np.ndarray]]) -> pd.DataFrame:
    """
    Takes lists and turns them into a dictionary of
    """
    rows = it.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())


def estimate_grid_from_draws(grid_priors: _PriorGrid, draws: np.ndarray,
                             ages: np.ndarray, times: np.ndarray,
                             new_prior_distribution: Optional[str] = 'gaussian'):
    """
    Estimates using MLE the parameters for the grid using prior draws.
    Updates the grid_priors _PriorGrid object in place, so returns nothing.
    Also overrides existing prior distributions with another one, defaulting
    to Gaussian. Will skip if the age or time didn't exist in the draws (
    for example with dage and dtime for one age/time point).

    Arguments
    ---------
    grid_priors
        Prior grids that have the .mle() method that can be used to estimate
    draws
        3-d array coming out of `DismodExtractor.gather_draws_for_prior_grid()`
    ages
        Array of ages
    times
        Array of times
    new_prior_distribution
        The new prior distribution to override the existing priors.
    """
    assert isinstance(draws, np.ndarray)
    assert len(draws.shape) == 3
    for idx, row in grid_priors.grid.loc[:, grid_priors.columns + ["age", "time"]].iterrows():
        if row.age in ages:
            age_idx = ages.tolist().index(row.age)
        else:
            continue
        if row.time in times:
            time_idx = times.tolist().index(row.time)
        else:
            continue
        if new_prior_distribution is not None:
            grid_priors.grid['density'] = new_prior_distribution
        grid_priors[row.age, row.time] = grid_priors[row.age, row.time].mle(
            draws[age_idx, time_idx, :]
        )
