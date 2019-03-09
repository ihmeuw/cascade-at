"""
Tests major statistical operations on the model.
"""
from math import nan, isnan
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.interpolate import SmoothBivariateSpline
from scipy.stats import norm

from cascade.model.operations import (
    _estimates_from_one_grid, _dataframe_to_bivariate_spline
)


def make_grid_priors(ages, times, grid_priors, hyper_priors, smooth_id=7):
    age = pd.DataFrame(dict(
        age_id=list(range(0, len(ages) * 2, 2)),  # non-consecutive
        age=ages,  # Use the given order.
    ))
    time = pd.DataFrame(dict(
        time_id=list(range(0, len(times) * 3, 3)),  # non-consecutive
        time=times,  # Use the given order.
    ))
    prior = pd.DataFrame(dict(
        prior_id=list(range(7)),
        prior_name=["value", "dage", "dtime", "lvalue", "ldage", "ldtime", "const"],
        lower=0,
        upper=1,
        mean=0,
        std=0.1,
        eta=0,
        # If the hyper-priors are nan, then these entries are garbage, but
        # nothing references them.
        density_id=[grid_priors["value"], grid_priors["dage"], grid_priors["dtime"],
                    hyper_priors["value"], hyper_priors["dage"], hyper_priors["dtime"], 0],
    ))
    # Make the last one const. If tests want to make a point const, use prior_id=6.
    prior.loc[6, "lower"] = 0.5
    prior.loc[6, "upper"] = 0.5

    smooth = pd.DataFrame(dict(
        smooth_id=[smooth_id],
        smooth_name=[f"test_{smooth_id}"],
        n_age=[len(ages)],
        n_time=[len(times)],
        mulstd_value_prior_id=[3 if not isnan(hyper_priors["value"]) else nan],
        mulstd_dage_prior_id=[4 if not isnan(hyper_priors["dage"]) else nan],
        mulstd_dtime_prior_id=[5 if not isnan(hyper_priors["dtime"]) else nan],
    ))
    smooth_grid = pd.DataFrame(dict(
        smooth_grid_id=list(range(len(ages) * len(times))),
        const_value=nan,
        value_prior_id=0,
        dage_prior_id=1,
        dtime_prior_id=2,
        smooth_id=7,
        age_id=np.tile(age.age_id.values, len(times)),
        time_id=np.repeat(time.time_id.values, len(ages)),
    ))
    # Set impossible dage and dtime to nan
    no_dage = age.loc[age.age.idxmax()].age_id
    no_dtime = time.loc[time.time.idxmax()].time_id
    smooth_grid.loc[smooth_grid.age_id == no_dage, "dage_prior_id"] = nan
    smooth_grid.loc[smooth_grid.time_id == no_dtime, "dtime_prior_id"] = nan

    dismod_file = SimpleNamespace(
        age=age, time=time, prior=prior, smooth=smooth, smooth_grid=smooth_grid
    )
    return dismod_file


def vars_from_priors(dismod_file):
    """Adds a vars table to ``dismod_file`` respecting constant value priors
    Assumes there is only one smoothing.
    """
    # This ignores that dage or dtime can be constant distributions b/c that
    # is a degenerate choice.
    smooth_grid_value = dismod_file.smooth_grid.merge(
        dismod_file.prior, how="left", left_on="value_prior_id", right_on="prior_id")
    no_const_value = smooth_grid_value.const_value.isna()
    not_squeezed = smooth_grid_value.upper > smooth_grid_value.lower
    non_constant = smooth_grid_value[no_const_value & not_squeezed]
    only_relevant_columns = non_constant[["age_id", "time_id"]]
    # Going to ignore columns that identify the type of the var.
    grid_vars = only_relevant_columns.assign(var_type="rate")
    for odd_hyper in ["dage", "dtime", "value"]:
        odd_name = f"mulstd_{odd_hyper}_prior_id"
        if dismod_file.smooth.loc[0].notna()[odd_name]:
            odd_id = int(dismod_file.smooth.loc[0][odd_name])  # noqa: F841
            prior = dismod_file.prior.query("prior_id == @odd_id")
            if (prior["upper"] > prior["lower"]).bool():
                grid_vars = grid_vars.append(pd.DataFrame(
                    {"var_type": [f"mulstd_{odd_hyper}"]},
                ), sort=False)
    dismod_file.var = grid_vars.assign(
        var_id=list(range(len(grid_vars))),
        smooth_id=7,
    )  # Going to ignore a bunch of columns.


def make_bilinear_function(ages, times, corners):
    """This is a nice way to make a plane in space. It behaves like Brad's
    when outside the domain. Stays constant."""
    sp = SmoothBivariateSpline(np.tile(ages, 2), np.repeat(times, 2), corners, kx=1, ky=1)

    def bilinear_f(age, time):
        return sp(age, time)[0, 0]

    return bilinear_f


def draws_at_value(mean, std, cnt):
    """Generates evenly-spaced draws that will have nearly the desired
    mean and std. These are ordered. Why? This takes away the possibility
    that randomly-generated draws will not have the desired mean and std."""
    return norm.isf(np.linspace(0.01, 0.99, cnt), loc=mean, scale=std)


def test_point_grid():
    """SUT is estimate_single_grid. Doing one point value, no hyper-priors."""
    mean = 0.01
    std = 0.001
    cnt = 100
    draws0 = draws_at_value(mean, std, cnt)
    draws1 = draws_at_value(0.1, 0.03, cnt)
    draws_df = pd.DataFrame(dict(
        fit_var_id=np.concatenate([np.repeat([0, 1], cnt), [0]]),
        var_type=["rate"] * 2 * cnt + ["mulstd_exclude"],
        fit_var_value=np.concatenate([draws0, draws1, [9e9]]),
        sample_index=np.concatenate([np.tile(list(range(cnt)), 2), [cnt]]),
        age=np.concatenate([np.repeat([0.0, 10.0], cnt), [0.0]]),
        time=np.concatenate([np.repeat([2000, 2010], cnt), [2000]]),
    ))
    summary = _estimates_from_one_grid(draws_df)
    assert len(summary) == 2
    d0 = int(summary.query("time == 2000").index[0])
    d1 = int(summary.query("time == 2010").index[0])
    assert np.isclose(summary.iloc[d0]["mean"], mean, rtol=0.01)
    assert np.isclose(summary.iloc[d0]["std"], std, rtol=0.1)
    assert np.isclose(summary.iloc[d1]["mean"], 0.1, rtol=0.1)
    assert np.isclose(summary.iloc[d1]["std"], 0.03, rtol=0.1)


def test_bivariate_spline():
    df = pd.DataFrame(dict(
        age=[1.0, 1.0, 0.0, 0.0],
        time=[2000, 2010, 2000, 2010],
        mean=[3, 5, -2, 17],
    ))
    f = _dataframe_to_bivariate_spline(df)
    assert np.isclose(f(1.0, 2000), 3)
    assert np.isclose(f(1.0, 2010), 5)
    assert np.isclose(f(0.0, 2000), -2)
    assert np.isclose(f(0.0, 2010), 17)
