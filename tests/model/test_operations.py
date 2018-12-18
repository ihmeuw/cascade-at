"""
Tests major statistical operations on the model.
"""
from math import nan, isnan
from types import SimpleNamespace  # Since Python 3.3

import numpy as np
from numpy.random import shuffle
from scipy.stats import norm
from scipy.interpolate import SmoothBivariateSpline
import pandas as pd

from cascade.model.operations import (
    estimate_single_grid, expand_priors, reduce_priors, concatenate_grids_and_priors,
    random_field_iterator
)


# Construction of posteriors from priors.
# Let's make smooth grids as tests, generate draws on them,
# and then ensure the priors look like the distributions of the draws.

# Begin from a DismodFile. The relevant tables are:
#     age: age_id, age
#     time: time_id, time
#     prior: prior_id, prior_name, lower, upper, mean, std, eta, nu, density_id
#     smooth: smooth_id, smooth_name, n_age, n_time, mulstd_value_prior,
#        mulstd_dage_prior, mulstd_dtime_prior
#     smooth_grid: smooth_grid_id, const_value, value_prior_id, dage_prior_id,
#        dtime_prior_id, smooth_id, age_id, time_id
#
# Let's specify one of these.

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
    prior.loc[6]["lower"] = 0.5
    prior.loc[6]["upper"] = 0.5

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
                    [[f"mulstd_{odd_hyper}"]],
                    columns=["var_type"],
                ))
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


def test_var_grid_iterator():
    # parent rate 0, parent rate 1,
    # child 1 rate 0, child 2 rate 0, with shared smooth grid
    # child 1 rate 1, different smooth grid.
    # Note there can be two rates with the _same smooth id_ but different nodes.
    var_df = pd.DataFrame(dict(
        var_id=[0, 1, 2, 3, 4, 5, 6, 7],
        smooth_id=[0, 1, 2, 2, 3, 4, 5, 6],
        var_type=["rate", "rate", "rate", "rate", "rate", "mulcov_meas_value",
                  "mulcov_rate_value", "mulcov_meas_std"],
        node_id=[0, 0, 1, 2, 1, nan, nan, nan],
        rate_id=[0, 1, 0, 0, 1, nan, 2, nan],
        integrand_id=[nan, nan, nan, nan, nan, 7, nan, 6],
        covariate_id=[nan, nan, nan, nan, nan, 8, 9, 10],
    ))
    assert len(list(random_field_iterator(var_df))) == 8
    for kind, index, grid in random_field_iterator(var_df.iloc[[0]]):
        assert kind == "rate"
        assert index[0] == 0
    for kind, index, grid in random_field_iterator(var_df.iloc[2:4]):
        assert index[0] == 2
    a = list()
    for kind, index, grid in random_field_iterator(var_df):
        a.append((kind, index[0], index[1], index[2]))
    assert len(set(a)) == 8, "two are alike"


def test_array_grid():
    """Testing with grid of ages and times"""
    ages = [0, 1, 5, 10, 20, 40]
    times = [1990, 1995, 2000]
    shuffle(ages)  # We can choose to make these out of order.
    dismod_file = make_grid_priors(
        ages=ages,
        times=times,
        grid_priors=dict(value=1, dage=0, dtime=0),
        hyper_priors=dict(value=0, dage=3, dtime=0),
    )
    n = len(ages)
    m = len(times)
    assert len(dismod_file.smooth_grid == n * m)
    vars_from_priors(dismod_file)
    # One var for value at point, one for single hyper-prior.
    assert len(dismod_file.var) == n * m + 3

    mean = 0.01
    std = 0.001
    cnt = 20
    draws = draws_at_value(mean, std, cnt)
    draws_df = pd.DataFrame(dict(
        fit_var_id=np.repeat(dismod_file.var["var_id"].tolist(), cnt),
        fit_var_value=np.tile(draws.tolist(), len(dismod_file.var)),
        sample_index=list(range(cnt * len(dismod_file.var)))
    ))
    assert not draws_df.empty
    smooth, smooth_grid, prior = estimate_single_grid(draws_df, dismod_file, 7)
    assert len(prior) == (n - 1) * m + (m - 1) * n + n * m + 3
    has_prior = smooth_grid[smooth_grid.value_prior_id.isna()]
    grid_with_prior = has_prior.merge(prior, left_on="value_prior_id", right_on="prior_id", how="left")
    for row_idx, row in grid_with_prior.iterrows():
        assert np.isclose(row["mean"], mean, rtol=0.01)
        assert np.isclose(row["std"], std, rtol=0.15)


def test_point_grid():
    """SUT is estimate_single_grid. Doing one point value, no hyper-priors."""
    dismod_file = make_grid_priors(
        ages=[40],
        times=[1990],
        grid_priors=dict(value=1, dage=0, dtime=0),
        hyper_priors=dict(value=nan, dage=nan, dtime=nan),
    )
    vars_from_priors(dismod_file)
    # One var for value at point, one for single hyper-prior.
    assert len(dismod_file.var) == 1

    mean = 0.01
    std = 0.001
    cnt = 100
    draws = draws_at_value(mean, std, cnt)
    draws_df = pd.DataFrame(dict(
        fit_var_id=0,
        fit_var_value=draws,
        sample_index=list(range(cnt))
    ))
    smooth, smooth_grid, prior = estimate_single_grid(draws_df, dismod_file, 7)
    assert len(smooth) == 1
    assert len(smooth_grid) == 1
    assert len(prior) == 1  # The value and three const.
    assert np.isclose(prior.iloc[0]["mean"], mean, rtol=0.01)
    assert np.isclose(prior.iloc[0]["std"], std, rtol=0.1)


def test_expand_priors():
    """different grids to expand"""
    point_db = make_grid_priors(
        ages=[40],
        times=[1990],
        grid_priors=dict(value=2, dage=1, dtime=1),
        hyper_priors=dict(value=nan, dage=nan, dtime=nan),
    )
    assert len(point_db.smooth_grid) == 1, "there is one grid point"
    assert point_db.smooth_grid.value_prior_id.notna().bool()
    assert point_db.smooth_grid.dage_prior_id.isna().bool()
    assert point_db.smooth_grid.dtime_prior_id.isna().bool()

    expanded = expand_priors(point_db.smooth, point_db.smooth_grid, point_db.prior)
    assert len(expanded) == 6
    assert len(expanded[expanded.density_id.notna()]) == 1
    smooth, smooth_grid, priors = reduce_priors(point_db.smooth, expanded)
    assert "prior_id" in priors.columns
    assert len(smooth_grid) == 1
    assert len(priors) == 1
    assert isnan(smooth.loc[0, "mulstd_value_prior_id"])
    assert isnan(smooth.loc[0, "mulstd_dage_prior_id"])
    assert isnan(smooth.loc[0, "mulstd_dtime_prior_id"])


# This section tests concatenation of results in
# estimate_priors_from_posterior_draws.
def make_point_dismod_file(smooth_id, age, time):
    dismod_file = make_grid_priors(
        ages=[age],
        times=[time],
        grid_priors=dict(value=1, dage=0, dtime=0),
        hyper_priors=dict(value=nan, dage=nan, dtime=nan),
        smooth_id=smooth_id
    )
    vars_from_priors(dismod_file)
    # One var for value at point, one for single hyper-prior.
    assert len(dismod_file.var) == 1

    mean = 0.01
    std = 0.001
    cnt = 100
    draws = draws_at_value(mean, std, cnt)
    draws_df = pd.DataFrame(dict(
        fit_var_id=0,
        fit_var_value=draws,
        sample_index=list(range(cnt))
    ))
    smooth, smooth_grid, prior = estimate_single_grid(draws_df, dismod_file, smooth_id)
    return smooth, smooth_grid, prior


def make_grid_dismod_file(smooth_id, ages, times):
    shuffle(ages)  # We can choose to make these out of order.
    dismod_file = make_grid_priors(
        ages=ages,
        times=times,
        grid_priors=dict(value=1, dage=0, dtime=0),
        hyper_priors=dict(value=0, dage=3, dtime=0),
        smooth_id=smooth_id
    )
    n = len(ages)
    m = len(times)
    assert len(dismod_file.smooth_grid == n * m)
    vars_from_priors(dismod_file)
    # One var for value at point, one for single hyper-prior.
    assert len(dismod_file.var) == n * m + 3

    mean = 0.01
    std = 0.001
    cnt = 20
    draws = draws_at_value(mean, std, cnt)
    draws_df = pd.DataFrame(dict(
        fit_var_id=np.repeat(dismod_file.var["var_id"].tolist(), cnt),
        fit_var_value=np.tile(draws.tolist(), len(dismod_file.var)),
        sample_index=list(range(cnt * len(dismod_file.var)))
    ))
    assert not draws_df.empty
    smooth, smooth_grid, prior = estimate_single_grid(draws_df, dismod_file, smooth_id)
    return smooth, smooth_grid, prior


def test_concatenate_grids():
    smooths = list()
    grids = list()
    priors = list()
    smooth_idx = 0

    for age, time in [(0, 1990), (40, 2010), (20, 2000)]:
        smooth, grid, prior = make_point_dismod_file(smooth_idx, age, time)
        smooths.append(smooth)
        grids.append(grid)
        priors.append(prior.assign(smooth_id=smooth_idx))
        smooth_idx += 1

    for age, time in [([0, 5, 10], [1990, 2000]),
                      ([0, 0.19, 100], [2010, 2015, 2016]),
                      ([20, 60], [2000, 2001, 2002, 2003, 2004, 2005, 2010])]:
        smooth, grid, prior = make_grid_dismod_file(smooth_idx, age, time)
        smooths.append(smooth)
        grids.append(grid)
        priors.append(prior.assign(smooth_id=smooth_idx))
        smooth_idx += 1

    all_smooth, all_grid, all_priors = concatenate_grids_and_priors(smooths, grids, priors)
    assert len(all_smooth) == smooth_idx
    assert len(all_grid) == sum(len(g) for g in grids)
    assert len(all_priors) == sum(len(p) for p in priors)
    assert len(all_priors.prior_id.unique()) == len(all_priors)
