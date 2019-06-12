"""
Predict data and then fit it.
"""
import logging
from math import nan, inf
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.random import RandomState

from cascade.core import getLoggers
from cascade.dismod import DismodATException
from cascade.model import (
    Model, SmoothGrid, Covariate, DismodGroups, Var,
    ObjectWrapper, Gaussian, Uniform
)
from cascade.dismod.process_behavior import get_fit_output

CODELOG, MATHLOG = getLoggers(__name__)


def reasonable_grid_from_var(var, age_time, strictly_positive):
    """
    Create a smooth grid with priors that are Uniform and
    impossibly large, in the same shape as a Var.

    Args:
        var (Var): A single var grid.
        age_time (List[float],List[float]): Tuple of ages and times
            on which to put the prior distributions.
        strictly_positive (bool): Whether the value prior is positive.

    Returns:
        SmoothGrid: A single smooth grid with Uniform distributions.
    """
    if age_time is None:
        age_time = (var.ages, var.times)
    smooth_grid = SmoothGrid(*age_time)
    if strictly_positive:
        small = 1e-9
        for age, time in smooth_grid.age_time():
            meas_value = max(var(age, time), small)
            meas_std = 0.1 * meas_value + 0.01
            smooth_grid.value[age, time] = Gaussian(meas_value, meas_std, lower=small, upper=5)
    else:
        for age, time in smooth_grid.age_time():
            meas_value = var(age, time)
            meas_std = 0.1
            smooth_grid.value[age, time] = Gaussian(meas_value, meas_std)

    # Set all grid points so that the edges are set.
    smooth_grid.dage[:, :] = Uniform(-inf, inf, 0)
    smooth_grid.dtime[:, :] = Uniform(-inf, inf, 0)
    # Then calculate actual dage and dtime.
    ages = smooth_grid.ages
    times = smooth_grid.times
    mid_time = times[0]
    for age_start, age_finish in zip(ages[:-1], ages[1:]):
        dage = var(age_finish, mid_time) - var(age_start, mid_time)
        smooth_grid.dage[age_start, :] = Gaussian(dage, 0.01)
    mid_age = ages[0]
    for time_start, time_finish in zip(times[:-1], times[1:]):
        dtime = var(mid_age, time_finish) - var(mid_age, time_start)
        smooth_grid.dtime[:, time_start] = Gaussian(dtime, 0.01)

    return smooth_grid


def model_from_var(var, parent_location, age_time=None,
                   multiple_random_effects=False, covariates=None):
    """
    Given values across all rates, construct a model with loose priors
    in order to be able to predict from those rates.

    Args:
        var (DismodGroups[Var]): Values on grids.
        parent_location (int): A parent location, because that isn't
            in the keys.
        multiple_random_effects (bool): Create a separate smooth grid for
            each random effect in the Var. The default is to create a
            single smooth grid for all random effects.
        covariates (List[Covariate]): List of covariate names.
    Returns:
        Model: with Uniform distributions everywhere and no mulstd.
    """
    child_locations = list(sorted({k[1] for k in var.random_effect.keys()}))
    nonzero_rates = list(var.rate.keys())
    model = Model(
        nonzero_rates,
        parent_location,
        child_locations,
        covariates=covariates,
    )

    strictly_positive = dict(rate=True)
    for group_name, group in var.items():
        is_random_effect = group_name == "random_effect"
        skip_re_children = is_random_effect and not multiple_random_effects
        for key, var in group.items():
            if skip_re_children:
                # Make one smooth grid for all children.
                assign_key = (key[0], None)
            else:
                assign_key = key

            if assign_key not in model[group_name]:
                must_be_positive = strictly_positive.get(group_name, False)
                model[group_name][assign_key] = reasonable_grid_from_var(
                    var, age_time, must_be_positive)

    return model


TOPOLOGY = dict(
    single_measure_a=["omega"],
    single_measure_b=["iota"],
    born=["omega", "chi"],
    no_death=["iota", "rho"],
    no_remission=["iota", "chi", "omega"],
    born_remission=["rho", "omega", "chi"],
    illness_death=["iota", "chi", "rho", "omega"],
)
"""There are 16 ways to set up rates on SCR, and these
are the ones that makes some sense."""


def choose_ages(age_cnt, age_range, expansion=1.5):
    """Chooses ages using a geometric series so the ranges increase."""
    interval_cnt = age_cnt - 1
    interval = age_range[1] - age_range[0]
    base_interval = interval * (expansion - 1) / (expansion**interval_cnt - 1)
    intervals = [base_interval * expansion**idx for idx in range(interval_cnt)]
    return np.concatenate([[0], np.cumsum(intervals)])


def fit_sim():
    """user_fit_sim.py from Dismod-AT done with file movement."""
    rng = RandomState(3947242)
    n_children = 2  # You can change the number of children.
    topology_choice = "born_remission"
    age_cnt = 5
    time_cnt = 3
    covariate_cnt = 20
    data_integrands = [
        "Sincidence", "prevalence", "mtother", "mtspecific", "mtexcess",
        "remission", "relrisk", "mtall", "susceptible", "Tincidence",
        "mtwith", "withC", "mtstandard",
    ]
    parent_location = 1
    child_locations = [parent_location + 1 + add_child
                       for add_child in range(n_children)]
    locations = pd.DataFrame(dict(
        name=["Universe"] + [f"child_{cidx}" for cidx in range(n_children)],
        parent_id=[nan] + n_children * [parent_location],
        location_id=[parent_location] + child_locations,
    ))

    base_year = 2000
    covariates = [
        Covariate(f"sdi{cov_create_idx}", reference=0.1)
        for cov_create_idx in range(covariate_cnt)
    ]

    # The model sits on one age and time and has only incidence rate, iota.
    topology = TOPOLOGY[topology_choice]
    truth_var = DismodGroups()
    if "iota" in topology:
        iota = Var([0, 50], [base_year])
        iota[0, :] = 0
        iota[50, :] = 0.01
        truth_var.rate["iota"] = iota
    if "rho" in topology:
        rho = Var([50], [base_year])
        rho[:, :] = 0.005
        truth_var.rate["rho"] = rho
    if "omega" in topology:
        omega = Var([0, 1, 10, 100], [base_year])
        omega[0, base_year] = 0.08
        omega[1, base_year] = 0.04
        omega[10, base_year] = 0.001
        omega[100, base_year] = 0.2
        truth_var.rate["omega"] = omega
    if "chi" in topology:
        chi = Var([0, 100], [base_year])
        chi[0, base_year] = 0.0
        chi[100, base_year] = 0.2
        truth_var.rate["chi"] = chi

    for make_cov_var in covariates:
        cov_var = Var([50], base_year)
        cov_var[:, :] = 0.1
        if rng.choice([False, True], p=[0.2, 0.8]):
            apply_to = rng.choice(topology)
            truth_var.alpha[(make_cov_var.name, apply_to)] = cov_var
        else:
            apply_to = rng.choice(data_integrands)
            truth_var.beta[(make_cov_var.name, apply_to)] = cov_var

    model = model_from_var(
        truth_var, parent_location, covariates=covariates)

    db_file = Path("running.db")
    if db_file.exists():
        db_file.unlink()
    dismod_objects = ObjectWrapper(db_file)
    dismod_objects.locations = locations
    dismod_objects.parent_location_id = parent_location
    dismod_objects.model = model
    dismod_objects.age_extents = [0, 100]
    dismod_objects.time_extents = [1990, 2010]

    dismod_objects.run_dismod("init")
    dismod_objects.truth_var = truth_var

    data_ages = np.linspace(0, 100, 21)
    data_times = np.linspace(1990, 2010, 5)
    covariate_values = {
        make_cov_val.name: 0.1
        for make_cov_val in covariates
    }
    data_predictions = pd.DataFrame([dict(
        integrand=integrand,
        location=location,
        age=age,
        time=time,
        **covariate_values,
    )
        for integrand in data_integrands
        for location in child_locations
        for age in data_ages
        for time in data_times
    ])

    dismod_objects.avgint = data_predictions
    dismod_objects.run_dismod(["predict", "truth_var"])
    predicted, not_predicted = dismod_objects.predict
    print(predicted.columns)

    data = predicted.drop(columns=["sample_index"])
    data = data.assign(
        density="gaussian",
        std=data["mean"] * 0.1 + 0.001,
        eta=1e-5,
    )
    dismod_objects.data = data

    # Now make a new model with different ages and times.
    ages = np.linspace(0, 100, age_cnt)
    times = np.linspace(1990, 2010, time_cnt)
    model = model_from_var(
        truth_var,
        parent_location,
        age_time=(ages, times),
        covariates=covariates
    )
    dismod_objects.model = model
    option = dict(random_seed=0,
                  ode_step_size=10,
                  zero_sum_random="iota",  # Zero-sum random affects result.
                  quasi_fixed="true",
                  derivative_test_fixed="first-order",
                  max_num_iter_fixed=100,
                  print_level_fixed=5,
                  tolerance_fixed=1e-8,
                  derivative_test_random="second-order",
                  max_num_iter_random=100,
                  tolerance_random=1e-8,
                  print_level_random=0,
                  )
    dismod_objects.set_option(**option)
    dismod_objects.run_dismod("init")
    CODELOG.info("fit both")
    stdout, stderr = dismod_objects.run_dismod("fit both")
    exit_kind, exit_string, iteration_cnt = get_fit_output(stdout)
    print(f"{exit_string} with {iteration_cnt} iterations")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        fit_sim()
    except DismodATException as dat_exc:
        print(dat_exc)
