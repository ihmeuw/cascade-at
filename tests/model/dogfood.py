"""
Predict data and then fit it.
"""
import logging
from math import nan, inf

import numpy as np
import pandas as pd

from cascade.core import getLoggers
from cascade.model import (
    Model, SmoothGrid, Covariate, DismodGroups, Var,
    ObjectWrapper
)

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
    smooth_grid = SmoothGrid(*age_time)
    set_cols = ["density", "mean", "lower", "upper", "std"]
    if strictly_positive:
        smooth_grid.value.grid.loc[:, set_cols] = [
            "uniform", 1e-2, 1e-9, 5, 1e-3
        ]
    else:
        smooth_grid.value.grid.loc[:, set_cols] = ["uniform", -inf, inf, 0, inf]
    smooth_grid.dage.grid.loc[:, set_cols] = ["uniform", -inf, inf, 0, inf]
    smooth_grid.dtime.grid.loc[:, set_cols] = ["uniform", -inf, inf, 0, inf]
    return smooth_grid


def model_from_var(var, parent_location, multiple_random_effects=False,
                   covariates=None):
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
                    var, (var.ages, var.times), must_be_positive)

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


def fit_sim():
    """user_fit_sim.py from Dismod-AT done with file movement."""
    n_children = 2  # You can change the number of children.
    topology_choice = "born_remission"

    parent_location = 1
    child_locations = [parent_location + 1 + add_child for add_child in range(n_children)]
    locations = pd.DataFrame(dict(
        name=["Universe"] + [f"child_{cidx}" for cidx in range(n_children)],
        parent_id=[nan] + n_children * [parent_location],
        location_id=[parent_location] + child_locations,
    ))

    income_covariate = Covariate("income", reference=0.5)
    covariates = [income_covariate]

    # The model sits on one age and time and has only incidence rate, iota.
    topology = TOPOLOGY[topology_choice]
    truth_var = DismodGroups()
    base_year = 2000
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

    model = model_from_var(truth_var, parent_location, covariates=covariates)

    dismod_objects = ObjectWrapper("running.db")
    dismod_objects.locations = locations
    dismod_objects.parent_location_id = parent_location
    dismod_objects.model = model
    dismod_objects.age_extents = [0, 100]
    dismod_objects.time_extents = [1990, 2010]

    dismod_objects.run_dismod("init")
    dismod_objects.truth_var = truth_var
    data_integrands = [
        "Sincidence", "prevalence", "mtother", "mtspecific", "mtexcess",
        "remission", "relrisk", "mtall", "susceptible", "Tincidence",
        "mtwith", "withC", "mtstandard",
    ]
    data_ages = np.linspace(0, 100, 21)
    data_times = np.linspace(1990, 2010, 5)
    data_predictions = pd.DataFrame([dict(
        integrand=integrand,
        location=location,
        age=age,
        time=time,
        income=0.3,
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
    dismod_objects.run_dismod("init")
    CODELOG.info("fit random")
    dismod_objects.run_dismod("fit random")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fit_sim()
