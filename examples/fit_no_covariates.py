"""
This example constructs makes a test disease model, similar to
diabetes, and sends that data to DismodAT.

1.  The test model is constructed by specifying functions for
    the primary rates, incidence, remission, excess mortality rate,
    and total mortality. Then this is solved to get prevalence over time.

2.  Then construct demographic observations of these rates. This means
    averaging over the rates for given ages and times.

3.  Load a subset of these observations into a DismodAT file and run
    DismodAT on it.

This example works in cohort time, so that rates don't change over
years.

.. autofunction: pretend_diabetes


.. autofunction: observe_demographic_rates

"""
from argparse import Namespace
import itertools as it
import logging
from pathlib import Path
import pickle
from timeit import default_timer as timer

import numpy as np
import pandas as pd

import cascade.input_data.db.bundle
from cascade.testing_utilities import make_execution_context
from cascade.dismod.db.metadata import IntegrandEnum, DensityEnum, RateName
from dmfile_create import write_to_file

LOGGER = logging.getLogger("fit_no_covariates")


def cached_bundle_load(context, bundle_id, tier_idx):
    cache_bundle = Path(f"{bundle_id}.pkl")
    if cache_bundle.exists():
        LOGGER.info(f"Reading bundle from {cache_bundle}. "
                    f"If you want to get a fresh copy, delete this file.")
        return pickle.load(cache_bundle.open("rb"))

    LOGGER.debug(f"Begin getting study covariates {bundle_id}")
    bundle_begin = timer()
    bundle, covariate = cascade.input_data.db.bundle.bundle_with_study_covariates(
        context, bundle_id, tier_idx)
    LOGGER.debug(f"bundle is {bundle} time {timer() - bundle_begin}")

    pickle.dump((bundle, covariate),
                cache_bundle.open("wb"),
                pickle.HIGHEST_PROTOCOL)
    return bundle, covariate


def choose_constraints(bundle, measure):
    LOGGER.debug(f"measures in bundle {bundle['measure'].unique()}")
    observations = pd.DataFrame(bundle[bundle["measure"] != measure])
    constraints = pd.DataFrame(bundle[bundle["measure"] == measure])
    return observations, constraints


def fake_mtother():
    ages = list(range(0, 105, 5))
    years = list(range(1965, 2020, 5))
    age_time = np.array(list(it.product(ages, years)), dtype=np.float)
    return pd.DataFrame(dict(
        measure="mtother",
        mean=0.001,
        sex="Male",
        standard_error=0.0007,
        age_start=age_time[:, 0],
        age_end=age_time[:, 0],
        year_start=age_time[:, 1],
        year_end=age_time[:, 1],
    ))


def retrieve_external_data(config):
    context = make_execution_context()
    bundle, covariate = cached_bundle_load(context, config.bundle_id, config.tier_idx)
    with_mtother = pd.concat([bundle, fake_mtother()], ignore_index=True)
    LOGGER.debug(f"Data now {with_mtother}")

    # Split the input data into observations and constraints.
    bundle_observations, bundle_constraints = choose_constraints(with_mtother, "mtother")
    return Namespace(observations=bundle_observations, constraints=bundle_constraints)


def bundle_to_observations(config, bundle_df):
    """Convert bundle into an internal format."""
    if "incidence" in bundle_df["measure"].values:
        LOGGER.info("Bundle has incidence. Replacing with Sincidence. Is this correct?")
        bundle_df["measure"] = bundle_df["measure"].replace("incidence", "Sincidence")

    for check_measure in bundle_df["measure"].unique():
        if check_measure not in IntegrandEnum.__members__:
            raise KeyError(f"{check_measure} isn't a name known to Cascade.")

    if "location_id" in bundle_df.columns:
        location_id = bundle_df["location_id"]
    else:
        location_id = np.full(len(bundle_df), config.location_id, dtype=np.int)

    # assume using demographic notation because this bundle uses it.
    demographic_interval_specification = 1

    # Stick with year_start instead of time_start because that's what's in the
    # bundle, so it's probably what modelers use. Would be nice to pair
    # start with finish or begin with end.
    return pd.DataFrame({
        "measure": bundle_df["measure"],
        "location_id": location_id,
        "density": DensityEnum.gaussian,
        "weight": "constant",
        "age_start": bundle_df["age_start"],
        "age_end": bundle_df["age_end"] + demographic_interval_specification,
        # The years should be floats in the bundle.
        "year_start": bundle_df["year_start"].astype(np.float),
        "year_end": bundle_df["year_end"].astype(np.float) + demographic_interval_specification,
        "mean": bundle_df["mean"],
        "standard_error": bundle_df["standard_error"],
    })


def age_year_from_data(df):
    results = dict()
    for topic in ["age", "year"]:
        values = np.unique(np.hstack([df[f"{topic}_start"], df[f"{topic}_end"]]))
        values.sort()
        results[topic] = pd.DataFrame({topic: values})
    return results["age"], results["year"]


RATE_TO_INTEGRAND = dict(
    iota=IntegrandEnum.Sincidence,
    rho=IntegrandEnum.remission,
    chi=IntegrandEnum.mtexcess,
    omega=IntegrandEnum.mtother,
    prevalence=IntegrandEnum.prevalence,
)


def integrand_outputs(rates, location_id, age, time):
    """
    The internal model declares what outputs it wants.
    """
    age_time = np.array(list(it.product(age["age"].values, time["year"].values)), dtype=np.float)
    entries = list()
    for rate in (RATE_TO_INTEGRAND.get(r) for r in rates):
        entries.append(pd.DataFrame({
            "integrand": rate,
            "location_id": location_id,  # Uses location_id, not node id
            "weight": "constant",  # Assumes a constant rate exists.
            # Uses modeler versions of age and year, not age and time.
            "age_start": age_time[:, 0],
            "age_end": age_time[:, 0],
            "year_start": age_time[:, 1],
            "year_end": age_time[:, 1]
        }))
    return pd.concat(entries, ignore_index=True)


def build_smoothing_grid(age, time):
    """Builds a default smoothing grid with uniform priors."""
    age_time = np.array(list(it.product(age["age"].values, time["year"].values)), dtype=np.float)
    return pd.DataFrame({
        "age": age_time[:, 0],
        "year": age_time[:, 1],
        "value_prior": "uniform01",
        "age_difference_prior": "uniform",
        "time_difference_prior": "uniform",
        "const_value": np.NaN,
    })


def build_constraint(constraint):
    """
    This makes a smoothing grid where the mean value is set to a given
    set of values.
    """
    return pd.DataFrame({
        "age": constraint["age_start"],
        "year": constraint["year_start"],
        "value_prior": None,
        "age_difference_prior": "uniform",
        "time_difference_prior": "uniform",
        "const_value": constraint["mean"],
    })


def internal_model(config, inputs):
    model = Namespace()
    # convert the observations to a normalized format.
    model.observations = bundle_to_observations(config, inputs.observations)
    model.constraints = bundle_to_observations(config, inputs.constraints)

    rates_to_calculate_str = config.options["non_zero_rates"].split()
    age_df, time_df = age_year_from_data(inputs.constraints)
    desired_outputs = integrand_outputs(
        rates_to_calculate_str + ["prevalence"],
        config.location_id, age_df, time_df)
    model.outputs = desired_outputs

    model.priors = pd.DataFrame({
        "prior_name": ["uniform", "uniform01"],
        "density_id": 0,  # uniform
        "lower": [1e-10, None],
        "upper": [1.0, None],
        "mean": [0.01, 0.0],
        "std": np.array([np.NaN, np.NaN], dtype=np.float),
        "eta": np.array([np.NaN, np.NaN], dtype=np.float),
        "nu": np.array([np.NaN, np.NaN], dtype=np.float),
    })

    smoothing_default = build_smoothing_grid(age_df, time_df)
    pini = build_smoothing_grid(age_df, time_df)
    # For initial prevalence, cut off all grid points outside birth.
    smoothing_initial_prevalence = pd.DataFrame(pini[pini["age"] < 1e-6])
    model.smoothers = {
        RateName.iota: smoothing_default,
        RateName.rho: smoothing_default,
        RateName.chi: smoothing_default,
        RateName.pini: smoothing_initial_prevalence,
        RateName.omega: build_constraint(inputs.constraints),
    }
    return model


def construct_database():
    # Configuration
    config = Namespace()
    config.bundle_id = 3209
    config.tier_idx = 2
    config.location_id=26
    config.options = dict(
        non_zero_rates = "iota rho chi omega"
    )

    # Get the bundle and process it.
    inputs = retrieve_external_data(config)
    LOGGER.debug(f"inputs has {[x for x in dir(inputs) if not x.startswith('_')]}")
    model = internal_model(config, inputs)
    LOGGER.debug(f"model has {[x for x in dir(model) if not x.startswith('_')]}")
    write_to_file(config, model)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    construct_database()
