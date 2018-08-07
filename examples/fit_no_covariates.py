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
from cascade.dismod.db.wrapper import DismodFile, _get_engine
from cascade.testing_utilities import make_execution_context
from cascade.dismod.db.metadata import IntegrandEnum, DensityEnum

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
    observations = pd.DataFrame(bundle[bundle["measure"] != measure])
    constraints = pd.DataFrame(bundle[bundle["measure"] == measure])
    return observations, constraints


def retrieve_external_data(config):
    context = make_execution_context()
    bundle, covariate = cached_bundle_load(context, config.bundle_id, config.tier_idx)
    # Split the input data into observations and constraints.
    bundle_observations, bundle_constraints = choose_constraints(bundle, "mtother")
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

    demographic_interval_specification = 1

    return pd.DataFrame({
        "measure": bundle_df["measure"],
        "location_id": location_id,
        "density": DensityEnum.gaussian,
        "weight": "constant",
        "age_start": bundle_df["age_start"],
        "age_end": bundle_df["age_end"] + demographic_interval_specification,
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
    iota="Sincidence",
    rho="remission",
    chi="mtexcess",
    omega="mtother",
    prevalence="prevalence",
)


def integrand_outputs(rates, age, time):
    age_time = np.array(list(it.product(age, time)))
    entries = list()
    for rate in (RATE_TO_INTEGRAND.get(r) for r in rates):
        entries.append(pd.DataFrame({
            "integrand": rate,
            "location_id": 26,
            "weight": "constant",
            "age_lower": age_time[:, 0],
            "age_upper": age_time[:, 0],
            "time_lower": age_time[:, 1],
            "time_upper": age_time[:, 1]
        }))
    return pd.concat(entries, ignore_index=True)


def internal_model(config, inputs):
    # convert the observations to a normalized format.
    observations = bundle_to_observations(config, inputs.observations)

    age_df, time_df = age_year_from_data(inputs.constraints)
    desired_outputs = integrand_outputs(config.options["non_zero_rates"].split(), age_df, time_df)
    return Namespace(observations=observations, outputs=desired_outputs)


def enum_to_dataframe(enum_name):
    """Given an enum, return a dataframe with two columns, name and value."""
    return pd.DataFrame.from_records(
        np.array(
            [(measure, enum_value.value) for (measure, enum_value) in enum_name.__members__.items()],
            dtype=np.dtype([('name', object), ('value', np.int)])
        )
    )


def default_integrand_names():
    # Converting an Enum to a DataFrame with specific parameters
    integrands = enum_to_dataframe(IntegrandEnum)
    df = pd.DataFrame({"integrand_name": integrands["name"]})
    df["minimum_meas_cv"] = 0.0
    return df


def simplest_weight():
    """Defines one weight for everything by defining it on one age-time point."""
    weight = pd.DataFrame({
        "weight_name": ["constant"],
        "n_age": [1],
        "n_time": [1],
    })
    weight_grid = pd.DataFrame({
        "weight_id": [0],
        "age_id": [0],
        "time_id": [0],
        "weight": [1.0],
    })
    return weight, weight_grid


def observations_to_data(dismodel, observations_df):
    """Turn an internal format into a Dismod format."""
    measure_to_integrand = dict(
        incidence=IntegrandEnum.Sincidence.value,
        mtexcess=IntegrandEnum.mtexcess.value,
    )
    return pd.DataFrame({
        "measure": observations_df["measure"].apply(measure_to_integrand.get),
        # Translate node id from location_id
        "node_id": observations_df["location_id"],
        # Translate density from string
        "density_id": observations_df["density"],
        # Translate weight from string
        "weight_id": observations_df["weight"],
        "age_lower": observations_df["age_start"],
        "age_upper": observations_df["age_end"],
        "time_lower": observations_df["year_start"].astype(np.float),
        "time_upper": observations_df["year_end"],
        "meas_value": observations_df["mean"],
        "meas_std": observations_df["standard_error"],
        "hold_out": 0,
    })


def write_to_file(config, model):
    avgint_columns = dict()
    data_columns = dict()
    bundle_dismod_db = Path("fit_no.db")
    bundle_file_engine = _get_engine(bundle_dismod_db)
    bundle_fit = DismodFile(bundle_file_engine, avgint_columns, data_columns)

    # Standard Density table.
    density_enum = enum_to_dataframe(DensityEnum)
    densities = pd.DataFrame({"density_name": density_enum["name"]})
    bundle_fit.density = densities

    # Standard integrand naming scheme.
    all_integrands = default_integrand_names()
    bundle_fit.integrands = all_integrands

    # Assume we have one location, so no parents.
    unique_locations = model.observations["location_id"].unique()
    assert len(unique_locations) == 1
    node_table = pd.DataFrame({
        "node_name": unique_locations.astype(int).astype(str),
        "parent": None,
    })

    bundle_fit.age = model.age
    bundle_fit.time = model.time

    # These are helpers to convert from ages and times to age and time indexes.
    # pd.merge_asof will do an approximate merge.
    age_idx = pd.DataFrame(model.age)
    age_idx["index0"] = age_idx.index

    time_idx = pd.DataFrame(model.time)
    time_idx["index0"] = time_idx.index

    bundle_fit.node = node_table
    bundle_fit.weight, bundle_fit.weight_grid = simplest_weight()

    # The avgint needs to be translated.
    bundle_fit.avgint = model.outputs

    observations = observations_to_data(bundle_fit, model.observations)
    constraints = model.constraints
    bundle_fit.data = pd.concat([observations, constraints], ignore_index=True)

    bundle_fit.covariate = pd.DataFrame({
        "covariate_name": np.array(0, object),
        "reference": np.array(0, np.float),
        "max_difference": np.array(0, np.float),
    })

    flush_begin = timer()
    bundle_fit.flush()
    LOGGER.debug(f"Flush db {timer() - flush_begin}")


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
