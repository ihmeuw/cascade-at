"""
This example constructs a test disease model, similar to
diabetes, and sends that data to DismodAT.

1.  The test model is constructed by specifying functions for
    the primary rates, incidence, remission, excess mortality rate,
    and total mortality. Then this is solved to get prevalence over time.

2.  Then construct demographic observations of these rates. This means
    averaging over the rates for given ages and times.

3.  Load a subset of these observations into a DismodAT file and run
    DismodAT on it. The commands to run are::

        set option quasi_fixed false
        set option ode_step_size 1
        init
        fit fixed
        predict fit_var

This example works in cohort time, so that rates don't change over
years.
"""
from argparse import Namespace
import itertools as it
from pathlib import Path
import pickle
from timeit import default_timer as timer

import numpy as np
import pandas as pd

import cascade.input_data.configuration.construct_bundle
from cascade.executor.argument_parser import DMArgumentParser
import cascade.input_data.db.bundle
from cascade.input_data.configuration.builder import build_constraint
from cascade.testing_utilities import make_execution_context
from cascade.dismod.db.metadata import IntegrandEnum
from cascade.dismod.db.wrapper import _get_engine
from cascade.core.context import ModelContext
from cascade.dismod.serialize import model_to_dismod_file
from cascade.model.grids import AgeTimeGrid, PriorGrid
from cascade.model.priors import Uniform
from cascade.model.rates import Smooth
from cascade.input_data.configuration.construct_bundle import bundle_to_observations

from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)

RATE_TO_INTEGRAND = dict(
    iota=IntegrandEnum.Sincidence,
    rho=IntegrandEnum.remission,
    chi=IntegrandEnum.mtexcess,
    omega=IntegrandEnum.mtother,
    prevalence=IntegrandEnum.prevalence,
)


def cached_bundle_load(context, bundle_id, tier_idx):
    cache_bundle = Path(f"{bundle_id}.pkl")
    if cache_bundle.exists():
        CODELOG.info(f"Reading bundle from {cache_bundle}. " f"If you want to get a fresh copy, delete this file.")
        return pickle.load(cache_bundle.open("rb"))

    CODELOG.debug(f"Begin getting bundle and study covariates {bundle_id}")
    bundle_begin = timer()
    bundle = cascade.input_data.configuration.construct_bundle.normalized_bundle_from_database(
        context, bundle_id, tier_idx)
    CODELOG.debug(f"bundle is {bundle} time {timer() - bundle_begin}")
    covariate = None

    pickle.dump((bundle, covariate), cache_bundle.open("wb"), pickle.HIGHEST_PROTOCOL)
    return bundle, covariate


def choose_constraints(bundle, measure):
    CODELOG.debug(f"measures in bundle {bundle['measure'].unique()}")
    observations = pd.DataFrame(bundle[bundle["measure"] != measure])
    constraints = pd.DataFrame(bundle[bundle["measure"] == measure])
    return observations, constraints


def fake_mtother():
    ages = list(range(0, 105, 5))
    years = list(range(1965, 2020, 5))
    age_time = np.array(list(it.product(ages, years)), dtype=np.float)
    return pd.DataFrame(
        dict(
            measure="mtother",
            mean=0.001,
            sex="Male",
            standard_error=0.0007,
            age_lower=age_time[:, 0],
            age_upper=age_time[:, 0],
            time_lower=age_time[:, 1],
            time_upper=age_time[:, 1],
        )
    )


def retrieve_external_data(config):
    context = make_execution_context()
    bundle, covariate = cached_bundle_load(context, config.bundle_id, config.tier_idx)
    with_mtother = pd.concat([bundle, fake_mtother()], ignore_index=True)
    CODELOG.debug(f"Data now {with_mtother}")

    # Split the input data into observations and constraints.
    bundle_observations, bundle_constraints = choose_constraints(with_mtother, "mtother")
    return Namespace(observations=bundle_observations, constraints=bundle_constraints)


def data_from_csv(data_path):
    data = pd.read_csv(data_path)
    data = data.rename(
        index=str,
        columns={
            "integrand": "measure",
            "age_lower": "age_start",
            "age_upper": "age_end",
            "time_lower": "year_start",
            "time_upper": "year_end",
            "meas_value": "mean",
            "meas_std": "standard_error",
        },
    )
    # Split the input data into observations and constraints.
    bundle_observations, bundle_constraints = choose_constraints(data, "mtother")
    return Namespace(observations=bundle_observations, constraints=bundle_constraints)


def age_year_from_data(df):
    results = dict()
    for topic in ["age", "year"]:
        values = df[f"{topic}_start"].unique().tolist()
        values.extend(df[f"{topic}_end"].unique().tolist())
        values = list(set(values))
        values.sort()
        results[topic] = pd.DataFrame({topic: values})
        CODELOG.debug(f"{topic}: {values}")
    return AgeTimeGrid(results["age"].age, results["year"].year)


def internal_model(model, inputs):
    config = model.parameters
    # convert the observations to a normalized format.
    model.input_data.observations = bundle_to_observations(config, inputs.observations)
    model.input_data.constraints = bundle_to_observations(config, inputs.constraints)

    grid = age_year_from_data(inputs.constraints)

    priors = PriorGrid(grid)
    priors[:, :].prior = Uniform(1e-10, 1.0, 0.01)
    default_smoothing = Smooth(priors, priors, priors)

    # No smoothing for initial prevalence in Brad's example.
    for rate in config.non_zero_rates:
        getattr(model.rates, rate).parent_smooth = default_smoothing

    avgint_rows = []
    for rate in config.non_zero_rates + ["prevalence"]:
        avgint_rows.extend([{
            "integrand_name": rate,
            "age_lower": age,
            "age_upper": age,
            "time_lower": time,
            "time_upper": time,
            "weight_id": 0,
            "node_id": 0,
            "x_sex": sex,
        } for age in grid.ages for time in grid.times for sex in [-0.5, 0.5]])
    model.average_integrand_cases = pd.DataFrame(avgint_rows, columns=["integrand_name", "age_lower", "age_upper",
                                                                       "time_lower", "time_upper", "weight_id",
                                                                       "node_id", "x_sex"])

    model.rates.omega.parent_smooth = build_constraint(model.input_data.constraints)

    return model


def construct_database(input_path, output_path, non_zero_rates):
    # Configuration
    model_context = ModelContext()
    model_context.parameters.bundle_id = 3209
    model_context.parameters.tier_idx = 2
    model_context.parameters.location_id = 26
    model_context.parameters.non_zero_rates = non_zero_rates

    # Get the bundle and process it.
    raw_inputs = data_from_csv(Path(input_path))
    internal_model(model_context, raw_inputs)

    CODELOG.info(f"Creating file {output_path}")
    dismod_file = model_to_dismod_file(model_context)
    flush_begin = timer()
    dismod_file.engine = _get_engine(Path(output_path))
    dismod_file.flush()
    CODELOG.debug(f"Flush db {timer() - flush_begin}")


def entry():
    """This is the entry that setuptools turns into an installed program."""
    parser = DMArgumentParser("Reads csv for a run without covariates.")
    parser.add_argument("input_path", help="Path to the csv file to load")
    parser.add_argument("output_path", help="Path to the dismod database file to create")
    parser.add_argument(
        "--non-zero-rates",
        default=[],
        choices=["iota", "rho", "chi", "omega"],
        nargs="*",
        help="Rates to estimate, all others will be zero",
    )

    args = parser.parse_args()
    if args.non_zero_rates:
        non_zero_rates = args.non_zero_rates
    else:
        non_zero_rates = ["iota", "rho", "chi", "omega"]
    construct_database(args.input_path, args.output_path, non_zero_rates)


if __name__ == "__main__":
    entry()
