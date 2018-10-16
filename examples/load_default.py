import os
import logging
from pathlib import Path
from pprint import pformat
import json

import pandas as pd
import numpy as np

from cascade.executor.argument_parser import DMArgumentParser
from cascade.input_data.db.demographics import age_groups_to_ranges
from cascade.dismod.db.wrapper import _get_engine
from cascade.dismod.db.metadata import DensityEnum
from cascade.testing_utilities import make_execution_context
from cascade.input_data.db.configuration import settings_for_model
from cascade.input_data.db.csmr import load_csmr_to_t3
from cascade.input_data.db.asdr import load_asdr_to_t3
from cascade.input_data.db.mortality import get_cause_specific_mortality_data, get_age_standardized_death_rate_data
from cascade.executor.no_covariate_main import bundle_to_observations, build_constraint
from cascade.executor.dismod_runner import run_and_watch, DismodATException
from cascade.input_data.configuration.form import Configuration
from cascade.input_data.db.bundle import bundle_with_study_covariates, freeze_bundle
from cascade.dismod.serialize import model_to_dismod_file
from cascade.model.integrands import make_average_integrand_cases_from_gbd
from cascade.saver.save_model_results import save_model_results
from cascade.input_data.configuration import SettingsError
from cascade.input_data.configuration.builder import (
    initial_context_from_epiviz,
    fixed_effects_from_epiviz,
    random_effects_from_epiviz,
)

CODELOG = logging.getLogger(__name__)
MATHLOG = logging.getLogger("cascade_a.math.runner")


def execution_context_from_settings(settings):
    return make_execution_context(
        modelable_entity_id=settings.model.modelable_entity_id,
        model_version_id=settings.model.model_version_id,
        model_title=settings.model.title,
        gbd_round_id=settings.gbd_round_id,
        bundle_id=settings.model.bundle_id,
        add_csmr_cause=settings.model.add_csmr_cause,
        location_id=settings.model.drill_location,
    )

def model_context_from_settings(execution_context, settings):
    model_context = initial_context_from_epiviz(settings)

    fixed_effects_from_epiviz(model_context, settings)
    random_effects_from_epiviz(model_context, settings)

    add_mortality_data(model_context, execution_context)
    add_omega_constraint(model_context, execution_context)
    model_context.average_integrand_cases = make_average_integrand_cases_from_gbd(execution_context)

    return model_context


def meas_bounds_to_stdev(df):
    """
    Given data that includes a measurement upper bound and measurement lower
    bound, assume those are 95% confidence intervals. Convert them to
    standard error using:

    .. math::

        \mbox{stderr} = \frac{\mbox{upper} - \mbox{lower}}{2 1.96}

    Standard errors become Gaussian densities.
    Replace any zero values with :math:`10^{-9}`.
    """
    MATHLOG.debug("Assigning standard error from measured upper and lower.")
    df["standard_error"] = (df.meas_upper - df.meas_lower) / (2 * 1.96)
    df["standard_error"] = df.standard_error.replace({0: 1e-9})
    df = df.rename(columns={"meas_value": "mean"})
    df["density"] = DensityEnum.gaussian
    df["weight"] = "constant"
    return df.drop(["meas_lower", "meas_upper"], axis=1)


def add_mortality_data(model_context, execution_context):
    """
    Gets cause-specific mortality rate and adds that data as an ``mtspecific``
    measurement by appending it to the bundle. Uses ranges for ages and years.
    This doesn't determine point-data values.
    """
    MATHLOG.debug(f"Creating a set of mtspecific observations from IHME CSMR database.")
    csmr = meas_bounds_to_stdev(
        age_groups_to_ranges(execution_context, get_cause_specific_mortality_data(execution_context))
    )
    csmr["measure"] = "mtspecific"
    csmr = csmr.rename(columns={"location_id": "node_id"})
    model_context.input_data.observations = pd.concat([model_context.input_data.observations, csmr])


def add_omega_constraint(model_context, execution_context):
    """
    Adds a constraint to other-cause mortality rate. Removes mtother,
    mtall, and mtspecific from observation data.
    """
    asdr = meas_bounds_to_stdev(
        age_groups_to_ranges(execution_context, get_age_standardized_death_rate_data(execution_context))
    )
    asdr["measure"] = "mtall"
    asdr = asdr.rename(columns={"location_id": "node_id"})
    min_time = np.min(list(model_context.input_data.times))  # noqa: F841
    max_time = np.max(list(model_context.input_data.times))  # noqa: F841
    asdr = asdr.query("year_start >= @min_time and year_end <= @max_time and year_start % 5 == 0")
    model_context.rates.omega.parent_smooth = build_constraint(asdr)

    mask = model_context.input_data.observations.measure.isin(["mtall", "mtother", "mtspecific"])
    model_context.input_data.constraints = pd.concat([model_context.input_data.observations[mask], asdr])
    model_context.input_data.observations = model_context.input_data.observations[~mask]

def go():
    settings = Configuration(json.load(Path("1989.json").open()))
    errors = settings.validate_and_normalize()
    if errors:
        print(errors)

    ec = execution_context_from_settings(settings)
    mc = model_context_from_settings(ec, settings)


if __name__ == "__main__":
    go()
