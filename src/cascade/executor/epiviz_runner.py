import os
from pathlib import Path
from pprint import pformat

import pandas as pd
import numpy as np

from cascade.executor.argument_parser import DMArgumentParser
from cascade.input_data.db.demographics import age_groups_to_ranges
from cascade.dismod.db.wrapper import _get_engine
from cascade.dismod.db.metadata import DensityEnum
from cascade.testing_utilities import make_execution_context
from cascade.input_data.db.configuration import load_settings
from cascade.input_data.db.csmr import load_csmr_to_t3
from cascade.input_data.db.asdr import load_asdr_to_t3
from cascade.input_data.db.mortality import get_cause_specific_mortality_data, get_age_standardized_death_rate_data
from cascade.executor.no_covariate_main import bundle_to_observations, build_constraint
from cascade.executor.dismod_runner import run_and_watch, DismodATException
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

from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)


def add_settings_to_execution_context(ec, settings):
    to_append = dict(
        modelable_entity_id=settings.model.modelable_entity_id,
        model_version_id=settings.model.model_version_id,
        model_title=settings.model.title,
        gbd_round_id=settings.gbd_round_id,
        bundle_id=settings.model.bundle_id,
        add_csmr_cause=settings.model.add_csmr_cause,
        location_id=settings.model.drill_location,
    )
    for param, value in to_append.items():
        setattr(ec.parameters, param, value)


def meas_bounds_to_stdev(df):
    r"""
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
    MATHLOG.debug(f"Add omega constraint from age-standardized death rate data.")
    asdr = meas_bounds_to_stdev(
        age_groups_to_ranges(execution_context, get_age_standardized_death_rate_data(execution_context))
    )
    asdr["measure"] = "mtall"
    asdr = asdr.rename(columns={"location_id": "node_id"})
    min_time = np.min(list(model_context.input_data.times))  # noqa: F841
    max_time = np.max(list(model_context.input_data.times))  # noqa: F841
    asdr = asdr.query("year_start >= @min_time and year_end <= @max_time and year_start % 5 == 0")
    model_context.rates.omega.parent_smooth = build_constraint(asdr)

    mask = model_context.input_data.observations.measure == "mtall"
    model_context.input_data.constraints = pd.concat([model_context.input_data.observations[mask], asdr])
    model_context.input_data.observations = model_context.input_data.observations[~mask]


def model_context_from_settings(execution_context, settings):
    model_context = initial_context_from_epiviz(settings)

    fixed_effects_from_epiviz(model_context, execution_context, settings)
    random_effects_from_epiviz(model_context, settings)

    freeze_bundle(execution_context, execution_context.parameters.bundle_id)
    load_csmr_to_t3(execution_context)
    load_asdr_to_t3(execution_context)

    bundle, study_covariates = bundle_with_study_covariates(
        execution_context, bundle_id=model_context.parameters.bundle_id
    )
    bundle = bundle.query("location_id == @execution_context.parameters.location_id")
    observations = bundle_to_observations(model_context.parameters, bundle)
    observations = observations.rename(columns={"location_id": "node_id"})
    model_context.input_data.observations = observations

    mask = model_context.input_data.observations.standard_error > 0
    mask &= model_context.input_data.observations.measure != "relrisk"
    if mask.any():
        remove_cnt = mask.sum()
        MATHLOG.warning(f"removing {remove_cnt} rows from bundle where standard_error == 0.0")
        model_context.input_data.observations = model_context.input_data.observations[mask]

    add_mortality_data(model_context, execution_context)
    add_omega_constraint(model_context, execution_context)
    model_context.average_integrand_cases = make_average_integrand_cases_from_gbd(execution_context)

    return model_context


def write_dismod_file(mc, ec, db_file_path):
    dismod_file = model_to_dismod_file(mc, ec)
    dismod_file.engine = _get_engine(Path(db_file_path))
    dismod_file.flush()
    return dismod_file


def run_dismod(dismod_file, with_random_effects):
    dm_file_path = dismod_file.engine.url.database
    if dm_file_path == ":memory:":
        raise ValueError("Cannot run dismodat on an in-memory database")

    command_prefix = ["dmdismod", dm_file_path]

    run_and_watch(command_prefix + ["init"], False, 1)
    dismod_file.refresh()
    if "end init" not in dismod_file.log.message.iloc[-1]:
        raise DismodATException("DismodAt failed to complete 'init' command")

    random_or_fixed = "both" if with_random_effects else "fixed"
    # FIXME: both doesn't work. Something about actually having parents in the node table
    random_or_fixed = "fixed"
    run_and_watch(command_prefix + ["fit", random_or_fixed], False, 1)
    dismod_file.refresh()
    if "end fit" not in dismod_file.log.message.iloc[-1]:
        raise DismodATException("DismodAt failed to complete 'fit' command")

    run_and_watch(command_prefix + ["predict", "fit_var"], False, 1)
    dismod_file.refresh()
    if "end predict" not in dismod_file.log.message.iloc[-1]:
        raise DismodATException("DismodAt failed to complete 'predict' command")


def has_random_effects(model):
    return any([bool(r.child_smoothings) for r in model.rates])


def main(args):
    ec = make_execution_context()
    settings = load_settings(ec, args.meid, args.mvid, args.settings_file)
    add_settings_to_execution_context(ec, settings)
    mc = model_context_from_settings(ec, settings)

    ec.dismodfile = write_dismod_file(mc, ec, args.db_file_path)

    if not args.db_only:
        run_dismod(ec.dismodfile, has_random_effects(mc))

        if not args.no_upload:
            save_model_results(ec)


def entry():
    parser = DMArgumentParser("Run DismodAT from Epiviz")
    parser.add_argument("db_file_path")
    parser.add_argument("--settings-file")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--db-only", action="store_true")
    parser.add_argument("--pdb", action="store_true")
    args = parser.parse_args()

    CODELOG.debug(args)
    try:
        main(args)
    except SettingsError as e:
        MATHLOG.error(str(e))
        MATHLOG.error(f"Form data:{os.linesep}{pformat(e.form_data)}")
        error_lines = list()
        for error_spot, human_spot, error_message in e.form_errors:
            if args.settings_file is not None:
                error_location = error_spot
            else:
                error_location = human_spot
            error_lines.append(f"\t{error_location}: {error_message}")
        MATHLOG.error(f"Form validation errors:{os.linesep}{os.linesep.join(error_lines)}")
        exit(1)
    except Exception:
        if args.pdb:
            import pdb
            import traceback

            traceback.print_exc()
            pdb.post_mortem()
        else:
            CODELOG.exception(f"Uncaught exception in {os.path.basename(__file__)}")
            raise


if __name__ == "__main__":
    entry()
