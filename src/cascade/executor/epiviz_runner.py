import os
import logging
from pathlib import Path
from pprint import pformat
import json

from cascade.executor.argument_parser import DMArgumentParser
from cascade.input_data.db.demographics import get_age_groups, get_years
from cascade.dismod.db.wrapper import _get_engine
from cascade.testing_utilities import make_execution_context
from cascade.input_data.db.configuration import settings_for_model
from cascade.executor.no_covariate_main import bundle_to_observations
from cascade.executor.dismod_runner import run_and_watch, DismodATException
from cascade.input_data.configuration.form import Configuration
from cascade.input_data.db.bundle import bundle_with_study_covariates, freeze_bundle
from cascade.dismod.serialize import model_to_dismod_file
from cascade.saver.save_model_results import save_model_results
from cascade.input_data.configuration import SettingsError
from cascade.input_data.configuration.builder import (
    initial_context_from_epiviz,
    fixed_effects_from_epiviz,
    random_effects_from_epiviz,
)

CODELOG = logging.getLogger(__name__)


def load_settings(meid=None, mvid=None, settings_file=None):
    if len([c for c in [meid, mvid, settings_file] if c is not None]) != 1:
        raise ValueError("Must supply exactly one of mvid, meid or settings_file")

    if settings_file is not None:
        with open(settings_file, "r") as f:
            raw_settings = json.load(f)
    else:
        raw_settings = settings_for_model(meid, mvid)

    settings = Configuration(raw_settings)
    errors = settings.validate_and_normalize()
    if errors:
        raise SettingsError("Configuration does not validate", errors, raw_settings)

    return settings


def execution_context_from_settings(settings):
    return make_execution_context(
        modelable_entity_id=settings.model.modelable_entity_id,
        model_version_id=settings.model.model_version_id,
        model_title=settings.model.title,
        gbd_round_id=settings.gbd_round_id,
    )


def model_context_from_settings(execution_context, settings):
    model_context = initial_context_from_epiviz(settings)

    fixed_effects_from_epiviz(model_context, settings)
    random_effects_from_epiviz(model_context, settings)

    freeze_bundle(execution_context)

    bundle, study_covariates = bundle_with_study_covariates(
        execution_context, bundle_id=model_context.parameters.bundle_id
    )
    model_context.input_data.observations = bundle_to_observations(model_context.parameters, bundle)

    integrand_grids_from_gbd(model_context, execution_context)

    return model_context


def integrand_grids_from_gbd(model_context, execution_context):
    gbd_age_groups = get_age_groups(execution_context)
    age_ranges = [(r.age_group_years_start, r.age_group_years_end) for _, r in gbd_age_groups.iterrows()]
    time_ranges = [(y, y) for y in get_years(execution_context)]

    for integrand in model_context.outputs.integrands:
        integrand.age_ranges = age_ranges
        integrand.time_ranges = time_ranges


def write_dismod_file(mc, db_file_path):
    dismod_file = model_to_dismod_file(mc)
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
    run_and_watch(command_prefix + ["fit", random_or_fixed], False, 1)
    dismod_file.refresh()
    if "end fit" not in dismod_file.log.message.iloc[-1]:
        raise DismodATException("DismodAt failed to complete 'fit' command")
    return

    run_and_watch(command_prefix + ["predict", "fit_var"], False, 1)
    dismod_file.refresh()
    if "end predict" not in dismod_file.log.message.iloc[-1]:
        raise DismodATException("DismodAt failed to complete 'predict' command")


def has_random_effects(model):
    return any([bool(r.child_smoothings) for r in model.rates])


def main():
    parser = DMArgumentParser("Run DismodAT from Epiviz")
    parser.add_argument("db_file_path")
    parser.add_argument("--settings_file")
    parser.add_argument("--no-upload", action="store_true")
    args, _ = parser.parse_known_args()

    settings = load_settings(args.meid, args.mvid, args.settings_file)

    ec = execution_context_from_settings(settings)
    mc = model_context_from_settings(ec, settings)

    ec.dismodfile = write_dismod_file(mc, args.db_file_path)

    run_dismod(ec.dismodfile, has_random_effects(mc))

    if not args.no_upload:
        save_model_results(ec)


if __name__ == "__main__":
    try:
        main()
    except SettingsError as e:
        CODELOG.error(str(e))
        CODELOG.error(f"Form data: {pformat(e.form_data)}")
        CODELOG.error(f"Form validation errors: {pformat(e.form_errors)}")
        exit(1)
    except Exception:
        CODELOG.exception(f"Uncaught exception in {os.path.basename(__file__)}")
        raise
