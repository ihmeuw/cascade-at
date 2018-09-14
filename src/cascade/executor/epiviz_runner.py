import sys
import logging
from pathlib import Path
from pprint import pprint
import argparse
import json

from cascade.dismod.db.wrapper import _get_engine
from cascade.testing_utilities import make_execution_context
from cascade.input_data.db.configuration import settings_for_model
from cascade.executor.no_covariate_main import bundle_to_observations
from cascade.executor.dismod_runner import run_and_watch, DismodATException
from cascade.input_data.configuration.form import Configuration
from cascade.input_data.db.bundle import bundle_with_study_covariates, freeze_bundle
from cascade.dismod.serialize import model_to_dismod_file
from cascade.saver.save_model_results import save_model_results
from cascade.input_data.configuration.builder import (
    initial_context_from_epiviz,
    fixed_effects_from_epiviz,
    integrand_grids_from_epiviz,
    random_effects_from_epiviz,
)


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
        pprint(raw_settings)
        pprint(errors)
        raise ValueError("Configuration does not validate")

    return settings


def execution_context_from_settings(settings):
    return make_execution_context(
        modelable_entity_id=settings.model.modelable_entity_id,
        model_version_id=settings.model.model_version_id,
        model_title=settings.model.title,
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

    integrand_grids_from_epiviz(model_context, settings)

    return model_context


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

    random_or_fixed = "random" if with_random_effects else "fixed"
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


def main():
    parser = argparse.ArgumentParser("Run DismodAT from Epiviz")
    parser.add_argument("db_file_path")
    parser.add_argument("--meid", type=int)
    parser.add_argument("--mvid", type=int)
    parser.add_argument("--settings_file")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    root.addHandler(ch)

    settings = load_settings(args.meid, args.mvid, args.settings_file)

    ec = execution_context_from_settings(settings)
    mc = model_context_from_settings(ec, settings)

    ec.dismodfile = write_dismod_file(mc, args.db_file_path)

    run_dismod(ec.dismodfile, has_random_effects(mc))

    if not args.no_upload:
        save_model_results(ec)


if __name__ == "__main__":
    main()
