from pathlib import Path
from pprint import pprint
from argparse import ArgumentParser

from cascade.dismod.db.wrapper import _get_engine
from cascade.testing_utilities import make_execution_context
from cascade.core.db import latest_model_version
from cascade.input_data.db.configuration import settings_json_from_epiviz
from cascade.executor.no_covariate_main import bundle_to_observations
from cascade.input_data.configuration.form import Configuration
from cascade.input_data.db.bundle import bundle_with_study_covariates, freeze_bundle
from cascade.dismod.serialize import model_to_dismod_file
from cascade.input_data.configuration.builder import (
    initial_context_from_epiviz,
    fixed_effects_from_epiviz,
    integrand_grids_from_epiviz,
)


def model_context_from_epiviz(execution_context):
    config_data = settings_json_from_epiviz(execution_context)
    configuration = Configuration(config_data)
    errors = configuration.validate_and_normalize()
    if errors:
        pprint(config_data)
        pprint(errors)
        raise ValueError("Configuration does not validate")

    model_context = initial_context_from_epiviz(configuration)

    fixed_effects_from_epiviz(model_context, execution_context, configuration)

    freeze_bundle(execution_context)

    bundle, study_covariates = bundle_with_study_covariates(
        execution_context, bundle_id=model_context.parameters.bundle_id
    )
    model_context.input_data.observations = bundle_to_observations(model_context.parameters, bundle)

    integrand_grids_from_epiviz(model_context, configuration)

    return model_context


def main():
    parser = ArgumentParser("Build a simple model from Epiviz")
    parser.add_argument("db_file_path")
    parser.add_argument("--meid", type=int)
    parser.add_argument("--mvid", type=int)
    args = parser.parse_args()

    if args.mvid:
        mvid = args.mvid
    elif args.meid:
        modelable_entity_id = args.meid
        ec = make_execution_context(modelable_entity_id=modelable_entity_id)
        mvid = latest_model_version(ec)
    else:
        raise ValueError("Must supply either mvid or meid")

    ec = make_execution_context(model_version_id=mvid)
    mc = model_context_from_epiviz(ec)

    dismod_file = model_to_dismod_file(mc)
    dismod_file.engine = _get_engine(Path(args.db_file_path))
    dismod_file.flush()


if __name__ == "__main__":
    main()
