import logging
import json
from typing import Optional

from cascade_at.context.arg_utils import ArgumentList
from cascade_at.context.args import ModelVersionID, BoolArg, LogLevel, StrArg
from cascade_at.context.model_context import Context
from cascade_at.settings.settings import settings_json_from_model_version_id, load_settings
from cascade_at.inputs.measurement_inputs import MeasurementInputsFromSettings
from cascade_at.core.log import get_loggers, LEVELS

LOG = get_loggers(__name__)


ARG_LIST = ArgumentList([
    ModelVersionID(),
    BoolArg('--make', help='whether or not to make the file structure for the cascade'),
    BoolArg('--configure', help='whether or not to configure for the IHME cluster'),
    LogLevel(),
    StrArg('--json-file', help='for testing, pass a json file directly by filepath'
                               'instead of referencing a model version ID.'),
    StrArg('--test-dir', help='if set, will save files to the directory specified.'
                              'Invalidated if --configure is set.')
])


def create_inputs(model_version_id: int, make: bool, configure: bool,
                  test_dir: Optional[str] = None, json_file: Optional[str] = None) -> None:
    """
    Grabs the inputs for a specific model version ID, sets up the folder
    structure, and pickles the inputs object plus writes the settings json
    for use later on.

    Optionally use a json file for settings instead of a model version ID's json file.
    """
    LOG.info(f"Configuring inputs for model version ID {model_version_id}.")

    context = Context(
        model_version_id=model_version_id,
        make=make,
        configure_application=configure,
        root_directory=test_dir
    )
    if json_file:
        LOG.info(f"Reading settings from file: {json_file}")
        with open(json_file, 'r') as json_file:
            parameter_json = json.load(json_file)
    else:
        parameter_json = settings_json_from_model_version_id(
            model_version_id=model_version_id,
            conn_def=context.model_connection
        )
    settings = load_settings(settings_json=parameter_json)

    inputs = MeasurementInputsFromSettings(settings=settings)
    inputs.get_raw_inputs()
    inputs.configure_inputs_for_dismod(settings=settings)

    context.write_inputs(inputs=inputs, settings=parameter_json)


def main():
    args = ARG_LIST.parse_args()
    logging.basicConfig(level=LEVELS[args.loglevel])

    create_inputs(
        model_version_id=args.model_version_id,
        make=args.make,
        configure=args.configure,
        test_dir=args.test_dir,
        json_file=args.json_file
    )


if __name__ == '__main__':
    main()
