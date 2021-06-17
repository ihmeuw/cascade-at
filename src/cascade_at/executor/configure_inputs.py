#!/usr/bin/env python
import json
import logging
import sys
from typing import Optional

from cascade_at.executor.args.arg_utils import ArgumentList
from cascade_at.executor.args.args import ModelVersionID, BoolArg, LogLevel, StrArg
from cascade_at.context.model_context import Context
from cascade_at.core.log import get_loggers, LEVELS
from cascade_at.inputs.measurement_inputs import MeasurementInputsFromSettings
from cascade_at.settings.settings import settings_json_from_model_version_id, load_settings

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


def configure_inputs(model_version_id: int, make: bool, configure: bool,
                     test_dir: Optional[str] = None, json_file: Optional[str] = None) -> None:
    """
    Grabs the inputs for a specific model version ID, sets up the folder
    structure, and pickles the inputs object plus writes the settings json
    for use later on. Also uploads CSMR to the database attached to the model version,
    if applicable.

    Optionally use a json file for settings instead of a model version ID's json file.

    Parameters
    ----------
    model_version_id
        The model version ID to configure inputs for
    make
        Whether or not to make the directory structure for the model version ID
    configure
        Configure the application for the IHME cluster, otherwise will use the
        test_dir for the directory tree instead.
    test_dir
        A test directory to use rather than the directory specified by the
        model version context in the IHME file system.
    json_file
        An optional filepath pointing to a different json than is attached to the
        model_version_id. Will use this instead for settings.
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
        LOG.info(f"Replacing {context.settings_file}")
    else:
        parameter_json = settings_json_from_model_version_id(
            model_version_id=model_version_id,
            conn_def=context.model_connection
        )

    settings = load_settings(settings_json=parameter_json)

    inputs = MeasurementInputsFromSettings(settings=settings)
    inputs.get_raw_inputs()
    inputs.configure_inputs_for_dismod(settings=settings)

    try:
        if not inputs.csmr.raw.empty:
            LOG.info("Uploading CSMR to t3 table.")
            inputs.csmr.attach_to_model_version_in_db(
                model_version_id=model_version_id,
                conn_def=context.model_connection
            )
    except Exception as ex:
        msg = str(ex)
        if len(msg) > 1000:
            msg = msg[:500] + ' ... \n' + msg[-500:]
        LOG.error(msg)

    context.write_inputs(inputs=inputs, settings=parameter_json)
    return context, inputs

def main():

    args = ARG_LIST.parse_args(sys.argv[1:])
    logging.basicConfig(level=LEVELS[args.log_level])

    configure_inputs(
        model_version_id=args.model_version_id,
        make=args.make,
        configure=args.configure,
        test_dir=args.test_dir,
        json_file=args.json_file,
    )


if __name__ == '__main__':
    main()
