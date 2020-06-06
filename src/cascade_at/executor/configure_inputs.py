import logging
import json
from argparse import ArgumentParser

from cascade_at.context.model_context import Context
from cascade_at.settings.settings import settings_json_from_model_version_id, load_settings
from cascade_at.inputs.measurement_inputs import MeasurementInputsFromSettings
from cascade_at.core.log import get_loggers, LEVELS

LOG = get_loggers(__name__)


def get_args(args):
    """
    Parse the arguments for configuring inputs.
    :return:
    """
    if args:
        return args
    parser = ArgumentParser()
    parser.add_argument("-model-version-id", type=int, required=True,
                        help="model version ID (need this from database entry)")
    parser.add_argument("--make", action='store_true',
                        help="whether or not to make the file structure for cascade")
    parser.add_argument("--configure", action='store_true',
                        help="whether or not to configure the application")
    parser.add_argument("--loglevel", type=str, required=False, default='info')
    parser.add_argument("--json_file", type=str, required=False, default=None,
                        help=("for testing, pass a json file directly by "
                              "file path instead of referencing by model "
                              "version ID"))
    parser.add_argument("--test_dir", type=str, required=False, default=None,
                        help=("if set, will save files to the directory "
                              "specified. Invalidated if --configure is "
                              "set"))
    return parser.parse_args()


def main(args=None):
    """
    Grabs the inputs for a specific model version ID, sets up the folder
    structure, and pickles the inputs object plus writes the settings json
    for use later on.

    If you're doing a drill, then only get input data from locations
    that will be used for the drilling for parent-children.
    """
    args = get_args(args=args)
    logging.basicConfig(level=LEVELS[args.loglevel])

    LOG.info(f"Configuring inputs for model version ID {args.model_version_id}.")
    LOG.debug(f"Arguments: {args}.")

    context = Context(
        model_version_id=args.model_version_id,
        make=args.make,
        configure_application=args.configure,
        root_directory=args.test_dir
    )
    if args.json_file:
        LOG.info(f"Reading settings from file: {args.json_file}")
        with open(args.json_file, 'r') as json_file:
            parameter_json = json.load(json_file)
    else:
        parameter_json = settings_json_from_model_version_id(
            model_version_id=args.model_version_id,
            conn_def=context.model_connection
        )
    settings = load_settings(settings_json=parameter_json)

    inputs = MeasurementInputsFromSettings(settings=settings)
    inputs.get_raw_inputs()
    inputs.configure_inputs_for_dismod(settings=settings)

    context.write_inputs(inputs=inputs, settings=parameter_json)


if __name__ == '__main__':
    main()
