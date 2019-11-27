import logging
from argparse import ArgumentParser

from cascade_at.context.model_context import Context
from cascade_at.settings.settings import settings_json_from_model_version_id, load_settings
from cascade_at.collector.measurement_inputs import MeasurementInputsFromSettings
from cascade_at.core.log import get_loggers, LEVELS

LOG = get_loggers(__name__)


def get_args():
    """
    Parse the arguments for configuring inputs.
    :return:
    """
    parser = ArgumentParser()
    parser.add_argument("-model-version-id", type=int, required=True,
                        help="model version ID (need this from database entry)")
    parser.add_argument("-conn-def", type=str, required=False, default='dismod-at-dev',
                        help="the connection definition to use from the .odbc.ini file")
    parser.add_argument("--make", action='store_true',
                        help="whether or not to make the file structure for cascade")
    parser.add_argument("--configure", action='store_true',
                        help="whether or not to configure the application")
    parser.add_argument("--drill", type=int, required=False,
                        help="if doing a drill, which parent ID to drill the model from?")
    parser.add_argument("--loglevel", type=str, required=False, default='info')
    return parser.parse_args()


def main():
    """
    Grabs the inputs for a specific model version ID, sets up the folder
    structure, and pickles the inputs object plus writes the settings json
    for use later on.

    If you're doing a drill, then only get input data from locations
    that will be used for the drilling for parent-children.
    """
    args = get_args()
    logging.basicConfig(level=LEVELS[args.loglevel])

    LOG.info(f"Configuring inputs for model version ID {args.model_version_id}.")
    LOG.debug(f"Arguments: {args}.")

    context = Context(
        model_version_id=args.model_version_id,
        make=args.make,
        configure_application=args.configure
    )
    parameter_json = settings_json_from_model_version_id(
        model_version_id=args.model_version_id,
        conn_def=args.conn_def
    )
    settings = load_settings(settings_json=parameter_json)

    inputs = MeasurementInputsFromSettings(settings=settings)

    if args.drill:
        drill_descendants = inputs.location_dag.descendants(location_id=args.drill)
        inputs.demographics.location_id = [args.drill] + drill_descendants

    inputs.get_raw_inputs()
    inputs.configure_inputs_for_dismod(settings=settings)

    context.write_inputs(inputs=inputs, settings=parameter_json)


if __name__ == '__main__':
    main()
