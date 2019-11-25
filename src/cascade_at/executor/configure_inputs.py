from argparse import ArgumentParser

from cascade_at.context.model_context import Context
from cascade_at.settings.settings import settings_from_model_version_id
from cascade_at.collector.measurement_inputs import MeasurementInputsFromSettings
from cascade_at.collector.grid_alchemy import Alchemy
from cascade_at.core.log import get_loggers

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
    parser.add_argument("--make", "-make", type=bool, required=False, default=True,
                        help="whether or not to make the file structure for cascade")
    parser.add_argument("--conf", "-configure", type=bool, required=False, default=True,
                        help="whether or not to configure the application")
    return parser.parse_args()


if __name__ == '__main__':
    """
    Grabs the inputs for a specific model version ID, sets up the folder
    structure, and pickles the inputs, settings, and alchemy objects
    for use later on.
    """
    args = get_args()

    context = Context(
        model_version_id=args.model_version_id,
        make=args.make,
        configure_application=args.configure
    )
    settings = settings_from_model_version_id(
        model_version_id=args.model_version_id,
        conn_def=args.conn_def
    )
    inputs = MeasurementInputsFromSettings(settings=settings)
    inputs.get_raw_inputs()
    inputs.configure_inputs_for_dismod(settings=settings)
    alchemy = Alchemy(settings=settings)

    context.write_inputs(
        inputs=inputs,
        alchemy=alchemy,
        settings=settings
    )
