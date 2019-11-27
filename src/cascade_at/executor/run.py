import logging
from argparse import ArgumentParser

from cascade_at.core.log import get_loggers, LEVELS
from cascade_at.cascade.cascade_commands import CASCADE_COMMANDS
from cascade_at.settings.settings import settings_from_model_version_id
from cascade_at.inputs.locations import LocationDAG

LOG = get_loggers(__name__)


def get_args():
    """
    Parse the arguments for running a Cascade model!
    :return: parsed args
    """
    parser = ArgumentParser()
    parser.add_argument("-model-version-id", type=int, required=True)
    parser.add_argument("-conn-def", type=int, required=True)
    parser.add_argument("--loglevel", type=str, required=False, default="info")
    return parser.parse_args()


def main():
    args = get_args()
    logging.basicConfig(level=LEVELS[args.loglevel])

    settings = settings_from_model_version_id(
        model_version_id=args.model_version_id,
        conn_def=args.conn_def
    )

    if settings.model.drill == 'drill':
        cascade_command = CASCADE_COMMANDS['drill'](
            model_version_id=args.model_version_id,
            conn_def=args.conn_def,
            drill_location_start=settings.model.drill_location_start,
            drill_sex=settings.model.drill_sex
        )
    elif settings.model.drill == 'cascade':
        raise NotImplementedError("Cascade is not implemented yet for Cascade-AT.")
    else:
        raise NotImplementedError(f"The drill/cascade setting {settings.model.drill} is not implemented.")


if __name__ == '__main__':
    main()
