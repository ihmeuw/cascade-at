import logging
import os
from argparse import ArgumentParser

from cascade_at.core.log import get_loggers, LEVELS
from cascade_at.context.model_context import Context

LOG = get_loggers(__name__)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model-version-id", type=int, required=True)
    parser.add_argument("--loglevel", type=str, required=False, default='info')
    return parser.parse_args()


def main():
    """
    Cleans up all dismod databases (.db files) associated with the model version ID.
    :return:
    """
    args = get_args()
    logging.basicConfig(level=LEVELS[args.loglevel])
    context = Context(model_version_id=args.model_version_id)

    for root, dirs, files in os.walk(context.database_dir):
        for f in files:
            if f.endswith(".db"):
                file = context.database_dir / root / f
                LOG.info(f"Deleting {file}.")
                os.remove(file)


if __name__ == '__main__':
    main()
