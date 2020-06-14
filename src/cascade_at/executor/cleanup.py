import logging
import os
import sys

from cascade_at.executor.args.arg_utils import ArgumentList
from cascade_at.executor.args.args import ModelVersionID, LogLevel
from cascade_at.context.model_context import Context
from cascade_at.core.log import get_loggers, LEVELS

LOG = get_loggers(__name__)


ARG_LIST = ArgumentList([ModelVersionID(), LogLevel()])


def cleanup(model_version_id: int) -> None:
    """
    Delete all databases (.db) files attached to a model version.

    Parameters
    ----------
    model_version_id
        The model version ID to delete databases for
    """
    context = Context(model_version_id=model_version_id)

    for root, dirs, files in os.walk(context.database_dir):
        for f in files:
            if f.endswith(".db"):
                file = context.database_dir / root / f
                LOG.info(f"Deleting {file}.")
                os.remove(file)


def main():

    args = ARG_LIST.parse_args(sys.argv[1:])
    logging.basicConfig(level=LEVELS[args.log_level])

    cleanup(model_version_id=args.model_version_id)


if __name__ == '__main__':
    main()
