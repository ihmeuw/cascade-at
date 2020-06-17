import logging
import sys
from typing import List

from cascade_at.executor.args.arg_utils import ArgumentList
from cascade_at.executor.args.args import StrArg, DmCommands, LogLevel
from cascade_at.core.log import get_loggers, LEVELS
from cascade_at.dismod.api.run_dismod import run_dismod

LOG = get_loggers(__name__)


ARG_LIST = ArgumentList([
    StrArg('--file', help='Which database file to execute commands on'),
    DmCommands(),
    LogLevel()
])


def run_dmdismod(file, dm_commands: List[str]) -> None:
    """
    Runs commands on a dismod file.

    Parameters
    ----------
    file
        Filepath to a database
    dm_commands
        List of commands that dismod_at understands
    """
    for c in dm_commands:
        process = run_dismod(dm_file=file, command=c)
        if process.exit_status:
            LOG.error(f"{c} failed with exit_status {process.exit_status}:")
            LOG.error(f"{process.stderr}")
            raise
        else:
            print(process.stdout)
            print(process.stderr)


def main():

    args = ARG_LIST.parse_args(sys.argv[1:])
    logging.basicConfig(level=LEVELS[args.log_level])

    run_dmdismod(
        file=args.file,
        dm_commands=args.dm_commands
    )


if __name__ == '__main__':
    main()
