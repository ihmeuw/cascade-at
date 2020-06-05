import logging
from argparse import ArgumentParser

from cascade_at.context.arg_utils import parse_commands
from cascade_at.dismod.api.run_dismod import run_dismod
from cascade_at.core.log import get_loggers, LEVELS

LOG = get_loggers(__name__)


def get_args():
    """
    Parse the arguments for creating a dismod sqlite database.
    :return: parsed args, plus additional parsing for
    """
    parser = ArgumentParser()
    parser.add_argument("-file", type=str, required=True)
    parser.add_argument("--commands", nargs="+", required=False, default=[])
    parser.add_argument("--loglevel", type=str, required=False, default='info')

    arguments = parser.parse_args()

    # Turn the commands argument into a list than can run on dismod as commands
    # e.g. "fit-fixed" will translate to the command "fit fixed"
    if arguments.commands:
        arguments.commands = parse_commands(arguments.commands)
    else:
        arguments.commands = list()
    return arguments


def main():
    """
    Creates a dismod database using the saved inputs and the file
    structure specified in the context.

    Then runs an optional set of commands on the database passed
    in the --commands argument.

    Also passes an optional argument --options as a dictionary to
    the dismod database to fill/modify the options table.
    """
    args = get_args()
    logging.basicConfig(level=LEVELS[args.loglevel])

    for c in args.commands:
        process = run_dismod(dm_file=args.file, command=c)
        if process.exit_status:
            LOG.error(f"{c} failed with exit_status {process.exit_status}:")
            LOG.error(f"{process.stderr}")
            raise
        else:
            print(process.stdout)
            print(process.stderr)


if __name__ == '__main__':
    main()
