import logging
from argparse import ArgumentParser

from cascade_at.context.model_context import Context
from cascade_at.dismod.api.dismod_alchemy import DismodAlchemy
from cascade_at.context.arg_utils import parse_options, parse_commands
from cascade_at.dismod.api.run_dismod import run_dismod
from cascade_at.core.log import get_loggers, LEVELS

LOG = get_loggers(__name__)


def get_args():
    """
    Parse the arguments for creating a dismod sqlite database.
    :return: parsed args, plus additional parsing for
    """
    parser = ArgumentParser()
    parser.add_argument("-model-version-id", type=int, required=True)
    parser.add_argument("-parent-location-id", type=int, required=True)
    parser.add_argument("-sex-id", type=int, required=True)
    parser.add_argument("--options", metavar="KEY=VALUE=TYPE", nargs="+", required=False,
                        help="optional key-value-type pairs to set in the option table of dismod")
    parser.add_argument("--commands", nargs="+", required=False, default=[])
    parser.add_argument("--loglevel", type=str, required=False, default='info')

    arguments = parser.parse_args()
    # Turn the options argument into a dictionary that can be passed
    #  to the options table rather than a list of "KEY=VALUE=TYPE"
    if arguments.options:
        arguments.options = parse_options(arguments.options)
    else:
        arguments.options = dict()
    
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

    context = Context(model_version_id=args.model_version_id)

    inputs, alchemy, settings = context.read_inputs()
    da = DismodAlchemy(
        path=context.db_file(location_id=args.parent_location_id, sex_id=args.sex_id),
        settings_configuration=settings,
        measurement_inputs=inputs,
        grid_alchemy=alchemy,
        parent_location_id=args.parent_location_id,
        sex_id=args.sex_id
    )
    da.fill_for_parent_child(**args.options)

    for c in args.commands:
        process = run_dismod(dm_file=da.path.absolute(), command=c)
        if process.exit_status:
            import pdb; pdb.set_trace()
            LOG.error(f"{c} failed with exit_status {process.exit_status}:")
            LOG.error(f"{process.stderr}")


if __name__ == '__main__':
    main()
