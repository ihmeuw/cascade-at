import logging
from pathlib import Path
import numpy as np
from typing import Union, List, Dict, Any
from argparse import ArgumentParser

from cascade_at.context.model_context import Context
from cascade_at.settings.settings_config import SettingsConfig
from cascade_at.dismod.api.dismod_filler import DismodFiller
from cascade_at.dismod.api.dismod_extractor import DismodExtractor
from cascade_at.context.arg_utils import parse_options, parse_commands
from cascade_at.dismod.api.run_dismod import run_dismod_commands
from cascade_at.core.log import get_loggers, LEVELS
from cascade_at.inputs.measurement_inputs import MeasurementInputs
from cascade_at.model.grid_alchemy import Alchemy

LOG = get_loggers(__name__)


def get_args(args=None):
    """
    Parse the arguments for creating a dismod sqlite database.
    :return: parsed args, plus additional parsing for
    """
    if args:
        return args

    parser = ArgumentParser()
    parser.add_argument("--model-version-id", type=int, required=True)
    parser.add_argument("--parent-location-id", type=int, required=True)
    parser.add_argument("--sex-id", type=int, required=True)
    parser.add_argument("--options", metavar="KEY=VALUE=TYPE", nargs="+", required=False,
                        help="optional key-value-type pairs to set in the option table of dismod")
    parser.add_argument("--fill", required=False, action='store_true')
    parser.add_argument("--prior-parent", type=int, required=False, default=None)
    parser.add_argument("--prior-sex", type=int, required=False, default=None)
    parser.add_argument("--dm-commands", nargs="+", required=False, default=[])
    parser.add_argument("--loglevel", type=str, required=False, default='info')
    parser.add_argument("--test_dir", type=str, required=False, default=None)
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


def get_prior(path: Union[str, Path], location_id: int, sex_id: int,
              rates: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Gets priors from a path to a database for a given location ID and sex ID.
    """
    child_prior = DismodExtractor(path=path).gather_draws_for_prior_grid(
        location_id=location_id,
        sex_id=sex_id,
        rates=rates
    )
    return child_prior


def fill(path: Union[str, Path], settings: SettingsConfig,
         inputs: MeasurementInputs, alchemy: Alchemy,
         parent_location_id: int, sex_id: int, child_prior: Dict[str, Dict[str, np.ndarray]],
         options: Dict[str, Any]) -> None:
    """
    Fill a DisMod database at the specified path with the inputs, model, and settings
    specified, for a specific parent and sex ID, with options to override the priors.
    """
    df = DismodFiller(
        path=path, settings_configuration=settings, measurement_inputs=inputs,
        grid_alchemy=alchemy, parent_location_id=parent_location_id, sex_id=sex_id,
        child_prior=child_prior
    )
    df.fill_for_parent_child(**options)


def main(args=None):
    """
    Creates a dismod database using the saved inputs and the file
    structure specified in the context. Alternatively it will
    skip the filling stage and move straight to the command
    stage if you don't pass --fill.

    Then runs an optional set of commands on the database passed
    in the --commands argument.

    Also passes an optional argument --options as a dictionary to
    the dismod database to fill/modify the options table.
    """
    args = get_args(args=args)
    logging.basicConfig(level=LEVELS[args.loglevel])

    if args.test_dir:
        context = Context(model_version_id=args.model_version_id,
                          configure_application=False,
                          root_directory=args.test_dir)
    else:
        context = Context(model_version_id=args.model_version_id)

    db_path = context.db_file(location_id=args.parent_location_id, sex_id=args.sex_id)
    inputs, alchemy, settings = context.read_inputs()

    # If we want to override the rate priors with posteriors from a previous
    # database, pass them in here.
    if args.prior_parent or args.prior_sex:
        if not (args.prior_parent and args.prior_sex):
            raise RuntimeError("Need to pass both prior parent and sex or neither.")
        child_prior = get_prior(
            path=context.db_file(
                location_id=args.prior_parent,
                sex_id=args.prior_sex
            ),
            location_id=args.parent_location_id, sex_id=args.sex_id,
            rates=[r.rate for r in settings.rate]
        )
    else:
        child_prior = None

    if args.fill:
        fill(
            path=db_path, inputs=inputs, alchemy=alchemy, settings=settings,
            parent_location_id=args.parent_location_id, sex_id=args.sex_id,
            child_prior=child_prior, options=args.options
        )

    if args.dm_commands:
        run_dismod_commands(dm_file=str(db_path), commands=args.dm_commands)


if __name__ == '__main__':
    main()
