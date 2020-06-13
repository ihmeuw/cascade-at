import logging
from argparse import ArgumentParser
import os
import pandas as pd

from cascade_at.context.model_context import Context
from cascade_at.dismod.api.dismod_filler import DismodFiller
from cascade_at.dismod.api.dismod_extractor import DismodExtractor
from cascade_at.context.arg_utils import parse_options, parse_commands
from cascade_at.dismod.api.run_dismod import run_dismod_commands
from cascade_at.core.log import get_loggers, LEVELS
from cascade_at.model.priors import Gaussian

LOG = get_loggers(__name__)


def get_args(args=None):
    """
    Parse the arguments for creating a dismod sqlite database.
    :return: parsed args, plus additional parsing for
    """
    if args:
        return args

    parser = ArgumentParser()
    parser.add_argument("-model-version-id", type=int, required=True)
    parser.add_argument("-parent-location-id", type=int, required=True)
    parser.add_argument("-sex-id", type=int, required=True)
    parser.add_argument("--options", metavar="KEY=VALUE=TYPE", nargs="+", required=False,
                        help="optional key-value-type pairs to set in the option table of dismod")
    parser.add_argument("--prior-parent", type=int, required=False, default=None)
    parser.add_argument("--prior-sex", type=int, required=False, default=None)
    parser.add_argument('--prior-mulcov', type=int, required=False, default=None)
    parser.add_argument("--commands", nargs="+", required=False, default=[])
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


def main(args=None):
    """
    Creates a dismod database using the saved inputs and the file
    structure specified in the context.

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

    inputs, alchemy, settings = context.read_inputs()

    # If we want to override the rate priors with posteriors from a previous
    # database, pass them in here.
    if args.prior_parent or args.prior_sex:
        if not (args.prior_parent and args.prior_sex):
            raise RuntimeError("Need to pass both prior parent and sex or neither.")
        child_prior = DismodExtractor(path=context.db_file(
            location_id=args.prior_parent,
            sex_id=args.prior_sex
        )).gather_draws_for_prior_grid(
            location_id=args.parent_location_id,
            sex_id=args.sex_id,
            rates=[r.rate for r in settings.rate]
        )
    else:
        child_prior = None

    if args.prior_mulcov is not None:
        mulcov_priors = {}
        ctx = Context(model_version_id=args.prior_mulcov)
        path = os.path.join(ctx.outputs_dir, 'mulcov_stats.csv')
        mulcov_stats_df = pd.read_csv(path, index=False)
        for row in mulcov_stats_df.iterrows():
            if row['rate_name'] is not None:
                mulcov_priors[
                    (row['mulcov_type'], row['c_covariate_name'], row['rate_name'])
                ] = Gaussian(mean=row['mean'], standard_deviation=row['std'])
            if row['integrand_name'] is not None:
                mulcov_priors[
                    (row['mulcov_type'], row['c_covariate_name'], row['integrand_name'])
                ] = Gaussian(mean=row['mean'], standard_deviation=row['std'])
    else:
        mulcov_priors = None

    df = DismodFiller(
        path=context.db_file(location_id=args.parent_location_id, sex_id=args.sex_id),
        settings_configuration=settings,
        measurement_inputs=inputs,
        grid_alchemy=alchemy,
        parent_location_id=args.parent_location_id,
        sex_id=args.sex_id,
        child_prior=child_prior,
        mulcov_prior=mulcov_prior,
    )
    df.fill_for_parent_child(**args.options)
    run_dismod_commands(dm_file=df.path.absolute(), commands=args.commands)


if __name__ == '__main__':
    main()
