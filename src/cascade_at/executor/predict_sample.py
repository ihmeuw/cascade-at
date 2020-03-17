import logging
from argparse import ArgumentParser

from cascade_at.context.model_context import Context
from cascade_at.dismod.api.dismod_io import DismodIO
from cascade_at.dismod.api.fill_extract_helpers.data_tables import prep_data_avgint
from cascade_at.dismod.api.fill_extract_helpers.posterior_to_prior import get_prior_avgint_grid
from cascade_at.core.log import get_loggers, LEVELS
from cascade_at.dismod.api.run_dismod import run_dismod_commands


LOG = get_loggers(__name__)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-model-version-id", type=int, required=True)
    parser.add_argument("-source-location", type=int, required=True)
    parser.add_argument("-source-sex", type=int, required=True)
    parser.add_argument("-target-locations", nargs="+", required=True, default=[], type=int)
    parser.add_argument("-target-sexes", nargs="+", required=True, default=[], type=int)
    parser.add_argument("--loglevel", type=str, required=False, default='info')
    return parser.parse_args()


def main():
    args = get_args()
    logging.basicConfig(level=LEVELS[args.loglevel])

    context = Context(model_version_id=args.model_version_id)
    inputs, alchemy, settings = context.read_inputs()

    sourceDB = DismodIO(path=context.db_file(
        location_id=args.source_location, sex_id=args.source_sex, make=False
    ))

    rates = [r.rate for r in settings.rate]
    posterior_grid = get_prior_avgint_grid(
        settings=settings,
        integrands=rates,
        sexes=args.target_sexes,
        locations=args.target_locations,
        midpoint=False
    )
    posterior_grid = inputs.add_covariates_to_data(df=posterior_grid)
    posterior_grid = prep_data_avgint(
        df=posterior_grid,
        node_df=sourceDB.node,
        covariate_df=sourceDB.covariate
    )
    posterior_grid.rename(columns={'sex_id': 'c_sex_id'}, inplace=True)
    sourceDB.avgint = posterior_grid
    run_dismod_commands(
        dm_file=sourceDB,
        commands=['predict sample']
    )


if __name__ == '__main__':
    main()
