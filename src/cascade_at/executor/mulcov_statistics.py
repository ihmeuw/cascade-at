import logging
from argparse import ArgumentParser
import pandas as pd
import numpy as np

from cascade_at.context.model_context import Context
from cascade_at.dismod.api.dismod_io import DismodIO
from cascade_at.core.log import get_loggers, LEVELS


LOG = get_loggers(__name__)


def get_args():
    """

    Returns:

    """
    parser = ArgumentParser()
    parser.add_argument("-model-version-id", type=int, required=True)
    parser.add_argument("-locations", nargs="+", required=True, default=[], type=int)
    parser.add_argument("-sexes", nargs="+", required=True, default=[], type=int)
    parser.add_argument("-outfile-name", type=str, required=True)
    parser.add_argument("--sample", action='store_true', required=False,
                        help="Are the results in the sample table or fit_var -- False = fit_var")
    parser.add_argument("--mean", action='store_true', required=False)
    parser.add_argument("--std", action='store_true', required=False)
    parser.add_argument("--quantile", required=False, nargs="+", type=float)
    parser.add_argument("--loglevel", type=str, required=False, default='info')
    return parser.parse_args()


def common_covariate_names(dbs):
    return set.intersection(
        *map(set, [d.covariate.c_covariate_name.tolist() for d in dbs])
    )


def get_mulcovs(dbs, covs, table='fit_var'):
    """
    Get mulcov values from all of the dbs, with all of the common covariates.
    Args:
        dbs: list of cascade_at.dismod.api.dismod_io.DismodIO
        covs: set of covariate names
        table: name of the table to pull from (can be fit_var or sample)

    Returns:

    """
    if table == 'fit_var':
        id_col = 'fit_var_id'
        val_col = 'fit_var_value'
    elif table == 'sample':
        id_col = 'var_id'
        val_col = 'var_value'
    else:
        raise ValueError("Must pass tables fit_var or sample.")

    dfs = pd.DataFrame()
    for db in dbs:
        mulcov = db.covariate[db.covariate.c_covariate_name.isin(covs)].merge(db.mulcov)
        try:
            df = db.var.merge(getattr(db, table), left_on='var_id', right_on=id_col)
            df = df.fillna(np.nan)

            df = df.merge(db.integrand, on='integrand_id', how='left')
            df = df.merge(db.rate, on='rate_id', how='left')
            mulcov = mulcov.astype({'integrand_id': 'float64'})
            df = mulcov.merge(df)
            df.rename(columns={val_col: 'mulcov_value'}, inplace=True)
            df = df[[
                'c_covariate_name', 'mulcov_type', 'rate_name',
                'integrand_name', 'mulcov_value'
            ]]
        except AttributeError:
            df = pd.DataFrame()
        dfs = dfs.append(df)
    return dfs


def compute_statistics(df, mean=True, std=True, quantile=None):
    """
    Compute statistics on a data frame with mulcovs.
    Args:
        df: pd.DataFrame
        mean: bool
        std: bool
        quantile: optional list

    Returns: dictionary with requested statistics

    """
    stats_df = pd.DataFrame()
    group_cols = ['c_covariate_name', 'mulcov_type', 'rate_name', 'integrand_name']
    df_groups = df.fillna('none').copy().groupby(group_cols, sort=False)

    if mean:
        ds = df_groups.mean().reset_index()
        ds['stat'] = 'mean'
        stats_df = stats_df.append(ds)
    if std:
        degrees_of_freedom = int(df_groups.ngroups > len(df))
        ds = df_groups.std(ddof=degrees_of_freedom).reset_index()
        ds['stat'] = 'std'
        stats_df = stats_df.append(ds)
    if quantile is not None:
        for q in quantile:
            ds = df_groups.quantile(q=q).reset_index()
            ds['stat'] = f'quantile_{q}'
            stats_df = stats_df.append(ds)
    return stats_df


def main():
    """

    Returns:

    """
    args = get_args()
    logging.basicConfig(level=LEVELS[args.loglevel])

    context = Context(model_version_id=args.model_version_id)
    db_files = [DismodIO(context.db_file(location_id=loc, sex_id=sex))
                for loc in args.locations for sex in args.sexes]
    LOG.info(f"There are {len(db_files)} databases that will be aggregated.")

    common_covariates = common_covariate_names(db_files)
    LOG.info(f"The common covariates in the passed databases are {common_covariates}.")

    if args.sample:
        table_name = 'sample'
    else:
        table_name = 'fit_var'

    LOG.info(f"Will pull from the {table_name} table from each database.")
    mulcov_estimates = get_mulcovs(
        dbs=db_files, covs=common_covariates, table=table_name
    )
    mulcov_statistics = compute_statistics(
        df=mulcov_estimates, mean=args.mean, std=args.std, quantile=args.quantile
    )
    LOG.info('Write to output file.')
    mulcov_statistics.to_csv(context.outputs_dir / f'{args.outfile_name}.csv', index=False)


if __name__ == '__main__':
    main()
