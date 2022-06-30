#!/usr/bin/env python
import logging
import sys
from typing import List, Optional

import numpy as np
import pandas as pd

from cascade_at.executor.args.arg_utils import ArgumentList
from cascade_at.executor.args.args import ModelVersionID, BoolArg, ListArg, StrArg, LogLevel
from cascade_at.context.model_context import Context
from cascade_at.core.log import get_loggers, LEVELS
from cascade_at.dismod.api.dismod_io import DismodIO

LOG = get_loggers(__name__)


ARG_LIST = ArgumentList([
    ModelVersionID(),
    ListArg('--locations', help='The locations to pull mulcov statistics from', type=int, required=True),
    ListArg('--sexes', help='The sexes to pull mulcov statistics from', type=int, required=True),
    StrArg('--outfile-name', help='Filepath where mulcov statistics will be saved', required=False, default='mulcov_stats'),
    BoolArg('--sample', help='If true, the results will be pulled from the sample table rather'
                             'than the fit_var table'),
    BoolArg('--mean', help='Whether or not to compute the mean'),
    BoolArg('--std', help='Whether or not to compute the standard deviation'),
    ListArg('--quantile', help='Quantiles to compute', type=float),
    LogLevel()
])


def common_covariate_names(dbs):
    return set.intersection(
        *map(set, [d.covariate.c_covariate_name.tolist() for d in dbs])
    )


def get_mulcovs(dbs: List[DismodIO], covs: List[str],
                table: str = 'fit_var') -> pd.DataFrame:
    """
    Get mulcov values from all of the dbs, with all of the common covariates.

    Parameters
    dbs
        A list of dismod i/o objects
    covs
        A list of covariate names
    table
        Name of the table to pull from (can be fit_var or sample)
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
            mulcov = mulcov.astype({'integrand_id': 'float64', 'rate_id': 'float64'})
            df = mulcov.merge(df)
            df.rename(columns={val_col: 'mulcov_value'}, inplace=True)
            df = df[[
                'c_covariate_name', 'mulcov_type', 'rate_name',
                'integrand_name', 'mulcov_value'
            ]]
        except AttributeError:
            df = pd.DataFrame()
        dfs = pd.merge([dfs, df])
    return dfs


def compute_statistics(df, mean=True, std=True, quantile=None):
    """
    Compute statistics on a data frame with covariate multipliers.
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
    stats_df = df_groups.count().reset_index()[group_cols]
    if mean:
        ds = df_groups.mean().reset_index()
        stats_df['mean'] = ds['mulcov_value']
    if std:
        degrees_of_freedom = int(df_groups.ngroups > len(df))
        ds = df_groups.std(ddof=degrees_of_freedom).reset_index()
        stats_df['std'] = ds['mulcov_value']
    if quantile is not None:
        for q in quantile:
            ds = df_groups.quantile(q=q).reset_index()
            stats_df[f'quantile_{q}'] = ds['mulcov_value']
    return stats_df


def mulcov_statistics(model_version_id: int, locations: List[int], sexes: List[int],
                      outfile_name: str, sample: bool = True,
                      mean: bool = True, std: bool = True,
                      quantile: Optional[List[float]] = None) -> None:
    """
    Compute statistics for the covariate multipliers on a dismod database,
    and save them to a file.

    Parameters
    ----------
    model_version_id
        The model version ID
    locations
        A list of locations that, when used in combination with sexes, point to the databases
        to pull covariate multiplier estimates from
    sexes
        A list of sexes that, when used in combination with locations, point to the databases
        to pull covariate multiplier estimates from
    outfile_name
        A filepath specifying where to save the covariate multiplier statistics.
    sample
        Whether or not the results are stored in the sample table or the fit_var table.
    mean
        Whether or not to compute the mean
    std
        Whether or not to compute the standard deviation
    quantile
        An optional list of quantiles to compute
    """

    context = Context(model_version_id=model_version_id)
    db_files = [DismodIO(context.db_file(location_id=loc, sex_id=sex))
                for loc in locations for sex in sexes]
    LOG.info(f"There are {len(db_files)} databases that will be aggregated.")

    common_covariates = common_covariate_names(db_files)
    LOG.info(f"The common covariates in the passed databases are {common_covariates}.")

    if sample:
        table_name = 'sample'
    else:
        table_name = 'fit_var'

    LOG.info(f"Will pull from the {table_name} table from each database.")
    mulcov_estimates = get_mulcovs(
        dbs=db_files, covs=common_covariates, table=table_name
    )
    if not mulcov_estimates.empty:
        stats = compute_statistics(
            df=mulcov_estimates, mean=mean, std=std, quantile=quantile
        )
    else:
        stats = mulcov_estimates
    LOG.info('Write to output file.')
    stats.to_csv(context.outputs_dir / f'{outfile_name}.csv', index=False)


def main():

    args = ARG_LIST.parse_args(sys.argv[1:])
    logging.basicConfig(level=LEVELS[args.log_level])

    mulcov_statistics(
        model_version_id=args.model_version_id,
        locations=args.locations,
        sexes=args.sexes,
        outfile_name=args.outfile_name,
        sample=args.sample,
        mean=args.mean,
        std=args.std,
        quantile=args.quantile
    )


if __name__ == '__main__':
    main()
