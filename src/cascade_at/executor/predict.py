import sys
from pathlib import Path
from typing import List, Union

import logging
import pandas as pd

from cascade_at.context.model_context import Context
from cascade_at.core.log import get_loggers, LEVELS
from cascade_at.dismod.api.dismod_io import DismodIO
from cascade_at.dismod.api.fill_extract_helpers.data_tables import prep_data_avgint
from cascade_at.dismod.api.fill_extract_helpers.posterior_to_prior import get_prior_avgint_grid
from cascade_at.dismod.api.multithreading import _DismodThread, dmdismod_in_parallel
from cascade_at.dismod.api.run_dismod import run_dismod_commands
from cascade_at.executor.args.arg_utils import ArgumentList
from cascade_at.executor.args.args import LogLevel, BoolArg, ListArg
from cascade_at.executor.args.args import ModelVersionID, ParentLocationID, SexID, NSim, NPool
from cascade_at.executor.dismod_db import save_predictions
from cascade_at.inputs.measurement_inputs import MeasurementInputs
from cascade_at.model.grid_alchemy import Alchemy
from cascade_at.model.utilities.integrand_grids import integrand_grids
from cascade_at.settings.settings import SettingsConfig

LOG = get_loggers(__name__)


ARG_LIST = ArgumentList([
    ModelVersionID(),
    ParentLocationID(),
    SexID(),
    NSim(),
    NPool(),
    ListArg('--child-locations', help='child locations to make predictions for', type=int, required=False),
    ListArg('--child-sexes', help='sexes to make predictions for', type=int, required=False),
    BoolArg('--prior-grid', help='whether to predict on the prior grid or the regular avgint grid'),
    BoolArg('--save-fit', help='whether to save the results of the predict sample as the fit'),
    BoolArg('--save-final', help='whether to save results as final'),
    BoolArg('--sample', help='whether to predict from the sample table or the fit_var table'),
    LogLevel()
])


def fill_avgint_with_priors_grid(inputs: MeasurementInputs, alchemy: Alchemy, settings: SettingsConfig,
                                 source_db_path: Union[str, Path],
                                 child_locations: List[int], child_sexes: List[int]):
    """
    Fill the average integrand table with the grid that the priors are on.
    This is so that we can "predict" the prior for the next level of the cascade.

    Parameters
    ----------
    inputs
        An inputs object
    alchemy
        A grid alchemy object
    settings
        A settings configuration object
    source_db_path
        The path of the source database that has had a fit on it
    child_locations
        The child locations to predict for
    child_sexes
        The child sexes to predict for
    """

    sourceDB = DismodIO(path=source_db_path)
    rates = [r.rate for r in settings.rate]
    grids = integrand_grids(alchemy=alchemy, integrands=rates)

    posterior_grid = get_prior_avgint_grid(
        grids=grids,
        sexes=child_sexes,
        locations=child_locations,
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


class Predict(_DismodThread):
    """
    Predicts for a database in parallel. Chops up the sample table
    into a bunch of copies, each with only one sample.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _process(self, db: str):

        dbio = DismodIO(path=db)
        n_var = len(dbio.var)

        this_sample = dbio.sample.loc[dbio.sample.sample_index == self.index].copy()
        this_sample['sample_index'] = 0
        this_sample['sample_id'] = this_sample['var_id']
        dbio.sample = this_sample
        del dbio

        run_dismod_commands(
            dm_file=db,
            commands=[f'predict sample']
        )
        dbio = DismodIO(path=db)
        predict = dbio.predict
        predict['sample_index'] = self.index
        return predict


def predict_sample_sequence(path: Union[str, Path], table: str):
    """
    Runs predict for either fit_var or sample, based on the table.
    """
    run_dismod_commands(
        dm_file=path,
        commands=[f'predict {table}']
    )


def predict_sample_pool(main_db: Union[str, Path], index_file_pattern: str,
                        n_sim: int, n_pool: int):
    """
    Run predict sample in a pool by making copies of the existing database
    and splitting out the sample table into n_sim databases, running
    predict sample on each of them, and combining the results back
    into the main database.
    """
    predict = Predict(
        main_db=main_db,
        index_file_pattern=index_file_pattern
    )
    predictions = dmdismod_in_parallel(
        dm_thread=predict,
        sims=list(range(n_sim)),
        n_pool=n_pool
    )
    predictions = pd.DataFrame().append(predictions).reset_index(drop=True)
    d = DismodIO(path=main_db)
    return predictions[['sample_index', 'avgint_id', 'avg_integrand']]


def predict_sample(model_version_id: int, parent_location_id: int, sex_id: int,
                   child_locations: List[int], child_sexes: List[int],
                   prior_grid: bool = True, save_fit: bool = False, save_final: bool = False,
                   sample: bool = False, n_sim: int = 1, n_pool: int = 1) -> None:
    """
    Takes a database that has already had a fit and simulate sample run on it,
    fills the avgint table for the child_locations and child_sexes you want to make
    predictions for, and then predicts on that grid. Makes predictions on the grid
    that is specified for the primary rates in the model, for the primary rates only.

    Parameters
    ----------
    model_version_id
        The model version ID
    parent_location_id
        The parent location ID that specifies where the database is stored
    sex_id
        The sex ID that specifies where the database is stored
    child_locations
        The child locations to make predictions for on the rate grid
    child_sexes
        The child sexes to make predictions for on the rate grid
    prior_grid
        Whether or not to replace the default gbd-avgint grid with
        a prior grid for the rates.
    save_fit
        Whether or not to save the fit for upload later.
    save_final
        Whether or not to save the final for upload later.
    sample
        Whether to predict from the sample table or the fit_var table
    n_sim
        The number of simulations to predict for
    n_pool
        The number of multiprocessing pools to create. If 1, then will not
        run with pools but just run all simulations together in one dmdismod command.

    """
    predictions = None

    context = Context(model_version_id=model_version_id)
    inputs, alchemy, settings = context.read_inputs()
    main_db = context.db_file(location_id=parent_location_id, sex_id=sex_id)
    index_file_pattern = context.db_index_file_pattern(location_id=parent_location_id, sex_id=sex_id)
    
    if sample:
        table = 'sample'
    else:
        table = 'fit_var'

    if prior_grid:
        fill_avgint_with_priors_grid(
            inputs=inputs, alchemy=alchemy, settings=settings, source_db_path=main_db,
            child_locations=child_locations, child_sexes=child_sexes
        )

    if sample and (n_pool > 1):
        predictions = predict_sample_pool(
            main_db=main_db, index_file_pattern=index_file_pattern,
            n_sim=n_sim, n_pool=n_pool
        )
    else:
        predict_sample_sequence(path=main_db, table=table)

    if save_fit or save_final:
        if len(child_locations) == 0:
            locations = inputs.location_dag.parent_children(parent_location_id)
        else:
            locations = child_locations
        if len(child_sexes) == 0:
            sexes = [sex_id]
        else:
            sexes = child_sexes
        out_dirs = []
        if save_fit:
            out_dirs.append(context.fit_dir)
        if save_final:
            out_dirs.append(context.draw_dir)
        for folder in out_dirs:
            save_predictions(
                db_file=main_db,
                locations=locations, sexes=sexes,
                model_version_id=model_version_id,
                gbd_round_id=settings.gbd_round_id,
                out_dir=folder,
                sample=sample,
                predictions=predictions
            )


def main():

    args = ARG_LIST.parse_args(sys.argv[1:])
    logging.basicConfig(level=LEVELS[args.log_level])

    predict_sample(
        model_version_id=args.model_version_id,
        parent_location_id=args.parent_location_id,
        sex_id=args.sex_id,
        child_locations=args.child_locations,
        child_sexes=args.child_sexes,
        prior_grid=args.prior_grid,
        save_fit=args.save_fit,
        save_final=args.save_final,
        sample=args.sample,
        n_sim=args.n_sim,
        n_pool=args.n_pool
    )


if __name__ == '__main__':
    main()
