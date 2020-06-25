import logging
import sys
from pathlib import Path
from typing import List, Union

from cascade_at.context.model_context import Context
from cascade_at.core.log import get_loggers, LEVELS
from cascade_at.dismod.api.dismod_io import DismodIO
from cascade_at.dismod.api.fill_extract_helpers.data_tables import prep_data_avgint
from cascade_at.dismod.api.fill_extract_helpers.posterior_to_prior import get_prior_avgint_grid
from cascade_at.dismod.api.run_dismod import run_dismod_commands
from cascade_at.executor.args.arg_utils import ArgumentList
from cascade_at.executor.args.args import ModelVersionID, BoolArg, ListArg, ParentLocationID, SexID, LogLevel
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
    ListArg('--child-locations', help='child locations to make predictions for', type=int, required=False),
    ListArg('--child-sexes', help='sexes to make predictions for', type=int, required=False),
    BoolArg('--prior-grid', help='whether to predict on the prior grid or the regular avgint grid'),
    BoolArg('--save-fit', help='whether to save the results of the predict sample as the fit'),
    BoolArg('--sample', help='whether to predict from the sample table or the fit_var table'),
    LogLevel()
])


def fill_avgint_with_priors_grid(inputs: MeasurementInputs, alchemy: Alchemy, settings: SettingsConfig,
                                 source_db_path: Union[str, Path],
                                 child_locations: List[int], child_sexes: List[int]):

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


def predict_sample(model_version_id: int, parent_location_id: int, sex_id: int,
                   child_locations: List[int], child_sexes: List[int],
                   prior_grid: bool = True, save_fit: bool = False, sample: bool = False) -> None:
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
    sample
        Whether to predict from the sample table or the fit_var table
    """
    context = Context(model_version_id=model_version_id)
    inputs, alchemy, settings = context.read_inputs()
    path = context.db_file(location_id=parent_location_id, sex_id=sex_id)
    
    if sample:
        table = 'sample'
    else:
        table = 'fit_var'

    if prior_grid:
        fill_avgint_with_priors_grid(
            inputs=inputs, alchemy=alchemy, settings=settings, source_db_path=path,
            child_locations=child_locations, child_sexes=child_sexes
        )
    run_dismod_commands(
        dm_file=path,
        commands=[f'predict {table}']
    )
    if save_fit:
        if len(child_locations) == 0:
            locations = inputs.location_dag.parent_children(parent_location_id)
        else:
            locations = child_locations
        if len(child_sexes) == 0:
            sexes = [sex_id]
        else:
            sexes = child_sexes
        save_predictions(
            db_file=path,
            locations=locations, sexes=sexes,
            model_version_id=model_version_id,
            gbd_round_id=settings.gbd_round_id,
            out_dir=context.fit_dir,
            sample=sample
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
        sample=args.sample
    )


if __name__ == '__main__':
    main()
