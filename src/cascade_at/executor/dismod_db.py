import logging
import sys
from pathlib import Path
from typing import Union, List, Dict, Any, Optional

import numpy as np

from cascade_at.context.model_context import Context
from cascade_at.core.log import get_loggers, LEVELS
from cascade_at.dismod.api.dismod_extractor import DismodExtractor
from cascade_at.dismod.api.dismod_filler import DismodFiller
from cascade_at.dismod.api.run_dismod import run_dismod_commands
from cascade_at.executor.args.arg_utils import ArgumentList
from cascade_at.executor.args.args import DmCommands, DmOptions, ParentLocationID, SexID
from cascade_at.executor.args.args import ModelVersionID, BoolArg, LogLevel, StrArg, IntArg
from cascade_at.inputs.measurement_inputs import MeasurementInputs
from cascade_at.model.grid_alchemy import Alchemy
from cascade_at.saver.results_handler import ResultsHandler
from cascade_at.settings.settings_config import SettingsConfig

LOG = get_loggers(__name__)


ARG_LIST = ArgumentList([
    ModelVersionID(),
    ParentLocationID(),
    SexID(),
    DmCommands(),
    DmOptions(),
    BoolArg('--fill', help='whether or not to fill the dismod database with data'),
    IntArg('--prior-parent', help='the location ID of the parent database to grab the prior for'),
    IntArg('--prior-sex', help='the sex ID of the parent database to grab prior for'),
    BoolArg('--save', help='whether or not to save the fit'),
    LogLevel(),
    StrArg('--test-dir', help='if set, will save files to the directory specified')
])


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


def fill_database(path: Union[str, Path], settings: SettingsConfig,
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


def save_fit(db_file: Union[str, Path], location_id: int, sex_id: int,
             model_version_id: int, out_dir: Path) -> None:
    """
    Save the fit from this dismod database for a specific location and sex to be
    uploaded later on.
    """
    LOG.info("Extracting results from DisMod SQLite Database.")
    da = DismodExtractor(path=db_file)
    predictions = da.format_predictions_for_ihme(locations=[location_id], sexes=[sex_id])

    LOG.info(f"Saving the results for location {location_id} and sex {sex_id} to {out_dir}.")
    rh = ResultsHandler(model_version_id=model_version_id)
    rh.save_draw_files(df=predictions, directory=out_dir)


def dismod_db(model_version_id: int, parent_location_id: int, sex_id: int,
              dm_commands: List[str], dm_options: Dict[str, Union[int, str, float]],
              prior_parent: Optional[int] = None, prior_sex: Optional[int] = None,
              test_dir: Optional[str] = None, fill: bool = False,
              save: bool = True) -> None:
    """
    Creates a dismod database using the saved inputs and the file
    structure specified in the context. Alternatively it will
    skip the filling stage and move straight to the command
    stage if you don't pass --fill.

    Then runs an optional set of commands on the database passed
    in the --commands argument.

    Also passes an optional argument --options as a dictionary to
    the dismod database to fill/modify the options table.

    Parameters
    ----------
    model_version_id
        The model version ID
    parent_location_id
        The parent location for the database
    sex_id
        The parent sex for the database
    dm_commands
        A list of commands to pass to the run_dismod_commands function, executed
        directly on the dismod database
    dm_options
        A dictionary of options to pass to the the dismod option table
    prior_parent
        An optional parent location ID that specifies where to pull the prior
        information from.
    prior_sex
        An optional parent sex ID that specifies where to pull the prior information from.
    test_dir
        A test directory to create the database in rather than the database
        specified by the IHME file system context.
    fill
        Whether or not to fill the database with new inputs based on the model_version_id,
        parent_location_id, and sex_id. If not filling, this script can be used
        to just execute commands on the database instead.
    save
        Whether or not to save the fit from this database as the parent fit.
    """
    if test_dir is not None:
        context = Context(model_version_id=model_version_id,
                          configure_application=False,
                          root_directory=test_dir)
    else:
        context = Context(model_version_id=model_version_id)

    db_path = context.db_file(location_id=parent_location_id, sex_id=sex_id)
    inputs, alchemy, settings = context.read_inputs()

    # If we want to override the rate priors with posteriors from a previous
    # database, pass them in here.
    if prior_parent or prior_sex:
        if not (prior_parent and prior_sex):
            raise RuntimeError("Need to pass both prior parent and sex or neither.")
        child_prior = get_prior(
            path=context.db_file(
                location_id=prior_parent,
                sex_id=prior_sex
            ),
            location_id=parent_location_id, sex_id=sex_id,
            rates=[r.rate for r in settings.rate]
        )
    else:
        child_prior = None

    if fill:
        fill_database(
            path=db_path, inputs=inputs, alchemy=alchemy, settings=settings,
            parent_location_id=parent_location_id, sex_id=sex_id,
            child_prior=child_prior, options=dm_options
        )

    if dm_commands:
        run_dismod_commands(dm_file=str(db_path), commands=dm_commands)

    if save:
        save_fit(
            db_file=context.db_file(location_id=parent_location_id, sex_id=sex_id),
            location_id=parent_location_id, sex_id=sex_id,
            model_version_id=model_version_id,
            out_dir=context.fit_dir
        )


def main():

    args = ARG_LIST.parse_args(sys.argv[1:])
    logging.basicConfig(level=LEVELS[args.log_level])

    dismod_db(
        model_version_id=args.model_version_id,
        parent_location_id=args.parent_location_id,
        sex_id=args.sex_id,
        dm_commands=args.dm_commands,
        dm_options=args.dm_options,
        fill=args.fill,
        prior_parent=args.prior_parent,
        prior_sex=args.prior_sex,
        test_dir=args.test_dir,
        save=args.save
    )


if __name__ == '__main__':
    main()
