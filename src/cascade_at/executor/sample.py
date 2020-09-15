import pandas as pd

import logging
import sys
from pathlib import Path
from typing import Union

from cascade_at.executor.args.arg_utils import ArgumentList
from cascade_at.executor.args.args import ModelVersionID, ParentLocationID, SexID, NPool, NSim
from cascade_at.executor.args.args import StrArg, BoolArg, LogLevel
from cascade_at.context.model_context import Context
from cascade_at.core.log import get_loggers, LEVELS
from cascade_at.dismod.api.dismod_io import DismodIO
from cascade_at.dismod.process.process_behavior import check_sample_asymptotic, SampleAsymptoticError
from cascade_at.dismod.api.multithreading import _DismodThread, dmdismod_in_parallel
from cascade_at.dismod.api.run_dismod import run_dismod_commands
from cascade_at.executor import ExecutorError

LOG = get_loggers(__name__)


ARG_LIST = ArgumentList([
    ModelVersionID(),
    ParentLocationID(),
    SexID(),
    NSim(),
    NPool(),
    StrArg('--fit-type', help='what type of fit to simulate for, fit fixed or both', default='both'),
    BoolArg('--asymptotic', help='whether or not to do asymptotic statistics or fit-refit'),
    LogLevel()
])


class SampleError(ExecutorError):
    """Raised when there are issues with sample simulate."""
    pass


def simulate(path: Union[str, Path], n_sim: int):
    """
    Simulate from a database, within a database.

    Parameters
    ----------
    path
        A path to the database object to create simulations in.
    n_sim
        Number of simulations to create.
    """
    d = DismodIO(path=path)
    try:
        if d.fit_var.empty:
            raise SampleError("Cannot run sample simulate on a database without fit_var!")
    except ValueError:
        raise SampleError("Cannot run sample simulate on a database without fit_var!"
                          "Does not have the fit_var table yet.")

    # Create n_sim simulation datasets based on the fitted parameters
    run_dismod_commands(
        dm_file=path,
        commands=[
            'set start_var fit_var',
            'set truth_var fit_var',
            'set scale_var fit_var',
            f'simulate {n_sim}'
        ]
    )


class FitSample(_DismodThread):
    """
    Fit Sample for a database in parallel. Copies the sample table and fits for just
    one sample index. Will use the __call__ method from _DismodThread.

    Parameters
    ----------
    main_db
        Path to the main database to sample from.
    index_file_pattern
        File pattern to create the index databases with different samples.
    fit_type
        The type of fit to run, one of "fixed" or "both".
    """
    def __init__(self, fit_type: str, **kwargs):
        super().__init__(**kwargs)
        self.fit_type = fit_type

    def _process(self, db: str):
        run_dismod_commands(
            dm_file=db, commands=[f'fit {self.fit_type} {self.index}']
        )

        db = DismodIO(path=db)
        fit = db.fit_var
        fit['sample_index'] = self.index
        fit.rename(columns={
            'fit_var_id': 'var_id',
            'fit_var_value': 'var_value'
        }, inplace=True)
        return fit


def sample_simulate_pool(main_db: Union[str, Path], index_file_pattern: str,
                         fit_type: str, n_sim: int, n_pool: int):
    """
    Fit the samples in a database in parallel by making copies of the database, fitting them
    separately, and then combining them back together in the sample table of main_db.

    Parameters
    ----------
    main_db
        Path to the main database that will be spawned.
    index_file_pattern
        File pattern for the new databases that will have index equal to the simulation number.
    fit_type
        The type of fit to run, one of "fixed" or "both".
    n_sim
        Number of simulations that will be fit.
    n_pool
        Number of pools for the multiprocessing.
    """
    if fit_type not in ["fixed", "both"]:
        raise SampleError(f"Unrecognized fit type {fit_type}.")

    fit_sample = FitSample(
        main_db=main_db,
        index_file_pattern=index_file_pattern,
        fit_type=fit_type
    )
    fits = dmdismod_in_parallel(
        dm_thread=fit_sample,
        sims=list(range(n_sim)),
        n_pool=n_pool
    )
    # Reconstruct the sample table with all n_sim fits
    samp = pd.DataFrame().append(fits).reset_index(drop=True)
    d = DismodIO(path=main_db)
    d.sample = samp[['sample_index', 'var_id', 'var_value']]


def sample_simulate_sequence(path: Union[str, Path], n_sim: int, fit_type: str):
    """
    Fit the samples in a database in sequence.

    Parameters
    ----------
    path
        A path to the database object to create simulations in.
    n_sim
        Number of simulations to create.
    fit_type
        Type of fit -- fixed or both
    """
    run_dismod_commands(
        dm_file=path,
        commands=[
            f'sample simulate {fit_type} {n_sim}'
        ]
    )


def sample_asymptotic(path: Union[str, Path], n_sim: int, fit_type: str):
    """
        Fit the samples in a database in sequence.

        Parameters
        ----------
        path
            A path to the database object to create simulations in.
        n_sim
            Number of simulations to create.
        fit_type
            Type of fit -- fixed or both
        """
    return run_dismod_commands(
        dm_file=path,
        commands=[
            'set start_var fit_var',
            'set truth_var fit_var',
            'set scale_var fit_var',
            f'sample asymptotic {fit_type} {n_sim}'
        ]
    )


def sample(model_version_id: int, parent_location_id: int, sex_id: int,
           n_sim: int, n_pool: int, fit_type: str, asymptotic: bool = False) -> None:
    """
    Simulates from a dismod database that has already had a fit run on it. Does so
    optionally in parallel.

    Parameters
    ----------
    model_version_id
        The model version ID
    parent_location_id
        The parent location ID specifying location of database
    sex_id
        The sex ID specifying location of database
    n_sim
        The number of simulations to do
    n_pool
        The number of multiprocessing pools to create. If 1, then will not
        run with pools but just run all simulations together in one dmdismod command.
    fit_type
        The type of fit that was performed on this database, one of fixed or both.
    asymptotic
        Whether or not to do asymptotic samples or fit-refit
    """

    context = Context(model_version_id=model_version_id)
    main_db = context.db_file(location_id=parent_location_id, sex_id=sex_id)
    index_file_pattern = context.db_index_file_pattern(location_id=parent_location_id, sex_id=sex_id)

    if asymptotic:
        result = sample_asymptotic(path=main_db, n_sim=n_sim, fit_type=fit_type)
        try:
            check_sample_asymptotic(result[f'sample asymptotic {fit_type} {n_sim}'].stderr)
        except SampleAsymptoticError:
            asymptotic = False
            LOG.info("Jumping to sample simulate because sample asymptotic failed.")
            LOG.warning("Please review the warning from sample asymptotic.")
    if not asymptotic:
        simulate(path=main_db, n_sim=n_sim)
        if n_pool > 1:
            sample_simulate_pool(
                main_db=main_db, index_file_pattern=index_file_pattern, fit_type=fit_type,
                n_pool=n_pool, n_sim=n_sim
            )
        else:
            sample_simulate_sequence(path=main_db, n_sim=n_sim, fit_type=fit_type)


def main():

    args = ARG_LIST.parse_args(sys.argv[1:])
    logging.basicConfig(level=LEVELS[args.log_level])

    sample(
        model_version_id=args.model_version_id,
        parent_location_id=args.parent_location_id,
        sex_id=args.sex_id,
        n_sim=args.n_sim,
        n_pool=args.n_pool,
        fit_type=args.fit_type,
        asymptotic=args.asymptotic
    )


if __name__ == '__main__':
    main()
