import logging
from shutil import copy2
from argparse import ArgumentParser
import pandas as pd
from multiprocessing import Pool
from pathlib import Path
from typing import Union

from cascade_at.context.model_context import Context
from cascade_at.executor import ExecutorError
from cascade_at.dismod.api.dismod_io import DismodIO
from cascade_at.dismod.api.run_dismod import run_dismod_commands
from cascade_at.core.log import get_loggers, LEVELS


LOG = get_loggers(__name__)


class SampleSimulateError(ExecutorError):
    """Raised when there are issues with sample simulate."""
    pass


def get_args():
    """
    Parse the arguments for simulating and sampling from a dismod database.
    """
    parser = ArgumentParser()
    parser.add_argument("--model-version-id", type=int, required=True)
    parser.add_argument("--parent-location-id", type=int, required=True)
    parser.add_argument("--sex-id", type=int, required=True)
    parser.add_argument("--n-sim", type=int, required=False, default=5)
    parser.add_argument("--n-pool", type=int, required=False, default=1,
                        help="How many multiprocessing pools (1 means not in parallel, so no pools)")
    parser.add_argument("--fit-type", type=str, required=False, default='both')
    parser.add_argument("--loglevel", type=str, required=False, default='info')

    return parser.parse_args()


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
            raise SampleSimulateError("Cannot run sample simulate on a database without fit_var!")
    except ValueError:
        raise SampleSimulateError("Cannot run sample simulate on a database without fit_var!"
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


class FitSample:
    """
    Fit Sample for a database in parallel. Copies the sample table and fits for just
    one sample index.

    Parameters
    ----------
    main_db
        Path to the main database to sample from.
    index_file_pattern
        File pattern to create the index databases with different samples.
    fit_type
        The type of fit to run, one of "fixed" or "both".
    """
    def __init__(self, main_db: Union[str, Path], index_file_pattern: str, fit_type: str):
        self.main_db = main_db
        self.index_file_pattern = index_file_pattern
        self.fit_type = fit_type

    def __call__(self, index: int):
        index_db = self.index_file_pattern.format(index=index)
        copy2(src=str(self.main_db), dst=str(index_db))

        run_dismod_commands(dm_file=index_db, commands=[f'fit {self.fit_type} {index}'])

        db = DismodIO(path=index_db)
        fit = db.fit_var
        fit['sample_index'] = index
        return fit


def sample_simulate_pool(main_db, index_file_pattern, fit_type, n_sim, n_pool):
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
        raise SampleSimulateError(f"Unrecognized fit type {fit_type}.")

    fit_sample = FitSample(main_db=main_db, index_file_pattern=index_file_pattern, fit_type=fit_type)

    p = Pool(n_pool)
    fits = list(p.map(fit_sample, range(n_sim)))
    p.close()

    # Reconstruct the sample table with all n_sim fits
    sample = pd.DataFrame().append(fits).reset_index(drop=True)
    sample.rename(columns={'fit_var_id': 'var_id', 'fit_var_value': 'var_value'}, inplace=True)
    
    d = DismodIO(path=main_db)
    d.sample = sample[['sample_index', 'var_id', 'var_value']]


def sample_simulate_sequence(path: Union[str, Path], n_sim: int):
    """
    Fit the samples in a database in sequence.

    Parameters
    ----------
    path
        A path to the database object to create simulations in.
    n_sim
        Number of simulations to create.
    """
    run_dismod_commands(
        dm_file=path,
        commands=[
            f'sample simulate {n_sim}'
        ]
    )


def main():
    args = get_args()
    logging.basicConfig(level=LEVELS[args.loglevel])

    context = Context(model_version_id=args.model_version_id)
    main_db = context.db_file(location_id=args.parent_location_id, sex_id=args.sex_id)
    index_file_pattern = context.db_index_file_pattern(location_id=args.parent_location_id, sex_id=args.sex_id)

    simulate(path=main_db, n_sim=args.n_sim)

    if args.n_pool > 1:
        sample_simulate_pool(main_db=main_db, index_file_pattern=index_file_pattern, fit_type=args.fit_type,
                             n_pool=args.n_pool, n_sim=args.n_sim)
    else:
        sample_simulate_sequence(path=main_db, n_sim=args.n_sim)


if __name__ == '__main__':
    main()
