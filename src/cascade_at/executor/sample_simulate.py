import logging
from shutil import copy2
from argparse import ArgumentParser
import pandas as pd
from multiprocessing import Pool

from cascade_at.context.model_context import Context
from cascade_at.dismod.api.dismod_io import DismodIO
from cascade_at.dismod.api.run_dismod import run_dismod_commands
from cascade_at.core.log import get_loggers, LEVELS


LOG = get_loggers(__name__)


def get_args():
    """
    Parse the arguments for simulating and sampling from a dismod database.
    """
    parser = ArgumentParser()
    parser.add_argument("-model-version-id", type=int, required=True)
    parser.add_argument("-parent-location-id", type=int, required=True)
    parser.add_argument("-sex-id", type=int, required=True)
    parser.add_argument("-n-sim", type=int, required=False, default=5)
    parser.add_argument("-n-pool", type=int, required=False, default=1,
                        help="How many multiprocessing pools (1 means no parallelizing)")
    parser.add_argument("-fit-type", type=str, required=False, default='both')
    parser.add_argument("--loglevel", type=str, required=False, default='info')

    return parser.parse_args()


class FitSample:
    def __init__(self, context, location_id, sex_id, fit_type):
        """
        Fits a sample on a database.
        Args:
            context: (cascade_at.context.model_context.Context)
            location_id: (int)
            sex_id: (int)
            fit_type: (str)
        """
        self.context = context
        self.location_id = location_id
        self.sex_id = sex_id
        self.fit_type = fit_type

        self.main_db = context.db_file(
            location_id=self.location_id,
            sex_id=self.sex_id
        )

    def __call__(self, index=None):
        index_db = self.context.db_file(
            location_id=self.location_id,
            sex_id=self.sex_id,
            index=index
        )
        if index is not None:
            copy2(src=str(self.main_db), dst=str(index_db))
        run_dismod_commands(dm_file=index_db, commands=[f'fit {self.fit_type} {index}'])
        db = DismodIO(path=index_db)
        fit = db.fit_var
        fit['sample_index'] = index
        return fit


def main():
    """
    Takes dismod databases that have already had a fit run on them and simulates new datasets, refitting
    on all of them, then combining the results back into one database.
    Returns:

    """
    args = get_args()
    logging.basicConfig(level=LEVELS[args.loglevel])

    context = Context(model_version_id=args.model_version_id)
    main_db = context.db_file(location_id=args.parent_location_id, sex_id=args.sex_id)

    d = DismodIO(path=main_db)
    if d.fit_var.empty:
        raise RuntimeError("Cannot run sample / simulate on a database without fit_var!")

    # Create n_sim simulation datasets based on the fitted parameters
    run_dismod_commands(
        dm_file=main_db,
        commands=[
            'set start_var fit_var'
            'set truth_var fit_var',
            'set scale_var fit_var',
            f'simulate {args.n_sim}'
        ]
    )

    if args.n_pool > 1:
        # Make a pool and fit to each of the simulations (uses the __call__ method)
        fit_sample = FitSample(context=context, location_id=args.location_id, sex_id=args.sex_id,
                               fit_type=args.fit_type)
        p = Pool(args.n_pool)
        fits = list(p.map(fit_sample, range(args.n_sim)))
        p.close()

        # Reconstruct the sample table with all n_sim fits
        sample = pd.DataFrame().append(fits).reset_index(drop=True)
        sample.rename(columns={'fit_var_id': 'var_id', 'fit_var_value': 'var_value'}, inplace=True)
        d.sample = sample
    else:
        # If we only have one pool that means we aren't going to run in parallel
        run_dismod_commands(
            dm_file=main_db,
            commands=[
                f'sample simulate {args.n_sim}'
            ]
        )
