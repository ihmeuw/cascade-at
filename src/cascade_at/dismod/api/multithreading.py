from typing import Union, List
from pathlib import Path
from shutil import copy2
from multiprocessing import Pool

from cascade_at.dismod.api.run_dismod import run_dismod


class _DismodThread:
    """
    Splits a dismod database into multiple databases to run parallel
    processes on the database. The work happens when you call
    an instantiated _DismodThread.
    """
    def __init__(self, main_db: Union[str, Path], index_file_pattern: str):
        self.main_db = main_db
        self.index_file_pattern = index_file_pattern
        self.index = None

    def __call__(self, index: int):
        self.index = index
        index_db = self.index_file_pattern.format(index=index)
        copy2(src=str(self.main_db), dst=str(index_db))
        
        # Set the seed to null so each process will have a unique random sequence
        run_dismod(str(index_db), "set option random_seed ''")
        return self._process(db=index_db)

    def _process(self, db: str):
        raise NotImplementedError


def dmdismod_in_parallel(dm_thread: _DismodThread,
                         sims: List[int], n_pool: int):
    """
    Run a dismod thread in parallel by constructing
    a multiprocessing pool. A dismod thread is
    anything that is based off of _DismodThread so it has
    a __call__ method with an overridden _process method.
    """
    p = Pool(n_pool)
    processes = list(p.map(dm_thread, sims))
    p.close()
    return processes
