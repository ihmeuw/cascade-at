from copy import deepcopy
import numpy as np
from typing import List, Dict

from cascade_at.model.grid_alchemy import Alchemy


def integrand_grids(alchemy: Alchemy, integrands: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Get the age-time grids associated with a list of integrands.
    Should be used for converting priors to posteriors. Uses the default grid unless
    another one has been specified.

    Parameters:
    ----------
    alchemy
        An alchemy object for the model
    integrands
        A list of integrands

    Returns
    -------
    Dict, a dictionary of grids with keys for each integrand, which are dictionaries for "age" and "time".
    """
    grids = dict()

    default_grid = alchemy.construct_age_time_grid()
    for integrand in integrands:
        grids[integrand] = deepcopy(default_grid)

    rate_grids = alchemy.get_all_rates_grids()
    for k, v in rate_grids.items():
        if k in integrands:
            grids[k].update({'age': v.ages})
            grids[k].update({'time': v.times})
    return grids
