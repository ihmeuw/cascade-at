import pandas as pd
from typing import Dict, List
import numpy as np

from cascade_at.dismod.api.fill_extract_helpers.utils import vec_to_midpoint
from cascade_at.model.utilities.grid_helpers import expand_grid
from cascade_at.dismod.constants import RateToIntegrand, IntegrandEnum, INTEGRAND_TO_WEIGHT


def get_prior_avgint_grid(grids: Dict[str, Dict[str, np.ndarray]],
                          sexes: List[int],
                          locations: List[int],
                          midpoint: bool = False) -> pd.DataFrame:
    """
    Get a data frame to use for setting up posterior predictions on a grid.
    The grids are specified in the grids parameter.

    Will still need to have covariates added to it, and prep data from
    dismod.api.data_tables.prep_data_avgint to convert nodes and covariate names
    before it can be input into the avgint table in a database.

    Parameters
    ---------
    grids
        A dictionary of grids with keys for each integrand,
        which are dictionaries for "age" and "time".
    sexes
        A list of sexes
    locations
        A list of locations
    midpoint
        Whether to midpoint the grid lower and upper values (recommended for rates).

    Returns
    -------
    Dataframe with columns
        "avgint_id", "integrand_id", "location_id", "weight_id", "subgroup_id",
        "age_lower", "age_upper", "time_lower", "time_upper", "sex_id"

    """
    posterior_dfs = pd.DataFrame()
    for k, v in grids.items():
        if midpoint:
            time = vec_to_midpoint(v['time'])
            age = vec_to_midpoint(v['age'])
        else:
            time = v['time']
            age = v['age']

        posterior_df = expand_grid({
            'age_lower': age,
            'time_lower': time,
            'location_id': locations,
            'sex_id': sexes
        })
        posterior_df['time_upper'] = posterior_df['time_lower']
        posterior_df['age_upper'] = posterior_df['age_lower']

        posterior_df['rate'] = k
        posterior_df['integrand'] = posterior_df['rate'].map(RateToIntegrand)
        posterior_df['integrand_id'] = posterior_df['integrand'].apply(
            lambda x: IntegrandEnum[x].value
        )
        posterior_df['weight_id'] = posterior_df["integrand"].apply(
            lambda x: INTEGRAND_TO_WEIGHT[x].value
        )
        posterior_df['subgroup_id'] = 0

        posterior_dfs = posterior_dfs.append(posterior_df)

    return posterior_dfs[[
        "integrand_id", "location_id", "weight_id", "subgroup_id",
        "age_lower", "age_upper", "time_lower", "time_upper", "sex_id"
    ]]
