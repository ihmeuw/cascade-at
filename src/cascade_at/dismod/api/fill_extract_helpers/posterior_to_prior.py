import pandas as pd
from typing import Dict, List
import numpy as np
from scipy import stats

from cascade_at.dismod.api.fill_extract_helpers.utils import vec_to_midpoint
from cascade_at.model.utilities.grid_helpers import expand_grid
from cascade_at.dismod.constants import RateToIntegrand, IntegrandEnum, INTEGRAND_TO_WEIGHT
from cascade_at.inputs.utilities.gbd_ids import format_age_time
from cascade_at.dismod.integrand_mappings import RATE_TO_INTEGRAND, integrand_to_gbd_measures
from cascade_at.model.smooth_grid import SmoothGrid


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


def format_rate_grid_for_ihme(rates: Dict[str, SmoothGrid], gbd_round_id: int,
                              location_id: int, sex_id: int) -> pd.DataFrame:
    """
    Formats a grid of mean, upper, and lower for a prior rate
    for the IHME database. **Only does this for Gaussian priors.**

    Parameters
    ----------
    rates
         A dictionary of SmoothGrids, keyed by primary rates like "iota"
    gbd_round_id
        the GBD round
    location_id
        the location ID to append to this data frame
    sex_id
        the sex ID to append to this data frame

    Returns
    -------
    A data frame formatted for the IHME databases
    """
    dfs = []
    for rate, smooth_grid in rates.items():
        df = smooth_grid.value.grid.copy()
        if df.empty:
            continue

        df['age_lower'] = df['age']
        df['age_upper'] = df['age']
        df['time_lower'] = df['time']
        df['time_upper'] = df['time']

        df = format_age_time(df=df, gbd_round_id=gbd_round_id)

        group_cols = ['age', 'time']
        # TODO: Once we can upgrade to pandas 1.1.0, then we can use the groupby(..., dropna=False)
        #  feature, which we need because eta and nu can be null and that's ok, but pandas drops them.
        #  In the meantime, we will group on age and time which means we're looping over each row,
        #  which in some cases will be x 30 more computation than necessary.
        #  Once we upgrade, use the group_cols below and it will skip duplicate computation.
        #  group_cols = ['mean', 'std', 'lower', 'upper', 'density', 'eta', 'nu']

        for name, group in df.groupby(group_cols):
            at_row = smooth_grid.value[group.iloc[0]['age'], group.iloc[0]['time']].quantiles([0.025, 0.975])
            df.loc[group.index, 'lower'] = at_row[0]
            df.loc[group.index, 'upper'] = at_row[1]

        df['integrand'] = RATE_TO_INTEGRAND[rate].name
        df = integrand_to_gbd_measures(df=df, integrand_col='integrand')

        df['location_id'] = location_id
        df['sex_id'] = sex_id

        dfs.append(df)
    return pd.concat(dfs, axis=0, sort=False).reset_index()[[
        'location_id', 'year_id', 'age_group_id', 'sex_id', 'measure_id',
        'mean', 'upper', 'lower'
    ]]
