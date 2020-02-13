import numpy as np
import pandas as pd

from cascade_at.core.log import get_loggers
from cascade_at.dismod.api.fill_extract_helpers import utils
from cascade_at.dismod.constants import DensityEnum, IntegrandEnum, \
    INTEGRAND_TO_WEIGHT

LOG = get_loggers(__name__)

DEFAULT_DENSITY = ["uniform", 0, -np.inf, np.inf]


def prep_data_avgint(df, node_df, covariate_df):
    """
    Preps both the data table and the avgint table.
    Putting it in the same function because it does the same stuff,
    but they need to be called separately because dismod requires
    different columns.
    """
    data = df.copy()
    data = utils.map_locations_to_nodes(df=data, node_df=node_df)
    data = utils.map_covariate_names(df=data, covariate_df=covariate_df)

    data.reset_index(inplace=True, drop=True)
    return data


def construct_data_table(df, node_df, covariate_df):
    """
    Constructs the data table from input df.

    Parameters:
        df: (pd.DataFrame) data frame of inputs that have been prepped for dismod
        node_df: (pd.DataFrame) the dismod node table
        covariate_df: (pd.DataFrame) the dismod covariate table
    """
    LOG.info("Constructing data table.")

    data = df.copy()
    data = prep_data_avgint(
        df=data,
        node_df=node_df,
        covariate_df=covariate_df
    )
    data["data_name"] = data.index.astype(str)

    data["density_id"] = data["density"].apply(lambda x: DensityEnum[x].value)
    data["integrand_id"] = data["measure"].apply(lambda x: IntegrandEnum[x].value)
    data["weight_id"] = data["measure"].apply(lambda x: INTEGRAND_TO_WEIGHT[x].value)
    data["subgroup_id"] = 0

    columns = data.columns
    data = data[[
        'data_name', 'integrand_id', 'density_id', 'node_id', 'weight_id', 'subgroup_id',
        'hold_out', 'meas_value', 'meas_std', 'eta', 'nu',
        'age_lower', 'age_upper', 'time_lower', 'time_upper'
    ] + [x for x in columns if x.startswith('x_')]]
    return data


def construct_gbd_avgint_table(df, node_df, covariate_df, integrand_df):
    """
    Constructs the avgint table using the output df
    from the inputs.to_avgint() method.
    """
    LOG.info("Constructing the avgint table.")
    avgint = df.copy()
    avgint = prep_data_avgint(
        df=avgint,
        node_df=node_df,
        covariate_df=covariate_df
    )
    avgint_df = pd.DataFrame()
    for i in integrand_df.integrand_name.unique():
        if i == 'mtstandard' or i == 'relrisk':
            continue
        df = avgint.copy()
        df['measure'] = i
        avgint_df = avgint_df.append(df)

    avgint_df = avgint_df.reset_index(drop=True)

    avgint_df["integrand_id"] = avgint_df["measure"].apply(lambda x: IntegrandEnum[x].value)
    avgint_df["weight_id"] = avgint_df["measure"].apply(lambda x: INTEGRAND_TO_WEIGHT[x].value)
    avgint_df["subgroup_id"] = 0

    avgint_df = avgint_df[[
        'integrand_id', 'node_id', 'weight_id', 'subgroup_id', 'c_location_id',
        'age_group_id', 'year_id', 'sex_id',
        'age_lower', 'age_upper', 'time_lower', 'time_upper'
    ] + [x for x in avgint_df.columns if x.startswith('x_')]]
    
    gbd_id_cols = ['sex_id', 'age_group_id', 'year_id']
    avgint_df.rename(columns={x: 'c_' + x for x in gbd_id_cols}, inplace=True)

    return avgint_df
