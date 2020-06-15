import pandas as pd
import numpy as np
from typing import Optional
from intervaltree import IntervalTree

from cascade_at.core.db import db_queries
from cascade_at.core.db import db_tools
from cascade_at.core import CascadeATError
from cascade_at.core.log import get_loggers

LOG = get_loggers(__name__)


class IhmeIDError(CascadeATError):
    pass


DEMOGRAPHIC_ID_COLS = ['location_id', 'sex_id', 'age_group_id', 'year_id']


SEX_ID_TO_NAME = {
    1: 'Male',
    2: 'Female',
    3: 'Both'
}

SEX_NAME_TO_ID = {
    v: k for k, v in SEX_ID_TO_NAME.items()
}


class CascadeConstants:
    NON_SEX_SPECIFIC_ID = 3
    NON_AGE_SPECIFIC_ID = [22, 27]
    GLOBAL_LOCATION_ID = 1
    ESTIMATION_LOCATION_HIERARCHY_ID = 9
    AGE_GROUP_SET_ID = 12
    PRECISION_FOR_REFERENCE_VALUES = 1e-10
    WITH_HIV = 1
    WITH_SHOCK = 1


class StudyCovConstants:
    SEX_COV_VALUE_MAP = {
        'Male': 0.5,
        'Both': 0,
        'Female': -0.5
    }
    MAX_DIFFERENCE_SEX_COV = 0.5 + CascadeConstants.PRECISION_FOR_REFERENCE_VALUES
    ONE_COV_VALUE = 1.0
    MAX_DIFFERENCE_ONE_COV = None


def get_sex_ids():
    """
    Gets the sex IDs from db_queries.

    :return: (df)
    """
    return db_queries.get_ids(table='sex')


def get_measure_ids(conn_def):
    """
    Gets measure IDs because the output from get_ids(table='measure') does not
    include measure it only includes measure_name.

    :param conn_def: (str)
    :return: (df)
    """
    query = "SELECT measure_id, measure, measure_name FROM shared.measure"
    df = db_tools.ezfuncs.query(query, conn_def=conn_def)
    return df


def get_location_set_version_id(gbd_round_id):
    """
    Gets a location_set_version_id for the estimation hierarchy
    for GBD round passed.

    :param gbd_round_id: (int)
    :return: (int)
    """
    location_set_version_id = db_queries.get_location_metadata(
        location_set_id=CascadeConstants.ESTIMATION_LOCATION_HIERARCHY_ID,
        gbd_round_id=gbd_round_id
    )['location_set_version_id'].unique()[0]
    return location_set_version_id


def get_age_group_metadata(gbd_round_id: int) -> pd.DataFrame:
    """
    Gets age group metadata.
    """
    df = db_queries.get_age_metadata(age_group_set_id=CascadeConstants.AGE_GROUP_SET_ID,
                                     gbd_round_id=gbd_round_id)
    df.rename(columns={'age_group_years_start': 'age_lower', 'age_group_years_end': 'age_upper'}, inplace=True)
    df.age_lower = df.age_lower.astype(float)
    df.age_upper = df.age_upper.astype(float)
    df.age_group_id = df.age_group_id.astype(int)
    return df[['age_group_id', 'age_lower', 'age_upper']]


def make_age_intervals(df: Optional[pd.DataFrame] = None,
                       gbd_round_id: Optional[int] = None) -> IntervalTree:
    """
    Makes an interval tree out of age lower and upper for age group IDs.
    The interval tree can be made from an existing data frame with those columns
    or it can be made from getting the full set of age groups from the IHME databases.

    Parameters
    ----------
    df
        Data frame from which to construct the interval tree. Must have the
        columns ['age_group_id', 'age_lower', 'age_upper']. If passed, ignores gbd_round_id.
    gbd_round_id
        The gbd round ID from which to pull the age group metadata which is used
        to construct the interval tree. Ignored if df is specified instead.
    """
    if df is None and gbd_round_id is None:
        raise IhmeIDError("Need to pass either a data frame with columns"
                          "['age_group_id', 'age_lower', 'age_upper' or a valid"
                          "gbd_round_id to get the full set of age groups.")
    if df is None:
        df = get_age_group_metadata(gbd_round_id=gbd_round_id)
    else:
        for col in ['age_group_id', 'age_lower', 'age_upper']:
            if col not in df.columns:
                raise IhmeIDError(f"The data frame columns {df.columns} do not contain"
                                  f"the required column {col}.")
    age_intervals = IntervalTree.from_tuples(
        df[['age_lower', 'age_upper', 'age_group_id']].values
    )
    return age_intervals


def make_time_intervals(df: Optional[pd.DataFrame] = None) -> IntervalTree:
    """
    Makes an interval tree out of year_id.
    The interval tree can be made from an existing data frame with that column
    or it can be made from the knowledge that the year ID == year.

    Parameters
    ----------
    df
        Optional data frame from which to construct the interval tree.
        Must have 'year_id' as a column.
    """
    if df is None:
        df = pd.DataFrame({
            'year_id': np.arange(1950, 2050)
        })
    else:
        if 'year_id' not in df.columns:
            raise IhmeIDError(f"The data frame columns {df.columns} do not contain the"
                              "one required column year_id.")
    time_intervals = IntervalTree.from_tuples([
        (t, t+1, t) for t in df.year_id.unique()
    ])
    return time_intervals


def get_study_level_covariate_ids():
    """
    Grabs the covariate names for study-level
    that will be used in DisMod-AT
    :return:
    """
    return {0: "s_sex", 1604: "s_one"}


def get_country_level_covariate_ids(country_covariate_id):
    """
    Grabs country-level covariate names associated with
    the IDs passed.
    :param country_covariate_id: (list of int)
    :return: (dict)
    """
    df = db_queries.get_ids(table='covariate')
    df = df.loc[df.covariate_id.isin(country_covariate_id)].copy()
    cov_dict = df[['covariate_id', 'covariate_name_short']].set_index('covariate_id').to_dict('index')
    return {k: f"c_{v['covariate_name_short']}" for k, v in cov_dict.items()}
