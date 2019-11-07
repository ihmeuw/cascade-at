from cascade_at.core.db import db_queries
from cascade_at.core.db import db_tools

from cascade_at.core.log import get_loggers

LOG = get_loggers(__name__)


SEX_MAP = {
    1: 'Male',
    2: 'Female',
    3: 'Both'
}


class CascadeConstants:
    NON_SEX_SPECIFIC_ID = [3]
    NON_AGE_SPECIFIC_ID = [22, 27]
    GLOBAL_LOCATION_ID = 1
    ESTIMATION_LOCATION_HIERARCHY_ID = 35
    AGE_GROUP_SET_ID = 12


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


def get_age_group_metadata():
    """
    Gets age group metadata.
    """
    df = db_queries.get_age_metadata(age_group_set_id=CascadeConstants.AGE_GROUP_SET_ID)
    df.rename(columns={'age_group_years_start': 'age_lower', 'age_group_years_end': 'age_upper'}, inplace=True)
    return df[['age_group_id', 'age_lower', 'age_upper']]


def get_age_id_to_range():
    """
    Gets the age group ID to range dictionary.
    :return: dict[int, tuple(float, float)]
    """
    df = get_age_group_metadata()
    return dict([(t.age_group_id, (t.age_lower, t.age_upper)) for t in df.itertuples()])


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

