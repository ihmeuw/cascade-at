import db_queries
import db_tools


class CascadeConstants:
    NON_SEX_SPECIFIC_ID = [3]
    NON_AGE_SPECIFIC_ID = [22, 27]
    GLOBAL_LOCATION_ID = 1
    ESTIMATION_LOCATION_HIERARCHY_ID = 35


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
