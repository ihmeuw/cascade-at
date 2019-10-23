import db_queries
import db_tools


def get_sex_ids():
    return db_queries.get_ids(table='sex')


def get_measure_ids(conn_def):
    query = "SELECT measure_id, measure, measure_name FROM shared.measure"
    df = db_tools.ezfuncs.query(query, conn_def=conn_def)
    return df


