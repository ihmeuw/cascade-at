import pandas as pd

from cascade.core.db import connection

from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)


def get_excess_mortality_data(execution_context):
    model_version_id = execution_context.parameters.model_version_id

    query = """
     SELECT
                model_version_dismod_id as data_id,
                location_id,
                year_start,
                year_end,
                age_start as age_lower,
                age_end as age_upper,
                sex_id as sex,
                mean as meas_value,
                lower as meas_lower,
                upper as meas_upper,
                standard_error
            FROM
                epi.t3_model_version_emr t3_emr
            WHERE model_version_id = %(model_version_id)s
            AND t3_emr.outlier_type_id = 0
    """

    with connection(execution_context) as c:
        data = pd.read_sql(query, c, params={"model_version_id": model_version_id})

    return data


def get_cause_specific_mortality_data(execution_context):
    model_version_id = execution_context.parameters.model_version_id

    query = """
            SELECT
                location_id,
                year_id as year_start,
                year_id as year_end,
                age_group_id,
                sex_id as sex,
                mean as meas_value,
                lower as meas_lower,
                upper as meas_upper
            FROM
                epi.t3_model_version_csmr
            WHERE model_version_id = %(model_version_id)s
    """

    with connection(execution_context) as c:
        data = pd.read_sql(query, c, params={"model_version_id": model_version_id})

    return data


def get_age_standardized_death_rate_data(execution_context):
    model_version_id = execution_context.parameters.model_version_id

    query = """
            SELECT
                location_id,
                year_id as year_start,
                year_id as year_end,
                age_group_id,
                sex_id as sex,
                mean as meas_value,
                lower as meas_lower,
                upper as meas_upper
            FROM
                epi.t3_model_version_asdr
            WHERE model_version_id = %(model_version_id)s
    """

    with connection(execution_context) as c:
        data = pd.read_sql(query, c, params={"model_version_id": model_version_id})

    return data
