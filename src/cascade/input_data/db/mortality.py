import pandas as pd

from cascade.core.db import connection

from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)


def get_excess_mortality_data(execution_context):
    """
    The year range is from start of year to end of year, so these
    measurements have a year duration. To make point data, take
    the midpoint of the year.
    """
    model_version_id = execution_context.parameters.model_version_id

    query = """
     SELECT
                model_version_dismod_id as data_id,
                location_id,
                year_start,
                age_start,
                age_end,
                sex_id,
                mean,
                lower,
                upper,
                standard_error
            FROM
                epi.t3_model_version_emr t3_emr
            WHERE model_version_id = %(model_version_id)s
            AND t3_emr.outlier_type_id = 0
    """

    with connection(execution_context) as c:
        data = pd.read_sql(query, c, params={"model_version_id": model_version_id})

    return data


def get_frozen_cause_specific_mortality_data(execution_context):
    """
    The year range is from start of year to end of year, so these
    measurements have a year duration. To make point data, take
    the midpoint of the year.
    """
    model_version_id = execution_context.parameters.model_version_id

    query = """
            SELECT
                location_id,
                year_id
                age_group_id,
                sex_id,
                mean,
                lower,
                upper
            FROM
                epi.t3_model_version_csmr
            WHERE model_version_id = %(model_version_id)s
    """

    with connection(execution_context) as c:
        data = pd.read_sql(query, c, params={"model_version_id": model_version_id})

    return data


def get_frozen_age_standardized_death_rate_data(execution_context):
    """
    The year range is from start of year to end of year, so these
    measurements have a year duration. To make point data, take
    the midpoint of the year.
    """
    model_version_id = execution_context.parameters.model_version_id

    query = """
            SELECT
                location_id,
                year_id
                age_group_id,
                sex_id,
                mean,
                lower,
                upper
            FROM
                epi.t3_model_version_asdr
            WHERE model_version_id = %(model_version_id)s
    """

    with connection(execution_context) as c:
        data = pd.read_sql(query, c, params={"model_version_id": model_version_id})

    return data


def normalize_mortality_data(df):
    df = df.rename(columns={
        "mean": "meas_value",
        "lower": "meas_lower",
        "upper": "meas_upper",
        "year_id": "time_lower",
    })
    df["time_upper"] = df.time_lower + 1
    return df
