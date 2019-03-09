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
                upper
            FROM
                epi.t3_model_version_emr t3_emr
            WHERE model_version_id = %(model_version_id)s
            AND t3_emr.outlier_type_id = 0
    """

    with connection(execution_context) as c:
        data = pd.read_sql(query, c, params={"model_version_id": model_version_id})

    return data


def get_frozen_cause_specific_mortality_data(execution_context, model_version_id):
    """
    The year range is from start of year to end of year, so these
    measurements have a year duration. To make point data, take
    the midpoint of the year.
    """

    query = """
            SELECT
                location_id,
                year_id,
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
    CODELOG.debug(f"csmr {data.head(5)}")
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
                year_id,
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
    rename_columns = {
        "mean": "meas_value",
        "lower": "meas_lower",
        "upper": "meas_upper",
        "year_id": "time_lower",
    }
    if not set(rename_columns.keys()).issubset(set(df.columns)):
        missing = ", ".join(str(c) for c in sorted(set(rename_columns.keys()) - set(df.columns)))
        raise RuntimeError(f"Mortality data missing columns. Missing {missing}.")
    df = df.rename(columns=rename_columns)
    df["time_upper"] = df.time_lower + 1
    return df
