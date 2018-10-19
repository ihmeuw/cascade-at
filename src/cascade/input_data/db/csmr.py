"""Upload csmr data to t3 table so EpiViz can access it for plotting."""

import pandas as pd

from cascade.core.db import cursor, db_queries
from cascade.input_data.db import GBD_ROUND_ID, METRIC_IDS, MEASURE_IDS


from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)


def _csmr_in_t3(execution_context):
    """Checks if data for the current model_version_id already exists in tier 3.
    """

    model_version_id = execution_context.parameters.model_version_id

    query = """
    SELECT exists(
             SELECT * FROM epi.t3_model_version_csmr
             WHERE model_version_id = %(model_version_id)s
    )
    """
    with cursor(execution_context) as c:
        c.execute(query, args={"model_version_id": model_version_id})
        exists = c.fetchone()[0]

    return exists == 1


def _get_csmr_data(execution_context):

    cause_id = execution_context.parameters.add_csmr_cause
    parent_loc = execution_context.parameters.location_id

    keep_cols = ["year_id", "location_id", "sex_id", "age_group_id", "val", "lower", "upper"]

    csmr = db_queries.get_outputs(
        topic="cause",
        cause_id=cause_id,
        location_id=parent_loc,
        metric_id=METRIC_IDS["per_capita_rate"],
        year_id="all",
        age_group_id="most_detailed",
        measure_id=MEASURE_IDS["deaths"],
        sex_id="all",
        gbd_round_id=GBD_ROUND_ID,
        version="latest",
    )[keep_cols]

    csmr = csmr[csmr["val"].notnull()]

    csmr.rename(columns={"val": "mean"})

    return csmr


def _upload_csmr_data_to_tier_3(cursor, model_version_id, csmr_data):
    """Uploads csmr data to tier 3 attached to the current model_version_id.
    """

    insert_query = f"""
        INSERT INTO epi.t3_model_version_csmr (
            model_version_id,
            year_id,
            location_id,
            sex_id,
            age_group_id,
            mean,
            lower,
            upper
        ) VALUES (
            {model_version_id}, {", ".join(["%s"]*7)}
        )
    """

    csmr_data = csmr_data.where(pd.notnull(csmr_data), None)
    cursor.executemany(insert_query, csmr_data.values.tolist())

    CODELOG.debug(f"uploaded {len(csmr_data)} lines of csmr data")


def load_csmr_to_t3(execution_context) -> bool:
    """
    Upload to t3_model_version_csmr if it's not already there.
    """

    model_version_id = execution_context.parameters.model_version_id

    database = execution_context.parameters.database

    if _csmr_in_t3(execution_context):
        CODELOG.info(
            f"""csmr data for model_version_id {model_version_id}
            on '{database}' already exists, doing nothing."""
        )
        return False
    else:
        CODELOG.info(
            f"""Uploading csmr data for model_version_id
            {model_version_id} on '{database}'"""
        )

        csmr_data = _get_csmr_data(execution_context)

        with cursor(execution_context) as c:
            _upload_csmr_data_to_tier_3(c, model_version_id, csmr_data)

        return True
