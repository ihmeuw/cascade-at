"""Upload asdr data to t3 table so EpiViz can access it for plotting."""

import pandas as pd

from cascade.core.db import cursor, db_queries
from cascade.input_data.db import AGE_GROUP_SET_ID, GBD_ROUND_ID


from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)


def _asdr_in_t3(execution_context):
    """Checks if data for the current model_version_id already exists in tier 3.
    """

    model_version_id = execution_context.parameters.model_version_id

    query = """
    SELECT exists(
             SELECT * FROM epi.t3_model_version_asdr
             WHERE model_version_id = %(model_version_id)s
    )
    """
    with cursor(execution_context) as c:
        c.execute(query, args={"model_version_id": model_version_id})
        exists = c.fetchone()[0]

    return exists == 1


def _get_asdr_data(execution_context):

    parent_loc = execution_context.parameters.location_id

    demo_dict = db_queries.get_demographics(gbd_team="epi", gbd_round_id=GBD_ROUND_ID)
    age_group_ids = demo_dict["age_group_id"]
    sex_ids = demo_dict["sex_id"]

    asdr = db_queries.get_envelope(
        location_id=parent_loc,
        year_id=-1,
        gbd_round_id=GBD_ROUND_ID,
        age_group_id=age_group_ids,
        sex_id=sex_ids,
        with_hiv=True,
        rates=True,
    ).drop(columns=["run_id"])

    asdr = asdr[asdr["mean"].notnull()]

    age_group_data = db_queries.get_age_metadata(age_group_set_id=AGE_GROUP_SET_ID, gbd_round_id=GBD_ROUND_ID)[
        ["age_group_id", "age_group_years_start", "age_group_years_end"]
    ]

    age_group_data.columns = ["age_group_id", "age_lower", "age_upper"]

    asdr = asdr.merge(age_group_data, how="left", on="age_group_id")

    ordered_cols = [
        "year_id",
        "location_id",
        "sex_id",
        "age_group_id",
        "age_upper",
        "age_lower",
        "mean",
        "upper",
        "lower",
    ]

    asdr = asdr[ordered_cols]

    return asdr


def _upload_asdr_data_to_tier_3(cursor, model_version_id, asdr_data):
    """Uploads asdr data to tier 3 attached to the current model_version_id.
    """

    insert_query = f"""
        INSERT INTO epi.t3_model_version_asdr (
            model_version_id,
            year_id,
            location_id,
            sex_id,
            age_group_id,
            age_upper,
            age_lower,
            mean,
            upper,
            lower
        ) VALUES (
            {model_version_id}, {", ".join(["%s"]*9)}
        )
    """

    asdr_data = asdr_data.where(pd.notnull(asdr_data), None)
    cursor.executemany(insert_query, asdr_data.values.tolist())

    CODELOG.debug(f"uploaded {len(asdr_data)} lines of asdr data")


def load_asdr_to_t3(execution_context) -> bool:
    """
    Upload to t3_model_version_asdr if it's not already there.
    """

    model_version_id = execution_context.parameters.model_version_id

    database = execution_context.parameters.database

    if _asdr_in_t3(execution_context):
        CODELOG.info(
            f"""asdr data for model_version_id {model_version_id}
            on '{database}' already exists, doing nothing."""
        )
        return False
    else:
        CODELOG.info(
            f"""Uploading asdr data for model_version_id
            {model_version_id} on '{database}'"""
        )

        asdr_data = _get_asdr_data(execution_context)

        with cursor(execution_context) as c:
            _upload_asdr_data_to_tier_3(c, model_version_id, asdr_data)

        return True
