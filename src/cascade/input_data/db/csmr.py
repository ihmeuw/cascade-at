"""Upload cmsr data to t3 table so EpiViz can access it for plotting."""

import logging

from db_queries import get_outputs

from cascade.core.db import cursor


CODELOG = logging.getLogger(__name__)


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

    model_version_id = execution_context.parameters.model_version_id
    add_csmr_cause = execution_context.parameters.add_csmr_cause
    parent_loc = add_csmr_cause = execution_context.parameters.drill

    csmr_age_group_ids = list(range(2, 22))
    csmr_measure_id = 1
    csmr_metric_id_rate = 3
    csmr_gbd_round_id = 5
    csmr_sex_id = [1, 2]
    csmr_year_ids = [1985, 1990, 1995, 2000, 2005, 2010, 2016]

    csmr = get_outputs(
        topic="cause",
        cause_id=add_csmr_cause,
        location_id=parent_loc,
        metric_id=csmr_metric_id_rate,
        year_id=csmr_year_ids,
        age_group_id=csmr_age_group_ids,
        measure_id=csmr_measure_id,
        sex_id=csmr_sex_id,
        gbd_round_id=csmr_gbd_round_id
        )[["year_id", "location_id", "sex_id", "age_group_id",
           "val", "lower", "upper"]]

    csmr["model_version_id"] = model_version_id

    csmr.rename(columns={"val": "mean"}, inplace=True)

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
            {model_version_id}, {", ".join(["%s"]*32)}
        )
    """

    cursor.executemany(insert_query, csmr_data.values())

    CODELOG.debug(f"uploaded {len(csmr_data)} lines of csmr data")


def load_csmr_to_t3(execution_context):
    """
    Upload to t3_model_version_csmr if the user requested that and if
    it's not already there.
    """

    model_version_id = execution_context.parameters.model_version_id
    add_csmr_cause = execution_context.parameters.add_csmr_cause

    if add_csmr_cause:
        CODELOG.info(f"""User requested csmr cause data {add_csmr_cause}
                     be added for model_version_id {model_version_id}""")

        database = execution_context.parameters.database

        if _csmr_in_t3(execution_context):
            CODELOG.info(
                f"""csmr data for model_version_id {model_version_id}
                on '{database}' already exists, doing nothing."""
            )
            return False
        else:
            CODELOG.info(f"""Uploading csmr data for model_version_id
                         {model_version_id} on '{database}'""")

            csmr_data = _get_csmr_data(execution_context)

            # and insert data into the t3_model_version_csmr table
            _upload_csmr_data_to_tier_3(cursor, model_version_id, csmr_data)

            return True

    else:
        CODELOG.info(f"""User did not request csmr data to be added for
                     model_version_id {model_version_id}""")

        return False
