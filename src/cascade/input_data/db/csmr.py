"""Upload CSMR data to t3 table so EpiViz can access it for plotting."""

import pandas as pd

from cascade.core import getLoggers
from cascade.core.db import cursor, db_queries
from cascade.input_data.db import METRIC_IDS, MEASURE_IDS, GBDDataError

CODELOG, MATHLOG = getLoggers(__name__)


def _csmr_in_t3(execution_context, model_version_id):
    """Checks if data for the current model_version_id already exists in tier 3.

    Returns:
        list of ``location_id`` that have data in tier 3 for
        this ``model_version_id``.
    """
    query = """
    SELECT DISTINCT location_id
    FROM epi.t3_model_version_csmr
    WHERE model_version_id = %(model_version_id)s
    """
    with cursor(execution_context) as c:
        c.execute(query, args={"model_version_id": model_version_id})
        location_rows = c.fetchall()

    return [row[0] for row in location_rows]


def _gbd_process_version_id_from_cod_version(cod_version):
    """Central comp uses process_version_id to track data versions independently
    from the versioning systems of the tools that produced the data. Thus we
    need to map from CODcorrect's versions to a process_version_id to do
    version constrained lookups using central comp's tools.
    """

    query = """
    SELECT gbd.gbd_process_version_metadata.gbd_process_version_id from gbd.gbd_process_version_metadata
        JOIN  gbd.gbd_process_version ON gbd.gbd_process_version_metadata.gbd_process_version_id =
                                         gbd.gbd_process_version.gbd_process_version_id
        WHERE gbd_process_id = 3                     -- codcorrect process
              and metadata_type_id = 1               -- codcorrect version
              and val = %(cod_version)s
              and gbd_process_version_status_id = 1  -- marked best
    """

    with cursor(database="gbd") as c:
        c.execute(query, args={"cod_version": cod_version})
        result = c.fetchone()

    if result is None:
        raise GBDDataError(f"No best gbd_process_version_id for COD version {cod_version}")

    return result[0]


def get_csmr_data(
        execution_context, location_set_id, cause_id, cod_version, gbd_round_id, decomp_step
):
    """Retrieve CSMR for all locations in a location set."""
    return get_csmr_base(
        execution_context,
        cause_id,
        cod_version,
        gbd_round_id,
        decomp_step,
        dict(location_set_id=location_set_id, location_id="all"),
    )


def get_csmr_location(
        execution_context, location_id, cause_id, cod_version, gbd_round_id, decomp_step
):
    """Retrieve CSMR for one location or for a list of locations."""
    return get_csmr_base(
        execution_context,
        cause_id,
        cod_version,
        gbd_round_id,
        decomp_step,
        dict(location_id=location_id),
    )


def get_csmr_base(
        execution_context, cause_id, cod_version, gbd_round_id, decomp_step, extra_args
):
    """Gets CSMR with minor formatting, limiting it to desired columns."""
    keep_cols = ["year_id", "location_id", "sex_id", "age_group_id", "val", "lower", "upper"]

    process_version_id = _gbd_process_version_id_from_cod_version(cod_version)

    csmr = db_queries.get_outputs(
        topic="cause",
        cause_id=cause_id,
        metric_id=METRIC_IDS["per_capita_rate"],
        year_id="all",
        age_group_id="most_detailed",
        measure_id=MEASURE_IDS["deaths"],
        sex_id="all",
        gbd_round_id=gbd_round_id,
        decomp_step=decomp_step,
        process_version_id=process_version_id,
        **extra_args,
    )[keep_cols]

    csmr = csmr[csmr["val"].notnull()]
    csmr = csmr.rename(columns={"val": "mean"})

    return csmr


def _upload_csmr_data_to_tier_3(cursor, model_version_id, csmr_data):
    """Uploads CSMR data to tier 3 attached to the current model_version_id.
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

    csmr_data = csmr_data[[
        "year_id",
        "location_id",
        "sex_id",
        "age_group_id",
        "mean",
        "lower",
        "upper",
    ]]
    cursor.executemany(insert_query, csmr_data.values.tolist())

    CODELOG.debug(f"uploaded {len(csmr_data)} lines of csmr data")


def load_csmr_to_t3(execution_context, data_access, location_and_children):
    """
    Upload to t3_model_version_csmr if it's not already there.
    """

    cause_id = data_access.add_csmr_cause
    gbd_round_id = data_access.gbd_round_id
    model_version_id = data_access.model_version_id
    cod_version = data_access.cod_version

    database = execution_context.parameters.database
    locations_with_data_in_t3 = _csmr_in_t3(execution_context, model_version_id)
    csmr_not_in_t3 = set(location_and_children) - set(locations_with_data_in_t3)
    if csmr_not_in_t3:
        CODELOG.info(f"Uploading CSMR data for model_version_id {model_version_id} on '{database}'")

        csmr_data = get_csmr_data(execution_context, list(csmr_not_in_t3), cause_id, cod_version, gbd_round_id)

        with cursor(execution_context) as c:
            _upload_csmr_data_to_tier_3(c, model_version_id, csmr_data)

        return True
    else:
        CODELOG.info(
            f"CSMR data for model_version_id {model_version_id} on '{database}' already exists, doing nothing."
        )
        return False
