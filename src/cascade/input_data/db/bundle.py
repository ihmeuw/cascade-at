import logging

from cascade.core.db import cursor

CODELOG = logging.getLogger(__name__)


def _bundle_is_frozen(execution_context):
    """Checks if data for the current model_version_id already exists in tier 3.
    """

    model_version_id = execution_context.parameters.model_version_id
    query = f"""
    select exists(
             select * from epi.t3_model_version_dismod
             where model_version_id = {model_version_id}
    )
    """
    with cursor(execution_context) as c:
        c.execute(query)
        exists = c.fetchone()[0]

    return exists == 1


def _get_bundle_id(execution_context):
    """Gets the bundle id associated with the current model_version_id.
    """
    model_version_id = execution_context.parameters.model_version_id

    query = f"""
    SELECT bundle_id
    FROM epi.model_version
    WHERE model_version_id = %(model_version_id)s
    """

    with cursor(execution_context) as c:
        CODELOG.debug(f"Looking up bundle_id for model_version_id {model_version_id}")
        c.execute(query, args={"model_version_id": model_version_id})
        bundle_ids = list(c)

        if not bundle_ids:
            raise ValueError(
                f"No bundle_id associated with model_version_id {model_version_id}"
            )

        if len(bundle_ids) > 1:
            raise ValueError(
                f"Multiple bundle_ids associated with model_version_id {model_version_id}"
            )

        return bundle_ids[0][0]


def _get_tier_2_bundle_data(execution_context):
    """Downloads the tier 2 data for the bundle associated with the current model_version_id.
    """
    database = execution_context.parameters.bundle_database

    bundle_id = _get_bundle_id(execution_context)

    query = f"""
    SELECT
         bundle_id ,
         seq ,
         request_id ,
         input_type_id ,
         nid ,
         underlying_nid ,
         location_id,
         sex_id ,
         year_start ,
         year_end ,
         age_start,
         age_end ,
         measure_id ,
         source_type_id ,
         sampling_type_id,
         representative_id ,
         urbanicity_type_id ,
         recall_type_id ,
         recall_type_value ,
         unit_type_id ,
         unit_value_as_published ,
         uncertainty_type_id ,
         uncertainty_type_value ,
         mean ,
         lower ,
         upper ,
         standard_error ,
         effective_sample_size ,
         sample_size ,
         cases ,
         design_effect,
         outlier_type_id
        FROM
         epi.bundle_dismod
        WHERE
         bundle_id = %(bundle_id)s and
         input_type_id NOT IN(5,6) and
         outlier_type_id IN (0,1)
         """
    with cursor(database=database) as c:
        c.execute(query, args={"bundle_id": bundle_id})
        bundle_rows = list(c)
        CODELOG.debug(
            f"Downloaded {len(bundle_rows)} lines of bundle_id {bundle_id} from '{database}'"
        )

    return bundle_rows


def _upload_bundle_data_to_tier_3(execution_context, bundle_data):
    """Updloads bundle data to tier 3 attached to the current model_version_id.
    """

    model_version_id = execution_context.parameters.model_version_id

    insert_query = f"""
    INSERT INTO epi.t3_model_version_dismod (
        model_version_id,
        bundle_id ,
        seq ,
        request_id ,
        input_type_id ,
        nid ,
        underlying_nid ,
        location_id,
        sex_id ,
        year_start ,
        year_end ,
        age_start,
        age_end ,
        measure_id ,
        source_type_id ,
        sampling_type_id,
        representative_id ,
        urbanicity_type_id ,
        recall_type_id ,
        recall_type_value ,
        unit_type_id ,
        unit_value_as_published ,
        uncertainty_type_id ,
        uncertainty_type_value ,
        mean ,
        lower ,
        upper ,
        standard_error ,
        effective_sample_size ,
        sample_size ,
        cases ,
        design_effect,
        outlier_type_id
    ) VALUES (
        {model_version_id}, {", ".join(["%s"]*32)}
    )
    """

    with cursor(execution_context) as c:
        c.executemany(insert_query, bundle_data)

    CODELOG.debug(
        f"uploaded {len(bundle_data)} lines of bundle data to '{execution_context.parameters.database}'"
    )


def freeze_bundle(execution_context) -> bool:
    """Freezes the bundle data attached to the current model_version_id if necessary.

    The freezing process works as follows:
    If there is any data in the tier 3 table for the model_version_id then it is assumed to already be frozen.
    Otherwise download all data for the bundle from the tier 2 table in the main epi database and reupload it
    to the dismodAT database (ie. promote it).

    Returns:
      True if any data was promoted and False otherwise
    """

    model_version_id = execution_context.parameters.model_version_id

    database = execution_context.parameters.database
    if _bundle_is_frozen(execution_context):
        CODELOG.info(
            f"Bundle data for model_version_id {model_version_id} on '{database}' already frozen, doing nothing."
        )
        return False
    else:
        CODELOG.info(
            f"Freezing bundle data for model_version_id {model_version_id} on '{database}'"
        )
        bundle_data = _get_tier_2_bundle_data(execution_context)
        _upload_bundle_data_to_tier_3(execution_context, bundle_data)
        return True
