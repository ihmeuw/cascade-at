import logging

import pandas as pd

from cascade.core.db import cursor, connection

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


def _get_tier_2_bundle_data(execution_context, bundle_id):
    """Downloads the tier 2 data for the bundle associated with the current model_version_id.
    """
    database = execution_context.parameters.bundle_database

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


def _get_tier_2_study_covariates(execution_context, bundle_id):
    """Downloads the tier 2 study covariate mappings for the bundle associated with the current model_version_id.
    """
    database = execution_context.parameters.bundle_database

    query = f"""
    SELECT
        bundle_id,
        seq,
        study_covariate_id
    FROM
        epi.bundle_dismod_study_covariate
    WHERE
        bundle_id = %(bundle_id)s
         """
    with cursor(database=database) as c:
        c.execute(query, args={"bundle_id": bundle_id})
        covariate_rows = list(c)
        CODELOG.debug(
            f"Downloaded {len(covariate_rows)} lines of study covariates for bundle_id {bundle_id} from '{database}'"
        )

    return covariate_rows


def _upload_bundle_data_to_tier_3(cursor, model_version_id, bundle_data):
    """Updloads bundle data to tier 3 attached to the current model_version_id.
    """

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

    cursor.executemany(insert_query, bundle_data)

    CODELOG.debug(f"uploaded {len(bundle_data)} lines of bundle data")


def _upload_study_covariates_to_tier_3(cursor, model_version_id, covariate_data):
    """Updloads study covariate mappings to tier 3 attached to the current model_version_id.
    """

    insert_query = f"""
    INSERT INTO epi.t3_model_version_study_covariate (
        model_version_id,
        bundle_id,
        seq,
        study_covariate_id
    ) VALUES (
        {model_version_id}, %s, %s, %s
    )
    """

    cursor.executemany(insert_query, covariate_data)

    CODELOG.debug(f"uploaded {len(covariate_data)} lines of covariate")


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
        bundle_id = _get_bundle_id(execution_context)
        bundle_data = _get_tier_2_bundle_data(execution_context, bundle_id)
        covariate_data = _get_tier_2_study_covariates(execution_context, bundle_id)
        with cursor(execution_context) as c:
            _upload_bundle_data_to_tier_3(c, model_version_id, bundle_data)
            _upload_study_covariates_to_tier_3(c, model_version_id, covariate_data)
        return True


def raw_bundle(execution_context):
    model_version_id = execution_context.parameters.model_version_id

    query = f"""
    SELECT
        model_version_dismod_id as data_id,
        nid,
        location_id,
        sm.measure as integrand,
        mean,
        year_start,
        year_end,
        age_start,
        age_end,
        lower,
        upper,
        sample_size,
        standard_error,
        sex_id
    FROM
        epi.model_version mv
    INNER JOIN
        epi.t3_model_version_dismod t3_dm
            USING(model_version_id)
    LEFT JOIN
        shared.measure sm ON (t3_dm.measure_id = sm.measure_id)
    WHERE mv.model_version_id = %(model_version_id)s
    AND t3_dm.outlier_type_id = 0
    """

    with connection(execution_context) as c:
        df = pd.read_sql(query, c, params={"model_version_id": model_version_id})

    df = df.set_index("data_id")
    return df


def study_covariates(execution_context):
    model_version_id = execution_context.parameters.model_version_id

    query = """
    select model_version_dismod_id, study_covariate_id
    from epi.t3_model_version_study_covariate
    left join epi.t3_model_version_dismod
      on epi.t3_model_version_dismod.model_version_id = epi.t3_model_version_study_covariate.model_version_id
      and epi.t3_model_version_dismod.seq = epi.t3_model_version_study_covariate.seq
    where epi.t3_model_version_study_covariate.model_version_id = %(model_version_id)s
    limit 100
    """
    with connection(execution_context) as c:
        df = pd.read_sql(query, c, params={"model_version_id": model_version_id})

    return df
