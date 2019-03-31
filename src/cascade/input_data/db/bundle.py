"""This module provides tools for working directly with bundle data in the external databases. Code which wants to
manipulate the bundles directly in the database should live here but bundle code which does not need to access the
databases directly should live outside the db package and use the functions here to retrieve the data in normalized
form.
"""

import pandas as pd

from cascade.core.db import cursor, connection
from cascade.input_data import InputDataError
from cascade.core.log import getLoggers
from cascade.input_data.db.study_covariates import get_study_covariates

CODELOG, MATHLOG = getLoggers(__name__)

BUNDLE_COLUMNS = [
    "bundle_id",
    "seq",
    "request_id",
    "input_type_id",
    "nid",
    "underlying_nid",
    "location_id",
    "sex_id",
    "year_start",
    "year_end",
    "age_start",
    "age_end",
    "measure_id",
    "source_type_id",
    "sampling_type_id",
    "representative_id",
    "urbanicity_type_id",
    "recall_type_id",
    "recall_type_value",
    "unit_type_id",
    "unit_value_as_published",
    "uncertainty_type_id",
    "uncertainty_type_value",
    "mean",
    "lower",
    "upper",
    "standard_error",
    "effective_sample_size",
    "sample_size",
    "cases",
    "design_effect",
    "outlier_type_id",
]


def _bundle_is_frozen(execution_context, model_version_id):
    """Checks if data for the current model_version_id already exists in tier 3.
    """
    query = """
    select exists(
             select * from epi.t3_model_version_dismod
             where model_version_id = %(model_version_id)s
    )
    """
    with cursor(execution_context) as c:
        c.execute(query, args={"model_version_id": model_version_id})
        exists = c.fetchone()[0]

    return exists == 1


def _get_bundle_id(execution_context, model_version_id):
    """Gets the bundle id associated with the current model_version_id.
    """
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
            raise InputDataError(f"No bundle_id associated with model_version_id {model_version_id}")

        if len(bundle_ids) > 1:
            raise InputDataError(f"Multiple bundle_ids associated with model_version_id {model_version_id}")

        return bundle_ids[0][0]


def _get_bundle_data(execution_context, model_version_id, bundle_id, tier=3, exclude_outliers=True):
    """
    Downloads the tier 2 or 3 data for the bundle associated with the current
    model_version_id.

    Returns:
        Bundle with two properties, ``tier`` and ``bundle_id``.
    """

    if tier == 2:
        table = "epi.bundle_dismod"
        model_version_query = ""
    elif tier == 3:
        table = "epi.t3_model_version_dismod"
        model_version_query = "and model_version_id = %(mvid)s"
    else:
        raise ValueError(f"Only tiers 2 and 3 are supported")

    if exclude_outliers:
        outlier_flags = "(0)"
    else:
        outlier_flags = "(0, 1)"

    query = f"""
    SELECT
        {", ".join(BUNDLE_COLUMNS)}
        FROM
         {table}
        WHERE
         bundle_id = %(bundle_id)s and
         input_type_id NOT IN(5,6) and
         outlier_type_id IN {outlier_flags} {model_version_query}
         """
    with connection(execution_context) as c:
        bundle_data = pd.read_sql(
            query,
            c,
            params={"bundle_id": bundle_id, "mvid": model_version_id})
        MATHLOG.debug(f"Downloaded {len(bundle_data)} lines of bundle_id {bundle_id}")
        if exclude_outliers:
            # The modelers input the group_review flag as group_review=0 but then elmo transforms it to
            # input_type_id = 6 which is what we actually filter on above.
            MATHLOG.debug("This excludes rows marked as outliers as well as those marked group_review=0")
        else:
            MATHLOG.debug("This excludes rows marked group_review=0")

    return bundle_data


def _upload_bundle_data_to_tier_3(cursor, model_version_id, bundle_data):
    """Uploads bundle data to tier 3 attached to the current model_version_id.
    """
    CODELOG.debug(f"Upload to tier3 {len(bundle_data)} lines to {model_version_id} mvid")
    insert_query = f"""
    INSERT INTO epi.t3_model_version_dismod (
        model_version_id,
        {", ".join(BUNDLE_COLUMNS)}
    ) VALUES (
        {model_version_id}, {", ".join(["%s"]*32)}
    )
    """

    bundle_data = bundle_data.where(pd.notnull(bundle_data), None)
    cursor.executemany(insert_query, bundle_data.values.tolist())

    CODELOG.debug(f"uploaded {len(bundle_data)} lines of bundle data")


def _upload_study_covariates_to_tier_3(cursor, model_version_id, covariate_data):
    """Uploads study covariate mappings to tier 3 attached to the current model_version_id.
    This isn't doing what it has to do. See
    https://stash.ihme.washington.edu/users/joewag/repos/cascade_ode/browse/cascade_ode/importer.py#13,170,259-260
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

    cursor.executemany(insert_query, covariate_data.values.tolist())

    CODELOG.debug(f"uploaded {len(covariate_data)} lines of covariate")


def freeze_bundle(execution_context, model_version_id, bundle_id=None) -> bool:
    """Freezes the bundle data attached to the current model_version_id if necessary.

    The freezing process works as follows:
    If there is any data in the tier 3 table for the model_version_id then it is assumed to already be frozen.
    Otherwise download all data for the bundle from the tier 2 table in the main epi database and re-upload it
    to the Dismod-AT database (ie. promote it).

    Returns:
      True if any data was promoted and False otherwise
    """
    if _bundle_is_frozen(execution_context, model_version_id):
        CODELOG.info(
            f"Bundle data for model_version_id {model_version_id} already frozen, doing nothing."
        )
        return False
    else:
        CODELOG.info(f"Freezing bundle data for model_version_id {model_version_id}")
        if bundle_id is None:
            bundle_id = _get_bundle_id(execution_context, model_version_id)
        bundle_data = _get_bundle_data(
            execution_context, model_version_id, bundle_id, tier=2, exclude_outliers=False)
        covariate_data = get_study_covariates(execution_context, bundle_id, model_version_id, tier=2)
        with cursor(execution_context) as c:
            _upload_bundle_data_to_tier_3(c, model_version_id, bundle_data)
            _upload_study_covariates_to_tier_3(c, model_version_id, covariate_data)
        return True
