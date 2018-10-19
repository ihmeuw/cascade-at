"""This module provides tools for working directly with bundle data in the external databases. Code which wants to
manipulate the bundles directly in the database should live here but bundle code which does not need to access the
databases directly should live outside the db package and use the functions here to retrieve the data in normalized
form.
"""

import pandas as pd

from cascade.core.db import cursor, connection
from cascade.input_data.configuration.id_map import make_integrand_map

from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)


def _bundle_is_frozen(execution_context):
    """Checks if data for the current model_version_id already exists in tier 3.
    """

    model_version_id = execution_context.parameters.model_version_id
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
            raise ValueError(f"No bundle_id associated with model_version_id {model_version_id}")

        if len(bundle_ids) > 1:
            raise ValueError(f"Multiple bundle_ids associated with model_version_id {model_version_id}")

        return bundle_ids[0][0]


def _get_bundle_data(execution_context, bundle_id, tier=3):
    """Downloads the tier 2 or 3 data for the bundle associated with the current model_version_id.
    """

    if tier == 2:
        database = execution_context.parameters.bundle_database
        table = "epi.bundle_dismod"
    elif tier == 3:
        database = execution_context.parameters.database
        table = "epi.t3_model_version_dismod"
    else:
        raise ValueError(f"Only tiers 2 and 3 are supported")

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
         {table}
        WHERE
         bundle_id = %(bundle_id)s and
         input_type_id NOT IN(5,6) and
         outlier_type_id IN (0,1)
         """
    with connection(database=database) as c:
        bundle_data = pd.read_sql(query, c, params={"bundle_id": bundle_id})
        CODELOG.debug(f"Downloaded {len(bundle_data)} lines of bundle_id {bundle_id} from '{database}'")

    return bundle_data


def _get_study_covariates(execution_context, bundle_id, tier=3):
    """Downloads the tier 2 or 3 study covariate mappings for the bundle associated with the current model_version_id.
    """

    if tier == 2:
        database = execution_context.parameters.bundle_database
        table = "epi.bundle_dismod_study_covariate"
    elif tier == 3:
        database = execution_context.parameters.database
        table = "epi.t3_model_version_study_covariate"
    else:
        raise ValueError(f"Only tiers 2 and 3 are supported")

    query = f"""
    SELECT
        bundle_id,
        seq,
        study_covariate_id
    FROM
        {table}
    WHERE
        bundle_id = %(bundle_id)s
         """
    with connection(database=database) as c:
        covariates = pd.read_sql(query, c, params={"bundle_id": bundle_id})
        CODELOG.debug(
            f"Downloaded {len(covariates)} lines of study covariates for bundle_id {bundle_id} from '{database}'"
        )

    return covariates


def _upload_bundle_data_to_tier_3(cursor, model_version_id, bundle_data):
    """Uploads bundle data to tier 3 attached to the current model_version_id.
    """
    CODELOG.debug(f"Upload to tier3 {len(bundle_data)} lines to {model_version_id} mvid")
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

    bundle_data = bundle_data.where(pd.notnull(bundle_data), None)
    cursor.executemany(insert_query, bundle_data.values.tolist())

    CODELOG.debug(f"uploaded {len(bundle_data)} lines of bundle data")


def _upload_study_covariates_to_tier_3(cursor, model_version_id, covariate_data):
    """Uploads study covariate mappings to tier 3 attached to the current model_version_id.
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


def freeze_bundle(execution_context, bundle_id=None) -> bool:
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
        CODELOG.info(f"Freezing bundle data for model_version_id {model_version_id} on '{database}'")
        if bundle_id is None:
            bundle_id = _get_bundle_id(execution_context)
        bundle_data = _get_bundle_data(execution_context, bundle_id, tier=2)
        covariate_data = _get_study_covariates(execution_context, bundle_id, tier=2)
        with cursor(execution_context) as c:
            _upload_bundle_data_to_tier_3(c, model_version_id, bundle_data)
            _upload_study_covariates_to_tier_3(c, model_version_id, covariate_data)
        return True


def _normalize_measures(data):
    """Transform measure_ids into canonical measure names
    """
    data = data.copy()
    gbd_measure_id_to_integrand = make_integrand_map()
    if any(data.measure_id == 6):
        MATHLOG.warn(f"Found incidence, measure_id=6, in data. Should be Tincidence or Sincidence.")
    try:
        data["measure"] = data.measure_id.apply(lambda k: gbd_measure_id_to_integrand[k].name)
    except KeyError as ke:
        raise RuntimeError(
            f"The bundle data uses measure {str(ke)} which doesn't map "
            f"to an integrand. The map is {gbd_measure_id_to_integrand}."
        )
    return data


def _normalize_sex(data):
    """Transform sex_ids into strings
    """
    data = data.copy()
    data["sex"] = data.sex_id.apply({1: "Male", 2: "Female", 3: "Both"}.get)
    return data


def _normalize_bundle_data(data):
    """Normalize bundle columns, strip extra columns and index on `seq`
    """
    data = _normalize_measures(data)
    data = _normalize_sex(data)

    data = data.set_index("seq")

    cols = ["measure", "mean", "sex", "standard_error", "age_start", "age_end", "year_start", "year_end", "location_id"]

    return data[cols]


def _covariate_ids_to_names(execution_context, study_covariates):
    """Convert study_covariate_ids to canonical study covariate names
    """
    study_covariate_ids = list(study_covariates.study_covariate_id.unique())
    study_covariates = study_covariates.rename(columns={"study_covariate_id": "name"})

    if study_covariate_ids:
        query = """
        select study_covariate_id, study_covariate
        from epi.study_covariate
        where study_covariate_id in %(covariate_ids)s
        """
        with cursor(execution_context) as c:
            c.execute(query, args={"covariate_ids": study_covariate_ids})
            covariate_mapping = dict(list(c))

        study_covariates["name"] = study_covariates.name.apply(covariate_mapping.get)
    else:
        MATHLOG.info(f"Found no study covariates to add to bundle.")

    return study_covariates


def _normalize_covariate_data(execution_context, study_covariates):
    study_covariates = _covariate_ids_to_names(execution_context, study_covariates)

    study_covariates = study_covariates.set_index("seq")

    return study_covariates.name


def bundle_with_study_covariates(execution_context, bundle_id=None, tier=3):
    """Get bundle data with associated study covariate labels.

    Args:
        execution_context (ExecutionContext): The context within which to make the query
        bundle_id (int): Bundle to load. Defaults to the bundle associated with the context
        tier (int): Tier to load data from. Defaults to 3 (frozen data) but will also accept 2 (scratch space)

    Returns:
        A tuple of (bundle data, study covariate labels) where the bundle data is a pd.DataFrame and the labels are a
        pd.Series with an index aligned with the bundle data
    """
    if bundle_id is None:
        bundle_id = _get_bundle_id(execution_context)

    bundle = _get_bundle_data(execution_context, bundle_id, tier=tier)
    bundle = _normalize_bundle_data(bundle)

    covariate_data = _get_study_covariates(execution_context, bundle_id, tier=tier)
    if not covariate_data.empty:
        normalized_covariate = _normalize_covariate_data(execution_context, covariate_data)
    else:
        normalized_covariate = covariate_data

    return (bundle, normalized_covariate)
