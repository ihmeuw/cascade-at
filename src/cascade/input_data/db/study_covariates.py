"""
Study covariates have a sparse representation in the database, which means
that, for each covariate_id, the database stores a list of seq numbers to which
that covariate applies. A seq number uniquely identifies the bundle record
for which that covariate is nonzero.

There is one case where a nonzero covariate is not 1. That is the "delta"
covariate.
"""
import pandas as pd
import numpy as np

from cascade.core.db import connection, cursor
from cascade.core.log import getLoggers

CODELOG, MATHLOG = getLoggers(__name__)


def get_study_covariates(execution_context, bundle_id, model_version_id, tier=3):
    """Downloads the tier 2 or 3 study covariate mappings for the bundle
    associated with the current model_version_id. This is used both to move
    covariates from tier 2 to tier 3 and to get them for construction of the
    study covariate column.
    """

    if tier == 2:
        table = "epi.bundle_dismod_study_covariate"
        mvid_clause = ""
    elif tier == 3:
        table = "epi.t3_model_version_study_covariate"
        mvid_clause = " and model_version_id = %(mvid)s"
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
        bundle_id = %(bundle_id)s {mvid_clause}
         """
    with connection(execution_context) as c:
        covariates = pd.read_sql(query, c,
                                 params={"bundle_id": bundle_id, "mvid": model_version_id})
        CODELOG.debug(
            f"Downloaded {len(covariates)} lines of study covariates for bundle_id {bundle_id}"
        )

    return covariates


def covariate_ids_to_names(execution_context, study_covariate_ids):
    """Convert study_covariate_ids to canonical study covariate names

    Args:
        execution_context: For access to databases.
        study_covariate_ids (List[int]): A list of unique study covariate ids.
    """
    study_covariate_ids = list(study_covariate_ids)

    if study_covariate_ids:
        query = """
        select study_covariate_id, study_covariate
        from epi.study_covariate
        where study_covariate_id in %(covariate_ids)s
        """
        with cursor(execution_context) as c:
            c.execute(query, args={"covariate_ids": np.array(study_covariate_ids).astype(int).tolist()})
            covariate_mapping = dict(list(c))

    else:
        MATHLOG.info(f"Found no study covariates to add to bundle.")
        covariate_mapping = {}

    return covariate_mapping
