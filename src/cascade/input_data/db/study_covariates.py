"""
Study covariates have a sparse representation in the database, which means
that, for each covariate_id, the database stores a list of seq numbers to which
that covariate applies. A seq number uniquely identifies the bundle record
for which that covariate is nonzero.

There is one case where a nonzero covariate is not 1. That is the "delta"
covariate.
"""
import pandas as pd

from cascade.core.db import connection, cursor
from cascade.core.log import getLoggers
from cascade.input_data import InputDataError

CODELOG, MATHLOG = getLoggers(__name__)


def _get_study_covariates(execution_context, bundle_id, tier=3):
    """Downloads the tier 2 or 3 study covariate mappings for the bundle
    associated with the current model_version_id. This is used both to move
    covariates from tier 2 to tier 3 and to get them for construction of the
    study covariate column.
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


def _covariate_ids_to_names(execution_context, study_covariate_ids):
    """Convert study_covariate_ids to canonical study covariate names
    """
    study_covariate_ids = list(study_covariate_ids)

    if study_covariate_ids:
        query = """
        select study_covariate_id, study_covariate
        from epi.study_covariate
        where study_covariate_id in %(covariate_ids)s
        """
        with cursor(execution_context) as c:
            c.execute(query, args={"covariate_ids": study_covariate_ids})
            covariate_mapping = dict(list(c))

    else:
        MATHLOG.info(f"Found no study covariates to add to bundle.")
        covariate_mapping = {}

    return covariate_mapping


def _normalize_covariate_data(bundle_index, id_to_name, study_covariates):
    """
    The input is study covariates in a sparse-columnar format, so it's a list
    of which covariates are nonzero for which seq numbers, where a seq
    number identifies a row in the bundle index.

    Args:
        bundle_index (pd.Index): The index of seq numbers for the bundle.
        execution_context: An execution context.
        study_covariates (pd.DataFrame): Contains seq numbers and covariate ids.
            Optionally contains the ``bundle_id``.

    Returns:
        pd.DataFrame: Each column is a full row of zeros and ones, and the row
        name is the name of the covariate, without the ``x_`` in front.
    """
    study_ids = study_covariates.set_index("seq").study_covariate_id
    study_covariate_columns = list()
    indices_not_found = list()
    for cov_id in sorted(id_to_name):  # Sort for stable behavior.
        cov_column =  pd.Series([0.0] * len(bundle_index), index=bundle_index.values, name=id_to_name[cov_id])
        try:
            cov_column.loc[study_ids[study_ids == cov_id].index] = 1.0
        except KeyError:
            indices_not_found.append((cov_id, id_to_name[cov_id]))
        study_covariate_columns.append(cov_column)
    if indices_not_found:
        raise InputDataError(f"Study covariates list ids not found in the bundle for "
                             f"covariates: {indices_not_found}.")

    return pd.concat(study_covariate_columns, axis=1)


def get_bundle_study_covariates(bundle_index, bundle_id, execution_context, tier):
    covariate_data = _get_study_covariates(execution_context, bundle_id, tier=tier)
    id_to_name = _covariate_ids_to_names(execution_context, covariate_data.study_covariate_id.unique())
    return _normalize_covariate_data(bundle_index, id_to_name, covariate_data)
