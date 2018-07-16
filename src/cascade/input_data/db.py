import logging

from cascade.core.db import cursor

CODELOG = logging.getLogger(__name__)


def _bundle_is_frozen(model_version_id, cursor):
    query = f"""
    select exists(
             select * from epi.t3_model_version_dismod
             where model_version_id = {model_version_id}
    )
    """
    cursor.execute(query)
    exists = cursor.fetchone()[0]

    return exists == 1


def freeze_bundle(execution_context):
    model_version_id = execution_context.parameters.model_version_id

    with cursor(execution_context) as c:
        database = execution_context.parameters.database
        if _bundle_is_frozen(model_version_id, c):
            CODELOG.info(
                f"Bundle data for model_version_id {model_version_id} on '{database}' already frozen, doing nothing."
            )
            return False
        else:
            CODELOG.info(
                f"Freezing bundle data for model_version_id {model_version_id} on '{database}'"
            )
            c.callproc("load_t3_model_version_dismod", [model_version_id])
            c.callproc("load_t3_model_version_study_covariate", [model_version_id])
            return True
