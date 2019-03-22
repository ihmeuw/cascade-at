import json

from cascade.core import getLoggers, __version__
from cascade.core.db import cursor

CODELOG, MATHLOG = getLoggers(__name__)


def create_model_version(execution_context, settings, username, title=None):
    new_mvid = _insert_model_version(execution_context, settings, username, title)

    _insert_settings_json(execution_context, settings, new_mvid)

    MATHLOG.info(f"Created new model version: {new_mvid}")


def _insert_model_version(execution_context, settings, username, title):
    if title is None:
        title = settings.model.title

    query = """
    insert into model_version
    (
        modelable_entity_id,
        bundle_id,
        single_measure_id,
        title,
        gbd_round_id,
        csmr_cod_output_version_id,
        csmr_mortality_output_version_id,
        inserted_by,
        model_version_status_id,
        code_version,
        location_set_version_id
    )
    values (
        %(modelable_entity_id)s,
        %(bundle_id)s,
        null,
        %(title)s,
        %(gbd_round_id)s,
        %(cod_version_id)s,
        %(mortality_version_id)s,
        %(username)s,
        2,
        %(code_version)s,
        %(location_set_version_id)s
    )
    """
    with cursor(execution_context) as c:
        c.execute(
            query,
            args=dict(
                modelable_entity_id=settings.model.modelable_entity_id,
                bundle_id=settings.model.bundle_id,
                title=title,
                gbd_round_id=settings.gbd_round_id,
                cod_version_id=settings.csmr_cod_output_version_id,
                # TODO: We don't distinguish this internally though theoretically they could differ
                mortality_version_id=settings.csmr_cod_output_version_id,
                code_version=__version__,
                location_set_version_id=settings.location_set_version_id,
                username=username,
            ),
        )
        return c.lastrowid


def _insert_settings_json(execution_context, settings, model_version_id):
    query = """
        insert into at_model_parameter
            (model_version_id, parameter_json)
        values (%(model_version_id)s, %(json)s)
    """

    with cursor(execution_context) as c:
        c.execute(query, args=dict(model_version_id=model_version_id, json=json.dumps(settings.to_dict())))
