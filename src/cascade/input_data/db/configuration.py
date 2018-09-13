import json

from cascade.core.db import cursor
from cascade.testing_utilities import make_execution_context
from cascade.core.db import latest_model_version


def settings_for_model(meid=None, mvid=None):
    if len([c for c in [meid, mvid] if c is not None]) != 1:
        raise ValueError("Must supply either mvid or meid")

    if mvid is not None:
        mvid = mvid
    else:
        modelable_entity_id = meid
        ec = make_execution_context(modelable_entity_id=modelable_entity_id)
        mvid = latest_model_version(ec)
    ec = make_execution_context(model_version_id=mvid)
    raw_settings = settings_json_from_epiviz(ec)

    if "model_version_id" not in raw_settings["model"] or not raw_settings["model"]["model_version_id"]:
        raw_settings["model"]["model_version_id"] = mvid

    return raw_settings


def settings_json_from_epiviz(execution_context):
    model_version_id = execution_context.parameters.model_version_id

    query = """select parameter_json from at_model_parameter where model_version_id = %(model_version_id)s"""
    with cursor(execution_context) as c:
        c.execute(query, args={"model_version_id": model_version_id})
        raw_data = c.fetchall()

    if len(raw_data) == 0:
        raise ValueError(f"No parameters for model version {model_version_id}")
    if len(raw_data) > 1:
        raise ValueError(f"Multiple parameter entries for model version {model_version_id}")

    config_data = json.loads(raw_data[0][0])

    # Fix bugs in epiviz
    # TODO: remove this once EPI-999 is resolved
    config_data = trim_config(config_data)
    if config_data is DO_REMOVE:
        config_data = {}

    return config_data


DO_REMOVE = object()


def trim_config(source):
    """ This function represents the approach to missing data which the viz
    team says it will enforce in the front end, though that hasn't happened
    yet.
    """
    trimmed = None
    remove = True
    if isinstance(source, dict):
        trimmed, remove = _trim_dict(source)
    elif isinstance(source, list):
        trimmed, remove = _trim_list(source)
    else:
        if source is not None and source != "":
            trimmed = source
            remove = False

    if remove:
        return DO_REMOVE
    else:
        return trimmed


def _trim_dict(source_dict):
    trimmed = {}
    remove = True
    for k, v in source_dict.items():
        # Removing keys prefixed by "__" because they represent garbage that
        # the GUI framework sticks in sometimes but isn't useful to us.
        if not k.startswith("__"):
            tv = trim_config(v)
            if tv is not DO_REMOVE:
                trimmed[k] = tv
                remove = False
    return trimmed, remove


def _trim_list(source_list):
    trimmed = []
    remove = True
    for v in source_list:
        tv = trim_config(v)
        if tv is not DO_REMOVE:
            trimmed.append(tv)
            remove = False
    return trimmed, remove
