import json

from cascade.core.db import cursor


def from_epiviz(execution_context):
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
    team says the will enforce in the front end, though that hasn't happened
    yet.
    """
    trimmed = None
    remove = True
    if isinstance(source, dict):
        trimmed = {}
        for k, v in source.items():
            if k.startswith("__"):
                continue
            tv = trim_config(v)
            if tv is not DO_REMOVE:
                trimmed[k] = tv
                remove = False
    elif isinstance(source, list):
        trimmed = []
        for v in source:
            tv = trim_config(v)
            if tv is not DO_REMOVE:
                trimmed.append(tv)
                remove = False
    else:
        if source is not None and source != "":
            trimmed = source
            remove = False

    if remove:
        return DO_REMOVE
    return trimmed
