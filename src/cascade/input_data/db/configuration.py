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
    if "bundle_id" not in config_data["model"]:
        config_data["model"]["bundle_id"] = None

    return config_data
