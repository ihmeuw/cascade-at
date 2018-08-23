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

    return json.loads(raw_data[0][0])
