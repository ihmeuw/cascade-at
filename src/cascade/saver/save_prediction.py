import numpy as np

from cascade.core.db import connection
from cascade.input_data.db.demographics import age_ranges_to_groups
from cascade.input_data.configuration.id_map import make_integrand_map


def save_predicted_value(execution_context, predicted, sex_id, fit_or_final):
    if fit_or_final == "fit":
        table = "model_estimate_fit"
    elif fit_or_final == "final":
        table = "model_estimate_final"
    else:
        raise ValueError("fit_or_final must be 'fit' or 'final'")

    predicted = predicted[["mean", "location", "integrand", "age_lower", "age_upper", "time_lower", "time_upper"]]
    predicted = predicted.rename(columns={"location": "location_id"})

    predicted = age_ranges_to_groups(execution_context, predicted)

    integrand_to_measure = {v.value: k for k, v in make_integrand_map().items()}
    predicted["measure_id"] = predicted.integrand.apply(lambda i: integrand_to_measure[i])
    predicted = predicted.drop("integrand", "columns")

    if np.any(predicted.time_lower != predicted.time_upper):
        raise ValueError("Can't turn time ranges into time_ids")
    predicted["time_id"] == predicted.time_lower
    predicted = predicted.drop(["time_lower", "time_upper"], "columns")

    predicted = predicted.assign(sex_id=sex_id, model_version_id=execution_context.parameters.model_version_id)

    with connection(execution_context) as c:
        predicted.to_sql(table, c)
