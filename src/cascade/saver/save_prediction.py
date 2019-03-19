import pandas as pd
import numpy as np

from cascade.core import getLoggers
from cascade.dismod.constants import IntegrandEnum
from cascade.input_data.db.demographics import age_ranges_to_groups
from cascade.input_data.configuration.id_map import make_integrand_map
from cascade.core.db import ezfuncs

CODELOG, MATHLOG = getLoggers(__name__)


def uncertainty_from_prediction_draws(predictions):
    predictions = pd.concat(predictions)
    CODELOG.debug(f"predictions columns {predictions.columns}")
    columns_to_remove = ["sample_index"] + [c for c in predictions.columns if c.startswith("s_") and c != "s_sex"]
    predictions = predictions.drop(columns_to_remove, "columns")
    predictions = predictions.groupby(
        ["location", "integrand", "age_lower", "age_upper", "time_lower", "time_upper", "s_sex"]
    )

    lower = predictions.quantile(0.025)
    lower.columns = ["lower"]
    upper = predictions.quantile(0.975)
    upper.columns = ["upper"]
    mean = predictions.mean()
    mean.columns = ["mean"]

    return pd.concat([lower, upper, mean], axis="columns").reset_index()


def _predicted_to_uploadable_format(execution_context, predicted):
    predicted = predicted[
        ["mean", "location", "integrand", "age_lower", "age_upper", "time_lower", "time_upper", "s_sex"]
    ]
    predicted = predicted.rename(columns={"location": "location_id"})

    predicted["sex_id"] = predicted.s_sex.apply(lambda s_sex: {-1: 2, 0: 3, 1: 1}[int(s_sex * 2)])
    predicted = predicted.drop("s_sex", "columns")

    predicted = age_ranges_to_groups(execution_context, predicted)

    integrand_to_measure = {v.name: k for k, v in make_integrand_map().items()}
    # TODO: This should actually be fixed in id_map but I'm not sure how best to do that so I'm punting
    integrand_to_measure[IntegrandEnum.prevalence.name] = 5
    integrand_to_measure[IntegrandEnum.Tincidence.name] = 42
    integrand_to_measure[IntegrandEnum.Sincidence.name] = 41

    predicted["measure_id"] = predicted.integrand.apply(lambda i: integrand_to_measure[i])
    predicted = predicted.drop("integrand", "columns")

    if np.any(predicted.time_lower != predicted.time_upper):
        raise ValueError("Can't turn time ranges into time_ids")
    predicted = predicted.rename(columns={"time_lower": "year_id"})
    predicted = predicted.drop("time_upper", "columns")

    return predicted


def save_predicted_value(execution_context, predicted, fit_or_final):
    if fit_or_final == "fit":
        table = "model_estimate_fit"
    elif fit_or_final == "final":
        table = "model_estimate_final"
    else:
        raise ValueError("fit_or_final must be 'fit' or 'final'")

    predicted = _predicted_to_uploadable_format(execution_context, predicted)

    predicted = predicted.assign(model_version_id=execution_context.parameters.model_version_id)

    engine = ezfuncs.get_engine(execution_context.parameters.database)
    CODELOG.debug(f"Saving {len(table)} records to {table} on db "
                  f"{execution_context.parameters.database}.")
    try:
        predicted.to_sql(table, engine, if_exists="append", index=False)
    except Exception as exc:
        raise RuntimeError(f"Could not save predictions") from exc
