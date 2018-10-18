import pandas as pd

from cascade.input_data.db.demographics import get_age_groups, get_years
from cascade.dismod.db.metadata import IntegrandEnum

from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)


def make_average_integrand_cases_from_gbd(execution_context):
    gbd_age_groups = get_age_groups(execution_context)
    age_ranges = [(r.age_group_years_start, r.age_group_years_end) for _, r in gbd_age_groups.iterrows()]
    time_ranges = [(y, y) for y in get_years(execution_context)]

    rows = [
        {
            "integrand_name": integrand.name,
            "age_lower": age_lower,
            "age_upper": age_upper,
            "time_lower": time_lower,
            "time_upper": time_upper,
            # Assuming using the first set of weights, which is constant.
            "weight_id": 0,
            "node_id": execution_context.parameters.location_id,
            "x_sex": sex,
        }
        for integrand in IntegrandEnum
        for age_lower, age_upper in age_ranges
        for time_lower, time_upper in time_ranges
        for sex in [-0.5, 0.5]
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "integrand_name",
            "age_lower",
            "age_upper",
            "time_lower",
            "time_upper",
            "weight_id",
            "node_id",
            "x_sex",
        ],
    )
