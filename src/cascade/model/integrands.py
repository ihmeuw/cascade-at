import pandas as pd

from cascade.input_data.db.demographics import get_age_groups, get_years
from cascade.dismod.db.metadata import IntegrandEnum

from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)


def make_average_integrand_cases_from_gbd(execution_context, sexes, include_birth_prevalence=False):
    """
    """
    gbd_age_groups = get_age_groups(execution_context)
    age_ranges = [(r.age_group_years_start, r.age_group_years_end) for _, r in gbd_age_groups.iterrows()]
    time_ranges = [(y, y) for y in get_years(execution_context)]

    # Assuming using the first set of weights, which is constant.
    weight_id = 0

    rows = [
        {
            "integrand_name": integrand.name,
            "age_lower": age_lower,
            "age_upper": age_upper,
            "time_lower": time_lower,
            "time_upper": time_upper,
            "weight_id": weight_id,
            "node_id": execution_context.parameters.location_id,
            "sex_id": sex_id,
        }
        for integrand in IntegrandEnum
        for age_lower, age_upper in age_ranges
        for time_lower, time_upper in time_ranges
        for sex_id in sexes
    ]

    if include_birth_prevalence:
        birth_prev_rows = [
            {
                "integrand_name": "prevalence",
                "age_lower": 0,
                "age_upper": 0,
                "time_lower": time_lower,
                "time_upper": time_upper,
                "weight_id": weight_id,
                "node_id": execution_context.parameters.location_id,
                "sex_id": sex_id,
            }
            for time_lower, time_upper in time_ranges
            for sex_id in sexes
        ]
        rows.extend(birth_prev_rows)

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
            "sex_id",
        ],
    )
