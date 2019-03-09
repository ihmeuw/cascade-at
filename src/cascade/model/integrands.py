import pandas as pd

from cascade.core.log import getLoggers
from cascade.dismod.constants import IntegrandEnum

CODELOG, MATHLOG = getLoggers(__name__)


def make_average_integrand_cases_from_gbd(
        ages_df, years_df, sexes, child_locations, include_birth_prevalence=False):
    """Determine what time and age ranges each integrand should be calculated
    for based on GBD's expected inputs for the rest of the pipeline.

    Args:
        execution_context
        sexes ([int]): a list of sex_ids to produces results for
        include_birth_prevalence (bool): If true, produce prevalence (and no
            other integrand for the Birth age group (id 164)
    """
    age_ranges = [(r.age_group_years_start, r.age_group_years_end) for _, r in ages_df.iterrows()]
    time_ranges = [(y, y) for y in years_df]

    rows = [
        {
            "integrand": integrand.name,
            "age_lower": age_lower,
            "age_upper": age_upper,
            "time_lower": time_lower,
            "time_upper": time_upper,
            "location": location_id,
            "sex_id": sex_id,
        }
        for integrand in IntegrandEnum
        for age_lower, age_upper in age_ranges
        for time_lower, time_upper in time_ranges
        for sex_id in sexes
        for location_id in child_locations
    ]

    if include_birth_prevalence:
        birth_prev_rows = [
            {
                "integrand": "prevalence",
                "age_lower": 0,
                "age_upper": 0,
                "time_lower": time_lower,
                "time_upper": time_upper,
                "location": location_id,
                "sex_id": sex_id,
            }
            for time_lower, time_upper in time_ranges
            for sex_id in sexes
            for location_id in child_locations
        ]
        rows.extend(birth_prev_rows)

    return pd.DataFrame(
        rows,
        columns=[
            "integrand",
            "age_lower",
            "age_upper",
            "time_lower",
            "time_upper",
            "location",
            "sex_id",
        ],
    )
