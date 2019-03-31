import pandas as pd
import numpy as np

from cascade.core.db import db_queries
from cascade.core.db import age_spans

from cascade.core.log import getLoggers

CODELOG, MATHLOG = getLoggers(__name__)

# FIXME: This should come from central comp tools but I don't see a spot...
EPI_AGE_GROUP_SET_ID = 12


def get_age_groups(execution_context):
    """Returns most-detailed age groups where Dismod-AT should produce output."""
    groups = db_queries.get_age_metadata(
        age_group_set_id=EPI_AGE_GROUP_SET_ID, gbd_round_id=execution_context.parameters.gbd_round_id
    )
    groups = groups[["age_group_id", "age_group_years_start", "age_group_years_end"]]

    return groups


def get_years(execution_context):
    return db_queries.get_demographics(gbd_team="epi", gbd_round_id=execution_context.parameters.gbd_round_id)[
        "year_id"
    ]


def get_locations(execution_context):
    return db_queries.get_demographics(gbd_team="epi", gbd_round_id=execution_context.parameters.gbd_round_id)[
        "location_id"
    ]


def age_groups_to_ranges(execution_context, data, keep_age_group_id=False):
    groups = get_age_groups(execution_context)
    with_group = data.merge(groups, on="age_group_id")
    if not keep_age_group_id:
        with_group = with_group.drop(columns="age_group_id")
    return with_group.rename(columns={"age_group_years_start": "age_lower", "age_group_years_end": "age_upper"})


def age_ranges_to_groups(execution_context, data, keep_age_range=False):
    age_groups = get_age_groups(execution_context)

    # TODO: Age group 164 (Birth) is a special case which is not included in the standard
    # age group set and which the model may optionally produce for prevalence (and no other
    # measure). I'm hard coding it in here but it really should be plumbed through better.
    age_groups = age_groups.append([{
        "age_group_id": 164,
        "age_group_years_start": 0,
        "age_group_years_end": 0,
    }])

    # TODO: This assumes there will never be two different age groups with the same
    # upper range. Be wary
    with_age_groups = pd.merge_asof(
        data.sort_values("age_upper"),
        age_groups.sort_values("age_group_years_end"),
        left_on="age_upper",
        right_on="age_group_years_end"
    )

    merge_is_good = np.allclose(with_age_groups.age_lower, with_age_groups.age_group_years_start)
    merge_is_good = merge_is_good and np.allclose(with_age_groups.age_upper, with_age_groups.age_group_years_end)
    merge_is_good = merge_is_good and len(data) == len(with_age_groups)
    if not merge_is_good:
        raise ValueError(
            "There are age_lowers or age_uppers that do not match GBD age group boundaries"
        )
    if not with_age_groups.query("age_group_id == 164 and integrand != 'prevalence'").empty:
        raise ValueError(
            "There are non-prevalence values for the Birth age group"
        )

    if not keep_age_range:
        return with_age_groups.drop(
            ["age_group_years_start", "age_group_years_end", "age_lower", "age_upper"],
            "columns"
        )
    else:
        return with_age_groups.drop(["age_group_years_start", "age_group_years_end"], "columns")


def get_years_from_lower_age_to_mean_age(execution_context):
    """
    Get the mean year estimate for age groups. This is the mean age beyond the
    start of the age group for people in a given age group.

    .. note::

        Unlike the overall span of the age group, this value is demographically
        specific.

    Returns:
        pd.DataFrame: with columns ["age_group_id", "location_id",
            "year_id", "sex_id", "mean"]
    """
    mean_years = db_queries.get_life_table(
        status="best", life_table_parameter_id=2, gbd_round_id=execution_context.parameters.gbd_round_id
    )
    return mean_years[["age_group_id", "location_id", "year_id", "sex_id", "mean"]]


def get_all_age_spans():
    """
    Every single age group id, mapped to its age span. This doesn't use a GBD
    round ID.

    Returns:
        pd.DataFrame: with columns ["age_group_id", "age_group_years_start",
            "age_group_years_end"]
    """
    return age_spans.get_age_spans()
