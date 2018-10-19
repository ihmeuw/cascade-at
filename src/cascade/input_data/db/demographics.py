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
    return groups[["age_group_id", "age_group_years_start", "age_group_years_end"]]


def get_years(execution_context):
    return db_queries.get_demographics(gbd_team="epi", gbd_round_id=execution_context.parameters.gbd_round_id)[
        "year_id"
    ]


def get_locations(execution_context):
    return db_queries.get_demographics(gbd_team="epi", gbd_round_id=execution_context.parameters.gbd_round_id)[
        "location_id"
    ]


def age_groups_to_ranges(execution_context, data):
    groups = get_age_groups(execution_context)
    with_group = data.merge(groups, on="age_group_id").drop(columns="age_group_id")
    return with_group.rename(columns={"age_group_years_start": "age_start", "age_group_years_end": "age_end"})


def get_all_age_spans(execution_context):
    """
    Every single age group id, mapped to its age span. This doesn't use a GBD
    round ID.

    Args:
        execution_context: Here for mocking ``db_queries`` when needed.

    Returns:
        pd.DataFrame: with columns ["age_group_id", "age_group_years_start",
            "age_group_years_end"]
    """
    return age_spans.get_age_spans()
