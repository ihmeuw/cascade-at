"""Upload and download of age-specific death rate, which is
all-cause mortality rate, by age."""
import pandas as pd
from numpy import nan

from cascade.core.db import cursor, db_queries, repeat_request
from cascade.core.log import getLoggers
from cascade.input_data.db import AGE_GROUP_SET_ID
from cascade.input_data.db.data_iterator import grouped_by_count
from cascade.runner.application_config import application_config
from cascade.stats.estimation import bounds_to_stdev

CODELOG, MATHLOG = getLoggers(__name__)


def _asdr_in_t3(execution_context, model_version_id):
    """Checks if data for the current model_version_id already exists in tier 3.
    """
    query = """
    SELECT DISTINCT location_id
    FROM epi.t3_model_version_asdr
    WHERE model_version_id = %(model_version_id)s
    """
    with cursor(execution_context) as c:
        c.execute(query, args={"model_version_id": model_version_id})
        location_rows = c.fetchall()

    return [row[0] for row in location_rows]


def get_asdr_formatted(choices_dict):
    """
    Retrieves ASDR data and formats it so that we know exactly what columns
    there are.
    """
    gbd_round_id = choices_dict["gbd_round_id"]
    demo_dict = db_queries.get_demographics(gbd_team="epi", gbd_round_id=gbd_round_id)
    choices_dict.update(dict(
        age_group_id=demo_dict["age_group_id"],
        sex_id=demo_dict["sex_id"],
    ))
    asdr = repeat_request(db_queries.get_envelope)(**choices_dict)

    nulls = asdr["mean"].isnull().sum()
    if nulls > 0:
        MATHLOG.info(f"Removing {nulls} null values from mean of age-specific death rate.")
        asdr = asdr[asdr["mean"].notnull()]

    cols = ["age_group_id", "location_id", "year_id", "sex_id", "mean", "upper", "lower"]
    return asdr[cols]


def get_asdr_global(gbd_round_id, decomp_step, location_set_version_id, with_hiv):
    r"""Gets the age-specific death rate from IHME databases.
    This is :math:`{}_nm_x`, the mortality rate. This gets rates, not counts.
    This gets data for all locations, specified by location set.
    """
    CODELOG.debug(f"get_asdr_global: round {gbd_round_id} decomp {decomp_step}")
    return get_asdr_formatted(dict(
        location_set_version_id=location_set_version_id,
        location_id="all",
        year_id=-1,
        gbd_round_id=gbd_round_id,
        decomp_step=decomp_step,
        with_hiv=with_hiv,
        rates=True,
    ))


def get_asdr_data(gbd_round_id, decomp_step, location_and_children, with_hiv):
    r"""Gets the age-specific death rate from IHME databases.
    This is :math:`{}_nm_x`, the mortality rate. This gets rates, not counts.
    Gets data only for locations specified.
    """
    CODELOG.debug(f"get_asdr_data round {gbd_round_id} decomp {decomp_step} "
                  f"for locations {location_and_children}")
    return get_asdr_formatted(dict(
        location_id=list(location_and_children),
        year_id=-1,
        gbd_round_id=gbd_round_id,
        decomp_step=decomp_step,
        with_hiv=with_hiv,
        rates=True,
    ))


def asdr_as_fit_input(
        location_set_version_id, included_locations, sexes, gbd_round_id,
        decomp_step, ages_df, with_hiv
):
    r"""Gets age-specific death rate (ASDR) from database and formats as
    input data. This is :math:`{}_nm_x`, the mortality rate by age group.
    Returns rates, not counts.

    Args:
        location_set_version_id (int): Location set version for which
            to retrieve the data. This is all locations.
        included_locations (List[int]): The locations in this run.
            If the locations we need are less than ``small-location-count``,
            then retrieve data only for those locations.
        sexes (List[int]): 1, 2, 3, or 4. Sex_id.
        gbd_round_id (int): GBD round identifies consistent data sets.
        ages_df (pd.DataFrame): Age_id to age mapping.
        with_hiv (bool): whether to include HIV deaths in mortality.

    Returns:
        pd.DataFrame: Columns are ``integrand``, ``hold_out``, ``density``,
        ``eta``, ``nu``, ``time_lower``, ``time_upper``, ``age_lower``,
        ``age_upper``, and ``location``, ``sex_id``.
    """
    parameters = application_config()["NonModel"]
    # Why not download all every time? It's really slow for testing.
    small_number_locations = parameters.getint("small-location-count")
    if included_locations is not None and len(included_locations) < small_number_locations:
        # Call db_queries multiple times because it uses the SQL in set()
        # syntax, which fails when the set is large.
        locations_per_query = parameters.getint("locations-per-query")
        multiple_asdr = list()
        for location_bunch in grouped_by_count(included_locations, locations_per_query):
            piece = get_asdr_data(gbd_round_id, decomp_step, location_bunch, with_hiv)
            multiple_asdr.append(piece)
        asdr = pd.concat(multiple_asdr, axis=0, ignore_index=True, sort=False)
        CODELOG.debug(f"asdr_as_fit_input Retrieving {len(asdr)} ASDR for {included_locations}.")
    else:
        asdr = get_asdr_global(gbd_round_id, decomp_step, location_set_version_id, with_hiv)
        CODELOG.debug(
            f"asdr_as_fit_input Retrieving {len(asdr)} ASDR for "
            f"location_set_version_id {location_set_version_id}.")
    assert not (set(asdr.age_group_id.unique()) - set(ages_df.age_group_id.values))
    return asdr_by_sex(asdr, ages_df, sexes)


def asdr_by_sex(asdr, ages, sexes):
    """Incoming age-specific death rate has ``age_id`` and upper and lower
    bounds. This translates those into age-ranges, time-ranges, and standard
    deviations."""
    CODELOG.debug(f"asdr_by_sex for sexes {sexes}")
    without_weight = ages.drop(columns=["age_group_weight_value"])
    as_up_low = without_weight.rename({"age_group_years_start": "age_lower", "age_group_years_end": "age_upper"},
                                      axis="columns")
    with_ages = asdr.merge(as_up_low, on="age_group_id", how="left")
    with_upper = with_ages.assign(time_upper=with_ages.year_id + 1)
    with_times = with_upper.rename(columns=dict(year_id="time_lower", location_id="location"))
    with_std = with_times.assign(std=bounds_to_stdev(with_times.lower, with_times.upper))
    rest = with_std.assign(
        integrand="mtother",
        hold_out=0,
        density="gaussian",
        eta=nan,
        nu=nan,
    )
    trimmed = rest.drop(columns=["age_group_id", "upper", "lower"])
    return trimmed.query("sex_id in @sexes")


def _upload_asdr_data_to_tier_3(gbd_round_id, cursor, model_version_id, asdr_data):
    """Uploads ASDR data to tier 3 attached to the current model_version_id.
    """

    insert_query = f"""
        INSERT INTO epi.t3_model_version_asdr (
            model_version_id,
            year_id,
            location_id,
            sex_id,
            age_group_id,
            mean,
            upper,
            lower,
            age_upper,
            age_lower
        ) VALUES (
            {model_version_id}, {", ".join(["%s"]*9)}
        )
    """

    age_group_data = db_queries.get_age_metadata(
        age_group_set_id=AGE_GROUP_SET_ID, gbd_round_id=gbd_round_id
    )[["age_group_id", "age_group_years_start", "age_group_years_end"]]

    age_group_data = age_group_data.rename(columns={
        "age_group_years_start": "age_lower",
        "age_group_years_end": "age_upper"
    })
    asdr_data = asdr_data.merge(age_group_data, how="left", on="age_group_id")
    asdr_data = asdr_data.where(pd.notnull(asdr_data), None)

    ordered_cols = [
        "year_id",
        "location_id",
        "sex_id",
        "age_group_id",
        "mean",
        "upper",
        "lower",
        "age_upper",
        "age_lower",
    ]
    asdr_data = asdr_data[ordered_cols]
    cursor.executemany(insert_query, asdr_data.values.tolist())
    CODELOG.debug(f"uploaded {len(asdr_data)} lines of asdr data")


def load_asdr_to_t3(execution_context, data_access, location_and_children):
    """
    Upload to t3_model_version_asdr if it's not already there.
    """
    model_version_id = data_access.model_version_id
    gbd_round_id = data_access.gbd_round_id
    database = execution_context.parameters.database
    locations_with_asdr_in_t3 = _asdr_in_t3(execution_context, model_version_id)
    missing_from_t3 = set(location_and_children) - set(locations_with_asdr_in_t3)
    if missing_from_t3:
        CODELOG.info(
            f"""Uploading ASDR data for model_version_id
            {model_version_id} on '{database}'"""
        )
        asdr_data = get_asdr_data(gbd_round_id, data_access.decomp_step, list(missing_from_t3), data_access.with_hiv)

        with cursor(execution_context) as c:
            _upload_asdr_data_to_tier_3(gbd_round_id, c, model_version_id, asdr_data)

        return True
    else:
        CODELOG.info(
            f"""ASDR data for model_version_id {model_version_id}
            on '{database}' already exists, doing nothing."""
        )
        return False
