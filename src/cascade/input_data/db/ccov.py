"""This module retrieves country covariates from the database.
"""

from cascade.core.db import db_queries

from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)


def country_covariates(covariate_id, demographics, gbd_round_id):
    """Retrieve country covariates from the database. Covariates can have a
    lower value and an upper value, in addition to their mean. This returns
    only the mean of the covariate on each demographic interval.

    Args:
        covariate_id (int): id of the country covariate to retrieve
        demographics (Dictionary): demographic ids needed to select the data;
            for example, keys for location_ids, age_group_ids, year_ids, and
            sex_ids.  The values can be int or list of ints.
        gbd_round_id (int): The number indicating which version of
            the GBD for which to retrieve these covariates.

    Returns:
        pd.DataFrame: Columns are `covariate_id`, `covariate_name_short`,
            `location_id`, `age_group_id`, `year_id`, `sex_id`,
            and `mean_value`.

    """

    covariates_df = db_queries.get_covariate_estimates(
        covariate_id,
        location_id=demographics["location_ids"],
        age_group_id=demographics["age_group_ids"],
        year_id=demographics["year_ids"],
        sex_id=demographics["sex_ids"],
        gbd_round_id=gbd_round_id
    )[[
        "covariate_id", "covariate_name_short", "location_id", "age_group_id",
        "year_id", "sex_id", "mean_value"
    ]]

    CODELOG.debug(
        f"Downloaded {len(covariates_df)} lines of country covariates data")

    return covariates_df
