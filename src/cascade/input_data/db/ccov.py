"""This module retrieves country covariates from the database.
"""

import logging

try:
    from db_queries import get_covariate_estimates
except ImportError:

    class DummyGetCovariateEstimates:
        def __getattr__(self, name):
            raise ImportError(f"Required package db_queries not found")

    get_covariate_estimates = DummyGetCovariateEstimates()


CODELOG = logging.getLogger(__name__)
MATHLOG = logging.getLogger(__name__)


def country_covariates(covariate_id, demographics):
    """Retrieve country covariates from the database.

    Args:
        covariate_id (int): id of the country covariate to retrieve
        demographics (Dictionary): demographic ids needed to select the data; for example,
            keys for location_ids, age_group_ids, year_ids, and sex_ids.  The values can
            be int or list of ints.

    Returns:
        Pandas DataFrame of Country Covariate Estimates

    """

    country_covariates = get_covariate_estimates(
        covariate_id,
        location_id=demographics["location_ids"],
        age_group_id=demographics["age_group_ids"],
        year_id=demographics["year_ids"],
        sex_id=demographics["sex_ids"])[
        ["covariate_id", "location_id", "age_group_id", "year_id", "sex_id", "mean_value"]]

    CODELOG.debug(
        f"Downloaded {len(country_covariates)} lines of country covariates data")

    return country_covariates
