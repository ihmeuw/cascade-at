"""This module retrieves country covariates from the database.
"""

import logging

import pandas as pd

try:
    from db_queries import get_covariate_estimates
except ImportError:

    class DummyGetCovariateEstimates:
        def __getattr__(self, name):
            raise ImportError(f"Required package db_queries not found")

    get_covariate_estimates = DummyGetCovariateEstimates()


CODELOG = logging.getLogger(__name__)
MATHLOG = logging.getLogger(__name__)


def country_covariates(execution_context, demographics):
    """Retrieve country covariates from the database.

    Args:
        execution_context (ExecutionContext): The context within which to make the query
        demographics (Dictionary): demographic ids needed to select the data; for example,
            keys for location_ids, age_group_ids, year_ids, and sex_ids.  The values can
            be int or list of ints.

    Returns:
        Pandas DataFrame of Country Covariate Estimates

    """

    country_covariate_ids = execution_context.parameters.country_covariate_ids

    country_covariates = list()

    for covariate_id in country_covariate_ids:

        country_covariates.append(get_covariate_estimates(
            covariate_id,
            location_id=demographics["location_ids"],
            age_group_id=demographics["age_group_ids"],
            year_id=demographics["year_ids"],
            sex_id=demographics["sex_ids"]
        ))

    ccov = pd.concat(country_covariates, ignore_index=True)[[
        "covariate_id", "location_id", "age_group_id", "year_id", "sex_id", "mean_value"]]

    CODELOG.debug(
        f"Downloaded {len(ccov)} lines of country covariates data")

    return ccov
