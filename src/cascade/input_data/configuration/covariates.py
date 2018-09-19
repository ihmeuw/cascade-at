"""
Decides how to assign covariates for IHME's EpiViz.
"""
import logging

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy import spatial

from cascade.input_data.db.ccov import country_covariates


MATHLOG = logging.getLogger(__name__)


def identity(x): return x


def squared(x): return np.power(x, 2)


def scale1000(x): return x * 1000


COVARIATE_TRANSFORMS = {
    0: identity,
    1: np.log,
    2: logit,
    3: squared,
    4: np.sqrt,
    5: scale1000
}
"""
These functions transform covariate data, as specified in EpiViz.
"""


def unique_covariate_transform(context):
    yield [(26, [1, 3])]


def assign_covariates(input_data):
    """
    The EpiViz interface allows assigning a covariate with a transformation
    to a specific target (rate, measure value, measure standard deviation).
    It is even the case, that the same covariate, say income, can be applied
    without transformation to iota on one smoothing and applied without
    transformation to chi with a *different smoothing*.
    Therefore, there can be multiple covariate columns built from the same
    covariate, one for each kind of transformation required.
    """
    locations = [6, 102]
    covariate_map = {}

    # This walks through all unique combinations of covariates and their
    # transformations. Then, later, we apply them to particular target
    # rates, meas_values, meas_stds.
    for country_covariate_id, transforms in unique_covariate_transform(input_data):
        demographics = dict(
            age_group_ids="all",
            year_ids="all",
            sex_ids="all",
            location_ids=locations,
        )
        ccov_df = country_covariates(country_covariate_id, demographics)
        covariate_name = ccov_df.loc[0]["covariate_short_name"]

        # Decide how to take the given data and extend / subset / interpolate.

        for transform in transforms:
            # This happens per application to integrand.
            settings_transform = COVARIATE_TRANSFORMS[transform]
            transform_name = settings_transform.__name__
            MATHLOG.info(f"Transforming {covariate_name} with {transform_name}")
            name = f"{covariate_name}_{transform_name}"
            covariate_map[(covariate_name, transform)] = name
            ccov_df["mean_value"] = settings_transform(ccov_df.mean_value)
            # Now attach the column to the observations.

    def column_id_func(covariate_name, transformation_id):
        return covariate_map[(covariate_name, transformation_id)]

    return column_id_func


def create_covariate_multipliers(context, column_id_func):
    # Assumes covariates exist.
    pass


def covariate_to_measurements_dummy(measurements, covariate):
    """
    Given a covariate that might not cover all of the age and time range
    of the measurements select a covariate value for each measurement.
    This version assigns 1.0 to every measurement.

    Args:
        measurements (pd.DataFrame):
            Columns include ``age_lower``, ``age_upper``, ``time_lower``,
            ``time_upper``. All others are ignored.
        covariate (pd.DataFrame):
            Columns include ``age_lower``, ``age_upper``, ``time_lower``,
            ``time_upper``, and ``value``.

    Returns:
        pd.Series: One row for every row in the measurements.
    """
    return pd.Series(np.ones((len(measurements),), dtype=np.float))


def covariate_to_measurements_nearest_favoring_same_year(measurements, covariates):
    """
    Given a covariate that might not cover all of the age and time range
    of the measurements select a covariate value for each measurement.
    This version chooses the covariate value whose mean age and time
    is closest to the mean age and time of the measurement in the same
    year. If that isn't found, it picks the covariate that is closest
    in age and time in the nearest year. In the case of a tie for distance,
    it averages.

    Args:
        measurements (pd.DataFrame):
            Columns include ``age_lower``, ``age_upper``, ``time_lower``,
            ``time_upper``. All others are ignored.
        covariate (pd.DataFrame):
            Columns include ``age_lower``, ``age_upper``, ``time_lower``,
            ``time_upper``, and ``value``.

    Returns:
        pd.Series: One row for every row in the measurements.
    """
    # Rescaling the age by 120 means that the nearest age within the year
    # will always be closer than the nearest time across a full year.
    tree = spatial.KDTree(list(zip(
        covariates[["age_lower", "age_upper"]].mean(axis=1) / 120,
        covariates[["time_lower", "time_upper"]].mean(axis=1)
    )))
    _, indices = tree.query(list(zip(
        measurements[["age_lower", "age_upper"]].mean(axis=1) / 120,
        measurements[["time_lower", "time_upper"]].mean(axis=1)
    )))
    return pd.Series(covariates.iloc[indices]["value"].values, index=measurements.index)
