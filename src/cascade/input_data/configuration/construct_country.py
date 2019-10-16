"""
Takes settings and creates a CovariateRecords object
"""

from collections.__init__ import namedtuple

import intervals as it
import numpy as np
import pandas as pd
from scipy import spatial
from scipy.interpolate import griddata

from cascade.core import getLoggers
from cascade.core.db import ezfuncs
from cascade.input_data import InputDataError

CODELOG, MATHLOG = getLoggers(__name__)

FEMALE = 2
MALE = 1
BOTH = 3
UNDEFINED = 4


def reference_value_for_covariate_mean_all_values(cov_df):
    """
    Strategy for choosing reference value for country covariate.
    This one takes the mean of all incoming covariate values.
    """
    return float(cov_df["mean_value"].mean())


def covariate_to_measurements_nearest_favoring_same_year(measurements, sex, covariates):
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
        sex (pd.Series): The sex covariate as [-0.5, 0, 0.5].
        covariates (pd.DataFrame):
            Columns include ``age_lower``, ``age_upper``, ``time_lower``,
            ``time_upper``, and ``value``.

    Returns:
        pd.Series: One row for every row in the measurements.
    """
    if measurements is None:
        return
    CODELOG.debug(f"measurements columns {measurements.columns}")
    # Rescaling the age means that the nearest age within the year
    # will always be closer than the nearest time across a full year.
    tree = spatial.KDTree(
        list(
            zip(
                covariates[["age_lower", "age_upper"]].mean(axis=1) / 240,
                covariates[["time_lower", "time_upper"]].mean(axis=1),
                covariates["sex_id"],
            )
        )
    )
    _, indices = tree.query(
        list(
            zip(
                measurements[["age_lower", "age_upper"]].mean(axis=1) / 240,
                measurements[["time_lower", "time_upper"]].mean(axis=1),
                sex,
            )
        )
    )
    return pd.Series(covariates.iloc[indices]["mean_value"].values, index=measurements.index)


def convert_gbd_ids_to_dismod_values(with_ids_df, age_groups_df):
    """
    Converts ``age_group_id`` into ``age_lower`` and ``age_upper`` and
    ``year_id`` into ``time_lower`` and ``time_upper``. This treats the year
    as a range from start of year to start of the next year.

    Args:
        with_ids_df (pd.DataFrame): Has ``age_group_id`` and ``year_id``.
        age_groups_df (pd.DataFrame): Has columns ``age_group_id``,
            ``age_group_years_start``, and ``age_group_years_end``.

    Returns:
        pd.DataFrame: New pd.DataFrame with four added columns and in the same
            order as the input dataset.
    """
    original_order = with_ids_df.copy()
    # This "original index" guarantees that the order of the output dataset
    # and the index of the output dataset match that of with_ids_df, because
    # the merge reorders everything, including creation of a new index.
    original_order["original_index"] = original_order.index
    merged = pd.merge(original_order, age_groups_df, on="age_group_id", sort=False)
    if len(merged) != len(with_ids_df):
        # This is a fault in the input data.
        incoming_age_group_ids = set(with_ids_df.age_group_id.unique())
        missing = incoming_age_group_ids - set(age_groups_df.age_group_id.unique())
        raise InputDataError(
            f"Not all age group ids from observations are found in the age group list "
            f"missing age groups {missing} other age ids in bundle {list(sorted(incoming_age_group_ids))} "
            f"Of the original {len(with_ids_df)} records, {len(merged)} had known ids.")
    reordered = merged.sort_values(by="original_index").reset_index()
    reordered["time_lower"] = reordered["year_id"]
    reordered["time_upper"] = reordered["year_id"] + 1
    dropped = reordered.drop(columns=["age_group_id", "year_id", "original_index", "index"])
    return dropped.rename(columns={"age_group_years_start": "age_lower", "age_group_years_end": "age_upper"})


def compute_covariate_age_time_dimensions(covariates):
    """
    Determines if the input covariate data is by_age and/or by_time.

    Returns:
        namedtuple: with bool fields age_1d and time_1d
    """

    covar_at_dims = namedtuple("Result", ["age_1d", "time_1d"])

    covar_at_dims.age_1d = len(covariates[["age_lower", "age_upper"]].drop_duplicates()) > 1

    covar_at_dims.time_1d = len(covariates[["time_lower", "time_upper"]].drop_duplicates()) > 1

    return covar_at_dims


def compute_covariate_age_interval(covariates):
    """
    Create an interval expressing all the ages in the covariates data frame.
    If the covariates have age ranges like: [5,10], [10,15], [30,35], [35-80],
    the overall age interval would be: [5,15], [30,80]
    This allows checks like: 9.5 in age_interval (yes), 20.25 in age_interval (no),
    0.667 in age_interval (no), 95 in age_interval (no), 50.1 in age_interval (yes)

    Returns:
        Interval: of ages in the covariates data frame
    """
    age_interval = it.empty()

    age_groups = covariates[["age_lower", "age_upper"]].drop_duplicates()

    for index, row in age_groups.iterrows():
        age_interval = age_interval | it.closed(row["age_lower"], row["age_upper"])

    return age_interval


def assign_interpolated_covariate_values(measurements, covariates, is_binary):
    """
    Compute a column of covariate values to assign to the measurements.

    Args:
        measurements (pd.DataFrame):
            Columns include ``age_lower``, ``age_upper``, ``time_lower``,
            ``time_upper``. All others are ignored.
        covariates (pd.DataFrame):
            Columns include ``age_lower``, ``age_upper``, ``time_lower``,
            ``time_upper``, ``sex``, and ``value``.
        is_binary (bool): Whether this is a binary covariate.

    Returns:
        pd.Series: One row for every row in the measurements.
    """
    is_binary = bool(is_binary)  # In case it's a Numpy bool

    # is the covariate by_age, does it have multiple years?
    covar_at_dims = compute_covariate_age_time_dimensions(covariates)
    # identify the overall interval for the covariate ages, could have middle gaps
    covar_age_interval = compute_covariate_age_interval(covariates)
    # find a matching covariate value for each measurement
    measurements_with_age, covariate_column = compute_interpolated_covariate_values_by_sex(
        measurements, covariates, covar_at_dims)
    # if the covariate is binary, make sure the assigned values are only 0 or 1
    if is_binary:
        covariate_column[covariate_column <= .5] = 0
        covariate_column[covariate_column > .5] = 1

    # set missings using covar_age_interval
    meas_mean_age_in_age_interval = [i in covar_age_interval for i in measurements_with_age["avg_age"]]
    covariate_column = pd.Series(np.where(meas_mean_age_in_age_interval, covariate_column, np.nan))

    return covariate_column


def get_measurement_data_by_sex(measurements):
    """Split the measurement data by sex values found in the measurements.

    Args:
        measurements (pandas.DataFrame): data for a specific measurement

    Returns:
        dict: possible sex keys (1, 2, 3, 4) and measurement data as values
    """
    measurements_by_sex = {}

    for sex in (FEMALE, MALE, BOTH, UNDEFINED):
        measurements_sex = measurements[measurements["sex_id"] == sex]

        if not measurements_sex.empty:
            measurements_by_sex[sex] = measurements_sex

    return measurements_by_sex


def compute_interpolated_covariate_values_by_sex(
        measurements, covariates, covar_at_dims):
    """
    Use the measurements data by sex as input to the corresponding interpolator
    to assign a covariate value to the measurement.

    Returns:
        pd.Series: One row for every row in the measurements.
    """

    covariates = covariates.assign(
        avg_age=covariates[["age_lower", "age_upper"]].mean(axis=1),
        avg_time=covariates[["time_lower", "time_upper"]].mean(axis=1)
    )

    covariates_by_sex = get_covariate_data_by_sex(covariates)

    measurements = measurements.assign(
        avg_age=measurements[["age_lower", "age_upper"]].mean(axis=1),
        avg_time=measurements[["time_lower", "time_upper"]].mean(axis=1)
    )

    measurements_by_sex = get_measurement_data_by_sex(measurements)

    cov_col = []
    cov_index = []

    for sex, measurements_sex in measurements_by_sex.items():

        cov_index = cov_index + list(measurements_sex.index)
        meas_sex_new_index = measurements_sex.reset_index()

        covariates_sex = covariates_by_sex[sex]

        # covariate is by_age and "by_time"
        if covar_at_dims.age_1d and covar_at_dims.time_1d:

            covariate_sex = griddata((covariates_sex["avg_age"], covariates_sex["avg_time"]),
                                     covariates_sex["mean_value"],
                                     (meas_sex_new_index["avg_age"], meas_sex_new_index["avg_time"]))

        # covariate is "by_time", but not by_age
        elif not covar_at_dims.age_1d and covar_at_dims.time_1d:

            covariate_sex = griddata((covariates_sex["avg_time"],),
                                     covariates_sex["mean_value"],
                                     (meas_sex_new_index["avg_time"],))

        # covariate is by_age, but not "by_time"
        elif covar_at_dims.age_1d and not covar_at_dims.time_1d:

            covariate_sex = griddata((covariates_sex["avg_age"],),
                                     covariates_sex["mean_value"],
                                     (meas_sex_new_index["avg_age"],))
        else:
            raise RuntimeError(f"Covariate sex neither by age nor time {covar_at_dims}.")

        cov_col = cov_col + list(covariate_sex)

    covariate_column = pd.Series(cov_col, index=cov_index).sort_index()

    return measurements, covariate_column


def check_binary_covariates(execution_context, covariate_ids):
    """Check the dichotomous value from shared.covariate to check if the covariate is binary.
    If it is, make sure the assigned value is only 0 or 1.
    """
    is_binary = dict()
    for covariate_id in covariate_ids:
        result_df = ezfuncs.query(
            "select dichotomous from shared.covariate where covariate_id=:covid",
            parameters=dict(covid=covariate_id),
            conn_def=execution_context.parameters.database)
        is_binary[covariate_id] = (result_df.dichotomous[0] == 1)
    return is_binary


def check_and_handle_binary_covariate(covariate_id, covariate_column, execution_context):
    """Check the dichotomous value from shared.covariate to check if the covariate is binary.
    If it is, make sure the assigned value is only 0 or 1.
    """
    result_df = ezfuncs.query(
        "select dichotomous from shared.covariate where covariate_id=:covid",
        parameters=dict(covid=covariate_id),
        conn_def=execution_context.parameters.database)

    if result_df.dichotomous[0] == 1:
        covariate_column[covariate_column <= .5] = 0
        covariate_column[covariate_column > .5] = 1

    return covariate_column


def get_covariate_data_by_sex(covariates):
    """Covariate data is expected to have sex values for (female, male) or for (both).
       This checks which are present, and selects the applicable covariate data
       for each sex type.  If (female, male) are present, then both is computed as the
       average.  If (both) is present, it is assigned to female and male.

    Args:
        covariates (pandas.DataFrame): data for a specific covariate_id

    Returns:
        dict: sex keys (1, 2, 3) and covariate data as values
    """

    covariates_by_sex = {}
    sex_values = set(covariates["sex_id"].unique())

    if not sex_values.difference({BOTH, UNDEFINED}):
        covariates_by_sex[FEMALE] = covariates
        covariates_by_sex[MALE] = covariates
        covariates_by_sex[BOTH] = covariates
    elif sex_values == {MALE, FEMALE}:
        covariates_by_sex[FEMALE] = covariates[covariates["sex_id"] == FEMALE]
        covariates_by_sex[MALE] = covariates[covariates["sex_id"] == MALE]
        covariates_both = covariates_by_sex[FEMALE].merge(
            covariates_by_sex[MALE],
            on=["age_lower", "age_upper", "time_lower", "time_upper"],
            how="inner")
        covariates_both["mean_value"] = covariates_both[
            ["mean_value_x", "mean_value_y"]].mean(axis=1)
        covariates_both["avg_age"] = covariates_both["avg_age_x"]
        covariates_both["avg_time"] = covariates_both["avg_time_x"]
        covariates_by_sex[BOTH] = covariates_both
    else:
        raise ValueError(
            f"Unexpected values for sex_id in covariates data.  Expected (3,4) or (1,2), found {sex_values}"
        )

    return covariates_by_sex
