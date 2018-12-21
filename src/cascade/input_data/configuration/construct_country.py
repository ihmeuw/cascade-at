"""
Takes settings and creates a CovariateRecords object
"""

from collections.__init__ import defaultdict, namedtuple

import numpy as np
import pandas as pd
from scipy import spatial
from scipy.interpolate import griddata

import intervals as it

from cascade.core import getLoggers
from cascade.core.db import ezfuncs
from cascade.input_data import InputDataError
from cascade.input_data.db.country_covariates import country_covariates
from cascade.input_data.db.demographics import get_all_age_spans
from cascade.input_data.configuration.construct_study import CovariateRecords

CODELOG, MATHLOG = getLoggers(__name__)

FEMALE = -0.5
MALE = 0.5
BOTH = 0


def unique_country_covariate_transform(configuration):
    """
    Iterates through all covariate IDs, including the list of ways to
    transform them, because each transformation is its own column for Dismod.
    This is used by ``assign_covariates``.
    """
    seen_covariate = defaultdict(set)
    if configuration.country_covariate:
        for covariate_configuration in configuration.country_covariate:
            seen_covariate[covariate_configuration.country_covariate_id].add(
                covariate_configuration.transformation)

    for cov_id, cov_transformations in seen_covariate.items():
        yield cov_id, list(sorted(cov_transformations))


def unique_country_covariate(configuration):
    """
    Iterates through all covariate IDs. This is used to create the
    initial CovariateRecords object. This is before the special covariates
    are set.
    """
    seen_covariate = set()
    if configuration.country_covariate:
        for covariate_configuration in configuration.country_covariate:
            seen_covariate.add(covariate_configuration.country_covariate_id)
    yield from sorted(seen_covariate)  # Sorted for stability.


def covariate_records_from_settings(model_context, execution_context,
                                    configuration, study_covariate_records):
    """
    The important choices are assignment of covariates to observations and
    integrands by
    :py:func:`covariate_to_measurements_nearest_favoring_same_year`
    and how reference values are chosen by
    :py:func:`reference_value_for_covariate_mean_all_values`.

    Args:
        model_context: The model context must have ``average_integrand_cases``
            a bundle as "observations", and a location id.
        execution_context: Context for execution of this program.
        configuration: Settings from EpiViz.
        study_covariate_records (CovariateRecords): Study covariates which
            have the sex column for measurements and average integrand cases.

    Returns:
        CovariateRecords object, completely filled out.
    """
    records = CovariateRecords("country")
    measurements = model_context.input_data.observations
    avgint = model_context.average_integrand_cases

    measurement_columns = list()
    avgint_columns = list()

    age_groups = get_all_age_spans()
    for covariate_id in unique_country_covariate(configuration):
        demographics = dict(
            age_group_ids="all", year_ids="all", sex_ids="all",
            location_ids=[model_context.parameters.location_id]
        )
        ccov_df = country_covariates(covariate_id, demographics,
                                     execution_context.parameters.gbd_round_id)
        covariate_name = ccov_df.loc[0]["covariate_name_short"]
        records.id_to_name[covariate_id] = covariate_name
        # There is an order dependency from whether we interpolate before we
        # transform or transform before we interpolate.
        # Decide how to take the given data and extend / subset / interpolate.
        ccov_ranges_df = convert_gbd_ids_to_dismod_values(ccov_df, age_groups)

        MATHLOG.info(f"Adding {covariate_name} using "
                     f"covariate_to_measurements_nearest_favoring_same_year()")
        if measurements is not None:
            observations_column = assign_interpolated_covariate_values(measurements, ccov_ranges_df, execution_context)
            observations_column.name = covariate_name
        else:
            observations_column = None

        if avgint is not None:
            avgint_column = assign_interpolated_covariate_values(measurements, ccov_ranges_df, execution_context)
            avgint_column.name = covariate_name
        else:
            avgint_column = None
        reference = reference_value_for_covariate_mean_all_values(ccov_df)
        records.id_to_reference[covariate_id] = reference
        measurement_columns.append(observations_column)
        avgint_columns.append(avgint_column)

    if all(isinstance(mmc, pd.Series) for mmc in measurement_columns) and measurement_columns:
        records.measurements = pd.concat(measurement_columns, axis=1)
    elif measurements is not None:
        records.measurements = pd.DataFrame(index=measurements.index)
    else:
        records.measurements = pd.DataFrame()

    if all(isinstance(aac, pd.Series) for aac in avgint_columns) and avgint_columns:
        records.average_integrand_cases = pd.concat(avgint_columns, axis=1)
    elif records.average_integrand_cases is not None:
        records.average_integrand_cases = pd.DataFrame(index=avgint.index)
    else:
        records.average_integrand_cases = pd.DataFrame()
    return records


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
                covariates["x_sex"],
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
    Also converts sex_id=[1, 2, 3] into x_sex=[0.5, -0.5, 0].

    Args:
        with_ids_df (pd.DataFrame): Has ``age_group_id`` and ``year_id``.
        age_groups_df (pd.DataFrame): Has columns ``age_group_id``,
            ``age_group_years_start``, and ``age_group_years_end``.

    Returns:
        pd.DataFrame: New pd.DataFrame with four added columns and in the same
            order as the input dataset.
    """
    sex_df = pd.DataFrame(dict(x_sex=[-0.5, 0, 0.5], sex_id=[2, 3, 1]))
    original_order = with_ids_df.copy()
    # This "original index" guarantees that the order of the output dataset
    # and the index of the output dataset match that of with_ids_df, because
    # the merge reorders everything, including creation of a new index.
    original_order["original_index"] = original_order.index
    aged = pd.merge(original_order, age_groups_df, on="age_group_id", sort=False)
    merged = pd.merge(aged, sex_df, on="sex_id")
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
    MATHLOG.info(f"Conversion of bundle assumes demographic notation for years, "
                 f"so it adds a year to time_upper.")
    reordered["time_upper"] = reordered["year_id"] + 1
    dropped = reordered.drop(columns=["age_group_id", "year_id", "original_index"])
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


def assign_interpolated_covariate_values(measurements, covariates, execution_context):
    """
    Compute a column of covariate values to assign to the measurements.

    Args:
        measurements (pd.DataFrame):
            Columns include ``age_lower``, ``age_upper``, ``time_lower``,
            ``time_upper``. All others are ignored.
        covariates (pd.DataFrame):
            Columns include ``age_lower``, ``age_upper``, ``time_lower``,
            ``time_upper``, ``sex``, and ``value``.
        execution_context: Context for execution of this program.

    Returns:
        pd.Series: One row for every row in the measurements.
    """

    # is the covariate by_age, does it have multiple years?
    covar_at_dims = compute_covariate_age_time_dimensions(covariates)

    # identify the overall interval for the covariate ages, could have middle gaps
    covar_age_interval = compute_covariate_age_interval(covariates)

    # find a matching covariate value for each measurement
    covariate_column = compute_interpolated_covariate_values_by_sex(
        measurements, covariates, covar_at_dims)

    # if the covariate is binary, make sure the assigned values are only 0 or 1
    covariate_id = covariates.loc[0, "covariate_id"]
    covariate_column = check_and_handle_binary_covariate(covariate_id, covariate_column,
                                                         execution_context)

    # set missings using covar_age_interval
    meas_mean_age_in_age_interval = [i in covar_age_interval for i in measurements["avg_age"]]
    covariate_column = pd.Series(np.where(meas_mean_age_in_age_interval, covariate_column, np.nan))

    return covariate_column


def get_measurement_data_by_sex(measurements):
    """Split the measurement data by sex values found in the measurements.

    Args:
        measurements (pandas.DataFrame): data for a specific measurement

    Returns:
        dict: possible sex keys (-0.5, 0, 0.5) and measurement data as values
    """
    measurements_by_sex = {}

    for sex in (FEMALE, MALE, BOTH):

        measurements_sex = measurements[measurements["x_sex"] == sex]

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

    covariates["avg_age"] = covariates[["age_lower", "age_upper"]].mean(axis=1)
    covariates["avg_time"] = covariates[["time_lower", "time_upper"]].mean(axis=1)

    covariates_by_sex = get_covariate_data_by_sex(covariates)

    measurements["avg_age"] = measurements[["age_lower", "age_upper"]].mean(axis=1)
    measurements["avg_time"] = measurements[["time_lower", "time_upper"]].mean(axis=1)

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

        cov_col = cov_col + list(covariate_sex)

    covariate_column = pd.Series(cov_col, index=cov_index).sort_index()

    return covariate_column


def check_and_handle_binary_covariate(covariate_id, covariate_column, execution_context):
    """Check the dichotomous value from shared.covariate to check if the covariate is binary.
    If it is, make sure the assigned value is only 0 or 1.
    """
    result_df = ezfuncs.query(f"select dichotomous from shared.covariate where covariate_id={covariate_id}",
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
        dict: sex keys (-0.5, 0, 0.5) and covariate data as values
    """

    covariates_by_sex = {}
    sex_values = covariates["x_sex"].unique()

    if (len(sex_values) == 1) and (0 in sex_values):
        covariates_by_sex[FEMALE] = covariates
        covariates_by_sex[MALE] = covariates
        covariates_by_sex[BOTH] = covariates
    elif (len(sex_values) == 2) and (-0.5 in sex_values) and (0.5 in sex_values):
        covariates_by_sex[FEMALE] = covariates[covariates["x_sex"] == FEMALE]
        covariates_by_sex[MALE] = covariates[covariates["x_sex"] == MALE]
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
        raise ValueError(f"Unexpected values for x_sex in covariates data.  Expected 3 or (1,2), found {sex_values}")

    return covariates_by_sex
