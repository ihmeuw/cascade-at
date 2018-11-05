"""
Takes settings and creates a CovariateRecords object.
"""
from collections.__init__ import defaultdict

import pandas as pd
from scipy import spatial

from cascade.input_data import InputDataError
from cascade.input_data.db.country_covariates import country_covariates
from cascade.input_data.db.demographics import get_all_age_spans
from cascade.input_data.configuration.construct_study import CovariateRecords
from cascade.core import getLoggers

CODELOG, MATHLOG = getLoggers(__name__)


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

    age_groups = get_all_age_spans(execution_context)
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
            observations_column = covariate_to_measurements_nearest_favoring_same_year(
                measurements, study_covariate_records.measurements["sex"], ccov_ranges_df)
            observations_column.name = covariate_name
        else:
            observations_column = None

        if avgint is not None:
            avgint_column = covariate_to_measurements_nearest_favoring_same_year(
                avgint, study_covariate_records.average_integrand_cases["sex"], ccov_ranges_df)
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
