from textwrap import dedent
import warnings

from scipy.interpolate import InterpolatedUnivariateSpline, LSQBivariateSpline

import numpy as np
import pandas as pd

from cascade.input_data.db.mortality import get_frozen_cause_specific_mortality_data, normalize_mortality_data
from cascade.input_data.db.demographics import age_groups_to_ranges, get_years_from_lower_age_to_mean_age
from cascade.stats import meas_bounds_to_stdev

from cascade.core import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)


def _prepare_prevalence(observations):
    prevalence = observations.query("measure == 'prevalence' and sex_id in [1, 2, 3]")

    zero_means = prevalence["mean"] == 0
    if np.any(zero_means):
        not_quite_zero = 1e-6
        MATHLOG.debug(f"{np.sum(zero_means)} prevalence rows with 0 means. Changing them to {not_quite_zero}.")
        prevalence.loc[zero_means, "mean"] = not_quite_zero

    MATHLOG.debug(f"{len(prevalence)} prevalence observations available, excluding sex_id 4")
    return prevalence


def _collapse_times(csmr):
    MATHLOG.debug("Treating CSMR measurements over time ranges as measurements at the ranges midpoint")
    csmr["time"] = (csmr["time_lower"] + csmr["time_upper"]) / 2
    return csmr.drop(columns=["time_lower", "time_upper"])


def _collapse_ages_unweighted(csmr):
    csmr["age"] = (csmr["age_lower"] + csmr["age_upper"]) / 2
    return csmr


def _collapse_ages_weighted(execution_context, csmr):
    """This converts age ranges into a single point which is the average age of
    an individual within the range based on mortality estimates from GBD.
    """
    mean_years = get_years_from_lower_age_to_mean_age(execution_context)
    mean_years = mean_years.rename(columns={"mean": "age", "location_id": "node_id", "year_id": "time_lower"})
    csmr = csmr.merge(mean_years, on=["age_group_id", "node_id", "sex_id", "time_lower"])
    csmr["age"] += csmr["age_lower"]
    return csmr


def _prepare_csmr(execution_context, csmr, use_weighted_age_group_midpoints=False):
    MATHLOG.debug("Preparing CSMR data from GBD")
    csmr = csmr.rename(columns={"location_id": "node_id"})

    MATHLOG.debug("Assigning standard error from measured upper and lower.")
    csmr = meas_bounds_to_stdev(csmr)

    null_means = csmr["mean"].isnull()
    if np.any(null_means):
        MATHLOG.warn(
            f"{np.sum(null_means)} CSMR rows with NaN means in EMR calculation. "
            "Removing them but there may be something wrong"
        )
        csmr = csmr.loc[~null_means]

    csmr = age_groups_to_ranges(execution_context, csmr, keep_age_group_id=True)
    if use_weighted_age_group_midpoints:
        MATHLOG.debug("Treating CSMR measurements over age ranges as measurements at the mortality weighted midpoint")
        csmr = _collapse_ages_weighted(execution_context, csmr)
    else:
        MATHLOG.debug("Treating CSMR measurements over age ranges as measurements at the unweighted midpoint")
        csmr = _collapse_ages_unweighted(csmr)
    csmr = csmr.drop(columns=["age_lower", "age_upper"])

    csmr = _collapse_times(csmr)

    return csmr


def _make_interpolators(csmr):
    """ Calculate interpolations over CSMR so we can integrate the CSMR value
    over the age/time range of each prevalence measurement.
    """
    mean = {}
    stderr = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="scipy.interpolate",
                                category=UserWarning, message=r"\s*The coefficient")
        mean["both"] = LSQBivariateSpline(
            csmr.age, csmr.time, csmr["mean"], sorted(csmr.age.unique()), sorted(csmr.time.unique()), kx=1, ky=1
        )
        stderr["both"] = LSQBivariateSpline(
            csmr.age, csmr.time, csmr.standard_error, sorted(csmr.age.unique()), sorted(csmr.time.unique()), kx=1, ky=1
        )

    csmr_by_age = csmr.sort_values("age").groupby("age").mean().reset_index()
    mean["age"] = InterpolatedUnivariateSpline(csmr_by_age.age, csmr_by_age["mean"], k=1)
    stderr["age"] = InterpolatedUnivariateSpline(csmr_by_age.age, csmr_by_age.standard_error, k=1)

    csmr_by_time = csmr.sort_values("time").groupby("time").mean().reset_index()
    mean["time"] = InterpolatedUnivariateSpline(csmr_by_time.time, csmr_by_time["mean"], k=1)
    stderr["time"] = InterpolatedUnivariateSpline(csmr_by_time.time, csmr_by_time.standard_error, k=1)

    return mean, stderr


def _emr_from_sex_and_node_specific_csmr_and_prevalence(csmr, prevalence):
    mean_interp, stderr_interp = _make_interpolators(csmr)

    emr = prevalence[["age_lower", "age_upper", "time_lower", "time_upper", "sex_id", "node_id", "density", "weight"]]
    emr = emr.assign(measure="mtexcess")

    def emr_mean(prevalence_measurement):
        time_is_point = prevalence_measurement.time_lower == prevalence_measurement.time_upper
        age_is_point = prevalence_measurement.age_lower == prevalence_measurement.age_upper
        if (time_is_point and age_is_point):
            # No integral, just interpolate to a point
            csmr_mean = mean_interp["both"](prevalence_measurement.age_lower, prevalence_measurement.time_lower)
            csmr_stderr = stderr_interp["both"](prevalence_measurement.age_lower, prevalence_measurement.time_lower)
        elif time_is_point:
            # Integrate over age
            divisor = prevalence_measurement.age_upper - prevalence_measurement.age_lower

            csmr_mean = mean_interp["age"].integral(prevalence_measurement.age_lower, prevalence_measurement.age_upper)
            csmr_mean /= divisor

            csmr_stderr = stderr_interp["age"].integral(
                prevalence_measurement.age_lower, prevalence_measurement.age_upper
            )
            csmr_stderr /= divisor
        elif age_is_point:
            # Integrate over time
            divisor = prevalence_measurement.time_upper - prevalence_measurement.time_lower

            csmr_mean = mean_interp["time"].integral(
                prevalence_measurement.time_lower, prevalence_measurement.time_upper
            )
            csmr_mean /= divisor

            csmr_stderr = stderr_interp["time"].integral(
                prevalence_measurement.time_lower, prevalence_measurement.time_upper
            )
            csmr_stderr /= divisor
        else:
            # Integrate over both
            divisor = (prevalence_measurement.time_upper - prevalence_measurement.time_lower) * (
                prevalence_measurement.age_upper - prevalence_measurement.age_lower
            )

            csmr_mean = mean_interp["both"].integral(
                prevalence_measurement.age_lower,
                prevalence_measurement.age_upper,
                prevalence_measurement.time_lower,
                prevalence_measurement.time_upper,
            )
            csmr_mean /= divisor

            csmr_stderr = stderr_interp["both"].integral(
                prevalence_measurement.age_lower,
                prevalence_measurement.age_upper,
                prevalence_measurement.time_lower,
                prevalence_measurement.time_upper,
            )
            csmr_stderr /= divisor

        emr_mean = csmr_mean / prevalence_measurement["mean"]

        # Propagation of uncertainty for f = A/B gives a standard deviation of
        # sqrt( (sigma_a)^2 / A^2 + (sigma_b)^2 / B^2 - 2 (sigma_ab) / (A B) )
        # but we don't know the covariance sigma_ab, so we use an upper bound on the error by
        # assuming it's zero.
        emr_stderr = np.sqrt(
            (prevalence_measurement.standard_error / prevalence_measurement["mean"]) ** 2
            + (csmr_stderr / csmr_mean) ** 2
        )
        return pd.Series({"mean": float(emr_mean), "standard_error": float(emr_stderr)})

    return emr.merge(prevalence.apply(emr_mean, "columns"), left_index=True, right_index=True)


def add_emr_from_prevalence(model_context, execution_context):
    r"""Estimate excess mortality from the supplied prevalence measurements and
    cause specific mortality estimates from GBD using the formula:
    :math:`\chi=\frac{P}{{}_nm_x^c}
    """
    prevalence = _prepare_prevalence(model_context.input_data.observations)
    MATHLOG.debug("Calculating excess mortality using: EMR=CSMR/prevalence from {len(prevalence)} observations")
    csmr = normalize_mortality_data(get_frozen_cause_specific_mortality_data(
        execution_context, execution_context.parameters.model_version_id))
    csmr = _prepare_csmr(execution_context, csmr, model_context.policies["use_weighted_age_group_midpoints"])

    emr = _calculate_emr_from_csmr_and_prevalence(csmr, prevalence)

    if not emr.empty:
        model_context.input_data.observations = pd.concat([model_context.input_data.observations, emr])


def _calculate_emr_from_csmr_and_prevalence(all_csmr, all_prevalence):
    MATHLOG.debug(
        dedent(
            """
    For every sex/location combination in the prevalence data we construct a
    linear interpolation of CSMR over age and time and use the integral of that
    to estimate the mean CSMR in the age/time region of each prevalence
    measurement which is then used to calculate EMR.
    """
        )
    )
    excess_mortality_chunks = []
    non_interpolatable_groups = all_prevalence[["sex_id", "node_id"]].drop_duplicates()
    for (sex_id, node_id) in non_interpolatable_groups.values:
        csmr = all_csmr.query("sex_id == @sex_id and node_id == @node_id")
        prevalence = all_prevalence.query("sex_id == @sex_id and node_id == @node_id")
        if csmr.empty:
            MATHLOG.debug(
                f"No CSMR data for the {len(prevalence)} prevalence observations "
                f"with sex_id {sex_id} and location_id {node_id}. These points will not have EMR data."
            )
        else:
            emr = _emr_from_sex_and_node_specific_csmr_and_prevalence(csmr, prevalence)
            excess_mortality_chunks.append(emr)

    if excess_mortality_chunks:
        excess_mortality = pd.concat(excess_mortality_chunks)
        MATHLOG.debug(f"Calculated {len(excess_mortality)} EMR points from {len(all_prevalence)} prevalence points.")
        return excess_mortality
    else:
        return pd.DataFrame()
