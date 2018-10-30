from scipy import stats
from scipy.interpolate import SmoothBivariateSpline

import numpy as np
import pandas as pd

from cascade.input_data.db.mortality import get_cause_specific_mortality_data
from cascade.input_data.db.demographics import age_groups_to_ranges


def _standard_error_from_uncertainty(p, lower, upper, confidence=0.95):
    quantile = 1 - (1 - confidence) / 2
    return np.amax([upper - p, p - lower], 0) / stats.norm.ppf(quantile)


def emr_from_prevalence(execution_context, observations):
    all_prevalence = observations.query("measure == 'prevalence'")

    # Exclude very small points which would produce impossibly high excess mortality values
    all_prevalence = all_prevalence.query("mean >= 1/100000")

    all_csmr = get_cause_specific_mortality_data(execution_context)
    all_csmr = all_csmr.rename(columns={"location_id": "node_id"})

    # Exclude 0 and NaN csmr values which will produce NaN outputs
    # FIXME: Do these actually exist?
    all_csmr = all_csmr.query("meas_value > 0")
    all_csmr = all_csmr.loc[
        all_csmr.meas_value.notnull() & all_csmr.meas_lower.notnull() & all_csmr.meas_upper.notnull()
    ]

    all_csmr["standard_error"] = _standard_error_from_uncertainty(
        all_csmr.meas_value, all_csmr.meas_lower, all_csmr.meas_upper
    )
    all_csmr = all_csmr.drop(columns=["meas_lower", "meas_upper"])

    # CSMR is actually always on a single point in time
    all_csmr["time"] = all_csmr.time_lower
    all_csmr = all_csmr.drop(columns=["time_lower", "time_upper"])

    # Collapse CSMR to the midpoints of age ranges
    all_csmr = age_groups_to_ranges(execution_context, all_csmr)
    all_csmr["age"] = (all_csmr["age_lower"] + all_csmr["age_upper"]) / 2
    all_csmr = all_csmr.drop(columns=["age_lower", "age_upper"])

    excess_mortality_chunks = []
    for sex_id in all_prevalence.sex_id.unique():
        for node_id in all_prevalence.node_id.unique():
            csmr = all_csmr.query("sex_id == @sex_id and node_id == @node_id")
            if not csmr.empty:
                prevalence = all_prevalence.query("sex_id == @sex_id and node_id == @node_id")
                csmr_mean_interpolator = SmoothBivariateSpline(csmr.age, csmr.time, csmr.meas_value, kx=1, ky=1)
                csmr_stderr_interpolator = SmoothBivariateSpline(csmr.age, csmr.time, csmr.standard_error, kx=1, ky=1)
                emr = prevalence[["age_lower", "age_upper", "time_lower", "time_upper", "sex_id", "node_id"]]

                def emr_mean(row):
                    time_lower = row.time_lower
                    time_upper = row.time_upper
                    if time_lower == time_upper:
                        # FIXME demographer notation?
                        time_upper += 1
                    age_lower = row.age_lower
                    age_upper = row.age_upper
                    if age_lower == age_upper:
                        # FIXME demographer notation?
                        age_upper += 1

                    csmr_mean = csmr_mean_interpolator.integral(age_lower, age_upper, time_lower, time_upper)
                    csmr_stderr = csmr_stderr_interpolator.integral(age_lower, age_upper, time_lower, time_upper)

                    emr_mean = row["mean"] / csmr_mean
                    emr_stderr = np.sqrt((row.standard_error / row["mean"]) ** 2 + (csmr_stderr / csmr_mean) ** 2)
                    return pd.Series({"mean": emr_mean, "standard_error": emr_stderr})

                emr = emr.merge(prevalence.apply(emr_mean, "columns"), left_index=True, right_index=True)
                excess_mortality_chunks.append(emr)

    return pd.concat(excess_mortality_chunks).dropna()
