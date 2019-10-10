from cascade.dismod.constants import DensityEnum
from cascade.core import getLoggers

import numpy as np
from scipy import stats

CODELOG, MATHLOG = getLoggers(__name__)


def meas_bounds_to_stdev(df):
    r"""
    Given data that includes a measurement upper bound and measurement lower
    bound, assume those are 95% confidence intervals. Convert them to
    standard error using:

    .. math::

        \mbox{stderr} = \frac{\mbox{upper} - \mbox{lower}}{2 1.96}

    Standard errors become Gaussian densities.
    Replace any zero values with :math:`10^{-9}`.
    """
    MATHLOG.debug("Assigning standard error from measured upper and lower.")
    bad_rows = np.sum((df.meas_lower > df.meas_value) | (df.meas_value > df.meas_upper))
    if bad_rows > 0:
        raise ValueError(f"Found data where meas_lower > meas_value or meas_upper < meas_value on {bad_rows} rows")
    df["standard_error"] = bounds_to_stdev(df.meas_lower, df.meas_upper)
    df["standard_error"] = df.standard_error.replace({0: 1e-9})
    df = df.rename(columns={"meas_value": "mean"})
    df["density"] = DensityEnum.gaussian
    df["weight"] = "constant"
    return df.drop(["meas_lower", "meas_upper"], axis=1)


def ess_to_stdev(mean, ess, proportion=False):
    """
    Takes an array of values (like mean), and
    effective sample size (ess) and transforms it
    to standard error assuming that the sample
    size is Poisson distributed. If you pass
    a proportion rather than a rate it will
    calculate the Wilson's Score Interval instead.

    Args:
        mean: pd.Series
        ess: pd.Series
        proportion: (bool) whether or not the measure is a proportion

    Returns:

    """
    count = mean * ess
    if proportion:
        # Calculate the Wilson's Score Interval
        z = stats.norm.ppf(quantile=0.975)
        std = np.sqrt(mean * (1 - mean) / ess + z**2 / (4 * ess**2))
    else:
        # Standard deviation for binomial with measure zero is approximately:
        std_0 = 1.0 / ess

        # When counts are >= 5, use standard deviation assuming that the
        # count is Poisson.
        # Note that when count = 5, mean is 5 / sample size.
        under_5 = count < 5
        std_5 = np.sqrt(5.0 / ess**2)

        std = np.sqrt(mean / ess)

        # For counts < 5, linearly interpolate between std_0 and std_5,
        # replacing the regular standard deviation.
        std[under_5] = ((5.0 - count[under_5]) * std_0[under_5] +
                        count[under_5] * std_5[under_5]) / 5.0
    return std


def bounds_to_stdev(lower, upper):
    r"""Given an upper and lower bound from a 95% confidence interval (CI),
    calculate a standard deviation.

    .. math::

        \mbox{stderr} = \frac{\mbox{upper} - \mbox{lower}}{2 1.96}

    Args:
        lower (pd.Series): The lower boound of the CI.
        upper (pd.Series): The upper boound of the CI.

    Returns:
        pd.Series: The standard deviation for that CI.
    """
    return (upper - lower) / (2 * 1.96)


def stdev_from_bundle_data(bundle_df):
    """
    Takes a bundle data frame and figures out the standard deviation
    from what is included in the bundle.

    There are other functions that will still use the bounds_to_stdev
    function rather than this because they're not dealing with
    bundle data. This function should only be used for bundle data.

    We prefer standard deviation (has_se), then uncertainty intervals (has_ui),
    then effective sample size (has_es), then sample size (has_ss).

    Args:
        bundle_df:

    Returns:

    """
    df = bundle_df.copy()
    standard_error = df.standard_error.values.copy()

    has_se = ~df.standard_error.values.isnull() & df.standard_error.values > 0
    MATHLOG.info(f"{sum(has_se)} rows have standard error.")
    has_ui = ~df.lower.values.isnull() & ~df.upper.values.isnull()
    MATHLOG.info(f"{sum(has_ui)} rows have uncertainty.")
    has_ess = ~df.effective_sample_size.values.isnull() & df.effective_sample_size.values > 0
    MATHLOG.info(f"{sum(has_ess)} rows have effective sample size.")
    has_ss = ~df.sample_size.values.isnull() & df.sample_size.values > 0
    MATHLOG.info(f"{sum(has_ss)} rows have sample size.")

    if sum(has_se | has_ui | has_ess | has_ss) < len(df):
        raise ValueError("Some rows have no valid uncertainty.")

    replace_ess_with_ss = ~has_ess & has_ss
    MATHLOG.info(f"{sum(replace_ess_with_ss)} rows will have their effective sample size filled by sample size.")
    replace_se_with_ui = ~has_se & has_ui
    MATHLOG.info(f"{sum(replace_se_with_ui)} rows will have their standard error filled by uncertainty intervals.")
    replace_se_with_ess = ~has_se & ~has_ui
    MATHLOG.info(f"{sum(replace_se_with_ess)} rows will have their standard error filled by effective sample size.")

    # Replace effective sample size with sample size
    df[replace_ess_with_ss, 'effective_sample_size'] = df[replace_ess_with_ss, 'sample_size']

    # Calculate standard deviation different ways (also
    stdev_from_bounds = bounds_to_stdev(lower=df.lower, upper=df.upper)
    stdev_from_es = ess_to_stdev(mean=df.mean, ess=df.effective_sample_size)

    # Use boolean arrays representing the pecking order to replacing standard error
    standard_error[replace_se_with_ui] = stdev_from_bounds[replace_se_with_ui]
    standard_error[replace_se_with_ess] = stdev_from_es[replace_se_with_ess]

    # Do a final check on standard error
    if (standard_error <= 0 | standard_error.isnull()).any():
        raise ValueError("There are still negative / 0 / or null values for standard error.")

    return standard_error
