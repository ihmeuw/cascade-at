import numpy as np
from scipy import stats

from cascade_at.dismod.constants import DensityEnum
from cascade_at.core.log import get_loggers

LOG = get_loggers(__name__)


def meas_bounds_to_stdev(df):
    """
    Given data that includes a measurement upper bound and measurement lower
    bound, assume those are 95% confidence intervals. Convert them to
    standard error using.

    Standard errors become Gaussian densities.
    Replace any zero values with :math:`10^{-9}`.
    """
    LOG.debug("Assigning standard error from measured upper and lower.")
    bad_rows = np.sum((df.meas_lower > df.meas_value) | (df.meas_value > df.meas_upper))
    if bad_rows > 0:
        raise ValueError(f"Found data where meas_lower > meas_value or meas_upper < meas_value on {bad_rows} rows")
    df["standard_error"] = bounds_to_stdev(df.meas_lower, df.meas_upper)
    df["standard_error"] = df.standard_error.replace({0: 1e-9})
    df = df.rename(columns={"meas_value": "mean"})
    df["density"] = DensityEnum.gaussian
    df["weight"] = "constant"
    return df.drop(["meas_lower", "meas_upper"], axis=1)


def wilson_interval(prop, ess):
    """
    Calculate Wilson Score Interval
    Args:
        prop: proportion
        ess: effective sample size
    Returns:
    """
    z = stats.norm.ppf(q=0.975)
    return np.sqrt(prop * (1 - prop) / ess + z**2 / (4 * ess**2))


def ess_to_stdev(mean, ess, proportion=False):
    """
    Takes an array of values (like mean), and
    effective sample size (ess) and transforms it
    to standard error assuming that the sample
    size is Poisson distributed. (Unless you pass a small
    sample, in that case will interpolate between
    Binomial and Poisson SE).

    If you pass a proportion rather than a rate it will
    calculate the Wilson's Score Interval instead.

    Args:
        mean: pd.Series
        ess: pd.Series
        proportion: (bool) whether or not the measure is a proportion

    Returns:
    """
    if proportion:
        # Calculate the Wilson's Score Interval
        std = wilson_interval(prop=mean, ess=ess)
    else:
        count = mean * ess
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
    return (upper - lower) / (2 * stats.norm.ppf(q=0.975))


def check_crosswalk_version_uncertainty_columns(df):
    """
    Checks for the validity of bundle columns
    representing uncertainty. Returns 4
    boolean pd.Series that represent
    where to index to replace values.
    """
    has_se = (~df['standard_error'].isnull()) & (df['standard_error'] > 0)
    LOG.info(f"{sum(has_se)} rows have standard error.")
    has_ui = (~df['lower'].isnull()) & (~df['upper'].isnull())
    LOG.info(f"{sum(has_ui)} rows have uncertainty.")
    has_ess = (~df['effective_sample_size'].isnull()) & (df['effective_sample_size'] > 0)
    LOG.info(f"{sum(has_ess)} rows have effective sample size.")
    has_ss = (~df['sample_size'].isnull()) & (df['sample_size'] > 0)
    LOG.info(f"{sum(has_ss)} rows have sample size.")

    if sum(has_se | has_ui | has_ess | has_ss) < len(df):
        raise ValueError("Some rows have no valid uncertainty.")

    return has_se, has_ui, has_ess, has_ss


def stdev_from_crosswalk_version(crosswalk_version):
    """
    Takes a crosswalk version data frame and figures out the standard deviation
    from what is included in the bundle.
    There are other functions that will still use the bounds_to_stdev
    function rather than this because they're not dealing with
    crosswalk versions. This function should only be used for crosswalk versions.
    We prefer standard deviation (has_se), then uncertainty intervals (has_ui),
    then effective sample size (has_es), then sample size (has_ss).
    Args:
        crosswalk_version (pd.DataFrame)
    Returns:
    """
    df = crosswalk_version.copy()
    standard_error = df['standard_error'].copy()

    has_se, has_ui, has_ess, has_ss = check_crosswalk_version_uncertainty_columns(df)

    replace_ess_with_ss = ~has_ess & has_ss
    LOG.info(f"{sum(replace_ess_with_ss)} rows will have their effective sample size filled by sample size.")
    replace_se_with_ui = ~has_se & has_ui
    LOG.info(f"{sum(replace_se_with_ui)} rows will have their standard error filled by uncertainty intervals.")
    replace_se_with_ess = ~has_se & ~has_ui
    LOG.info(f"{sum(replace_se_with_ess)} rows will have their standard error filled by effective sample size.")

    # Replace effective sample size with sample size
    df.loc[replace_ess_with_ss, 'effective_sample_size'] = df.loc[replace_ess_with_ss, 'sample_size']

    # Calculate standard deviation different ways (also
    stdev_from_bounds = bounds_to_stdev(lower=df['lower'], upper=df['upper'])
    stdev_from_es = ess_to_stdev(mean=df['mean'], ess=df['effective_sample_size'])
    
    # Check to see if the UI was bad to begin with, then go to ESS
    bad_ui = stdev_from_bounds <= 0
    replace_se_with_ess = replace_se_with_ess | (replace_se_with_ui & bad_ui)
    
    # Use boolean arrays representing the pecking order to replacing standard error
    standard_error[replace_se_with_ui] = stdev_from_bounds[replace_se_with_ui]
    standard_error[replace_se_with_ess] = stdev_from_es[replace_se_with_ess]

    # Do a final check on standard error
    if ((standard_error <= 0) | standard_error.isnull()).any():
        LOG.error("There are unrepairable negative or null values for standard error in the bundle.")

    return standard_error
