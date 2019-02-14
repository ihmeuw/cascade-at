from cascade.dismod.constants import DensityEnum
from cascade.core import getLoggers

import numpy as np

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
