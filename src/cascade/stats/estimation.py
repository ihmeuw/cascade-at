from cascade.dismod.db.metadata import DensityEnum
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
    if np.any(df.meas_lower > df.meas_value) or np.any(df.meas_value > df.meas_upper):
        raise ValueError(f"Found data where meas_lower > meas_value or meas_upper < meas_value")
    df["standard_error"] = (df.meas_upper - df.meas_lower) / (2 * 1.96)
    df["standard_error"] = df.standard_error.replace({0: 1e-9})
    df = df.rename(columns={"meas_value": "mean"})
    df["density"] = DensityEnum.gaussian
    df["weight"] = "constant"
    return df.drop(["meas_lower", "meas_upper"], axis=1)
