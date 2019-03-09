"""
Converts the internal representation to a Dismod File.
"""
import sys
import time

import numpy as np
import pandas as pd

from cascade.core.log import getLoggers
from cascade.dismod.constants import IntegrandEnum

CODELOG, MATHLOG = getLoggers(__name__)


def enum_to_dataframe(enum_name):
    """Given an enum, return a dataframe with two columns, name and value."""
    return pd.DataFrame.from_records(
        np.array(
            [(measure, enum_value.value) for (measure, enum_value) in enum_name.__members__.items()],
            dtype=np.dtype([("name", object), ("value", np.int)]),
        )
    )


def default_integrand_names():
    # Converting an Enum to a DataFrame with specific parameters
    integrands = enum_to_dataframe(IntegrandEnum)
    df = pd.DataFrame({"integrand_name": integrands["name"]})
    df["minimum_meas_cv"] = 0.0
    return df


def make_log_table():
    command_name = " ".join(sys.argv)
    return pd.DataFrame(
        {
            "message_type": ["command"],
            "table_name": np.array([None], dtype=np.object),
            "row_id": np.NaN,
            "unix_time": int(round(time.time())),
            "message": [command_name],
        }
    )
