import pytest

import pandas as pd

from cascade.stats.estimation import meas_bounds_to_stdev


def test_meas_bounds_to_stdev__bad_bounds():
    df = pd.DataFrame({"meas_lower": [0, 0, 1, 1], "meas_value": [0, -0.01, 1, 1], "meas_upper": [-10, 0, 1, 10]})

    with pytest.raises(ValueError):
        meas_bounds_to_stdev(df)
