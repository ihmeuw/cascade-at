import numpy as np
import pandas as pd

from cascade.dismod import serialize


def test_age_time_from_grids():
    smoothers = [
        pd.DataFrame({
            "age": [10.0, 5.0],
            "year": [1970, 1975],
        })
    ]
    total_data = pd.DataFrame({
        "age_lower": [0.0, 90.0],
        "age_upper": [10.0, 100.0],
        "time_lower": [1980, 1965],
        "time_upper": [1985, 1970],
    })
    age_df, time_df = serialize.age_time_from_grids(smoothers, total_data)
    # Note 90 is not included
    assert np.all(age_df["age"].values == np.array([0.0, 5.0, 10.0, 100.0]))
    # Note 1980 is not included
    assert np.all(time_df["time"].values == np.array([1965, 1970, 1975, 1985]))
