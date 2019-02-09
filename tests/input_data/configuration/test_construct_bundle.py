import numpy as np
import pandas as pd

from cascade.input_data.configuration.construct_bundle import bundle_to_observations


def test_bundle_to_observations__global_eta():
    df = pd.DataFrame(
        {
            "location_id": 90,
            "measure": "Tincidence",
            "age_lower": 0,
            "age_upper": 120,
            "time_lower": 1980,
            "time_upper": 2018,
            "mean": 0.1,
            "standard_error": 0.001,
            "sex_id": 3,
            "seq": 0,
            "hold_out": 0,
        },
        index=[0],
    )

    observations = bundle_to_observations(df, 90, None)
    assert np.isnan(observations.eta[0])

    observations = bundle_to_observations(df, 90, 1.0)
    assert observations.eta[0] == 1.0
