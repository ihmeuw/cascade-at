import pandas as pd
import numpy as np

from cascade.testing_utilities import make_execution_context
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

    ec = make_execution_context(global_data_eta=None)
    observations = bundle_to_observations(ec.parameters, df)
    assert np.isnan(observations.eta[0])

    ec = make_execution_context(global_data_eta=1.0)
    observations = bundle_to_observations(ec.parameters, df)
    assert observations.eta[0] == 1.0
