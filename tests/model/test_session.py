from math import nan
from pathlib import Path

import numpy as np
import pandas as pd

from cascade.model import Session, Var, DismodGroups


def test_options(dismod):
    locations = pd.DataFrame(dict(
        name=["global"],
        parent_id=[nan],
        location_id=[1],
    ))
    parent_location = 1
    db_file = Path("option.db")
    session = Session(locations, parent_location, db_file)
    opts = dict(
        meas_noise_effect="add_std_scale_all",
        parent_node_name="global",
        quasi_fixed="false",
        derivative_test_fixed="second-order",
        max_num_iter_fixed=100,
        print_level_fixed=0,
        tolerance_fixed=1e-10,
        age_avg_split=[0.19, 0.5, 1, 2],
    )
    session.set_option(**opts)
    session.age_extents = [0, 80]

    iota = Var([20], [2000])
    iota.grid.loc[:, "mean"] = 0.01
    model_var = DismodGroups()
    model_var.rate["iota"] = iota
    avgints = pd.DataFrame(dict(
        integrand="susceptible",
        location=parent_location,
        age_lower=np.linspace(0, 120, 2),
        age_upper=np.linspace(0, 120, 2),
        time_lower=2000,
        time_upper=2000,
    ))
    # Run a predict in order to verify the options are accepted.
    session.predict(model_var, avgints, parent_location)
