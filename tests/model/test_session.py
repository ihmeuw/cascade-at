from math import nan
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cascade.model import Session, Var, DismodGroups
from cascade.model.session import check_iterations_exceeded


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


@pytest.mark.parametrize("msg,expected", [
    ("""Lots of numbers 2342.2342
    And Dismod-AT exceeded iterations
    """, True),
    ("""Further text
    EXIT: Maximum Number of Iterations Exceeded
    dismod_at warning: ipopt failed to converge
    """, True),
    ("""and lastly
    lots of iterations
    and things exceeded
    but not both""", False),
    ("""totally nothing
    to do with either
    3249.3242
    number
    """, False)
])
def test_notice_exceeded(msg, expected):
    assert check_iterations_exceeded(msg) == expected
