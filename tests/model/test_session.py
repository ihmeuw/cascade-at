from math import nan
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
        meas_std_effect="add_std_scale_all",
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


def test_unknown_options():
    locations = pd.DataFrame(dict(
        name=["global"],
        parent_id=[nan],
        location_id=[1],
    ))
    parent_location = 1
    db_file = Path("option.db")
    session = Session(locations, parent_location, db_file)
    with pytest.raises(KeyError):
        session.set_option(unknown="hiya")


def test_locations():
    locations = pd.DataFrame(dict(
        parent_id=[nan, 1, 2, 2],
        location_id=[1, 2, 3, 4],
        name=["global", "North America", "United States", "Canada"],
    ))
    session = Session(locations, 1, "none.db")
    node_table = session.dismod_file.node
    assert len(node_table) == 4
    for find_col in ["node_id", "node_name", "parent", "c_location_id"]:
        assert find_col in node_table.columns

    assert node_table.at[1, "c_location_id"] == 2
    assert node_table.at[1, "parent"] == 0
    assert node_table.at[1, "node_name"] == "North America"


def test_locations_no_name():
    locations = pd.DataFrame(dict(
        parent_id=[nan, 1, 2, 2],
        location_id=[1, 2, 3, 4],
    ))
    session = Session(locations, 1, "none.db")
    node_table = session.dismod_file.node
    assert len(node_table) == 4
    for find_col in ["node_id", "node_name", "parent", "c_location_id"]:
        assert find_col in node_table.columns
    for loc, node_id in [(1, 0), (2, 1), (3, 2), (4, 3)]:
        assert session.location_func(loc) == node_id

    assert node_table.at[2, "node_name"] == "3"
