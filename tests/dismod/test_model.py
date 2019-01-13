from math import nan

import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
import pytest

from cascade.model.covariates import Covariate
from cascade.model.priors import Uniform, Gaussian
from cascade.dismod import Session, Model, DismodGroups, SmoothGrid, Var
from cascade.stats.compartmental import siler_default


@pytest.fixture
def basic_model():
    nonzero_rates = ["iota", "chi", "omega"]
    locations = nx.DiGraph()
    locations.add_edges_from([(1, 2), (1, 3), (1, 4)])
    eta = 1e-3

    parent_location = 1
    child_locations = list()
    model = Model(nonzero_rates, parent_location, child_locations)
    model.covariates = [Covariate("traffic", reference=0.0)]

    covariate_age_time = ([40], [2000])
    traffic = SmoothGrid(covariate_age_time)
    traffic.value[:, :] = Gaussian(lower=-1, upper=1, mean=0.00, standard_deviation=0.3, eta=eta)
    model.alpha[("traffic", "iota")] = traffic

    dense_age_time = (np.linspace(0, 120, 13), np.linspace(1990, 2015, 8))
    rate_grid = SmoothGrid(dense_age_time)
    rate_grid.value[:, :] = Uniform(lower=1e-6, upper=0.3, mean=0.001, eta=eta)
    rate_grid.dage[:, :] = Uniform(lower=-1, upper=1, mean=0.0, eta=eta)
    rate_grid.dtime[:, :] = Gaussian(lower=-1, upper=1, mean=0.0, standard_deviation=0.3, eta=eta)

    model.rate["omega"] = rate_grid
    model.rate["iota"] = rate_grid

    chi_grid = SmoothGrid(dense_age_time)
    chi_grid.value[:, :] = Uniform(lower=1e-6, upper=0.3, mean=0.004, eta=eta)
    chi_grid.dage[:, :] = Uniform(lower=-.9, upper=.9, mean=0.0, eta=eta)
    chi_grid.dtime[:, :] = Gaussian(lower=-.8, upper=.8, mean=0.0, standard_deviation=0.4, eta=eta)
    model.rate["chi"] = chi_grid
    return model


def test_write_rate(basic_model, dismod):
    locations = pd.DataFrame(dict(
        name=["global"],
        parent=[nan],
        c_location_id=[1],
    ))
    parent_location = 1
    db_file = Path("rftest.db")
    session = Session(locations, parent_location, db_file)

    data = None
    var = session.fit(basic_model, data)
    for name in basic_model:
        for key, grid in basic_model[name].items():
            field = var[name][key]
            print(f"{name}, {key} {len(grid)}, {len(field)}")

    # By 3 because there are three priors for every value,
    # and this model has no mulstds.
    assert 3 * var.count() == basic_model.count()


def test_predict(dismod):
    iota = Var(([0, 20, 120], [2000]))
    iota.grid.loc[np.isclose(iota.grid.age, 0), "mean"] = 0.0
    iota.grid.loc[np.isclose(iota.grid.age, 20), "mean"] = 0.02
    iota.grid.loc[np.isclose(iota.grid.age, 120), "mean"] = 0.02

    chi = Var(([20], [2000]))
    chi.grid.loc[:, "mean"] = 0.01

    mortality = siler_default()
    omega = Var([np.linspace(0, 120, 121), [2000]])
    omega.grid.loc[:, "mean"] = mortality(omega.grid.age.values)

    model_variables = DismodGroups()
    model_variables.rate["iota"] = iota
    model_variables.rate["chi"] = chi
    model_variables.rate["omega"] = omega

    parent_location = 1
    locations = pd.DataFrame(dict(
        name=["global"],
        parent=[nan],
        c_location_id=[parent_location],
    ))
    db_file = Path("prtest.db")
    session = Session(locations, parent_location, db_file)

    avgints = pd.DataFrame(dict(
        integrand="Sincidence",
        location=parent_location,
        age_lower=np.linspace(0, 120, 121),
        age_upper=np.linspace(0, 120, 121),
        time_lower=2000,
        time_upper=2000,
    ))

    predicted, not_predicted = session.predict(model_variables, avgints)
    assert not_predicted.empty
    assert not predicted.empty

    # Check that Sincidence is predicted correctly for every time point.
    iota_func = iota.as_function()
    for idx, row in predicted.iterrows():
        print(f"row {row}")
        input_iota = iota_func(row.age_lower, row.time_lower)
        assert np.isclose(input_iota, row["avg_integrand"])
