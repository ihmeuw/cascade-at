from math import nan

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from cascade.dismod.model_writer import DismodSession
from cascade.model.covariates import Covariate
from cascade.model.priors import Uniform, Gaussian
from cascade.model.random_field import Model, RandomField, FieldDraw


@pytest.fixture
def basic_model():
    nonzero_rates = ["iota", "chi", "omega"]
    locations = nx.DiGraph()
    locations.add_edges_from([(1, 2), (1, 3), (1, 4)])
    eta = 1e-3

    child_locations = list()
    model = Model(nonzero_rates, child_locations)
    model.covariates = [Covariate("traffic", reference=0.0)]

    covariate_age_time = ([40], [2000])
    traffic = RandomField(covariate_age_time)
    traffic.value[:, :] = Gaussian(lower=-1, upper=1, mean=0.00, standard_deviation=0.3, eta=eta)
    model.alpha[("traffic", "iota")] = traffic

    dense_age_time = (np.linspace(0, 120, 13), np.linspace(1990, 2015, 8))
    rate_grid = RandomField(dense_age_time)
    rate_grid.value[:, :] = Uniform(lower=1e-6, upper=0.3, mean=0.001, eta=eta)
    rate_grid.dage[:, :] = Uniform(lower=-1, upper=1, mean=0.0, eta=eta)
    rate_grid.dtime[:, :] = Gaussian(lower=-1, upper=1, mean=0.0, standard_deviation=0.3, eta=eta)

    model.rate["omega"] = rate_grid
    model.rate["iota"] = rate_grid

    chi_grid = RandomField(dense_age_time)
    chi_grid.value[:, :] = Uniform(lower=1e-6, upper=0.3, mean=0.004, eta=eta)
    chi_grid.dage[:, :] = Uniform(lower=-.9, upper=.9, mean=0.0, eta=eta)
    chi_grid.dtime[:, :] = Gaussian(lower=-.8, upper=.8, mean=0.0, standard_deviation=0.4, eta=eta)
    model.rate["chi"] = chi_grid

    model.weights["constant"] = FieldDraw(([40], [2000]))
    model.weights["constant"].values.loc[:, "mean"] = 1.0
    return model


def test_write_rate(basic_model):
    parent_location = 1
    session = DismodSession(pd.DataFrame(dict(
        name=["global"],
        parent=[nan],
        c_location_id=[1],
    )), parent_location, "rftest.db")
    session.write(basic_model)
