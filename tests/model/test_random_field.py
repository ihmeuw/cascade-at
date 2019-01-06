import networkx as nx

import numpy as np
import pytest

from cascade.model.random_field import Model, RandomField
from cascade.dismod.model_writer import ModelWriter


@pytest.fixture
def basic_model():
    nonzero_rates = ["iota", "chi", "omega"]
    locations = nx.DiGraph()
    locations.add_edges_from([(1, 2), (1, 3), (1, 4)])
    parent_location = 1

    model = Model(nonzero_rates, locations, parent_location)

    covariate_age_time = ([40], [2000])
    traffic = RandomField(covariate_age_time)
    traffic.priors.loc[traffic.priors.kind == "value", "density_id"] = 1
    traffic.priors.loc[traffic.priors.kind == "value", "mean"] = 0.1

    model.alpha[("traffic", "iota")] = traffic

    dense_age_time = (np.linspace(0, 120, 13), np.linspace(1990, 2015, 7))
    rate_grid = RandomField(dense_age_time)
    rate_grid.priors.loc[rate_grid.priors.kind == "value", "upper"] = 1
    rate_grid.priors.loc[rate_grid.priors.kind == "value", "lower"] = .0001
    rate_grid.priors.loc[rate_grid.priors.kind == "dage", "density_id"] = 1
    rate_grid.priors.loc[rate_grid.priors.kind == "dage", "lower"] = 0.0001
    rate_grid.priors.loc[rate_grid.priors.kind == "dage", "upper"] = 0.5
    rate_grid.priors.loc[rate_grid.priors.kind == "dage", "density_id"] = 1
    rate_grid.priors.loc[rate_grid.priors.kind == "dage", "lower"] = 0.0001
    rate_grid.priors.loc[rate_grid.priors.kind == "dage", "upper"] = 0.4

    model.rate["omega"] = rate_grid
    model.rate["iota"] = rate_grid
    model.rate["chi"] = rate_grid
    return model


def test_write_rate(basic_model):
    writer = ModelWriter()
    basic_model.write(writer)
