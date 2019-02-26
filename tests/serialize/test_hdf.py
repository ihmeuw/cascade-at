import numpy as np
from h5py import File
import networkx as nx
from numpy import isclose
import pytest

from cascade.model.dismod_groups import DismodGroups
from cascade.model.smooth_grid import SmoothGrid
from cascade.model.model import Model
from cascade.model.covariates import Covariate
from cascade.model.var import Var
from cascade.serialize.hdf import (
    write_var, read_var, write_var_group, read_var_group, write_smooth_grid,
    read_smooth_grid, write_grid_group, read_grid_group, write_model, read_model
)
from cascade.model.priors import Uniform, Gaussian


@pytest.fixture
def basic_model():
    nonzero_rates = ["iota", "chi", "omega"]
    locations = nx.DiGraph()
    locations.add_edges_from([(1, 2), (1, 3), (1, 4)])
    eta = 1e-3

    parent_location = 1
    child_locations = list()
    covariates = [Covariate("traffic", reference=0.0)]
    model = Model(nonzero_rates, parent_location, child_locations, covariates=covariates)

    covariate_age_time = ([40], [2000])
    traffic = SmoothGrid(*covariate_age_time)
    traffic.value[:, :] = Gaussian(lower=-1, upper=1, mean=0.00, standard_deviation=0.3, eta=eta)
    model.alpha[("traffic", "iota")] = traffic

    dense_age_time = (np.linspace(0, 120, 13), np.linspace(1990, 2015, 8))
    rate_grid = SmoothGrid(*dense_age_time)
    rate_grid.value[:, :] = Uniform(lower=1e-6, upper=0.3, mean=0.001, eta=eta)
    rate_grid.dage[:, :] = Uniform(lower=-1, upper=1, mean=0.0, eta=eta)
    rate_grid.dtime[:, :] = Gaussian(lower=-1, upper=1, mean=0.0, standard_deviation=0.3, eta=eta)

    model.rate["omega"] = rate_grid
    model.rate["iota"] = rate_grid

    chi_grid = SmoothGrid(*dense_age_time)
    chi_grid.value[:, :] = Uniform(lower=1e-6, upper=0.3, mean=0.004, eta=eta)
    chi_grid.dage[:, :] = Uniform(lower=-.9, upper=.9, mean=0.0, eta=eta)
    chi_grid.dtime[:, :] = Gaussian(lower=-.8, upper=.8, mean=0.0, standard_deviation=0.4, eta=eta)
    model.rate["chi"] = chi_grid
    return model


def test_write_vars_one_field(tmp_path):
    id_draw = Var([0, 50, 100], [1995, 2015])
    for a, t in id_draw.age_time():
        id_draw[a, t] = a * 2 + t

    with File(str(tmp_path / "test.hdf5"), "w") as f:
        g = f.create_group("var1")
        written_data = write_var(g, id_draw, "iota")
        assert written_data.shape == (3, 2)

    with File(str(tmp_path / "test.hdf5"), "r") as r:
        g = r["var1"]
        v = read_var(g["iota"])
        for a, t in id_draw.age_time():
            assert isclose(v[a, t], id_draw[a, t])


def new_var(idx):
    id_draw = Var([0, 50 + idx, 100], [1995, 2015])
    for a, t in id_draw.age_time():
        id_draw[a, t] = a * idx + t
    return id_draw


def test_write_groups(tmp_path):
    dg = DismodGroups()
    dg.rate["iota"] = new_var(1)
    dg.rate["omega"] = new_var(2)
    dg.random_effect[("iota", None)] = new_var(3)
    dg.random_effect[("omega", 1)] = new_var(4)
    dg.random_effect[("omega", 2)] = new_var(5)
    dg.alpha[("iota", "traffic")] = new_var(6)

    with File(str(tmp_path / "test.hdf5"), "w") as f:
        g = f.create_group("var1")
        write_var_group(g, dg)

    with File(str(tmp_path / "test.hdf5"), "r") as r:
        g = r["var1"]
        rg = read_var_group(g)

    assert "iota" in rg.rate
    assert "omega" in rg.rate
    assert ("iota", None) in rg.random_effect
    assert ("omega", 1) in rg.random_effect
    assert ("omega", 2) in rg.random_effect
    assert ("iota", "traffic") in dg.alpha
    assert not dg.beta
    assert not dg.gamma

    assert rg.rate["iota"] == dg.rate["iota"]
    assert rg.random_effect[("omega", 1)] == dg.random_effect[("omega", 1)]


def test_write_smooth_grid(tmp_path):
    ages = np.array([0, 20.0, 100])
    times = np.array([1990, 2015])
    sg = SmoothGrid(ages, times)
    sg.value[:, :] = Uniform(lower=0, upper=1, mean=0.1)
    sg.dage[:, :] = Gaussian(mean=0.1, standard_deviation=0.2)
    sg.dtime[:, :] = Gaussian(mean=-0.1, standard_deviation=0.3)

    with File(str(tmp_path / "dg.hdf5"), "w") as f:
        g = f.create_group("groupgroup")
        write_smooth_grid(g, sg, "mygrid")

    with File(str(tmp_path / "dg.hdf5"), "r") as r:
        g = r["groupgroup"]
        rg = read_smooth_grid(g["mygrid"])

    assert sg == rg


def test_write_smooth_grid_group(basic_model, tmp_path):
    with File(str(tmp_path / "dg.hdf5"), "w") as f:
        g = f.create_group("groupgroup")
        write_grid_group(g, basic_model)

    with File(str(tmp_path / "dg.hdf5"), "r") as r:
        g = r["groupgroup"]
        read_grid_group(g)


def test_write_model(basic_model, tmp_path):
    with File(str(tmp_path / "dg.hdf5"), "w") as f:
        g = f.create_group("groupgroup")
        write_model(g, basic_model)

    with File(str(tmp_path / "dg.hdf5"), "r") as r:
        g = r["groupgroup"]
        model = read_model(g)

    assert (model.nonzero_rates == basic_model.nonzero_rates).all()
    assert model.location_id == basic_model.location_id
    assert model.child_location == basic_model.child_location
    assert len(model.covariates) == len(basic_model.covariates)
    assert model.covariates == basic_model.covariates
