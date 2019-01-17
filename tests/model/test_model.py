from math import nan, inf
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from cascade.model import Session, Model, DismodGroups, SmoothGrid, Var, model_from_vars
from cascade.model.covariates import Covariate
from cascade.model.priors import Uniform, Gaussian
from cascade.stats.compartmental import siler_default, total_mortality_solution


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

    # This peeks inside the session to test some of its underlying functionality
    # without doing a fit.
    session.write_model(basic_model, ([], []))
    session._run_dismod(["init"])
    var = session.get_var("scale")
    for name in basic_model:
        for key, grid in basic_model[name].items():
            field = var[name][key]
            print(f"{name}, {key} {len(grid)}, {len(field)}")

    # By 3 because there are three priors for every value,
    # and this model has no mulstds, which don't always come in sets of 3.
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

    predicted, not_predicted = session.predict(model_variables, avgints, parent_location)
    assert not_predicted.empty
    assert not predicted.empty

    # Check that Sincidence is predicted correctly for every time point.
    iota_func = iota.as_function()
    for idx, row in predicted.iterrows():
        input_iota = iota_func(row.age_lower, row.time_lower)
        assert np.isclose(input_iota, row["avg_integrand"])


def test_survival(dismod):
    """Dismod-AT predicts mortality in agreement with another ODE solver.
    This is a single-parameter model.
    """
    mortality = siler_default()
    omega = Var([np.linspace(0, 120, 121), [2000]])
    omega.grid.loc[:, "mean"] = mortality(omega.grid.age.values)

    model_variables = DismodGroups()
    model_variables.rate["omega"] = omega

    parent_location = 1
    locations = pd.DataFrame(dict(
        name=["global"],
        parent=[nan],
        c_location_id=[parent_location],
    ))
    session = Session(locations, parent_location, Path("survtest.db"))
    session.set_option("ode_step_size", 1)
    avgints = pd.DataFrame(dict(
        integrand="susceptible",
        location=parent_location,
        age_lower=np.linspace(0, 120, 121),
        age_upper=np.linspace(0, 120, 121),
        time_lower=2000,
        time_upper=2000,
    ))

    predicted, not_predicted = session.predict(model_variables, avgints, parent_location)
    assert not_predicted.empty
    assert not predicted.empty

    # Check that susceptibles are predicted correctly for every time point.
    survival = total_mortality_solution(mortality)
    max_err = -inf
    for idx, row in predicted.iterrows():
        S = survival(row.age_lower)
        y = row["avg_integrand"]
        print(f"survival {row.age_lower} {S}-{y}")
        max_err = max(max_err, abs(S - y))
    print(f"Maximum error {max_err}")
    assert max_err < 0.015


def test_fit_mortality(dismod):
    """Create data for a single-parameter model and fit that data.
    """
    mortality = siler_default()
    omega = Var([np.linspace(0, 120, 121), [2000]])
    omega.grid.loc[:, "mean"] = mortality(omega.grid.age.values)

    model_variables = DismodGroups()
    model_variables.rate["omega"] = omega

    parent_location = 1
    locations = pd.DataFrame(dict(
        name=["global"],
        parent=[nan],
        c_location_id=[parent_location],
    ))
    session = Session(locations, parent_location, Path("fit0.db"))
    session.set_option("ode_step_size", 1)
    avgints = pd.DataFrame(dict(
        integrand="susceptible",
        location=parent_location,
        age_lower=np.linspace(0, 120, 121),
        age_upper=np.linspace(0, 120, 121),
        time_lower=2000,
        time_upper=2000,
    ))
    avgints = pd.concat([avgints, avgints.assign(integrand="mtother")])

    predicted, not_predicted = session.predict(model_variables, avgints, parent_location)
    assert not_predicted.empty
    assert not predicted.empty
    print(f"test_fit predicted\n{predicted}")

    # We asked for a prediction of mtother, which is exactly the omega that
    # we put in. Compare the two by constructing a continuous function from
    # the predicted values and comparing at age points.
    as_var = predicted[predicted.integrand == "mtother"] \
        .rename(columns={"avg_integrand": "mean", "age_lower": "age", "time_lower": "time"}) \
        .drop(columns=["predict_id", "sample_index", "location", "integrand", "age_upper", "time_upper"])
    mtother_var = Var((as_var.age.unique(), as_var.time.unique()))
    mtother_var.grid = as_var.assign(idx=0)
    mtother_func = mtother_var.as_function()

    for age in np.linspace(0, 120, 121):
        input_mx = mortality(age)
        output_mx = mtother_func(age, 2000)
        assert np.isclose(input_mx, output_mx)
        print(f"fit_mortality {age}\t{input_mx}\t{output_mx}")

    model = model_from_vars(model_variables, parent_location)
    priors = model.rate["omega"]
    print(f"test_fit priors\n{priors}")

    priors.value.grid.loc[:, ["density", "std", "eta"]] = [
        "gaussian", 0.5, 1e-4
    ]
    priors.value.grid.loc[:, "mean"] = omega.grid["mean"]
    priors.value.grid.loc[:, "upper"] = 5 + omega.grid["mean"]
    priors.dage.grid.loc[:, ["density", "mean", "std", "lower", "upper"]] = [
        "gaussian", 0.0, 0.1, -5, 5
    ]
    priors.dtime.grid.loc[:, ["density", "mean", "std", "lower", "upper"]] = [
        "gaussian", 0.0, 0.1, -5, 5
    ]

    data = predicted.drop(columns=["sample_index", "predict_id"]).rename(columns={"avg_integrand": "mean"})
    data = data.assign(density="gaussian", std=0.1, eta=1e-4, nu=nan)

    print(f"test_fit data\n{data}")
    result = session.fit(model, data, initial_guess=model_variables)
    assert result is not None
    result_omega = result.rate["omega"].grid
    print(f'test_fit\n{result_omega}')
    assert (np.abs(result_omega[result_omega.age < 100].residual_value) < 0.11).all()
