from math import nan, inf
from pathlib import Path
from sqlite3 import Connection

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from cascade.model import (
    Session, Model, DismodGroups, SmoothGrid, Var, Covariate,
    Uniform, Gaussian
)
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


def test_predict(dismod):
    iota = Var([0, 20, 120], [2000])
    iota.grid.loc[np.isclose(iota.grid.age, 0), "mean"] = 0.0
    iota.grid.loc[np.isclose(iota.grid.age, 20), "mean"] = 0.02
    iota.grid.loc[np.isclose(iota.grid.age, 120), "mean"] = 0.02

    chi = Var([20], [2000])
    chi.grid.loc[:, "mean"] = 0.01

    mortality = siler_default()
    omega = Var(np.linspace(0, 120, 121), [2000])
    omega.grid.loc[:, "mean"] = mortality(omega.grid.age.values)

    model_variables = DismodGroups()
    model_variables.rate["iota"] = iota
    model_variables.rate["chi"] = chi
    model_variables.rate["omega"] = omega

    parent_location = 1
    locations = pd.DataFrame(dict(
        name=["global"],
        parent_id=[nan],
        location_id=[parent_location],
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
    for idx, row in predicted.iterrows():
        # Each Var is a function of age and time.
        input_iota = iota(row.age_lower, row.time_lower)
        assert np.isclose(input_iota, row["mean"])


def test_survival(dismod):
    """Dismod-AT predicts mortality in agreement with another ODE solver.
    This is a single-parameter model.
    """
    mortality = siler_default()
    omega = Var(np.linspace(0, 120, 121), [2000])
    omega.grid.loc[:, "mean"] = mortality(omega.grid.age.values)

    model_variables = DismodGroups()
    model_variables.rate["omega"] = omega

    parent_location = 1
    locations = pd.DataFrame(dict(
        name=["global"],
        parent_id=[nan],
        location_id=[parent_location],
    ))
    session = Session(locations, parent_location, Path("survtest.db"))
    session.set_option(ode_step_size=1)
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
        y = row["mean"]
        print(f"survival {row.age_lower} {S}-{y}")
        max_err = max(max_err, abs(S - y))
    print(f"Maximum error {max_err}")
    assert max_err < 0.015


def test_fit_mortality(dismod):
    """Create data for a single-parameter model and fit that data.
    """
    mortality = siler_default()

    omega = Var(np.linspace(0, 120, 121), [2000])
    omega.grid.loc[:, "mean"] = mortality(omega.grid.age.values)

    model_variables = DismodGroups()
    model_variables.rate["omega"] = omega

    parent_location = 1
    locations = pd.DataFrame(dict(
        name=["global"],
        parent_id=[nan],
        location_id=[parent_location],
    ))
    session = Session(locations, parent_location, Path("fit0.db"))
    session.set_option(ode_step_size=1)
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
    assert not_predicted.empty and not predicted.empty

    # We asked for a prediction of mtother, which is exactly the omega that
    # we put in. Compare the two by constructing a continuous function from
    # the predicted values and comparing at age points.
    as_var = predicted[predicted.integrand == "mtother"] \
        .rename(columns={"age_lower": "age", "time_lower": "time"}) \
        .drop(columns=["sample_index", "location", "integrand", "age_upper", "time_upper"])
    mtother_var = Var(as_var.age.unique(), as_var.time.unique())
    mtother_var.grid = as_var.assign(idx=0)

    for age in np.linspace(0, 120, 121):
        input_mx = mortality(age)
        # The Var is a function (bivariate spline) of age and time.
        output_mx = mtother_var(age, 2000)
        assert np.isclose(input_mx, output_mx)
        print(f"fit_mortality {age}\t{input_mx}\t{output_mx}")

    model = Model.from_var(model_variables, parent_location)
    priors = model.rate["omega"]
    print(f"test_fit priors\n{priors}")

    for a, t in priors.age_time():
        target = omega(a, t)
        priors.value[a, t] = Gaussian(mean=target, standard_deviation=0.5, eta=1e-4, upper=target + 5, lower=0)
        priors.dage[a, t] = Gaussian(mean=0, standard_deviation=0.1, lower=-5, upper=5)
        priors.dtime[a, t] = Gaussian(mean=0, standard_deviation=0.1, lower=-5, upper=5)

    data = predicted.drop(columns=["sample_index"])
    data = data.assign(density="gaussian", std=0.1, eta=1e-4, nu=nan)

    result = session.fit(model, data, initial_guess=model_variables)
    assert result is not None
    # We need a way to get residuals separately.
    assert result.prior_residuals is not None
    omega_residuals = result.prior_residuals.rate["omega"].grid
    assert (np.abs(omega_residuals[omega_residuals.age < 100].residual_value) < 0.11).all()

    assert not result.data_residuals.empty
    assert {"avg_integrand", "name", "weighted_residual"} == set(result.data_residuals.columns)
    print(result.data_residuals)


def _ages_of_underlying_rate(rate_name, conn):
    rate_to_smooth = dict(
        conn.execute(
            """
        select rate_name, smooth_id from smooth join rate
        on rate.parent_smooth_id = smooth_id
        """
        )
    )
    age_tuples = conn.execute(
        """
    select age from age
    join smooth_grid on smooth_grid.age_id = age.age_id
    where smooth_id = ?
    """,
        str(rate_to_smooth[rate_name])
    ).fetchall()
    return [a[0] for a in age_tuples]


def test_ages_align(dismod):
    """Multiple rates have the correct ages when written.
    """
    mortality = siler_default()

    omega_ages = np.linspace(0, 120, 7)
    omega = Var(omega_ages, [2000])
    omega.grid.loc[:, "mean"] = mortality(omega.grid.age.values)

    iota_ages = np.linspace(0, 120, 27)
    iota = Var(iota_ages, 2001)
    iota[:, :] = 0.001

    model_variables = DismodGroups()
    model_variables.rate["omega"] = omega
    model_variables.rate["iota"] = iota

    parent_location = 1
    locations = pd.DataFrame(dict(
        name=["global"],
        parent_id=[nan],
        location_id=[parent_location],
    ))
    db_file = "ages_align.db"
    session = Session(locations, parent_location, Path(db_file))
    session.set_option(ode_step_size=1)
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
    assert not_predicted.empty and not predicted.empty

    conn = Connection(db_file)
    iota_ages_out = _ages_of_underlying_rate("iota", conn)
    assert len(iota_ages_out) == len(iota_ages)
    for a, b in zip(sorted(iota_ages), sorted(iota_ages_out)):
        assert np.isclose(a, b)

    omega_ages_out = _ages_of_underlying_rate("omega", conn)
    assert len(omega_ages_out) == len(omega_ages)
    for a, b in zip(sorted(omega_ages), sorted(omega_ages_out)):
        assert np.isclose(a, b)
