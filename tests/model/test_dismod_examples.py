"""
This set of tests derives from examples in Dismod-AT's distribution.
These are described at https://bradbell.github.io/dismod_at/doc/user.htm
"""
from math import nan, sqrt, exp
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from cascade.model import Session, Model, SmoothGrid, Covariate, Uniform, Gaussian, LogGaussian


def test_fit_random(dismod):
    parent_location = 1
    # The US and Canada are children of North America.
    locations = pd.DataFrame(dict(
        name=["North America", "United States", "Canada"],
        parent=[nan, parent_location, parent_location],
        c_location_id=[parent_location, 2, 3],
    ))

    iota_parent_true = 1e-2
    united_states_random_effect = +0.5
    canada_random_effect = -0.5

    # Data values are exactly the correct values.
    measured_value = np.array([
        iota_parent_true,
        iota_parent_true * exp(united_states_random_effect),
        iota_parent_true * exp(canada_random_effect)])
    data = pd.DataFrame(dict(
        integrand="Sincidence",
        location=[1, 2, 3],  # One data point for each location.
        age_lower=50,
        age_upper=50,
        time_lower=2000,
        time_upper=2000,
        density="gaussian",
        mean=measured_value,
        std=0.1 * measured_value,
        nu=nan,
        eta=nan,
    ))

    # The only nonlinear rate is iota. The three groups of model variables
    # are the underlying iota rate and random effects for US and Canada.
    # Rates change linearly between 1995 and 2015 but are constant across ages.
    iota = SmoothGrid(([50], [1995, 2015]))
    iota.value[:, :] = Uniform(
        mean=iota_parent_true * exp(united_states_random_effect),
        lower=1e-4,
        upper=1)
    iota.dtime[:, :] = LogGaussian(mean=0.0, standard_deviation=0.1, eta=1e-8)

    iota_child = SmoothGrid(([50], [1995, 2015]))
    # This large standard deviation makes the Gaussian uninformative.
    iota_child.value[:, :] = Gaussian(mean=0.0, standard_deviation=100.0)
    # But constrain to little change through time.
    iota_child.dtime[:, :] = Gaussian(mean=0.0, standard_deviation=0.1)

    model = Model(["iota"], parent_location, [2, 3])
    model.rate["iota"] = iota
    # Set the grid for all children by setting it for None.
    model.random_effect[("iota", None)] = iota_child

    session = Session(locations, parent_location, Path("fit_random.db"))
    option = dict(random_seed=0,
                  derivative_test_random="second-order",
                  max_num_iter_random=100,
                  tolerance_random=1e-10)
    session.set_option(**option)
    result = session.fit(model, data)

    parent_result = result.rate["iota"].grid.loc[:, "mean"]
    parent_expected = iota_parent_true * exp(united_states_random_effect)
    assert ((parent_result - parent_expected).abs() / parent_expected < 1e-10).all()
    print(f"Random effects are {list(result.random_effect.keys())}")
    canada_result = result.random_effect[("iota", 3)].grid.loc[:, "mean"]
    canada_expected = -2 * united_states_random_effect
    assert (((canada_result - canada_expected) / canada_expected).abs() < 1e-5).all()
    us_result = result.random_effect[("iota", 2)].grid.loc[:, "mean"]
    us_expected = 0
    assert ((us_result - us_expected).abs() < 1e-5).all()


@pytest.mark.parametrize("meas_std_effect", [
    "add_std_scale_all",
    "add_std_scale_log",
    "add_var_scale_all",
    "add_var_scale_log",
])
def test_fit_gamma(meas_std_effect):
    """The fit_gamma.py example in Dismod-AT's distribution"""
    rng = np.random.RandomState(3798427592)

    iota_true = 0.01
    gamma_true_scale = 2.0
    n_data = 2000
    data_std = iota_true / 3.0

    age_list = [0, 100]

    parent_location = 1
    child_locations = list()
    locations = pd.DataFrame(dict(
        name=["global"],
        parent=[nan],
        c_location_id=[parent_location],
    ))

    if meas_std_effect == 'add_std_scale_all':
        delta = data_std * (1.0 + gamma_true_scale)
        gamma_true = gamma_true_scale
    elif meas_std_effect == 'add_std_scale_log':
        delta = data_std * (1.0 + gamma_true_scale)
        gamma_true = gamma_true_scale * data_std
    elif meas_std_effect == 'add_var_scale_all':
        delta = data_std * sqrt(1.0 + gamma_true_scale)
        gamma_true = gamma_true_scale
    elif meas_std_effect == 'add_var_scale_log':
        delta = data_std * sqrt(1.0 + gamma_true_scale)
        gamma_true = gamma_true_scale * data_std * data_std
    else:
        assert False

    one = Covariate("one", 0)
    nonzero_rates = ["iota"]
    model = Model(nonzero_rates, parent_location, child_locations, covariates=[one])

    # There will be one rate, incidence, on two ages and two times.
    model.rate["iota"] = SmoothGrid(([0], [1990]))
    model.rate["iota"].value[:, :] = Uniform(lower=iota_true / 100, upper=1, mean=iota_true / 10)

    incidence_gamma = SmoothGrid([[0], [1990]])
    incidence_gamma.value[:, :] = Uniform(lower=0, upper=10 * gamma_true, mean=gamma_true / 10)
    model.gamma[("one", "Sincidence")] = incidence_gamma

    # No need to specify weight in data b/c appropriate weight for each integrand is chosen.
    data = pd.DataFrame(dict(
        integrand="Sincidence",
        location=1,
        age_lower=np.linspace(age_list[0], age_list[-1], n_data),
        age_upper=np.linspace(age_list[0], age_list[-1], n_data),
        time_lower=2000,
        time_upper=2000,
        density="gaussian",
        mean=norm.rvs(loc=iota_true, scale=delta, size=n_data, random_state=rng),
        std=data_std,
        one=1.0,
        nu=nan,
        eta=nan,
    ))

    # If you don't create a session with weights, they are automatically set to constant=1.
    session = Session(locations, parent_location, Path("example.db"))
    option = dict(meas_std_effect=meas_std_effect, random_seed=0,
                  zero_sum_random="iota", derivative_test_fixed="second-order",
                  max_num_iter_fixed=100, print_level_fixed=0,
                  tolerance_fixed=1e-10)
    session.set_option(**option)

    result = session.fit(model, data)
    rate_out = result.rate["iota"].grid["mean"]
    max_iota = ((rate_out - iota_true) / iota_true).abs().max()
    gamma_out = result.gamma[("one", "Sincidence")].grid["mean"]
    max_gamma = ((gamma_out - gamma_true) / gamma_true).abs().max()
    assert max_iota < 0.2, f"max iota error {max_iota}"
    assert max_gamma < 0.2, f"max gamma error {max_gamma}"
