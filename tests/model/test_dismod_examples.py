"""
This set of tests derives from examples in Dismod-AT's distribution.
"""
from math import nan, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from cascade.model import Session, Model, SmoothGrid, Covariate, Uniform


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
    time_list = [1990, 2200]

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
        assert(False)

    one = Covariate("one", 0)
    nonzero_rates = ["iota"]
    model = Model(nonzero_rates, parent_location, child_locations, covariates=[one])
    model.rate["iota"] = SmoothGrid((age_list, time_list))
    model.rate["iota"].value[:, :] = Uniform(lower=iota_true / 100, upper=1, mean=iota_true / 10)

    incidence_gamma = SmoothGrid([0, 1990])
    incidence_gamma.value[:, :] = Uniform(lower=0, upper=10 * gamma_true, mean=gamma_true / 10)
    model.gamma[("one", "iota")] = incidence_gamma

    # No need to specify weight in data b/c appropriate weight for each integrand is chosen.
    data = pd.DataFrame(dict(
        integrand="Sincidence",
        location=1,
        age_lower=np.linspace(age_list[0], age_list[-1], n_data),
        age_upper=np.linspace(age_list[0], age_list[-1], n_data),
        time_lower=2000,
        time_upper=2000,
        mean=norm(loc=iota_true, scale=delta, size=n_data, random_state=rng),
        std=data_std,
        one=1.0,
    ))

    # If you don't create a session with weights, they are automatically set to constant=1.
    session = Session(locations, parent_location, Path("example.db"))
    option = dict(meas_std_effect=meas_std_effect, random_seed=0,
                  zero_sum_random="iota", derivative_test_fixed="second-order",
                  max_num_iter_fixed=100, print_level_fixed=0,
                  tolerance_fixed=1e-10)
    for opt, value in option.items():
        session.set_option(opt, value)

    result = session.fit(model, data)
    assert np.allclose(result.rate["iota"].grid["mean"], iota_true)
    assert np.allclose(result.gamma[("one", "iota")].grid["mean"], gamma_true)
