import pytest

import pandas as pd

from cascade.input_data.configuration.construct_country import unique_country_covariate_transform
from cascade.input_data.configuration.form import Configuration
from cascade.input_data.configuration.builder import (
    initial_context_from_epiviz,
    random_effects_from_epiviz,
    make_smooth,
    assign_covariates,
    create_covariate_multipliers,
)
from cascade.model import priors
from cascade.input_data.configuration.covariate_records import CovariateRecords
from cascade.testing_utilities import make_execution_context


@pytest.fixture
def base_config():
    config = Configuration(
        {
            "model": {
                "modelable_entity_id": 12345,
                "model_version_id": 0xdeadbeef,
                "title": "Test Model",
                "constrain_omega": 1,
                "ode_step_size": 1,
                "description": "Test Model Description",
                "drill": "drill",
                "drill_location": 123,
                "drill_sex": 2,
                "split_sex": 4,
                "default_age_grid": "0 10 20 30 40 50 60 70 80",
                "default_time_grid": "1990 1995 2000 2005 2010",
            },
            "gbd_round_id": 5,
            "random_effect": [
                {
                    "rate": "rho",
                    "location": 180,
                    "time_grid": "1990 1991 1992 2000 2009",
                    "default": {
                        "dage": {"density": "uniform", "mean": 0, "min": 0, "max": 1},
                        "dtime": {"density": "gaussian", "mean": 1, "std": 0.2},
                        "value": {"density": "gaussian", "mean": 2, "std": 0.3},
                    },
                    "detail": [
                        {
                            "prior_type": "value",
                            "age_lower": 20,
                            "age_upper": 40,
                            "time_lower": 1991,
                            "time_upper": 2000,
                            "density": "log_students",
                            "mean": 0,
                            "std": 0.25,
                            "nu": 1,
                            "eta": 0,
                        }
                    ],
                }
            ],
            "rate": [
                {
                    "rate": "iota",
                    "age_grid": "0 20 40 60 80",
                    "default": {
                        "dage": {"density": "gaussian", "mean": 0, "std": 0.1},
                        "dtime": {"density": "gaussian", "mean": 0, "std": 0.2},
                        "value": {"density": "gaussian", "min": 0.00001, "mean": 0.00001, "std": 0.3},
                    },
                    "detail": [
                        {
                            "prior_type": "value",
                            "age_lower": 20,
                            "age_upper": 40,
                            "time_lower": 1995,
                            "time_upper": 2005,
                            "density": "students",
                            "min": 0.00001,
                            "mean": 0.00002,
                            "std": 0.25,
                            "nu": 1,
                        }
                    ],
                }
            ],
            "csmr_cod_output_version_id": 90,
        }
    )
    assert not config.validate_and_normalize()
    return config


def test_initial_context_from_epiviz(base_config):
    mc = initial_context_from_epiviz(base_config)
    assert mc.parameters.modelable_entity_id == 12345
    assert mc.parameters.gbd_round_id == 5
    assert mc.parameters.location_id == 123


def test_make_smooth(base_config):
    smooth = make_smooth(base_config, base_config.rate[0])

    assert smooth.d_age_priors.priors == {priors.Gaussian(mean=0, standard_deviation=0.1, name="dA")}
    assert smooth.d_time_priors.priors == {priors.Gaussian(mean=0, standard_deviation=0.2, name="dT")}
    assert smooth.value_priors.priors == {
        priors.Gaussian(mean=1e-05, standard_deviation=0.3, name="value", lower=1e-05),
        priors.StudentsT(mean=2e-05, standard_deviation=0.25, nu=1, lower=1e-05, name="value_20.0_40.0_1995.0_2005.0"),
    }
    assert set(smooth.grid.ages) == {0.0, 20.0, 40.0, 60.0, 80.0}
    assert set(smooth.grid.times) == {1990.0, 1995.0, 2000.0, 2005.0, 2010.0}

    for age in smooth.grid.ages:
        for time in smooth.grid.times:
            if age in {20.0, 40.0} and time in {1995.0, 2000.0, 2005.0}:
                expected = priors.StudentsT(mean=2e-05,
                                            standard_deviation=0.25,
                                            lower=1e-05,
                                            nu=1,
                                            name="value_20.0_40.0_1995.0_2005.0")
                assert smooth.value_priors[20, 2000].prior == expected
            else:
                expected = priors.Gaussian(mean=1e-05, standard_deviation=0.3, lower=1e-05, name="value")
                assert smooth.value_priors[0, 1990].prior == expected


def test_random_effects_from_epiviz(base_config):
    mc = initial_context_from_epiviz(base_config)
    random_effects_from_epiviz(mc, base_config)

    assert all([not rate.child_smoothings for rate in mc.rates if rate.name not in ["rho"]])

    rate = mc.rates.rho

    assert rate.child_smoothings[0][0] == 180

    expected_smooth = make_smooth(base_config, base_config.random_effect[0], name_prefix="rho")
    assert rate.child_smoothings[0][1] == expected_smooth


def test_covariates_from_settings_logic(base_config, ihme):
    # Can get rid of ihme by stubbing retrieval of covariates.
    mc = initial_context_from_epiviz(base_config)

    mc.input_data.observations = pd.DataFrame(dict(
        age_lower=[0, 10, 20, 50, 100],
        age_upper=[10, 20, 40, 70, 120],
        time_lower=[1970, 1980, 1990, 2000, 2005],
        time_upper=[1975, 1990, 2000, 2005, 2010],
        integrand=[6, 6, 6, 6, 6],
        x_sex=[0.5, 0.5, -0.5, -0.5, 0],
    ))
    configuration = base_config
    start = {
        "country_covariate_id": 26,
        "transformation": 0,
        "measure_id": 41,  # Sincidence
        "mulcov_type": "rate_value",
        "age_grid": [0, 20, 40, 60, 80],
        "default": {
            "dage": {"density": "gaussian", "mean": 0, "std": 0.1},
            "dtime": {"density": "gaussian", "mean": 0, "std": 0.2},
            "value": {"density": "gaussian", "mean": 0, "std": 0.3},
        },
        "detail": [
            {
                "prior_type": "value",
                "age_lower": 20,
                "age_upper": 40,
                "time_lower": 1995,
                "time_upper": 2005,
                "density": "students",
                "mean": 0,
                "std": 0.25,
                "nu": 1,
            }
        ],
    }
    configuration.country_covariate = [start]
    configuration.validate_and_normalize()
    assert configuration.model.default_time_grid
    ec = make_execution_context(gbd_round_id=5)
    assert ec.parameters.gbd_round_id == 5
    records = CovariateRecords("country")
    records.measurements = pd.DataFrame({"Funistan": [1.0, 2.0, 3.0, 4.0, 5.0]},
                                        index=mc.input_data.observations.index)
    records.id_to_name[26] = "Funistan"
    records.id_to_reference[26] = 3.7
    country_iter = unique_country_covariate_transform(configuration)
    column_id_func = assign_covariates(mc, records, country_iter)
    create_covariate_multipliers(mc, configuration, column_id_func)
