import pytest

from cascade.input_data.configuration.form import Configuration
from cascade.input_data.configuration.builder import (
    initial_context_from_epiviz,
    fixed_effects_from_epiviz,
    integrand_grids_from_epiviz,
)
from cascade.model import priors


@pytest.fixture
def base_config():
    config = Configuration(
        {
            "model": {
                "modelable_entity_id": 12345,
                "title": "Test Model",
                "description": "Test Model Description",
                "drill": "drill",
                "drill_location": 123,
                "default_age_grid": "0 10 20 30 40 50 60 70 80",
                "default_time_grid": "1990 1995 2000 2005 2010",
            },
            "gbd_round_id": 5,
            "random_effect": [],
            "rate": [
                {
                    "rate": 6,
                    "age_grid": "0 20 40 60 80",
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
            ],
        }
    )
    assert not config.validate_and_normalize()
    return config


def test_initial_context_from_epiviz(base_config):
    mc = initial_context_from_epiviz(base_config)
    assert mc.parameters.modelable_entity_id == 12345
    assert mc.parameters.gbd_round_id == 5
    assert mc.parameters.location_id == 123


def test_fixed_effects_from_epiviz(base_config):
    mc = initial_context_from_epiviz(base_config)
    fixed_effects_from_epiviz(mc, base_config)
    assert all([r.parent_smooth is None for r in [mc.rates.rho, mc.rates.pini, mc.rates.chi, mc.rates.omega]])
    smooth = mc.rates.iota.parent_smooth
    assert smooth.d_age_priors.priors == {priors.Gaussian(mean=0, standard_deviation=0.1)}
    assert smooth.d_time_priors.priors == {priors.Gaussian(mean=0, standard_deviation=0.2)}
    assert smooth.value_priors.priors == {
        priors.Gaussian(mean=0, standard_deviation=0.3),
        priors.StudentsT(mean=0, standard_deviation=0.25, nu=1),
    }
    assert set(smooth.grid.ages) == {0.0, 20.0, 40.0, 60.0, 80.0}
    assert set(smooth.grid.times) == {1990.0, 1995.0, 2000.0, 2005.0, 2010.0}

    for age in smooth.grid.ages:
        for time in smooth.grid.times:
            if age in {20.0, 40.0} and time in {1995.0, 2000.0, 2005.0}:
                assert smooth.value_priors[20, 2000].prior == priors.StudentsT(mean=0, standard_deviation=0.25, nu=1)
            else:
                assert smooth.value_priors[0, 1990].prior == priors.Gaussian(mean=0, standard_deviation=0.3)


def test_integrand_grids_from_epiviz(base_config):
    mc = initial_context_from_epiviz(base_config)
    fixed_effects_from_epiviz(mc, base_config)
    integrand_grids_from_epiviz(mc, base_config)

    assert all([integrand.grid is None for integrand in mc.outputs.integrands if integrand.name not in ["Sincidence"]])
    integrand = mc.outputs.integrands.Sincidence

    assert set(integrand.grid.ages) == {0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0}
    assert set(integrand.grid.times) == {1990.0, 1995.0, 2000.0, 2005.0, 2010.0}
