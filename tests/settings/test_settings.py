import pytest

from cascade_at.settings.base_case import BASE_CASE
from cascade_at.settings.settings import load_settings


@pytest.fixture
def settings():
    settings = load_settings(BASE_CASE)
    return settings


@pytest.fixture
def model(settings):
    return settings.model


@pytest.fixture
def policies(settings):
    return settings.policies


@pytest.fixture
def study_covariate(settings):
    return settings.study_covariate


@pytest.fixture
def country_covariate(settings):
    return settings.country_covariate


@pytest.fixture
def rate(settings):
    return settings.rate


@pytest.fixture
def random_effect(settings):
    return settings.random_effect


def test_settings(settings):
    assert settings.csmr_cod_output_version_id == 84


def test_model_metadata(model):
    assert model.random_seed == 495279142
    assert model.decomp_step_id == 3
    assert model.model_version_id == 472515
    assert model.add_csmr_cause == 587
    assert model.crosswalk_version_id == 5699


def test_model_setup(model):
    assert model.default_age_grid == [
        0.0, 0.01917808, 0.07671233, 1.0,
        5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0
    ]
    assert model.default_time_grid == [
        1990.0, 1995.0, 2000.0, 2005.0, 2010.0, 2015.0, 2016.0
    ]
    assert model.birth_prev == 0
    assert model.ode_step_size == 5
    assert model.minimum_meas_cv == 0.1
    assert model.rate_case == 'iota_pos_rho_zero'
    assert model.data_density == 'log_gaussian'
    assert model.constrain_omega == 1
    assert model.addl_ode_stpes == [0.01917808, 0.07671233, 1.0]
    assert model.relabel_incidence == 2


def test_model_cascade_method(model):
    assert model.drill == 'drill'
    assert model.drill_location_start == 70
    assert model.drill_sex == 2


def test_policies(policies):
    pass


def test_random_effect(random_effect):
    pass


def test_rate(rate):
    pass


def test_study_covariate(study_covariate):
    pass


def test_country_covariate(country_covariate):
    pass
