import pytest

from cascade_at.settings.base_case import BASE_CASE
from cascade_at.settings.settings import load_settings
from cascade_at.core.form.fields import FormList


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
    assert model.data_cv == 0.2
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
    assert policies.estimate_emr_from_prevalence == 0
    assert policies.use_weighted_age_group_midpoints == 1
    assert policies.with_hiv == 1
    assert policies.age_group_set_id == 12
    assert policies.exclude_relative_risk == 1
    assert policies.meas_std_effect == 'add_var_scale_log'
    assert policies.limited_memory_max_history_fixed == 30
    assert policies.gbd_round_id == 6


def test_random_effect(random_effect):
    assert type(random_effect) == FormList
    assert len(random_effect) == 1


def test_rate(rate):
    assert type(rate) == FormList
    assert len(rate) == 3


def test_study_covariate(study_covariate):
    assert type(study_covariate) == FormList
    assert len(study_covariate) == 1


def test_country_covariate(country_covariate):
    assert type(country_covariate) == FormList
    assert len(country_covariate) == 1
