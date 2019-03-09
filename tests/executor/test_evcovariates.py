from types import SimpleNamespace

import pytest

from cascade.executor.covariate_description import create_covariate_specifications, EpiVizCovariate


@pytest.fixture
def covariate_settings():
    study = list()
    country = list()
    setting = SimpleNamespace()
    setting.study_covariate_id = 1604  # one
    setting.mulcov_type = "rate_value"
    setting.measure_id = 41
    setting.transformation = 0
    study.append(setting)
    setting = SimpleNamespace()
    setting.study_covariate_id = 0  # sex
    setting.mulcov_type = "meas_value"
    setting.measure_id = 41
    setting.transformation = 0
    study.append(setting)
    setting = SimpleNamespace()
    setting.country_covariate_id = 156  # vehicles_4wheels_pc
    setting.mulcov_type = "meas_value"
    setting.measure_id = 41  # Sincidence
    setting.transformation = 1
    country.append(setting)
    setting = SimpleNamespace()
    setting.country_covariate_id = 1998  # med_schools
    setting.mulcov_type = "meas_std"
    setting.measure_id = 38  # prevalence
    setting.transformation = 3
    country.append(setting)
    setting = SimpleNamespace()
    setting.country_covariate_id = 156
    setting.mulcov_type = "meas_std"
    setting.measure_id = 41
    setting.transformation = 1
    country.append(setting)
    return study, country


def test_covariate_ordering():
    sex = EpiVizCovariate("study", 0, 0)
    one = EpiVizCovariate("study", 1604, 0)
    other = EpiVizCovariate("country", 1400, 3)
    # The spec comparison would make a different order.
    assert ("country", 1400, 3) < ("study", 1604, 0)
    assert sex < one
    assert sex < other
    assert one < other
    unsorted = [other, one, sex]
    ordered = sorted(unsorted)
    assert ordered[0] == sex
    assert ordered[1] == one
    assert ordered[2] == other


def test_create_covariates(covariate_settings):
    study, country = covariate_settings
    multipliers, covariates = create_covariate_specifications(study, country)
    assert len(multipliers) == 5
    assert len(covariates) == 4
    # Assign fake names so we can test this without IHME databases.
    for cov in covariates:
        cov.untransformed_covariate_name = cov.covariate_id

    check_multipliers = {
        ("alpha", ("1604", "iota")),
        ("beta", ("0", "Sincidence")),
        ("beta", ("156_log", "Sincidence")),
        ("gamma", ("1998_squared", "prevalence")),
        ("gamma", ("156_log", "Sincidence"))
    }
    for mult in multipliers:
        assert (mult.group, mult.key) in check_multipliers

    check_covs = {
        ("study", 1604, 0),
        ("study", 0, 0),
        ("country", 156, 1),
        ("country", 1998, 3),
    }
    for cov in covariates:
        assert cov.spec in check_covs
    # Ordering for covariates isn't working.
    # Cannot assert that the first two are the sex and one covariates.
