import pytest

import pandas as pd

from cascade.input_data.configuration import covariates


@pytest.fixture
def sample_measurements():
    return pd.DataFrame([
        # ranges on age and time
        dict(age_lower=0, age_upper=5, time_lower=1970, time_upper=1971),
        # point age
        dict(age_lower=20, age_upper=20, time_lower=1970, time_upper=1971),
        # point time
        dict(age_lower=0, age_upper=5, time_lower=1983, time_upper=1983),
        # point age and time
        dict(age_lower=62, age_upper=62, time_lower=1975, time_upper=1975),
        # age out of bounds and a range
        dict(age_lower=110, age_upper=120, time_lower=1970, time_upper=1971),
        # age out of bounds and a point
        dict(age_lower=120, age_upper=120, time_lower=1970, time_upper=1971),
        # time out of bounds and a range
        dict(age_lower=0, age_upper=5, time_lower=1900, time_upper=1910),
        # time out of bounds and a point
        dict(age_lower=0, age_upper=5, time_lower=1915, time_upper=1915),
    ])


@pytest.fixture
def sample_covariate():
    return pd.DataFrame([
        dict(age_lower=0, age_upper=5, time_lower=1969, time_upper=1969, value=2.0),
        dict(age_lower=5, age_upper=10, time_lower=1970, time_upper=1971, value=0.2),
        dict(age_lower=20, age_upper=20, time_lower=1970, time_upper=1971, value=0.5),
        dict(age_lower=30, age_upper=35, time_lower=1970, time_upper=1971, value=0.9),
        dict(age_lower=35, age_upper=40, time_lower=1970, time_upper=1971, value=0.8),
        dict(age_lower=60, age_upper=60, time_lower=1983, time_upper=1983, value=0.7),
    ])


def test_covariate_dummy(sample_measurements, sample_covariate):
    m = sample_measurements.copy()
    m["cov_name"] = covariates.covariate_to_measurements_dummy(sample_measurements, sample_covariate)


def test_covariate_nearest(sample_measurements, sample_covariate):
    m = sample_measurements.copy()
    m["cov_name"] = covariates.covariate_to_measurements_nearest_favoring_same_year(
        sample_measurements, sample_covariate)
    c = m["cov_name"]
    expected = [0.2, 0.5, 0.7, 0.8, 0.8, 0.8, 2.0, 2.0]
    for idx, ex in enumerate(expected):
        assert c.iloc[idx] == ex, f"idx {idx} versus {ex}"
