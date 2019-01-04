import pytest

import numpy as np
import pandas as pd

import cascade.input_data.configuration.construct_country
from cascade.input_data import InputDataError


@pytest.fixture
def sample_measurements():
    return pd.DataFrame([
        # ranges on age and time
        dict(age_lower=0, age_upper=5, time_lower=1970, time_upper=1971, sex_id=1),
        # point age
        dict(age_lower=20, age_upper=20, time_lower=1970, time_upper=1971, sex_id=1),
        # point time
        dict(age_lower=0, age_upper=5, time_lower=1983, time_upper=1983, sex_id=1),
        # point age and time
        dict(age_lower=62, age_upper=62, time_lower=1975, time_upper=1975, sex_id=1),
        # age out of bounds and a range
        dict(age_lower=110, age_upper=120, time_lower=1970, time_upper=1971, sex_id=1),
        # age out of bounds and a point
        dict(age_lower=120, age_upper=120, time_lower=1970, time_upper=1971, sex_id=1),
        # time out of bounds and a range
        dict(age_lower=0, age_upper=5, time_lower=1900, time_upper=1910, sex_id=1),
        # time out of bounds and a point
        dict(age_lower=0, age_upper=5, time_lower=1915, time_upper=1915, sex_id=1),
    ])


@pytest.fixture
def sample_covariate():
    return pd.DataFrame([
        dict(age_lower=0, age_upper=5, time_lower=1969, time_upper=1969, mean_value=2.0, sex_id=1),
        dict(age_lower=5, age_upper=10, time_lower=1970, time_upper=1971, mean_value=0.2, sex_id=1),
        dict(age_lower=20, age_upper=20, time_lower=1970, time_upper=1971, mean_value=0.5, sex_id=1),
        dict(age_lower=30, age_upper=35, time_lower=1970, time_upper=1971, mean_value=0.9, sex_id=1),
        dict(age_lower=35, age_upper=40, time_lower=1970, time_upper=1971, mean_value=0.8, sex_id=1),
        dict(age_lower=60, age_upper=60, time_lower=1983, time_upper=1983, mean_value=0.7, sex_id=1),
    ])


def test_covariate_nearest(sample_measurements, sample_covariate):
    m = sample_measurements.copy()
    name = cascade.input_data.configuration.construct_country.covariate_to_measurements_nearest_favoring_same_year(
        sample_measurements, sample_measurements.sex_id, sample_covariate
    )
    m["cov_name"] = name
    c = m["cov_name"]
    expected = [0.2, 0.5, 0.7, 0.8, 0.8, 0.8, 2.0, 2.0]
    for idx, ex in enumerate(expected):
        assert c.iloc[idx] == ex, f"idx {idx} versus {ex}"


def test_convert_age_groups():
    age_groups = pd.DataFrame({
        "age_group_id": [2, 3, 8],
        "age_group_years_start": [0.0, 1 / 52, 10],
        "age_group_years_end": [1 / 52, 1, 15],
    })
    by_id = pd.DataFrame({
        "age_group_id": [3, 2, 8, 3, 2, 8],
        "year_id": [1970, 1971, 1975, 1980, 1990, 1995],
        "sex_id": [1, 1, 1, 2, 2, 2],
        "value": [0, 1, 2, 3, 4, 5],
    })
    with_ranges = cascade.input_data.configuration.construct_country.convert_gbd_ids_to_dismod_values(by_id, age_groups)
    for column in ["age_lower", "age_upper", "time_lower", "time_upper"]:
        assert column in with_ranges.columns

    assert np.allclose(with_ranges["age_lower"].values, [1 / 52, 0, 10, 1 / 52, 0, 10])
    assert np.allclose(with_ranges["sex_id"].values, [1, 1, 1, 2, 2, 2])


def test_convert_age_groups_failure():
    groups = pd.DataFrame({
        "age_group_id": [2, 3, 8],
        "age_group_years_start": [0.0, 1 / 52, 10],
        "age_group_years_end": [1 / 52, 1, 15],
    })
    by_id = pd.DataFrame({
        "age_group_id": [3, 2, 8, 3, 2, 10],
        "year_id": [1970, 1971, 1975, 1980, 1990, 1995],
        "sex_id": [1, 1, 1, 2, 2, 2],
        "value": [0, 1, 2, 3, 4, 5],
    })
    with pytest.raises(InputDataError):
        cascade.input_data.configuration.construct_country.convert_gbd_ids_to_dismod_values(by_id, groups)
