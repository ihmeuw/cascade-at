from copy import copy

import pytest

from cascade_at.settings.base_case import BASE_CASE
from cascade_at.settings.convert import data_cv_from_settings
from cascade_at.settings.convert import min_cv_from_settings
from cascade_at.settings.settings import load_settings


@pytest.fixture
def settings():
    s = copy(BASE_CASE)
    s.update({
        "min_cv": [
            {
                "cascade_level_id": "most_detailed",
                "value": 0.1
            },
            {
                "cascade_level_id": "level1",
                "value": 1.0
            }
        ],
        "min_cv_by_rate": [
            {
                "rate_measure_id": "iota",
                "cascade_level_id": "most_detailed",
                "value": 0.2
            },
            {
                "rate_measure_id": "chi",
                "cascade_level_id": "most_detailed",
                "value": 0.3
            },
            {
                "rate_measure_id": "chi",
                "cascade_level_id": "level1",
                "value": 10.0
            }
        ]
    })
    return s


@pytest.mark.skip(reason="There is a bug in the viz tool.")
def test_min_cv_from_settings(settings):
    new = load_settings(settings)
    cv = min_cv_from_settings(settings=new)
    assert cv['most_detailed']['chi'] == 0.3
    assert cv['most_detailed']['iota'] == 0.2
    assert cv['most_detailed']['omega'] == 0.1
    assert cv['level1']['omega'] == 1.0
    assert cv['level1']['chi'] == 10.0
    assert cv['dummy']['omega'] == 0.0
    assert cv['dummy']['iota'] == 0.0
    assert cv['dummy']['chi'] == 0.0


# expected behavior while there is a bug in the viz tool
def test_min_cv_from_settings_TEMP(settings):
    new = load_settings(settings)
    cv = min_cv_from_settings(settings=new)
    assert cv['most_detailed']['chi'] == 0.3
    assert cv['most_detailed']['iota'] == 0.2
    assert cv['most_detailed']['omega'] == 0.1
    assert cv['level1']['omega'] == 1.0
    assert cv['level1']['chi'] == 10.0
    assert cv['dummy']['omega'] == 0.1
    assert cv['dummy']['iota'] == 0.1
    assert cv['dummy']['chi'] == 0.1


def test_data_cv_from_settings():
    settings = BASE_CASE.copy()
    s = load_settings(settings)
    cv = data_cv_from_settings(settings=s)
    assert cv['iota'] == 0.2


def test_data_cv_from_settings_by_integrand():
    settings = BASE_CASE.copy()
    settings.update({
        "data_cv_by_integrand": [{
            "integrand_measure_id": 5,
            "value": 0.5
        }]
    })
    s = load_settings(settings)
    cv = data_cv_from_settings(settings=s)
    assert cv['prevalence'] == 0.5
    assert cv['iota'] == 0.2
