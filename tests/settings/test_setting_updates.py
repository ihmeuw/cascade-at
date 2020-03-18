import pytest

from cascade_at.settings.base_case import BASE_CASE
from cascade_at.settings.settings import load_settings
from cascade_at.core.form.fields import FormList


@pytest.fixture
def data_cv_by_integrand():
    return {
        "data_cv_by_integrand": [
            {
                "integrand_measure_id": 5,
                "value": 0.01
            },
            {
                "integrand_measure_id": 9,
                "value": 0.3
            },
            {
                "integrand_measure_id": 15,
                "value": 0.2
            }
        ]
    }


@pytest.mark.parametrize("num,measure,value", [
    (0, 5, 0.01),
    (1, 9, 0.3),
    (2, 15, 0.2)
])
def test_data_cv_by_integrand_update(data_cv_by_integrand, num, measure, value):
    settings = BASE_CASE.copy()
    settings.update(data_cv_by_integrand)
    s = load_settings(settings)
    assert type(s.data_cv_by_integrand) == FormList
    assert s.data_cv_by_integrand[num].integrand_measure_id == measure
    assert s.data_cv_by_integrand[num].value == value
