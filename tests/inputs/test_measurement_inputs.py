import pytest

from cascade_at.settings.base_case import BASE_CASE
from cascade_at.settings.settings import load_settings
from cascade_at.inputs.measurement_inputs import MeasurementInputsFromSettings


@pytest.fixture(scope='module')
def mi():
    s = load_settings(BASE_CASE)
    m = MeasurementInputsFromSettings(settings=s)

