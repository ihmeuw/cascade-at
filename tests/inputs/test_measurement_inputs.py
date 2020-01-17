import pytest
from copy import deepcopy

from cascade_at.settings.base_case import BASE_CASE
from cascade_at.settings.settings import load_settings
from cascade_at.inputs.measurement_inputs import MeasurementInputsFromSettings


@pytest.fixture(scope='module')
def mi(asdr, cv, csmr, population, covariate_data):
    s = load_settings(BASE_CASE)
    m = MeasurementInputsFromSettings(settings=s)
    m.asdr = deepcopy(asdr)
    m.csmr = deepcopy(csmr)
    m.data = deepcopy(cv)
    m.covariate_data = [deepcopy(covariate_data)]
    m.population = deepcopy(population)
    return m


@pytest.fixture(scope='module')
def dismod_data(mi):
    return mi.configure_inputs_for_dismod(settings=load_settings(BASE_CASE)).dismod_data

