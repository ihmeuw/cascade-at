import pytest
import numpy as np
from copy import deepcopy

from cascade_at.settings.base_case import BASE_CASE
from cascade_at.context.model_context import Context
from cascade_at.settings.settings import load_settings
from cascade_at.inputs.measurement_inputs import MeasurementInputsFromSettings


@pytest.fixture
def context(tmp_path):
    c = Context(model_version_id=0, make=True, configure_application=False,
                root_directory=tmp_path)
    return c


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


@pytest.mark.parametrize("column,values", [
    ("age_lower", {0: 0.0, 1: 0.0, 2: 0.0}),
    ("age_upper", {0: 0.01917808, 1: 0.01917808, 2: 0.01917808}),
    ("hold_out", {0: 0.0, 1: 1.0, 2: 0.0}),
    ("location_id", {0: 70.0, 1: 70.0, 2: 70.0}),
    ("meas_std", {0: 3e-06, 1: 0.010204269138493082, 2: 1.020426913849308e-06}),
    ("meas_value", {0: 4e-05, 1: 0.17, 2: 5e-06}),
    ("measure", {0: 'Tincidence', 1: 'mtall', 2: 'mtspecific'}),
    ("name", {0: '342686', 1: np.nan, 2: np.nan}),
    ("sex_id", {0: 2.0, 1: 2.0, 2: 2.0}),
    ("time_lower", {0: 1990.0, 1: 1990.0, 2: 1990.0}),
    ("time_upper", {0: 1991.0, 1: 1991.0, 2: 1991.0}),
    ("density", {0: 'log_gaussian', 1: 'log_gaussian', 2: 'log_gaussian'}),
    ("eta", {0: 1e-05, 1: 1e-05, 2: 1e-05}),
    ("c_diabetes_fpg", {0: 0.96, 1: 0.96, 2: 0.96}),
    ("s_sex", {0: -0.5, 1: -0.5, 2: -0.5}),
    ("s_one", {0: 1.0, 1: 1.0, 2: 1.0})
])
def test_dismod_data(dismod_data, column, values):
    assert dismod_data.to_dict()[column] == values


def test_pickle(mi, context):
    settings = BASE_CASE
    context.write_inputs(inputs=mi, settings=settings)
    p_inputs, p_alchemy, p_settings = context.read_inputs()
    assert len(p_inputs.dismod_data) == len(mi.dismod_data)
