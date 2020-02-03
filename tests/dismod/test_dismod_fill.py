import pytest
from copy import deepcopy
from types import SimpleNamespace

from cascade_at.settings.base_case import BASE_CASE
from cascade_at.model.grid_alchemy import Alchemy
from cascade_at.context.model_context import Context
from cascade_at.settings.settings import load_settings
from cascade_at.inputs.measurement_inputs import MeasurementInputsFromSettings
from cascade_at.dismod.api.dismod_filler import DismodFiller


@pytest.fixture
def context(tmp_path):
    c = Context(model_version_id=0, make=True, configure_application=False,
                root_directory=tmp_path)
    return c


@pytest.fixture
def settings():
    return load_settings(BASE_CASE)


@pytest.fixture
def demographics():
    d = SimpleNamespace()
    d.age_group_id = [2]
    d.location_id = [70]
    d.sex_id = [2]
    d.year_id = [1990]
    return d


@pytest.fixture
def mi(asdr, cv, csmr, population, covariate_data, settings, demographics):
    m = MeasurementInputsFromSettings(settings=settings)
    m.asdr = deepcopy(asdr)
    m.csmr = deepcopy(csmr)
    m.data = deepcopy(cv)
    m.covariate_data = [deepcopy(covariate_data)]
    m.population = deepcopy(population)
    m.demographics = demographics
    m.configure_inputs_for_dismod(settings)
    return m


@pytest.fixture
def dismod_filler(mi, settings, tmp_path):
    alchemy = Alchemy(settings)
    d = DismodFiller(
        path=tmp_path / 'temp.db',
        settings_configuration=settings,
        measurement_inputs=mi,
        grid_alchemy=alchemy,
        parent_location_id=70,
        sex_id=2
    )
    return d


def test_dismod_filler(dismod_filler):
    dismod_filler.fill_for_parent_child()




