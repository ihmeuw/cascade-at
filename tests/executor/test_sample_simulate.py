import pytest
import os
import numpy as np

from cascade_at.dismod.api.dismod_io import DismodIO
from cascade_at.model.grid_alchemy import Alchemy
from cascade_at.dismod.api.dismod_filler import DismodFiller
from cascade_at.dismod.api.run_dismod import run_dismod_commands
from cascade_at.executor.sample_simulate import simulate, FitSample
from cascade_at.executor.sample_simulate import sample_simulate_pool, sample_simulate_sequence
from cascade_at.executor.sample_simulate import SampleSimulateError

NAME = 'sample.db'
if os.path.isfile(NAME):
    os.remove(NAME)


@pytest.fixture
def path():
    return NAME


@pytest.fixture(scope='module')
def filler(mi, settings):
    alchemy = Alchemy(settings)
    d = DismodFiller(
        path=NAME,
        settings_configuration=settings,
        measurement_inputs=mi,
        grid_alchemy=alchemy,
        parent_location_id=70,
        sex_id=2
    )
    d.fill_for_parent_child()
    return d


def test_sample_simulate_empty(filler):
    with pytest.raises(SampleSimulateError):
        simulate(NAME, n_sim=1)


def test_sample_simulate(filler):
    run_dismod_commands(NAME, commands=['init', 'fit fixed'])
    simulate(NAME, n_sim=2)


def test_fit_sample(filler):
    fit = FitSample(main_db=NAME, index_file_pattern='sample_{index}.db', fit_type='fixed')
    result = fit(1)
    assert all(result.sample_index) == 1
    assert len(result) == 250


def test_sample_simulate_sequence(filler):
    sample_simulate_sequence(NAME, n_sim=2)
    di = DismodIO(NAME)
    assert len(di.sample) == 500
    assert all(di.sample.columns == ['sample_id', 'sample_index', 'var_id', 'var_value'])
    assert all(di.sample.iloc[0:250].sample_index == 0)
    assert all(di.sample.iloc[250:500].sample_index == 1)
    assert all(~np.isnan(di.sample.var_value))


def test_sample_simulate_pool(filler):
    sample_simulate_pool(NAME, 'sample_{index}.db', fit_type='fixed', n_pool=1, n_sim=2)
    di = DismodIO(NAME)
    assert len(di.sample) == 500
    assert all(di.sample.columns == ['sample_id', 'sample_index', 'var_id', 'var_value'])
    assert all(di.sample.iloc[0:250].sample_index == 0)
    assert all(di.sample.iloc[250:500].sample_index == 1)
    assert all(~np.isnan(di.sample.var_value))

