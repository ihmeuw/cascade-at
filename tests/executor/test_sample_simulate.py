"""
These are tests for both executor/sample_simulate and executor/predict_sample since they will
typically be executed in sequence on the same database.
"""

import pytest
import os
import numpy as np

from cascade_at.dismod.api.dismod_io import DismodIO
from cascade_at.dismod.api.dismod_extractor import DismodExtractor
from cascade_at.model.grid_alchemy import Alchemy
from cascade_at.dismod.api.dismod_filler import DismodFiller
from cascade_at.dismod.api.run_dismod import run_dismod_commands
from cascade_at.executor.sample_simulate import simulate, FitSample
from cascade_at.executor.sample_simulate import sample_simulate_pool, sample_simulate_sequence
from cascade_at.executor.sample_simulate import SampleSimulateError
from cascade_at.executor.predict_sample import predict_sample


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


def test_sample_simulate_empty(filler, dismod):
    with pytest.raises(SampleSimulateError):
        simulate(NAME, n_sim=1)


def test_sample_simulate(filler, dismod):
    run_dismod_commands(NAME, commands=['init', 'fit fixed'])
    simulate(NAME, n_sim=2)


def test_fit_sample(filler, dismod):
    fit = FitSample(main_db=NAME, index_file_pattern='sample_{index}.db', fit_type='fixed')
    result = fit(1)
    assert all(result.sample_index) == 1
    assert len(result) == 250


def test_sample_simulate_sequence(filler, dismod):
    sample_simulate_sequence(NAME, n_sim=2)
    di = DismodIO(NAME)
    assert len(di.sample) == 500
    assert all(di.sample.columns == ['sample_id', 'sample_index', 'var_id', 'var_value'])
    assert all(di.sample.iloc[0:250].sample_index == 0)
    assert all(di.sample.iloc[250:500].sample_index == 1)
    assert all(~np.isnan(di.sample.var_value))


def test_sample_simulate_pool(filler, dismod):
    sample_simulate_pool(NAME, 'sample_{index}.db', fit_type='fixed', n_pool=1, n_sim=2)
    di = DismodIO(NAME)
    assert len(di.sample) == 500
    assert all(di.sample.columns == ['sample_id', 'sample_index', 'var_id', 'var_value'])
    assert all(di.sample.iloc[0:250].sample_index == 0)
    assert all(di.sample.iloc[250:500].sample_index == 1)
    assert all(~np.isnan(di.sample.var_value))


def test_predict_sample(mi, settings, dismod):
    alchemy = Alchemy(settings)
    predict_sample(
        inputs=mi, alchemy=alchemy, settings=settings,
        source_db_path=NAME, target_locations=[72], target_sexes=[2]
    )
    di = DismodIO(NAME)
    assert len(di.predict) == 2 * len(di.avgint)
    assert all(di.predict.c_sex_id == 2)


def test_gather_child_draws(mi, settings, dismod):
    de = DismodExtractor(NAME)
    draws = de.gather_draws_for_prior_grid(
        location_id=72, sex_id=2,
        rates=['iota', 'chi', 'pini']
    )
    for rate in ['iota', 'chi', 'pini']:
        assert rate in draws
        assert draws[rate]['value'].shape[-1] == 2

    # pini will not have any dage or dtime draws because it only has one age and time
    for rate in ['iota', 'chi']:
        assert draws[rate]['dage'].shape[-1] == 2
        assert draws[rate]['dtime'].shape[-1] == 2
