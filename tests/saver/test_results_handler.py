import pytest

import pandas as pd
import numpy as np

from cascade_at.saver.results_handler import ResultsHandler


@pytest.fixture
def means():
    return pd.DataFrame({
        'model_version_id': np.repeat([0], repeats=10),
        'location_id': np.repeat([0], repeats=10),
        'year_id': np.arange(1990, 2000),
        'sex_id': np.repeat([2], repeats=10),
        'age_group_id': np.repeat([2], repeats=10),
        'measure_id': np.repeat([6], repeats=10),
        'mean': np.random.randn(10)
    })


@pytest.fixture
def draws():
    return pd.DataFrame({
        'model_version_id': np.repeat([0], repeats=10),
        'location_id': np.repeat([0], repeats=10),
        'year_id': np.arange(1990, 2000),
        'sex_id': np.repeat([2], repeats=10),
        'age_group_id': np.repeat([2], repeats=10),
        'measure_id': np.repeat([6], repeats=10),
        'draw_0': np.random.randn(10),
        'draw_1': np.random.randn(10)
    })


def test_saver_summaries_mean(means):
    rh = ResultsHandler()
    df = rh.summarize_results(df=means)
    assert (df.columns == rh.draw_keys + ['mean', 'lower', 'upper']).all()
    assert (df['mean'] == means['mean']).all()
    assert (df['lower'] == means['mean']).all()
    assert (df['upper'] == means['mean']).all()


def test_saver_summaries_draws(draws):
    rh = ResultsHandler()
    df = rh.summarize_results(df=draws)
    assert (df.columns == rh.draw_keys + ['mean', 'lower', 'upper']).all()
    assert (df['mean'] == draws[['draw_0', 'draw_1']].mean(axis=1)).all()
    assert (df['lower'] == draws[['draw_0', 'draw_1']].quantile(0.025, axis=1)).all()
    assert (df['upper'] == draws[['draw_0', 'draw_1']].quantile(0.975, axis=1)).all()

