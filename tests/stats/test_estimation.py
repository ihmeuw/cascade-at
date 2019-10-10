import pytest

import pandas as pd
import numpy as np

from cascade.stats.estimation import meas_bounds_to_stdev, ess_to_stdev
from cascade.stats.estimation import wilson_interval, check_bundle_uncertainty_columns


def test_meas_bounds_to_stdev__bad_bounds():
    df = pd.DataFrame({"meas_lower": [0, 0, 1, 1], "meas_value": [0, -0.01, 1, 1], "meas_upper": [-10, 0, 1, 10]})

    with pytest.raises(ValueError):
        meas_bounds_to_stdev(df)


def test_wilson_interval():
    assert np.isclose(wilson_interval(0.2, 100), 0.04118296, atol=1e-6)
    assert np.isclose(
        wilson_interval(pd.Series([0.2, 0.2]), pd.Series([100, 100])),
        pd.Series([0.04118296, 0.04118296]),
        atol=1e-6
    ).all()
    assert np.isclose(
        wilson_interval(np.array([0.2, 0.2]), np.array([100, 100])),
        np.array([0.04118296, 0.04118296]),
        atol=1e-6
    ).all()


def test_ess_to_stdev():
    mean = pd.Series([0.1, 1.2])
    ess = pd.Series([20, 30])
    stdev = ess_to_stdev(mean=mean, ess=ess, proportion=False)
    assert np.isclose(stdev, np.array([0.07472136, 0.20000000]), atol=1e-6).all()


def test_check_bundle_uncertainty_cols():
    df = pd.DataFrame([(a, b, c, d, e)
                       for a in [0.1]
                       for b in [0.1, np.nan]
                       for c in [0.1, np.nan]
                       for d in [0.1, 0, np.nan]
                       for e in [0.1, 0, np.nan]])
    df.columns = ['standard_error', 'lower', 'upper',
                  'effective_sample_size', 'sample_size']
    has_se, has_ui, has_ess, has_ss = check_bundle_uncertainty_columns(df)

    assert has_se.all()
    assert has_ui[:int(len(df)/4)].all()
    assert ~has_ui[int(len(df)/4):].all()
    assert (has_ess == np.tile(
        np.concatenate([
            np.repeat([True], repeats=3), np.repeat([False], repeats=6),
        ]),
        reps=4
    )).all()
    assert (has_ss == np.tile(
        np.array([True, False, False]),
        reps=12
    )).all()


@pytest.fixture
def df():
    return pd.DataFrame({
        'mean': np.repeat([0.5], repeats=5),
        'standard_error': [0.1, 0.0, np.nan, 0.4, 0.2],
        'lower': [0.001, 0.000, 0.001, 0.004, 0.001],
        'upper': [0.8, 0.6, 0.9, 0.7, 0.6],
        'effective_sample_size': [80, 100, 100, 80, 100],
        'sample_size': [150, 200, 200, 150, 200]
    })


def test_stdev_from_bundle_data(df):
    pass