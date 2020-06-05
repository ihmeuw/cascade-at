import pytest

import pandas as pd
import numpy as np

from cascade_at.inputs.uncertainty import meas_bounds_to_stdev, ess_to_stdev, stdev_from_crosswalk_version
from cascade_at.inputs.uncertainty import wilson_interval, check_crosswalk_version_uncertainty_columns


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


@pytest.fixture
def simple_df():
    df = pd.DataFrame([(a, b, c, d, e)
                       for a in [0.1]
                       for b in [0.1, np.nan]
                       for c in [0.1, np.nan]
                       for d in [0.1, 0, np.nan]
                       for e in [0.1, 0, np.nan]])
    df.columns = ['standard_error', 'lower', 'upper',
                  'effective_sample_size', 'sample_size']
    return df


def test_check_bundle_uncertainty_cols(simple_df):
    has_se, has_ui, has_ess, has_ss = check_crosswalk_version_uncertainty_columns(simple_df)

    assert has_se.all()
    assert has_ui[:int(len(simple_df)/4)].all()
    assert ~has_ui[int(len(simple_df)/4):].all()
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


def test_check_bundle_uncertainty_cols_error(simple_df):
    df = simple_df.copy()
    df['standard_error'] = -1
    with pytest.raises(ValueError):
        check_crosswalk_version_uncertainty_columns(df)


@pytest.fixture
def df():
    return pd.DataFrame({
        'mean': np.repeat([0.5], repeats=7),
        'standard_error':           [0.1,   0.0,    np.nan, np.nan, np.nan, 0.1,    0.1],
        'lower':                    [0.001, np.nan, 0.001,  np.nan, 0.001,  0.001,  0.001],
        'upper':                    [0.8,   0.6,    0.9,    0.7,    np.nan, np.nan, 0.6],
        'effective_sample_size':    [0,     np.nan, 100,    np.nan, 9,      10,     np.nan],
        'sample_size':              [150,   200,    200,    150,    50,     100,    100]
    })


def test_stdev_from_bundle_data(df):
    standard_error = stdev_from_crosswalk_version(df)
    assert (np.isclose(standard_error, np.array([0.1, 0.05, 0.22934,
                                                 0.05773503, 0.2347179, 0.1, 0.1]), atol=1e-5)).all()
