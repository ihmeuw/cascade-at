import pytest
import pandas as pd
import numpy as np

from cascade_at.inputs.utilities.gbd_ids import make_age_intervals, make_time_intervals
from cascade_at.inputs.utilities.gbd_ids import map_id_from_interval_tree


@pytest.fixture
def time_df():
    return pd.DataFrame({
        'year_id': np.arange(1990, 1995)
    })


@pytest.fixture
def age_df():
    return pd.DataFrame({
        'age_group_id': np.arange(6, 11),
        'age_lower': np.arange(5, 30, 5),
        'age_upper': np.arange(10, 35, 5)
    })


def test_make_age_intervals(age_df):
    ints = make_age_intervals(df=age_df)
    for age in age_df.age_lower.unique():
        assert ints.overlaps(age)


def test_make_age_intervals_gbd(ihme):
    ints = make_age_intervals(gbd_round_id=6)
    for age in np.arange(0, 96):
        assert ints.overlaps(age)


def test_make_time_intervals(time_df):
    ints = make_time_intervals(df=time_df)
    for time in np.arange(1990, 1995):
        assert ints.overlaps(time)


def test_make_time_intervals_gbd():
    ints = make_time_intervals()
    for time in np.arange(1980, 2020):
        assert ints.overlaps(time)


def test_map_id_from_interval_tree(age_df):
    ints = make_time_intervals()
    assert map_id_from_interval_tree(1990, ints) == 1990
    ints = make_age_intervals(df=age_df)
    assert map_id_from_interval_tree(5, ints) == 6
