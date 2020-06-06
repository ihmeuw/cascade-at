import pytest
import pandas as pd
import numpy as np

from cascade_at.model.var import Var
from cascade_at.model.utilities.grid_helpers import rectangular_data_to_var


@pytest.fixture
def rectangular_data():
    return pd.DataFrame({
        'age_lower': np.tile([0.0, 1.0, 5.0], reps=2),
        'age_upper': np.tile([1., 5., 10.], reps=2),
        'time_lower': np.repeat([1950, 1951], repeats=3),
        'time_upper': np.repeat([1951, 1952], repeats=3),
        'mean': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        'std': [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]
    })


@pytest.mark.parametrize("shuffle", [False, True, True])
def test_rectangular_data_to_var(rectangular_data, shuffle):
    """ Tests that data can be transformed easily into rectangular data ``Var''.
        Also make sure that the data does not need to be sorted ahead of time."""
    if shuffle:
        data = rectangular_data.sample(frac=1)
    else:
        data = rectangular_data.copy()
    rectangular = rectangular_data_to_var(data)
    assert type(rectangular) is Var
    assert len(rectangular.grid) == 6
    assert rectangular.variable_count() == 6
    assert (np.array(rectangular.grid.columns) == np.array(['age', 'time', 'mean'])).all()
    assert (rectangular.grid.age == np.repeat([0.5, 3.0, 7.5], repeats=2)).all()
    assert (rectangular.grid.time == np.tile([1950.5, 1951.5], reps=3)).all()
    assert rectangular[0.5, 1950.5] == 0.01
    assert rectangular[3.0, 1950.5] == 0.02
    assert rectangular[7.5, 1950.5] == 0.03
    assert rectangular[0.5, 1951.5] == 0.04
    assert rectangular[3.0, 1951.5] == 0.05
    assert rectangular[7.5, 1951.5] == 0.06
