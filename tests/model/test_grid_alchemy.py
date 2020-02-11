import pytest
import numpy as np

from cascade_at.settings.settings import load_settings
from cascade_at.settings.base_case import BASE_CASE
from cascade_at.model.grid_alchemy import Alchemy


@pytest.fixture
def alchemy():
    return Alchemy(load_settings(BASE_CASE))


def test_construct_age_grid(alchemy):
    grid = alchemy.construct_age_time_grid()
    assert np.testing.assert_array_equal(grid['time'], np.array([1990., 1995., 2000., 2005., 2010., 2015., 2016.]))
    assert np.testing.assert_array_equal(grid['age'], np.array([
        0.000000e+00, 1.917808e-02, 7.671233e-02, 1.000000e+00,
        5.000000e+00, 1.000000e+01, 2.000000e+01, 3.000000e+01,
        4.000000e+01, 5.000000e+01, 6.000000e+01, 7.000000e+01,
        8.000000e+01, 9.000000e+01, 1.000000e+02
    ]))


def test_construct_single_age_time_grid(alchemy):
    grid = alchemy.construct_single_age_time_grid()
    assert np.testing.assert_array_equal(grid['time'], np.array([2005.]))
    assert np.testing.assert_array_equal(grid['age'], np.array([0.]))


def test_construct_two_level_model():
    pass