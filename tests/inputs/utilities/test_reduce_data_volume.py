import pytest
import pandas as pd
import numpy as np

from cascade_at.inputs.utilities.reduce_data_volume import decimate_years


@pytest.fixture
def fake_data():
    return pd.DataFrame({
        'location_id': np.repeat([101, 102], len(list(range(1990, 2000))) / 2),
        'time_lower': list(range(1990, 2000)),
        'time_upper': list(range(1991, 2001)),
        'meas_value': np.random.randn(len(list(range(1990, 2000)))),
        'meas_std': np.random.randn(len(list(range(1990, 2000))))
    })


def test_decimate_years(fake_data):
    location_101_mean = fake_data.loc[fake_data.location_id == 101][['meas_value', 'meas_std']].mean()
    location_102_mean = fake_data.loc[fake_data.location_id == 102][['meas_value', 'meas_std']].mean()

    decimated = decimate_years(data=fake_data)
    assert len(decimated) == 2
    assert all(decimated.columns == ['location_id', 'time_lower', 'time_upper', 'meas_value', 'meas_std'])
    pd.testing.assert_series_equal(
        decimated.loc[decimated.location_id == 101].iloc[0],
        pd.Series({
            'location_id': 101,
            'time_lower': 1992.5,
            'time_upper': 1992.5,
            'meas_value': location_101_mean['meas_value'],
            'meas_std': location_101_mean['meas_std']
        }),
        check_names=False
    )
    pd.testing.assert_series_equal(
        decimated.loc[decimated.location_id == 102].iloc[0],
        pd.Series({
            'location_id': 102,
            'time_lower': 1997.5,
            'time_upper': 1997.5,
            'meas_value': location_102_mean['meas_value'],
            'meas_std': location_102_mean['meas_std']
        }),
        check_names=False
    )
