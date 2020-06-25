import pytest
import pandas as pd
import numpy as np

from cascade_at.executor.mulcov_statistics import compute_statistics


@pytest.fixture
def mulcov_df():
    return pd.DataFrame({
        'c_covariate_name': ['c_diabetes_fpg', 's_sex', 's_sex'],
        'mulcov_type': ['rate_value', 'rate_value', 'rate_value'],
        'rate_name': ['iota', 'iota', 'chi'],
        'integrand_name': [np.nan, np.nan, np.nan],
        'mulcov_value': [0.99, 0.45, 0.30]
    })


def test_compute_statistics(mulcov_df):
    stat = compute_statistics(
        df=mulcov_df, mean=True, std=True, quantile=[0.025, 0.975]
    )
    assert type(stat) == pd.DataFrame
    assert len(stat) == len(mulcov_df)
    assert len(stat.columns) == 8
    assert all(stat['mean'].to_numpy()  ==  mulcov_df.mulcov_value.to_numpy())
    assert all(stat['std'].to_numpy() == np.zeros(3))
    assert all(stat['quantile_0.025'].to_numpy() == mulcov_df.mulcov_value.to_numpy())
    assert all(stat['quantile_0.975'].to_numpy() == mulcov_df.mulcov_value.to_numpy())
