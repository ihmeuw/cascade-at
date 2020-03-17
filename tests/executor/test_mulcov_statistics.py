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
    assert len(stat) == len(mulcov_df) * 4
    assert (stat.mulcov_value.values == np.concatenate((
        mulcov_df.mulcov_value.values,
        np.repeat(0, repeats=3),
        mulcov_df.mulcov_value.values,
        mulcov_df.mulcov_value.values
    ))).all()
    assert (stat.stat.values == np.repeat([
        'mean', 'std', 'quantile_0.025', 'quantile_0.975'
    ], repeats=3)).all()
