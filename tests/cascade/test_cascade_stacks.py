import pytest
import pandas as pd

from cascade_at.cascade.cascade_stacks import leaf_fit
from cascade_at.inputs.locations import LocationDAG


@pytest.fixture
def df():
    return pd.DataFrame({
        'location_id': [1, 2, 3, 4, 5],
        'parent_id': [0, 1, 1, 2, 2]
    })


@pytest.fixture
def dag(df):
    return LocationDAG(df=df)


def test_leaf_fit():
    lf = leaf_fit(
        model_version_id=0, location_id=5,
        sex_id=1, prior_parent=2, prior_sex=1, n_sim=100, n_pool=100
    )
    assert lf[1].command == (
        'sample --model-version-id 0 --parent-location-id 5 --sex-id 1 '
        '--n-sim 100 --n-pool 100 --fit-type both --asymptotic'
    )
