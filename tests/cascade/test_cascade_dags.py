import pytest

import pandas as pd

from cascade_at.cascade.cascade_dags import make_cascade_dag
from cascade_at.inputs.locations import LocationDAG
from cascade_at.cascade.cascade_operations import _CascadeOperation


@pytest.fixture
def df():
    return pd.DataFrame({
        'location_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'parent_id':   [0, 1, 1, 2, 2, 3, 3, 4, 4, 4]
    })


@pytest.fixture
def l_dag(df):
    return LocationDAG(df=df, root=1)


def test_make_dag(l_dag):
    tasks = make_cascade_dag(
        model_version_id=0, dag=l_dag,
        location_start=1, sex_start=2, split_sex=False
    )
    assert len(tasks) == 3 + 2 * 3 + 6 * 3 + 1
    for task in tasks:
        assert isinstance(task, _CascadeOperation)
