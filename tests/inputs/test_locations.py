import pytest

import pandas as pd
import numpy as np


from cascade_at.inputs.locations import LocationDAG
from cascade_at.inputs.locations import LocationDAGError


@pytest.fixture
def df():
    return pd.DataFrame({
        'location_id': [1, 2, 3, 4, 5],
        'parent_id': [0, 1, 1, 2, 2]
    })


def test_dag_from_df(df):
    dag = LocationDAG(df=df, root=1)
    assert set(dag.dag.successors(1)) == {2, 3}
    assert set(dag.dag.successors(2)) == {4, 5}
    assert set(dag.descendants(1)) == {2, 3, 4, 5}


def test_dag_error_noargs():
    with pytest.raises(LocationDAGError):
        LocationDAG()


def test_dag_error_missing_args():
    with pytest.raises(LocationDAGError):
        LocationDAG(location_set_version_id=0)


def test_dag_no_root(df):
    with pytest.raises(LocationDAGError):
        LocationDAG(df=df)


def test_not_empty_df(dag):
    assert not dag.df.empty


def test_known_node_names(dag):
    assert dag.dag.nodes[1]['location_name'] == 'Global'
    assert dag.dag.nodes[101]['location_name'] == 'Canada'
    assert dag.dag.nodes[102]['location_name'] == 'United States of America'


def test_known_node_parents(dag):
    assert dag.dag.nodes[dag.dag.nodes[555]['parent_id']]['location_name'] == 'United States of America'


def test_to_dataframe(dag):
    df = dag.to_dataframe()
    assert not df.empty
    assert df.loc[df.location_id == 102]['name'].iloc[0] == 'United States of America'
    assert df.loc[df.location_id == 555]['parent_id'].iloc[0] == 102.


def test_descendants(dag):
    assert not dag.descendants(101)
    assert len(dag.descendants(102)) == 51


def test_parent_children(dag):
    assert len(dag.parent_children(101)) == 1
    assert len(dag.parent_children(102)) == 52


def test_root(dag):
    assert dag.dag.graph["root"] == 1
