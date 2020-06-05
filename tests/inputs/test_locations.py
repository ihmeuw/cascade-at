
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
