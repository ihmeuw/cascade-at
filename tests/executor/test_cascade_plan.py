import networkx as nx

from cascade.core.parameters import _ParameterHierarchy
from cascade.executor.cascade_plan import CascadePlan
from cascade.testing_utilities import make_execution_context
from cascade.input_data.db.locations import location_hierarchy


def test_create(ihme):
    ec = make_execution_context(parent_location_id=0, gbd_round_id=5)
    locations = location_hierarchy(ec)
    settings = _ParameterHierarchy(
        model={"split_sex": 3, "drill_location": 6},
        policies=dict(),
    )
    c = CascadePlan.from_epiviz_configuration(locations, settings)
    assert len(c._task_graph.nodes) == 2
    print(nx.to_edgelist(c._task_graph))


def test_single(ihme):
    ec = make_execution_context(parent_location_id=0, gbd_round_id=5)
    locations = location_hierarchy(ec)
    settings = _ParameterHierarchy(
        model={"split_sex": 4, "drill_location": 6},
        policies=dict(),
    )
    c = CascadePlan.from_epiviz_configuration(locations, settings)
    assert len(c._task_graph.nodes) == 1
    print(nx.to_edgelist(c._task_graph))


def test_create_start_finish(ihme):
    ec = make_execution_context(parent_location_id=0, gbd_round_id=5)
    locations = location_hierarchy(ec)
    settings = _ParameterHierarchy(
        model={"split_sex": 3, "drill_location_start": 4, "drill_location_end": 6},
        policies=dict(),
    )
    c = CascadePlan.from_epiviz_configuration(locations, settings)
    assert len(c._task_graph.nodes) == 3
    print(nx.to_edgelist(c._task_graph))


def test_single_start_finish(ihme):
    ec = make_execution_context(parent_location_id=0, gbd_round_id=5)
    locations = location_hierarchy(ec)
    settings = _ParameterHierarchy(
        model={"split_sex": 4, "drill_location_start": 6, "drill_location_end": 6},
        policies=dict(),
    )
    c = CascadePlan.from_epiviz_configuration(locations, settings)
    assert len(c._task_graph.nodes) == 1
    print(nx.to_edgelist(c._task_graph))


def test_iterate_tasks(ihme):
    ec = make_execution_context(parent_location_id=0, gbd_round_id=5)
    locations = location_hierarchy(ec)
    settings = _ParameterHierarchy(
        model={"split_sex": 2, "drill_location": 6},
        policies=dict(),
    )
    c = CascadePlan.from_epiviz_configuration(locations, settings)
    cnt = 0
    last = -1
    parent = None
    for t in c.cascade_jobs:
        assert t[0] > last  # Only true in a drill
        last = t[0]

        which, local_settings = c.cascade_job(t)
        assert which in {"estimate_location", "bundle_setup"}
        assert hasattr(local_settings.model, "parent_location_id")
        assert local_settings.model.grandparent_location_id == parent
        parent = local_settings.model.parent_location_id
        assert len(local_settings.model.children) > 0

        cnt += 1
    # It would be 3, but this has been downgraded to use 1 if you
    # have split_sex
    assert cnt == 3
