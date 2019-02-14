import networkx as nx

from cascade.executor.cascade_plan import CascadePlan
from cascade.executor.dismodel_main import parse_arguments
from cascade.input_data.db.configuration import load_settings
from cascade.input_data.db.locations import location_hierarchy
from cascade.testing_utilities import make_execution_context


def test_create_start_finish(ihme):
    args = parse_arguments(["z.db"])
    ec = make_execution_context(parent_location_id=0, gbd_round_id=5)
    locations = location_hierarchy(ec)
    settings = load_settings(ec, None, 267845, None)
    settings.model.split_sex = 3
    settings.model.drill_location_start = 4
    settings.model.drill_location_end = 6
    c = CascadePlan.from_epiviz_configuration(locations, settings, args)
    assert len(c._task_graph.nodes) == 3
    print(nx.to_edgelist(c._task_graph))


def test_single_start_finish(ihme):
    args = parse_arguments(["z.db"])
    ec = make_execution_context(parent_location_id=0, gbd_round_id=5)
    locations = location_hierarchy(ec)
    settings = load_settings(ec, None, 267845, None)
    settings.model.split_sex = 3
    settings.model.drill_location_start = 6
    settings.model.drill_location_end = 6
    c = CascadePlan.from_epiviz_configuration(locations, settings, args)
    assert len(c._task_graph.nodes) == 1
    print(nx.to_edgelist(c._task_graph))


def test_iterate_tasks(ihme):
    args = parse_arguments(["z.db"])
    ec = make_execution_context(parent_location_id=0, gbd_round_id=5)
    locations = location_hierarchy(ec)
    settings = load_settings(ec, None, 267770, None)
    c = CascadePlan.from_epiviz_configuration(locations, settings, args)
    cnt = 0
    last = -1
    parent = None
    for t in c.cascade_jobs:
        assert t[0] > last  # Only true in a drill
        last = t[0]

        which, local_settings = c.cascade_job(t)
        assert which in {"estimate_location", "bundle_setup"}
        assert hasattr(local_settings, "parent_location_id")
        assert local_settings.grandparent_location_id == parent
        parent = local_settings.parent_location_id
        assert len(local_settings.children) > 0

        cnt += 1
    assert cnt == 2
