import networkx as nx
from numpy.random import RandomState

from cascade.executor.cascade_plan import CascadePlan
from cascade.executor.dismodel_main import parse_arguments
from cascade.input_data.db.configuration import load_settings
from cascade.input_data.db.locations import location_hierarchy
from cascade.testing_utilities import make_execution_context
from cascade.executor.create_settings import create_settings


def test_create_start_finish(ihme):
    args = parse_arguments(["z.db"])
    ec = make_execution_context(parent_location_id=0, gbd_round_id=5)
    locations = location_hierarchy(6, location_set_version_id=429)
    settings = load_settings(ec, None, 267845, None)
    settings.model.split_sex = 3
    settings.model.drill_location_start = 4
    settings.model.drill_location_end = 6
    c = CascadePlan.from_epiviz_configuration(locations, settings, args)
    assert len(c._task_graph.nodes) == 4
    print(nx.to_edgelist(c._task_graph))


def test_single_start_finish(ihme):
    args = parse_arguments(["z.db"])
    ec = make_execution_context(parent_location_id=0, gbd_round_id=5)
    locations = location_hierarchy(6, location_set_version_id=429)
    settings = load_settings(ec, None, 267845, None)
    settings.model.split_sex = 3
    settings.model.drill_location_start = 6
    settings.model.drill_location_end = 6
    c = CascadePlan.from_epiviz_configuration(locations, settings, args)
    assert len(c._task_graph.nodes) == 2
    print(nx.to_edgelist(c._task_graph))


def test_iterate_tasks(ihme):
    args = parse_arguments(["z.db"])
    ec = make_execution_context(parent_location_id=0, gbd_round_id=5)
    locations = location_hierarchy(6, location_set_version_id=429)
    settings = load_settings(ec, None, 267770, None)
    c = CascadePlan.from_epiviz_configuration(locations, settings, args)
    cnt = 0
    last = -1
    parent = None
    for idx, t in enumerate(c.cascade_jobs):
        if idx > 1:
            assert t[0] > last  # only true in drill
        last = t[0]

        which, local_settings = c.cascade_job(t)
        assert which in {"estimate_location", "bundle_setup"}
        assert hasattr(local_settings, "parent_location_id")
        if idx > 0:
            assert local_settings.grandparent_location_id == parent
            parent = local_settings.parent_location_id
            assert len(local_settings.children) > 0

        cnt += 1
    assert cnt == 3


def test_random_settings():
    rng = RandomState(342523)
    args = parse_arguments(["z.db"])
    locations = nx.DiGraph()
    children = [4, 31, 64, 103, 137, 158, 166]
    locations.add_edges_from([(1, c) for c in children])
    for i in range(100):
        settings = create_settings(rng, children)
        c = CascadePlan.from_epiviz_configuration(locations, settings, args)
        for idx, j in enumerate(c.cascade_jobs):
            job_kind, job_args = c.cascade_job(j)
            if idx > 0:
                assert job_kind == "estimate_location"
            else:
                assert job_kind == "bundle_setup"
            assert job_args is not None
