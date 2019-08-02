import networkx as nx
import pytest
from numpy.random import RandomState

from cascade.core.form import Form, FormList, FloatField
from cascade.executor.cascade_plan import (
    make_model_options, )
from cascade.executor.create_settings import create_settings
from cascade.executor.dismodel_main import DismodAT
from cascade.executor.execution_context import make_execution_context
from cascade.executor.job_definitions import job_graph_from_settings
from cascade.input_data.configuration.form import RandomEffectBound
from gridengineapp import execution_ordered

SUBJOBS_PER_LOCATION = 3


def test_create_start_finish(ihme):
    app = DismodAT()
    args = app.add_arguments().parse_args(["--mvid", "267845"])
    app.initialize(args)
    app.settings.model.split_sex = 3
    app.settings.model.drill_location_start = 4
    app.settings.model.drill_location_end = 6
    job_graph = app.job_graph()
    assert len(job_graph) == 1 + SUBJOBS_PER_LOCATION * 3


def test_single_start_finish(ihme):
    app = DismodAT()
    args = app.add_arguments().parse_args(["--mvid", "267845"])
    app.initialize(args)
    app.settings.model.split_sex = 3
    app.settings.model.drill_location_start = 6
    app.settings.model.drill_location_end = 6
    job_graph = app.job_graph()
    assert len(job_graph) == 1 + SUBJOBS_PER_LOCATION


def test_iterate_tasks(ihme):
    app = DismodAT()
    args = app.add_arguments().parse_args(["--mvid", "267770"])
    app.initialize(args)
    job_graph = app.job_graph()
    ordered = execution_ordered(job_graph)
    cnt = 0
    for idx, job_id in enumerate(ordered):
        if idx == 0:
            assert job_id.recipe == "bundle_setup"
        if idx > 1:
            assert job_id.location_id > 0
            assert job_id.recipe == "estimate_location"
        cnt += 1
    assert cnt == 1 + SUBJOBS_PER_LOCATION * 2


def test_random_settings():
    execution_context = make_execution_context(
        gbd_round_id=6, num_processes=4
    )
    rng = RandomState(342523)
    locations = nx.DiGraph()
    children = [4, 31, 64, 103, 137, 158, 166]
    locations.add_edges_from([(1, c) for c in children])
    for i in range(100):
        settings = create_settings(rng, locations)
        app = DismodAT(locations, settings, execution_context)
        args = app.add_arguments().parse_args(["--mvid", "267770"])
        job_graph = job_graph_from_settings(
            locations, settings, args, execution_context
        )
        for idx, job_id in enumerate(execution_ordered(job_graph)):
            job = job_graph.nodes[job_id]["job"]
            if idx > 0:
                assert job_id.recipe == "estimate_location"
            else:
                assert job_id.recipe == "bundle_setup"
            assert job.local_settings is not None


def field_set(name):
    return False


class MiniInnerForm(Form):
    """For test_model_options as a fake settings"""
    bound_random = FloatField()


class MiniForm(Form):
    """For test_model_options as a fake settings"""
    re_bound_location = FormList(RandomEffectBound)
    model = MiniInnerForm()


@pytest.mark.parametrize("loc,expected", [
    (1, 0.7), (2, None), (3, 0.7), (4, 0.3), (5, 0.2), (6, 0.2),
])
def test_model_options(loc, expected):
    locations = nx.DiGraph()
    locations.add_edges_from([(1, 2), (1, 3), (2, 4), (4, 5), (5, 6)])

    reb1 = RandomEffectBound(dict(location=2, value=0.4))
    assert not reb1.is_field_unset("location")
    assert not reb1.is_field_unset("value")

    reb2 = RandomEffectBound(dict(location=2))
    assert not reb2.is_field_unset("location")
    assert reb2.is_field_unset("value")

    bound_form = MiniForm(dict(
        model=dict(bound_random=0.7),  # This will be the default value.
    ))
    bound_form.re_bound_location = [
        dict(location=2),
        dict(location=4, value=0.3),
        dict(location=5, value=0.2),
    ]
    errors = bound_form.validate_and_normalize()
    assert not errors
    opts = make_model_options(locations, loc, bound_form)
    assert opts.bound_random == expected
