from types import SimpleNamespace

import networkx as nx
import pytest
from numpy.random import RandomState

from cascade.executor.cascade_plan import global_recipe_graph
from cascade.executor.create_settings import create_settings
from cascade.executor.execution_context import make_execution_context
from cascade.executor.job_definitions import (
    GlobalPrepareData, FindSingleMAP, add_job_list
)
from cascade.runner.job_graph import RecipeIdentifier, recipe_graph_to_job_graph


@pytest.fixture
def context(tmp_path):
    ec = make_execution_context(
        base_directory=tmp_path,
        modelable_entity_id=123,
        model_version_id=9876,
    )
    return dict(ec=ec, tmp=tmp_path)


@pytest.fixture
def pyramid_locations():
    locations = nx.balanced_tree(3, 2, create_using=nx.DiGraph)
    locations.graph["root"] = 0
    assign_levels(locations)
    return locations


def assign_levels(locations):
    current = [0]
    successors = list()
    level_idx = 0
    while current:
        for node in current:
            locations.node[node]["level"] = level_idx
            successors.extend(list(locations.successors(node)))
        current, successors = successors, list()
        level_idx += 1


def test_prepare_data(context):
    ec = context["ec"]
    recipe_id = RecipeIdentifier(21, "bundle_setup", "both")
    local_settings = SimpleNamespace()
    local_settings.parent_location_id = 22
    included_locations = None
    prepare = GlobalPrepareData(recipe_id, local_settings, included_locations, ec)
    assert not prepare.done()
    prepare.mock_run()
    assert prepare.done()


def test_global_estimate(context):
    ec = context["ec"]
    recipe_id = RecipeIdentifier(1, "bundle_setup", "both")
    local_settings = SimpleNamespace()
    local_settings.parent_location_id = 1
    included_locations = None
    prepare = GlobalPrepareData(recipe_id, local_settings, included_locations, ec)
    global_recipe = RecipeIdentifier(1, "estimate_location", "both")
    neighbors = dict(
        predecessors=[recipe_id],
    )
    single = FindSingleMAP(global_recipe, local_settings, neighbors, ec)
    assert not single.done()
    prepare.mock_run()
    single.mock_run()
    for outname, output in single.outputs.items():
        print(f"{outname}:\n\t{output.path}\n\t{output.validate()}")
    assert single.done()


@pytest.mark.skip("find how to run_mock")
def test_recipe_level(context, pyramid_locations):
    ec = context["ec"]
    args = SimpleNamespace(skip_cache=False)
    settings = create_settings(RandomState(342234), pyramid_locations)
    settings.model.split_sex = 1
    recipe_graph = global_recipe_graph(
        pyramid_locations, settings, args
    )
    for recipe_id in recipe_graph.nodes:
        local_settings = SimpleNamespace(
            parent_location_id=recipe_id.location_id,
            number_of_fixed_effect_samples=3,
            policies=SimpleNamespace(
                fit_strategy="fit_fixed_then_fit"
            ),
        )
        parent = list(recipe_graph.predecessors(recipe_id))
        if parent and parent[0].recipe == "estimate_location":
            local_settings.grandparent_location_id = parent[0].location_id

        recipe_graph.nodes[recipe_id]["local_settings"] = local_settings

    add_job_list(recipe_graph, ec)
    job_graph = recipe_graph_to_job_graph(recipe_graph)
    work = dict(execution_context=ec)
    run_mock(work, job_graph, continuation=False)
    for job_node in job_graph:
        job = job_graph.node[job_node]["job"]
        assert not job.output_missing(ec)
