"""
These tests represent the different ways we launch the Cascade.
It's TDD, but working from a very high level for the code.
There are three backends:

 1. Under Grid Engine (UGE, SGE), because this is what we use for production.
 2. As multiple processes, for analysis work done by modelers on laptop
    or a single node on the cluster.
 3. As one process, calling each part of the code as a function, because
    this is used to debug. You see it in Dismod-ODE code as an if-then
    that skips using multiprocessing.

The other key is to be able to run parts of the whole Cascade.
The whole cascade is constructed from these parts:

 * There is one **estimation** at each location of the location hierarchy.
 * Each estimation contains three kinds of stages,
   the **fit**, around 30 **simulation fits**, and one **summary**.

The modelers have discussed wanting to run portions of a total
Cascade. These are the portions they have listed so far.

 1. Run everything, known as a **global run**.
 2. Run the **drill**, which is all stages for an estimation and its parent
    estimations.
 3. Run all stages for **part of a drill**. For instance, run one estimation,
    so that we can debug how to improve the model for one estimation.
 4. Run every estimation that depends on a given one. This is a
    **subtree** of the location hierarchy.

Lastly, restarting is a common occurrence, which means finishing
those parts that are unfinished.

The tests here will use fakes for the implementation inside of
the stages, requested by using "--mock-stage". They will also
use a stub for settings, because we care about only one or two
parameters from the settings.
"""
from subprocess import run
from types import SimpleNamespace

import networkx as nx
import pytest
from numpy.random import RandomState

from cascade.executor.cascade_plan import (
    global_recipe_graph,
    job_graph_from_settings,
    drill_recipe_graph,
)
from cascade.executor.create_settings import create_settings


@pytest.mark.skip("target")
def test_multiple_process_run():
    command = "dmrun {mvid} --multiprocess --mock-stage"
    products = all_output(command[1:])
    run(command)
    assert products.exist()


@pytest.mark.skip("target")
def test_grid_engine_run():
    command = "dmrun {mvid} --mock-stage"
    products = all_output(command[1:])
    run(command)
    assert products.exist()


@pytest.mark.skip("target")
def test_single_process_run():
    command = "dmrun {mvid} --single-process --mock-stage"
    products = all_output(command[1:])
    run(command)
    assert products.exist()


@pytest.mark.skip("target")
def test_single_location_run():
    command = "dmrun {mvid} --location 102 --single-process --mock-stage"
    products = all_output(command[1:])
    run(command)
    assert products.exist()


@pytest.mark.skip("target")
def test_single_location_fit_stage_run():
    command = "dmrun {mvid} --location 102 --fit --single-process --mock-stage"
    products = all_output(command[1:])
    run(command)
    assert products.exist()


def add_level_to_graph(digraph, root=None):
    if root is None:
        root = digraph.graph["root"]
    digraph.node[root]["level"] = 0
    for start, finish in nx.bfs_edges(digraph, root):
        digraph.node[finish]["level"] = digraph.node[start]["level"] + 1


@pytest.fixture
def locations():
    tree_height = 3
    branching_factor = 3
    locations = nx.balanced_tree(branching_factor, tree_height, nx.DiGraph)
    assert len(locations) == 40
    locations.graph["root"] = 0
    add_level_to_graph(locations)
    return locations


@pytest.fixture
def basic_settings(locations):
    rng = RandomState(43234)
    return create_settings(rng, locations)


@pytest.fixture
def build_args():
    return SimpleNamespace(**dict(
        skip_cache=False,
        num_samples=5,
        pdb=False,
        num_processes=5,
        db_only=False,
        no_upload=True,
        settings_file=None,
        bundle_file=None,
        bundle_study_covariates_file=None,
    ))


def test_global_recipe_graph(locations, basic_settings, build_args):
    global_graph = global_recipe_graph(locations, basic_settings, build_args)
    assert nx.is_directed_acyclic_graph(global_graph)
    components = nx.number_connected_components(global_graph.to_undirected())
    print(f"Connected component count {components}")
    assert nx.is_connected(global_graph.to_undirected())
    print(nx.dag_longest_path_length(global_graph))
    location_height = nx.dag_longest_path_length(locations)
    assert nx.dag_longest_path_length(global_graph) == 1 + location_height


def test_global_recipe_most_detailed(locations, basic_settings, build_args):
    basic_settings.model.split_sex = "most_detailed"
    global_graph = global_recipe_graph(locations, basic_settings, build_args)
    setup = 1
    both = sum(3**n for n in range(4))
    split = 3
    other = sum(3**n for n in range(split, 4))
    assert len(global_graph) == setup + both + other


def test_global_recipe_skip_cache(locations, basic_settings, build_args):
    build_args.skip_cache = True
    global_graph = global_recipe_graph(locations, basic_settings, build_args)
    location_height = nx.dag_longest_path_length(locations)
    assert nx.dag_longest_path_length(global_graph) == location_height
    both = sum(3**n for n in range(4))
    split = int(basic_settings.model.split_sex)
    other = sum(3**n for n in range(split, 4))
    assert len(global_graph) == both + other


def test_drill_recipe_graph(locations, basic_settings, build_args):
    basic_settings.model.drill_location_start = 0
    basic_settings.model.drill_location_end = 9
    drill_graph = drill_recipe_graph(locations, basic_settings, build_args)
    assert nx.is_directed_acyclic_graph(drill_graph)
    assert nx.is_connected(drill_graph.to_undirected())
    print(nx.dag_longest_path_length(drill_graph))
    print(drill_graph.nodes)
    assert nx.dag_longest_path_length(drill_graph) == 3


def test_generate_job_graph(locations, basic_settings, build_args):
    """
    """
    basic_settings.model.drill = "global"
    job_graph = job_graph_from_settings(locations, basic_settings, build_args)
    assert isinstance(job_graph, nx.DiGraph)
    assert nx.is_directed_acyclic_graph(job_graph)
    assert nx.is_connected(job_graph.to_undirected())
    print(f"Longest path length {nx.dag_longest_path_length(job_graph)}")
    location_height = 4
    jobs_per_location = 3
    assert nx.dag_longest_path_length(job_graph) == jobs_per_location * location_height
