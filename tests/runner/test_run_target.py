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
from numpy.random import RandomState

from cascade.executor.cascade_plan import (
    recipe_graph_from_settings,
    location_specific_settings,
)
from cascade.executor.create_settings import create_settings


def test_multiple_process_run():
    command = "dmrun {mvid} --multiprocess --mock-stage"
    products = all_output(command[1:])
    run(command)
    assert products.exist()


def test_grid_engine_run():
    command = "dmrun {mvid} --mock-stage"
    products = all_output(command[1:])
    run(command)
    assert products.exist()


def test_single_process_run():
    command = "dmrun {mvid} --single-process --mock-stage"
    products = all_output(command[1:])
    run(command)
    assert products.exist()


def test_single_location_run():
    command = "dmrun {mvid} --location 102 --single-process --mock-stage"
    products = all_output(command[1:])
    run(command)
    assert products.exist()


def test_single_location_fit_stage_run():
    command = "dmrun {mvid} --location 102 --fit --single-process --mock-stage"
    products = all_output(command[1:])
    run(command)
    assert products.exist()


def generate_task_graph():
    """
    Stage, Transform, Job, Task, Estimation

    Names we use:
     * global fit
     * MAP estimate
     * pre-global fit
     * mid-hierarchy fit
     * aggregate
     * make draws

    LocationGraph, ByLocationGraph, EstimationGraph
    LocationWork, LocationJob, LocationTask
    Task
    """
    rng = RandomState(43234)
    locations = nx.balanced_tree(3, 3, nx.DiGraph)
    locations.graph["root"] = "0"
    assert len(locations) == 40
    settings = create_settings(rng, locations)
    args = SimpleNamespace(**dict(
        skip_cache=False,
        num_samples=5,
        pdb=False,
        num_processes=5,
        db_only=False,
        no_upload=True,
    ))
    recipe_graph = recipe_graph_from_settings(locations, settings, args)
    for node in recipe_graph:
        substeps, local_settings = location_specific_settings(locations, settings, args, node)
        tasks = recipe_to_tasks(node, local_settings)
        recipe_graph.node(subgraph=tasks)
    task_graph = recipe_graph_to_task_graph(recipe_graph)
