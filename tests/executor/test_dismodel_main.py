from argparse import ArgumentParser
from cascade.executor.dismodel_main import DismodAT, execution_context_without_settings
from cascade.runner.job_graph import JobIdentifier, RecipeIdentifier

from numpy.random import RandomState
from cascade.executor.create_settings import create_settings
import pytest
import networkx as nx


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


def test_add_arguments():
    app = DismodAT()
    parser = ArgumentParser()
    parser = app.add_arguments(parser)
    args = parser.parse_args(["--mvid", "23942", "--infrastructure", "--db-only"])
    assert args.mvid == 23942
    assert args.db_only
    assert args.bundle_file is None


def test_args_parses_job_identifier():
    """The application defines arguments that correspond to the identifier."""
    ji = JobIdentifier(RecipeIdentifier(32, "estimate_location", "female"), "draws")
    app = DismodAT()
    parser = ArgumentParser()
    parser = app.add_arguments(parser)
    arg_list = ["--mvid", "23942"] + ji.arguments
    args = parser.parse_args(arg_list)
    assert args.location_id == ji.location_id
    assert args.recipe == ji.recipe
    assert args.sex == ji.sex
    assert args.name == ji.name


def test_application_save_settings(pyramid_locations):
    """The application can save settings and load them again."""
    settings = create_settings(RandomState(342234), pyramid_locations)
    app = DismodAT()
    parser = app.add_arguments()
    args = parser.parse_args(
        ["--meid", "4242", "--mvid", "234243"]
    )
    ec = execution_context_without_settings(args)
    app = DismodAT(pyramid_locations, settings, ec)
    app.save_settings()
    jobs = app.graph_of_jobs(args)

    later_app = DismodAT()
    later_app.load_settings(args)
    assert len(later_app.locations) == len(app.locations)

    later_jobs = later_app.graph_of_jobs(args)
    assert len(later_jobs) == len(jobs)
