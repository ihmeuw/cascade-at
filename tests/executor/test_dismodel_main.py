import logging
from argparse import ArgumentParser
from getpass import getuser

import networkx as nx
import pytest
from numpy.random import RandomState

from cascade.core import getLoggers
from cascade.executor.create_settings import create_settings
from cascade.executor.dismodel_main import DismodAT, execution_context_without_settings
from cascade.runner.application_config import application_config
from cascade.runner.job_graph import JobIdentifier, RecipeIdentifier

CODELOG, MATHLOG = getLoggers(__name__)


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
    print(f"args are {dir(args)}")
    ec = execution_context_without_settings(args)
    app = DismodAT(pyramid_locations, settings, ec, args)
    app.save_settings()
    jobs = app.job_graph()

    later_app = DismodAT()
    later_app.load_settings(args)
    assert len(later_app.locations) == len(app.locations)

    later_jobs = later_app.job_graph()
    assert len(later_jobs) == len(jobs)


class FakeMVIDApp:
    def add_arguments(self, parser):
        parser.add_argument("--mvid", type=int)
        return parser


def test_entry_constructs_logs(monkeypatch, tmp_path):
    """Test whether logs are created correctly."""
    # The cases include
    # a) has mvid or not
    # b) directories exist or not
    # c) -v, -q, -v -v
    # Check for g+w for this user on this machine.
    directories = application_config()["DataLayout"]
    code_dir = tmp_path / directories["code-log-directory"]
    code_dir.mkdir(parents=True)
    math_dir = tmp_path / directories["epiviz-log-directory"]
    math_dir.mkdir(parents=True)

    ec = object()
    locs = nx.DiGraph()
    app = DismodAT(
        locations=locs, settings=dict(value=1), execution_context=ec
    )
    mvid = "2745"
    arg_list = [
        "--root-dir", str(tmp_path),
        "--mvid", mvid,
    ]
    args = app.add_arguments().parse_args(arg_list)

    app.initialize(args)

    code_log, math_log = getLoggers("testing")
    code_log.debug("CODELOG debug")
    code_log.info("CODELOG info")
    math_log.debug("MATHLOG debug")
    math_log.info("MATHLOG info")

    # Close all of the loggers so that they flush to disk.
    logger_list = [
        logging.root,
        logging.getLogger("cascade"),
        logging.getLogger("cascade.math")
    ]
    for logger in logger_list:
        for handler in logger.handlers:
            if hasattr(handler, "close"):
                handler.close()

    code_log_dir = code_dir / getuser() / "dismod"
    code_log_list = list(code_log_dir.glob("*.log"))
    assert len(code_log_list) == 1
    code_log = code_log_list[0]

    print(f"code log {code_log}")
    code_lines = code_log.open().readlines()
    assert len(code_lines) > 0

    math_log = math_dir / mvid / "log.log"

    print(f"math log {math_log}")
    math_lines = math_log.open().readlines()
    assert len(math_lines) > 0
