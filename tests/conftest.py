from getpass import getuser
from pathlib import Path
from uuid import UUID

import networkx as nx
import pytest

import cascade.core.db
from cascade.runner.application_config import application_config

cascade.core.db.BLOCK_SHARED_FUNCTION_ACCESS = True


@pytest.fixture
def mock_execution_context(mocker):
    mock_context = mocker.Mock()
    mock_context.parameters.database = "test_database"
    mock_context.parameters.model_version_id = 12345
    mock_context.parameters.add_csmr_cause = 173
    mock_context.parameters.run_id = UUID(bytes=b'1' * 16)
    return mock_context


@pytest.fixture
def mock_locations(mocker):
    locations = mocker.patch("cascade.input_data.db.locations.location_hierarchy")
    G = nx.DiGraph()
    G.add_nodes_from(list(range(8)))
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (6, 7)])
    assert len(G.nodes) == 8
    locations.return_value = G


@pytest.fixture
def mock_ezfuncs(mocker):
    return mocker.patch("cascade.core.db.ezfuncs")


@pytest.fixture
def mock_database_access(mock_ezfuncs):
    return {"cursor": mock_ezfuncs.get_connection().cursor(), "connection": mock_ezfuncs.get_connection()}


def pytest_addoption(parser):
    group = parser.getgroup("cascade")
    group.addoption("--ihme", action="store_true",
                    help="run functions requiring access to central comp and Dismod-AT")
    group.addoption("--signals", action="store_true",
                    help="tests using Unix signals can crash the Mac.")
    group.addoption("--dismod", action="store_true",
                    help="requires access to Dismod-AT command line")
    group.addoption("--cluster", action="store_true",
                    help="run functions requiring access to fair cluster")


@pytest.fixture
def ihme(request):
    return IhmeDbFuncArg(request)


class IhmeDbFuncArg:
    """
    Uses a pattern from https://pytest.readthedocs.io/en/2.0.3/example/attic.html
    """
    def __init__(self, request):
        if not request.config.getoption("ihme"):
            pytest.skip(f"specify --ihme to run tests requiring Central Comp databases")

        cascade.core.db.BLOCK_SHARED_FUNCTION_ACCESS = False


@pytest.fixture
def cluster(request):
    return ClusterFuncArg(request)


class ClusterFuncArg:
    """
    Uses a pattern from https://pytest.readthedocs.io/en/2.0.3/example/attic.html
    """
    def __init__(self, request):
        if not request.config.getoption("cluster"):
            pytest.skip(f"specify --cluster to run tests requiring the cluster")


@pytest.fixture
def signals(request):
    return SignalQuietArg(request)


class SignalQuietArg:
    """
    Some tests kill processes with UNIX signals that, on the Mac so far,
    cause the OS to try to notify Apple of a problem. This flag turns
    off those tests.
    """
    def __init__(self, request):
        if request.config.getoption("signals"):
            pytest.skip(f"specify --signals if using UNIX signals can stop pytest")


@pytest.fixture
def dismod(request):
    return DismodFuncArg(request)


class DismodFuncArg:
    """Must be able to run dmdismod."""
    def __init__(self, request):
        if not (request.config.getoption("dismod") or request.config.getoption("ihme")):
            pytest.skip("specify --dismod or --ihme to run tests requiring Dismod")


@pytest.fixture(scope="session")
def shared_cluster_tmp(tmp_path_factory):
    """This is a tmp_path that will be available from all cluster nodes
    inside Grid Engine jobs."""
    config = application_config()
    if not config.has_section("gridengineapp"):
        return None
    cluster_tmp = application_config()["gridengineapp"]["cluster-tmp"]
    tmp_path = Path(cluster_tmp.format(user=getuser())) / "tmp"
    if Path(*tmp_path.parts[:2]).exists():
        # the fixture still gets made, even if fair isn't chosen.
        tmp_path.mkdir(parents=True, exist_ok=True)
        tmp_path_factory._basetemp = tmp_path
        return tmp_path_factory.mktemp("run")
    else:
        return None
