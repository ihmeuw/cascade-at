import pytest

import cascade_at.core.db

cascade_at.core.db.BLOCK_SHARED_FUNCTION_ACCESS = True


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

        cascade_at.core.db.BLOCK_SHARED_FUNCTION_ACCESS = False


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
def dismod(request):
    return DismodFuncArg(request)


class DismodFuncArg:
    """Must be able to run dmdismod."""
    def __init__(self, request):
        if not (request.config.getoption("dismod") or request.config.getoption("ihme")):
            pytest.skip("specify --dismod or --ihme to run tests requiring Dismod")
