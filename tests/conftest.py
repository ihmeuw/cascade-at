import pytest

import cascade.core.db

cascade.core.db.BLOCK_SHARED_FUNCTION_ACCESS = True


@pytest.fixture
def mock_execution_context(mocker):
    mock_context = mocker.Mock()
    mock_context.parameters.database = "test_database"
    mock_context.parameters.model_version_id = 12345
    mock_context.parameters.add_csmr_cause = 173
    return mock_context


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
