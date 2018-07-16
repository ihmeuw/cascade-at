import pytest


@pytest.fixture
def mock_execution_context(mocker):
    mock_context = mocker.Mock()
    mock_context.parameters.database = "test_database"
    mock_context.parameters.model_version_id = 12345
    return mock_context


@pytest.fixture
def mock_ezfuncs(mocker):
    return mocker.patch("cascade.core.db.ezfuncs")
