import pytest


@pytest.fixture
def mock_execution_context(mocker):
    mock_context = mocker.Mock()
    mock_context.parameters.database = "test_database"
    mock_context.parameters.model_version_id = 12345
    mock_context.parameters.add_csmr_cause = 173
    mock_context.parameters.country_covariate_ids = [26, 28, 33]
    return mock_context


@pytest.fixture
def mock_ezfuncs(mocker):
    return mocker.patch("cascade.core.db.ezfuncs")


@pytest.fixture
def mock_database_access(mock_ezfuncs):
    return {"cursor": mock_ezfuncs.get_connection().cursor(), "connection": mock_ezfuncs.get_connection()}
