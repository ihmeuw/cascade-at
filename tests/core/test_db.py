from cascade.core.db import cursor

import pytest


def test_with_cursor__basic_connection__success(mock_execution_context, mock_ezfuncs):
    with cursor(mock_execution_context) as c:
        mock_ezfuncs.get_connection.assert_called_with("test_database")
        assert mock_ezfuncs.get_connection().cursor() is c

    c.close.assert_called_with()
    mock_ezfuncs.get_connection().commit.assert_called_with()


def test_with_cursor__basic_connection__raise(mock_execution_context, mock_ezfuncs):
    with pytest.raises(Exception) as excinfo:
        with cursor(mock_execution_context) as c:
            raise Exception("Woops")

    assert "Woops" in str(excinfo.value)
    c.close.assert_called_with()
    assert not mock_ezfuncs.get_connection().commit.called
