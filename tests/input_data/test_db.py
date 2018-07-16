from cascade.input_data.db import freeze_bundle


def test_freeze_bundle__did_freeze(mock_execution_context, mock_ezfuncs, mocker):
    mock_check = mocker.patch("cascade.input_data.db._bundle_is_frozen")
    mock_check.return_value = False

    assert freeze_bundle(mock_execution_context)

    model_version_id = mock_execution_context.parameters.model_version_id
    cursor = mock_ezfuncs.get_connection().cursor()
    cursor.callproc.called_with(mocker.ANY, [model_version_id])


def test_freeze_bundle__did_not_freeze(mock_execution_context, mock_ezfuncs, mocker):
    mock_check = mocker.patch("cascade.input_data.db._bundle_is_frozen")
    mock_check.return_value = True

    assert not freeze_bundle(mock_execution_context)

    model_version_id = mock_execution_context.parameters.model_version_id
    cursor = mock_ezfuncs.get_connection().cursor()
    assert not cursor.callproc.called
