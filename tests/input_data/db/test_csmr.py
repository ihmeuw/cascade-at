from cascade.input_data.db.csmr import load_csmr_to_t3


def test_load_csmr_to_t3_did_upload(mock_execution_context, mock_database_access, mocker):

    mock_check = mocker.patch("cascade.input_data.db.csmr._csmr_in_t3")
    mock_check.return_value = False

    mock_get_csmr_data = mocker.patch("cascade.input_data.db.csmr._get_csmr_data")
    mock_upload_csmr_data = mocker.patch("cascade.input_data.db.csmr._upload_csmr_data_to_tier_3")

    assert load_csmr_to_t3(mock_execution_context)

    cursor = mock_database_access["cursor"]
    model_version_id = mock_execution_context.parameters.model_version_id

    mock_upload_csmr_data.assert_called_with(cursor, model_version_id, mock_get_csmr_data())


def test_load_csmr_to_t3_no_upload(mock_execution_context, mock_database_access, mocker):

    mock_check = mocker.patch("cascade.input_data.db.csmr._csmr_in_t3")
    mock_check.return_value = True

    mock_upload_csmr_data = mocker.patch("cascade.input_data.db.csmr._upload_csmr_data_to_tier_3")

    assert not load_csmr_to_t3(mock_execution_context)

    assert not mock_upload_csmr_data.called
