from cascade.input_data.db.asdr import load_asdr_to_t3, get_asdr_data


def test_load_asdr_to_t3_did_upload(mock_execution_context, mock_database_access, mocker):

    mock_check = mocker.patch("cascade.input_data.db.asdr._asdr_in_t3")
    mock_check.return_value = [19]

    mock_locations = mocker.patch("cascade.input_data.db.asdr.location_hierarchy")
    mock_locations.return_value = None

    mock_location = mocker.patch("cascade.input_data.db.asdr.get_descendants")
    mock_location.return_value = [248]

    mock_get_asdr_data = mocker.patch("cascade.input_data.db.asdr.get_asdr_data")
    mock_upload_asdr_data = mocker.patch("cascade.input_data.db.asdr._upload_asdr_data_to_tier_3")

    model_version_id = mock_execution_context.parameters.model_version_id
    parent_id = mock_execution_context.parameters.parent_location_id
    gbd_round_id = mock_execution_context.parameters.gbd_round_id

    assert load_asdr_to_t3(mock_execution_context, model_version_id, parent_id, gbd_round_id)

    cursor = mock_database_access["cursor"]
    model_version_id = mock_execution_context.parameters.model_version_id

    mock_upload_asdr_data.assert_called_with(gbd_round_id, cursor, model_version_id, mock_get_asdr_data())


def test_load_asdr_to_t3_no_upload(mock_execution_context, mock_database_access, mocker):

    mock_check = mocker.patch("cascade.input_data.db.asdr._asdr_in_t3")
    mock_check.return_value = [248]

    mock_locations = mocker.patch("cascade.input_data.db.asdr.location_hierarchy")
    mock_locations.return_value = None

    mock_location = mocker.patch("cascade.input_data.db.asdr.get_descendants")
    mock_location.return_value = [248]

    mock_upload_asdr_data = mocker.patch("cascade.input_data.db.asdr._upload_asdr_data_to_tier_3")

    model_version_id = mock_execution_context.parameters.model_version_id
    parent_id = mock_execution_context.parameters.parent_location_id
    gbd_round_id = mock_execution_context.parameters.gbd_round_id

    assert not load_asdr_to_t3(mock_execution_context, model_version_id, parent_id, gbd_round_id)

    assert not mock_upload_asdr_data.called


def test_asdr_columns(ihme):
    asdr = get_asdr_data(5, [101], with_hiv=True)
    assert not asdr.duplicated(["age_group_id", "location_id", "year_id", "sex_id"]).any()
    assert (asdr.location_id == 101).all()
