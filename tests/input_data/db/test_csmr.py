from cascade.testing_utilities import make_execution_context
from cascade.input_data.db.csmr import load_csmr_to_t3, _csmr_in_t3


def test_load_csmr_to_t3_did_upload(mock_execution_context, mock_database_access, mocker):

    mock_check = mocker.patch("cascade.input_data.db.csmr._csmr_in_t3")
    mock_check.return_value = [19]

    mock_location = mocker.patch("cascade.input_data.db.csmr.get_descendants")
    mock_location.return_value = [248]

    mock_get_csmr_data = mocker.patch("cascade.input_data.db.csmr.get_csmr_data")
    mock_upload_csmr_data = mocker.patch("cascade.input_data.db.csmr._upload_csmr_data_to_tier_3")

    assert load_csmr_to_t3(mock_execution_context,
                           mock_execution_context.parameters.model_version_id)

    cursor = mock_database_access["cursor"]
    model_version_id = mock_execution_context.parameters.model_version_id

    mock_upload_csmr_data.assert_called_with(cursor, model_version_id, mock_get_csmr_data())


def test_load_csmr_to_t3_no_upload(mock_execution_context, mock_database_access, mocker):

    mock_location = mocker.patch("cascade.input_data.db.csmr.get_descendants")
    mock_location.return_value = [248]

    mock_check = mocker.patch("cascade.input_data.db.csmr._csmr_in_t3")
    mock_check.return_value = [248]

    mock_upload_csmr_data = mocker.patch("cascade.input_data.db.csmr._upload_csmr_data_to_tier_3")

    assert not load_csmr_to_t3(mock_execution_context,
                               mock_execution_context.parameters.model_version_id)

    assert not mock_upload_csmr_data.called


def test_csmr_check_in_t3_by_location(ihme):
    ec = make_execution_context()
    # This should be a list of location ids.
    locs = _csmr_in_t3(ec, 267245)
    assert locs == [80]
    # If the model version id doesn't exist, return an empty list.
    locs = _csmr_in_t3(ec, -3)
    assert not locs
