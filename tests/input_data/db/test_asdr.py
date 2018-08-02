# from unittest.mock import ANY

# import pytest

from cascade.input_data.db.asdr import (
    promote_asdr_t2_to_tier_3,
)


def test_load_asdr_to_t3_did_upload(mock_execution_context,
                                    mock_database_access, mocker):

    cursor = mock_database_access["cursor"]
    print(cursor)

    mock_check = mocker.patch(
        "cascade.input_data.db.asdr.exists_model_version_asdr_t3")
    mock_check.return_value = False

    mock_get_asdr_data = mocker.patch(
        "cascade.input_data.db.asdr._get_asdr_t2_data")
    mock_upload_asdr_data = mocker.patch(
        "cascade.input_data.db.asdr._upload_asdr_data_to_tier_3")

    assert promote_asdr_t2_to_tier_3(mock_execution_context)

    model_version_id = mock_execution_context.parameters.model_version_id

    mock_upload_asdr_data.assert_called_with(cursor, model_version_id,
                                             mock_get_asdr_data())


def test_load_asdr_to_t3_no_upload(mock_execution_context,
                                   mock_database_access, mocker):

    mock_check = mocker.patch(
        "cascade.input_data.db.asdr.exists_model_version_asdr_t3")
    mock_check.return_value = True

    mock_upload_asdr_data = mocker.patch(
        "cascade.input_data.db.asdr._upload_asdr_data_to_tier_3")

    assert not promote_asdr_t2_to_tier_3(mock_execution_context)
    assert not mock_upload_asdr_data.called
