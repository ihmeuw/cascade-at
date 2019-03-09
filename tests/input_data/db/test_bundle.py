from unittest.mock import ANY

import pytest

import pandas as pd

from cascade.input_data.db.bundle import _get_bundle_id, freeze_bundle
from cascade.input_data.db.study_covariates import covariate_ids_to_names
from cascade.input_data import InputDataError


def test_get_bundle_id__success(mock_execution_context, mock_database_access):
    cursor = mock_database_access["cursor"]
    cursor.__iter__.side_effect = lambda: iter([(123,)])

    bundle_id = _get_bundle_id(mock_execution_context,
                               mock_execution_context.parameters.model_version_id)
    assert bundle_id == 123
    cursor.execute.assert_called_with(
        ANY, args={"model_version_id": mock_execution_context.parameters.model_version_id}
    )


def test_get_bundle_id__no_bundle(mock_execution_context, mock_database_access):
    cursor = mock_database_access["cursor"]
    cursor.__iter__.side_effect = lambda: iter([])

    with pytest.raises(InputDataError) as excinfo:
        _get_bundle_id(mock_execution_context,
                       mock_execution_context.parameters.model_version_id)
    assert "No bundle_id" in str(excinfo.value)


def test_get_bundle_id__multiple_bundles(mock_execution_context, mock_database_access):
    cursor = mock_database_access["cursor"]
    cursor.__iter__.side_effect = lambda: iter([(123,), (456,)])

    with pytest.raises(InputDataError) as excinfo:
        _get_bundle_id(mock_execution_context,
                       mock_execution_context.parameters.model_version_id)
    assert "Multiple bundle_ids" in str(excinfo.value)


def test_freeze_bundle__did_freeze(mock_execution_context, mock_database_access, mocker):
    mock_check = mocker.patch("cascade.input_data.db.bundle._bundle_is_frozen")
    mock_check.return_value = False

    mock_bundle_id = mocker.patch("cascade.input_data.db.bundle._get_bundle_id")
    mock_bundle_id.return_value = 123

    mock_get_bundle_data = mocker.patch("cascade.input_data.db.bundle._get_bundle_data")
    mock_put_bundle_data = mocker.patch("cascade.input_data.db.bundle._upload_bundle_data_to_tier_3")
    mock_get_covariate_data = mocker.patch(
        "cascade.input_data.db.bundle.get_study_covariates")
    mock_put_covariate_data = mocker.patch("cascade.input_data.db.bundle._upload_study_covariates_to_tier_3")

    mvid = 12345
    assert freeze_bundle(mock_execution_context, mvid)

    cursor = mock_database_access["cursor"]
    model_version_id = mock_execution_context.parameters.model_version_id
    # TODO Test if these happen within the same transaction
    mock_put_bundle_data.assert_called_with(cursor, model_version_id, mock_get_bundle_data())
    mock_put_covariate_data.assert_called_with(cursor, model_version_id, mock_get_covariate_data())


def test_freeze_bundle__did_not_freeze(mock_execution_context, mock_database_access, mocker):
    mock_check = mocker.patch("cascade.input_data.db.bundle._bundle_is_frozen")
    mock_check.return_value = True

    mock_put_bundle_data = mocker.patch("cascade.input_data.db.bundle._upload_bundle_data_to_tier_3")
    mock_put_covariate_data = mocker.patch("cascade.input_data.db.bundle._upload_study_covariates_to_tier_3")

    mvid = 234
    assert not freeze_bundle(mock_execution_context, mvid)

    assert not mock_put_bundle_data.called
    assert not mock_put_covariate_data.called


def test_covariate_ids_to_names(mock_execution_context, mock_database_access):
    cursor = mock_database_access["cursor"]
    cursor.__iter__.side_effect = lambda: iter(enumerate("abcdefghijklmnopqrstuvwxyz"))

    raw_covariate_data = pd.DataFrame({"seq": [1, 2, 3, 5], "study_covariate_id": [1, 2, 3, 5]})

    mapping = covariate_ids_to_names(mock_execution_context, raw_covariate_data.study_covariate_id.unique())
    assert mapping[5] == "f"
