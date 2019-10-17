from unittest.mock import ANY

import pytest

import pandas as pd

from cascade.input_data.db.crosswalk_version import _get_crosswalk_version_id
from cascade.input_data.db.study_covariates import covariate_ids_to_names
from cascade.input_data import InputDataError


def test_get_crosswalk_version_id__success(mock_execution_context, mock_database_access):
    cursor = mock_database_access["cursor"]
    cursor.__iter__.side_effect = lambda: iter([(123,)])

    crosswalk_version_id = _get_crosswalk_version_id(
        mock_execution_context,
        mock_execution_context.parameters.model_version_id
    )
    assert crosswalk_version_id == 123
    cursor.execute.assert_called_with(
        ANY, args={"model_version_id": mock_execution_context.parameters.model_version_id}
    )


def test_get_bundle_id__no_bundle(mock_execution_context, mock_database_access):
    cursor = mock_database_access["cursor"]
    cursor.__iter__.side_effect = lambda: iter([])

    with pytest.raises(InputDataError) as excinfo:
        _get_crosswalk_version_id(
            mock_execution_context,
            mock_execution_context.parameters.model_version_id
        )
    assert "No crosswalk_version_ids" in str(excinfo.value)


def test_get_bundle_id__multiple_bundles(mock_execution_context, mock_database_access):
    cursor = mock_database_access["cursor"]
    cursor.__iter__.side_effect = lambda: iter([(123,), (456,)])

    with pytest.raises(InputDataError) as excinfo:
        _get_crosswalk_version_id(
            mock_execution_context,
            mock_execution_context.parameters.model_version_id
        )
    assert "Multiple crosswalk_version_ids" in str(excinfo.value)


def test_covariate_ids_to_names(mock_execution_context, mock_database_access):
    cursor = mock_database_access["cursor"]
    cursor.__iter__.side_effect = lambda: iter(enumerate("abcdefghijklmnopqrstuvwxyz"))

    raw_covariate_data = pd.DataFrame({"seq": [1, 2, 3, 5], "study_covariate_id": [1, 2, 3, 5]})

    mapping = covariate_ids_to_names(mock_execution_context, raw_covariate_data.study_covariate_id.unique())
    assert mapping[5] == "f"
