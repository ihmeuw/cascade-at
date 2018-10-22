"""
Study covariates are mostly either 0 or 1 and kept as a list of records
for which the study covariate is 1.
"""
import pandas as pd
import pytest

from cascade.input_data import InputDataError
from cascade.input_data.db.study_covariates import _normalize_covariate_data


@pytest.fixture
def basic_bundle():
    return pd.DataFrame(
        {"seq": [2, 4, 6, 8, 10],
         "mean": [2.0, 4.0, 6.0, 8.0, 10.0]}
    ).set_index("seq")


@pytest.fixture
def binary_covariate():
    return pd.DataFrame({"study_covariate_id": [64, 102, 64, 64, 102],
                         "bundle_id": [77, 77, 77, 77, 77],
                         "seq": [4, 4, 6, 8, 10]})


def test_create_columns(basic_bundle, binary_covariate):
    """
    Given covariate data and a bundle, create Series that correspond to
    the covariate values for the bundle bundle.
    """
    id_to_name = {64: "uses_blinker", 102: "likes_polka"}
    covs = pd.DataFrame({
        "uses_blinker": [0.0, 1.0, 1.0, 1.0, 0.0],
        "likes_polka": [0.0, 1.0, 0.0, 0.0, 1.0],
    },
    index=[2, 4, 6, 8, 10])
    normalized = _normalize_covariate_data(basic_bundle.index, id_to_name, binary_covariate)
    pd.testing.assert_frame_equal(normalized, covs)


def test_empty_columns(basic_bundle, binary_covariate):
    """
    If there are no entries for a covariate, it should be all zeros.
    """
    id_to_name = {64: "uses_blinker", 102: "likes_polka", 47: "meditates"}
    covs = pd.DataFrame({
        "likes_polka": [0.0, 1.0, 0.0, 0.0, 1.0],
        "meditates": [0.0, 0.0, 0.0, 0.0, 0.0],
        "uses_blinker": [0.0, 1.0, 1.0, 1.0, 0.0],
    },
    index=[2, 4, 6, 8, 10])
    normalized = _normalize_covariate_data(basic_bundle.index, id_to_name, binary_covariate)
    pd.testing.assert_frame_equal(normalized.sort_index("columns"), covs)


def test_id_disagrees(basic_bundle):
    """
    If there are no entries for a covariate, it should be all zeros.
    """
    not_in_bundle_index = 20
    id_to_name = {64: "uses_blinker", 102: "likes_polka"}
    covs_in = pd.DataFrame({"study_covariate_id": [64, 102, 64, 64, 102],
                  "bundle_id": [77, 77, 77, 77, 77],
                  "seq": [4, 4, 6, 8, not_in_bundle_index]})
    with pytest.raises(InputDataError):
        _normalize_covariate_data(basic_bundle.index, id_to_name, covs_in)
