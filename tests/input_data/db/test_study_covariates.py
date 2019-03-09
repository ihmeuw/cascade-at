"""
Study covariates are mostly either 0 or 1 and kept as a list of records
for which the study covariate is 1.
"""
import pandas as pd
import pytest

from cascade.input_data import InputDataError
from cascade.input_data.configuration.construct_study import normalize_covariate_data


@pytest.fixture
def basic_bundle():
    return pd.DataFrame(
        {"seq": [2, 4, 6, 8, 10, None, None],
         "mean": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0],
         "sex_id": [2, 1, 1, 3, 2, 1, 2],
         }
    )


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
    id_to_name = {102: "smoking", 64: "love_polka"}
    covs = pd.DataFrame({
        "covariate_sequence_number": [2, 4, 6, 8, 10, None, None],
        "love_polka": [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        "smoking": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    })
    normalized = normalize_covariate_data(basic_bundle, binary_covariate, id_to_name)
    pd.testing.assert_frame_equal(normalized.sort_index("columns"), covs)


def test_empty_columns(basic_bundle, binary_covariate):
    """
    If there are no entries for a covariate, it should be all zeros.
    """
    id_to_name = {102: "smoking", 64: "love_polka", 47: "has_cats"}
    covs = pd.DataFrame({
        "covariate_sequence_number": [2, 4, 6, 8, 10, None, None],
        "has_cats": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "love_polka": [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        "smoking": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    })
    normalized = normalize_covariate_data(basic_bundle, binary_covariate, id_to_name)
    pd.testing.assert_frame_equal(normalized.sort_index("columns"), covs)


@pytest.mark.skip(f"It looks like the ids should disagree. Waiting to hear.")
def test_id_disagrees(basic_bundle):
    """
    If there are no entries for a covariate, it should be all zeros.
    """
    id_to_name = {102: "smoking", 64: "love_polka"}
    not_in_bundle_index = 20
    covs_in = pd.DataFrame(
        {"study_covariate_id": [64, 102, 64, 64, 102],
         "bundle_id": [77, 77, 77, 77, 77],
         "seq": [4, 4, 6, 8, not_in_bundle_index]})
    with pytest.raises(InputDataError):
        normalize_covariate_data(basic_bundle, covs_in, id_to_name)


def test_no_covariates(basic_bundle):
    """
    Given covariate data and a bundle, create Series that correspond to
    the covariate values for the bundle bundle.
    """
    cov_in = pd.DataFrame(
        {"study_covariate_id": [],
         "bundle_id": [],
         "seq": []})
    covs = pd.DataFrame({"covariate_sequence_number": basic_bundle.seq})
    normalized = normalize_covariate_data(basic_bundle, cov_in, {})
    pd.testing.assert_frame_equal(normalized, covs)
