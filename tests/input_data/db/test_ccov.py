import pytest

import pandas as pd

from cascade.input_data.db.ccov import country_covariates


@pytest.fixture
def mock_get_covariate_estimates(mocker):
    return mocker.patch("cascade.input_data.db.ccov.get_covariate_estimates")


@pytest.fixture
def mock_ccov_estimates():
    mock_ccov_estimates = pd.DataFrame(columns=[
        "covariate_id", "location_id", "age_group_id", "year_id",
        "sex_id", "mean_value"])
    mock_ccov_estimates.loc[0] = [26, 101, 22, 1990, 1, 1000]
    return mock_ccov_estimates


@pytest.fixture
def expected_ccov():
    expected_ccov = pd.DataFrame(columns=[
        "covariate_id", "location_id", "age_group_id", "year_id",
        "sex_id", "mean_value"])
    expected_ccov.loc[0] = [26, 101, 22, 1990, 1, 1000]
    expected_ccov.loc[1] = [26, 101, 22, 1990, 1, 1000]
    expected_ccov.loc[2] = [26, 101, 22, 1990, 1, 1000]

    return expected_ccov


@pytest.mark.skip
def test_country_covariates_real(mock_execution_context):

    ccov = country_covariates(mock_execution_context)

    assert set(ccov.columns) == {"covariate_id", "location_id", "age_group_id",
                                 "year_id", "sex_id", "mean_value"}

    assert set(ccov['covariate_id'].unique()) == {26, 28, 33}


def test_country_covariates_mock(
        mock_execution_context, mock_get_covariate_estimates,
        mock_ccov_estimates, expected_ccov):

    mock_get_covariate_estimates.return_value = mock_ccov_estimates

    pd.testing.assert_frame_equal(
        country_covariates(mock_execution_context),
        expected_ccov, check_like=True)
