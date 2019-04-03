import pandas as pd
import pytest

from cascade.input_data.db.country_covariates import country_covariates


@pytest.fixture
def mock_db_queries(mocker):
    return mocker.patch("cascade.input_data.db.country_covariates.db_queries")


@pytest.fixture
def mock_ccov_estimates():
    mock_ccov_estimates = pd.DataFrame(columns=[
        "location_id", "age_group_id", "year_id",
        "sex_id", "mean_value"])
    mock_ccov_estimates.loc[0] = [101, 22, 1990, 1, 1000]
    return mock_ccov_estimates


@pytest.fixture
def expected_ccov():
    expected_ccov = pd.DataFrame(columns=[
        "location_id", "age_group_id", "year_id",
        "sex_id", "mean_value"])
    expected_ccov.loc[0] = [101, 22, 1990, 1, 1000]

    return expected_ccov


@pytest.fixture
def demographics_default():
    demographics_default = {}
    demographics_default["age_group_ids"] = [
        2, 3, 4, 5, 6,
        7, 8, 9, 10, 11,
        12, 13, 14, 15, 16,
        17, 18, 19, 20, 30,
        31, 32, 235, 27]
    demographics_default["year_ids"] = [
        1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989,
        1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
        2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
        2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 1950, 1951,
        1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961,
        1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971,
        1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979]
    demographics_default["sex_ids"] = [1, 2]
    demographics_default["location_ids"] = [102]

    return demographics_default


def test_country_covariates_real(ihme, demographics_default):

    country_covariate_id = 26
    gbd_round_id = 6
    decomp_step = "step1"

    ccov = country_covariates(country_covariate_id, demographics_default, gbd_round_id, decomp_step)

    assert set(ccov.columns) == {
        "location_id", "age_group_id", "year_id", "sex_id", "mean_value"}

    assert len(ccov) == 1 * 1 * 2 * 23 * 38
    assert set(ccov["location_id"].unique()) == {102}
    assert set(ccov["sex_id"].unique()) == {1, 2}
    # These age groups are fewer than requested. 27 is missing.
    assert set(ccov["age_group_id"].unique()) == {
        2, 3, 4, 5, 6,
        7, 8, 9, 10, 11,
        12, 13, 14, 15, 16,
        17, 18, 19, 20, 30,
        31, 32, 235}
    assert set(ccov["sex_id"].unique()) == {1, 2}
    # These years are 38 of the requested 68.
    assert set(ccov["year_id"].unique()) == {
        1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989,
        1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
        2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
        2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017}


def test_country_covariates_mock(
        demographics_default, mock_db_queries,
        mock_ccov_estimates, expected_ccov):

    mock_db_queries.get_covariate_estimates.return_value = mock_ccov_estimates

    country_covariate_id = 33
    gbd_round_id = 6
    decomp_step = "step1"

    pd.testing.assert_frame_equal(
        country_covariates(country_covariate_id, demographics_default, gbd_round_id, decomp_step),
        expected_ccov, check_like=True)
