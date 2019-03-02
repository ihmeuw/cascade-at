import pytest

import numpy as np
import pandas as pd

from cascade.core.context import ExecutionContext
from cascade.input_data.configuration.construct_country import (
    assign_interpolated_covariate_values,
    compute_covariate_age_interval,
    compute_covariate_age_time_dimensions,
    get_covariate_data_by_sex,)


@pytest.fixture
def execution_context():
    defaults = {
        "gbd_round_id": 5,
        "database": "dismod-at-dev",
    }
    execution_context = ExecutionContext()
    execution_context.parameters = defaults

    return execution_context


@pytest.fixture
def measurements_1():
    """
    Age and time ranges and sex values as seen in a particular measurements data set,
    notice that the ranges aren't regular, and the sexes aren't either (female and male) or
    (both) only, and aren't balanced.  The 1978 values will be out-of-time-range for the
    covariates fixture data.
    """
    measurements_1 = pd.DataFrame()
    measurements_1["age_lower"] = [0, 0, 0, 25, 25, 35, 0, 45, 45]
    measurements_1["age_upper"] = [18, 18, 18, 44, 34, 44, 99, 54, 54]
    measurements_1["time_lower"] = [1994, 1994, 1994, 2004, 2004, 2004, 2013, 1978, 1978]
    measurements_1["time_upper"] = [2011, 2006, 1996, 2004, 2004, 2004, 2015, 1978, 1978]
    measurements_1["sex_id"] = [3, 3, 3, 1, 2, 2, 3, 1, 2]

    return measurements_1


@pytest.fixture
def measurements_2():
    """
    Similar to measurements_1, but here sex is both only.
    """
    measurements_2 = pd.DataFrame()
    measurements_2["age_lower"] = [0, 0, 0, 25, 25, 35, 0, 45, 45]
    measurements_2["age_upper"] = [18, 18, 18, 44, 34, 44, 99, 54, 54]
    measurements_2["time_lower"] = [1994, 1994, 1994, 2004, 2004, 2004, 2013, 1978, 1978]
    measurements_2["time_upper"] = [2011, 2006, 1996, 2004, 2004, 2004, 2015, 1978, 1978]
    measurements_2["sex_id"] = [3, 3, 3, 3, 3, 3, 3, 3, 3]

    measurements_2["avg_age"] = measurements_2[["age_lower", "age_upper"]].mean(axis=1)
    measurements_2["avg_time"] = measurements_2[["time_lower", "time_upper"]].mean(axis=1)

    return measurements_2


@pytest.fixture
def covariates_1():
    """
    covariates have: one age group, many years (1990-2017), sex = both
    """
    covariates_1 = pd.DataFrame()
    covariates_1["age_lower"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0]
    covariates_1["age_upper"] = [125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
                                 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
                                 125, 125, 125, 125, 125, 125, 125, 125]
    covariates_1["time_lower"] = [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
                                  2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
                                  2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
    covariates_1["time_upper"] = [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
                                  2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
                                  2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
    covariates_1["sex_id"] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                              3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                              3, 3, 3, 3, 3, 3, 3, 3]
    covariates_1["mean_value"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0]
    covariates_1["covariate_id"] = 26

    return covariates_1


@pytest.fixture
def covariates_2():
    """
    covariates have: one age group, many years (1990-2017), sex = female, male
    """
    covariates_2 = pd.DataFrame()
    covariates_2["age_lower"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0]
    covariates_2["age_upper"] = [125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
                                 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
                                 125, 125, 125, 125, 125, 125, 125, 125,
                                 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
                                 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
                                 125, 125, 125, 125, 125, 125, 125, 125]
    covariates_2["time_lower"] = [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
                                  2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
                                  2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017,
                                  1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
                                  2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
                                  2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
    covariates_2["time_upper"] = [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
                                  2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
                                  2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017,
                                  1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
                                  2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
                                  2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
    covariates_2["sex_id"] = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                              2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                              2, 2, 2, 2, 2, 2, 2, 2,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1]
    covariates_2["mean_value"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0,
                                  2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                  2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                  2, 2, 2, 2, 2, 2, 2, 2]
    covariates_2["covariate_id"] = 26
    covariates_2["avg_age"] = covariates_2[["age_lower", "age_upper"]].mean(axis=1)
    covariates_2["avg_time"] = covariates_2[["time_lower", "time_upper"]].mean(axis=1)

    return covariates_2


@pytest.fixture
def covariates_3():
    """
    covariates have: two age groups, many years (1990-2017), sex = both
    """
    covariates_3 = pd.DataFrame()
    covariates_3["age_lower"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0,
                                 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
                                 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
                                 30, 30, 30, 30, 30, 30, 30, 30]
    covariates_3["age_upper"] = [25, 25, 25, 25, 25, 25, 25, 25, 25, 25,
                                 25, 25, 25, 25, 25, 25, 25, 25, 25, 25,
                                 25, 25, 25, 25, 25, 25, 25, 25,
                                 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
                                 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
                                 125, 125, 125, 125, 125, 125, 125, 125]
    covariates_3["time_lower"] = [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
                                  2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
                                  2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017,
                                  1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
                                  2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
                                  2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
    covariates_3["time_upper"] = [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
                                  2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
                                  2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017,
                                  1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
                                  2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
                                  2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
    covariates_3["sex_id"] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                              3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                              3, 3, 3, 3, 3, 3, 3, 3,
                              3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                              3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                              3, 3, 3, 3, 3, 3, 3, 3]
    covariates_3["mean_value"] = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                  4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                  4, 4, 4, 4, 4, 4, 4, 4,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0]
    covariates_3["covariate_id"] = 26

    return covariates_3


@pytest.fixture
def covariate_column_1():
    """Expected output"""
    covariate_column_1 = pd.Series([0.0, 0.0, np.nan, 0.0, np.nan, 0.0, 0.0, 0.0, 0.0],
                                   index=[4, 5, 8, 3, 7, 0, 1, 2, 6])

    return covariate_column_1.sort_index()


@pytest.fixture
def covariate_column_2():
    """Expected output"""
    covariate_column_2 = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.nan, np.nan],
                                   index=[0, 1, 2, 3, 4, 5, 6, 7, 8])

    return covariate_column_2


@pytest.fixture
def covariate_column_3():
    """Expected output"""
    covariate_column_3 = pd.Series([np.nan,
                                    np.nan,
                                    np.nan,
                                    2.646154,
                                    np.nan,
                                    2.338462,
                                    1.723077,
                                    np.nan,
                                    np.nan])

    return covariate_column_3.sort_index()


@pytest.fixture
def mean_value_covs_2():
    """Expected value for sex = both; i.e. the avg of female and male values"""

    mean_value_covs_2 = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                  index=np.arange(0, 28))

    return mean_value_covs_2


def test_compute_covariate_age_time_dimensions(covariates_1, covariates_2, covariates_3):

    cov_at_dims_1 = compute_covariate_age_time_dimensions(covariates_1)
    cov_at_dims_2 = compute_covariate_age_time_dimensions(covariates_2)
    cov_at_dims_3 = compute_covariate_age_time_dimensions(covariates_3)

    assert cov_at_dims_1.age_1d is False and cov_at_dims_1.time_1d is True
    assert cov_at_dims_2.age_1d is False and cov_at_dims_2.time_1d is True
    assert cov_at_dims_3.age_1d is True and cov_at_dims_3.time_1d is True


def test_age_interval(covariates_1, covariates_2, covariates_3):

    age_interval_1 = compute_covariate_age_interval(covariates_1)
    age_interval_2 = compute_covariate_age_interval(covariates_2)
    age_interval_3 = compute_covariate_age_interval(covariates_3)

    assert 0 in age_interval_1
    assert 50 in age_interval_1
    assert 125 in age_interval_1
    assert 126 not in age_interval_1

    assert 0 in age_interval_2
    assert 50 in age_interval_2
    assert 125 in age_interval_2
    assert 126 not in age_interval_2

    assert 0 in age_interval_3
    assert 50 in age_interval_3
    assert 125 in age_interval_3
    assert 126 not in age_interval_3
    assert 29 not in age_interval_3


def test_get_covariate_data_by_sex(covariates_1, covariates_2, covariates_3, mean_value_covs_2):
    """
    covariates_1 and covariates_3 have only sex = both, so covariate data
    for all three sexes should be equal.
    covariates_2 has sex = male and sex = female, so both should be the average
    """
    FEMALE = 2
    MALE = 1
    BOTH = 3

    cov_data_1 = get_covariate_data_by_sex(covariates_1)
    cov_data_2 = get_covariate_data_by_sex(covariates_2)
    cov_data_3 = get_covariate_data_by_sex(covariates_3)

    pd.testing.assert_frame_equal(cov_data_1[FEMALE], cov_data_1[BOTH])
    pd.testing.assert_frame_equal(cov_data_1[MALE], cov_data_1[BOTH])

    pd.testing.assert_frame_equal(cov_data_3[FEMALE], cov_data_3[BOTH])
    pd.testing.assert_frame_equal(cov_data_3[MALE], cov_data_3[BOTH])

    if not cov_data_2[MALE].empty and not cov_data_2[FEMALE].empty:
        assert not cov_data_2[BOTH].empty

    pd.testing.assert_series_equal(cov_data_2[BOTH]["mean_value"], mean_value_covs_2, check_names=False)


def test_assign_interpolated_covariate_values_sex_both_1d(ihme, measurements_1, covariates_1,
                                                          covariate_column_1, execution_context):
    """
    covariates have multiple time values, only one age group, and only both_sexes (no female, no male)
    measurements have all three sexes (female, male, both), and multiple age groups and time values
    """
    cov_col = assign_interpolated_covariate_values(measurements_1, covariates_1, False)

    pd.testing.assert_series_equal(covariate_column_1, cov_col)


def test_assign_interpolated_covariate_values_sex_mf_1d(ihme, measurements_2, covariates_2,
                                                        covariate_column_2, execution_context):
    """
    covariates have multiple time values, only one age group, and two sexes (female, male)
    measurements have only both_sexes (no female, no male), and multiple age groups and time values
    """
    cov_col = assign_interpolated_covariate_values(measurements_2, covariates_2, False)

    pd.testing.assert_series_equal(covariate_column_2, cov_col)


def test_assign_interpolated_covariate_values_sex_both_2d(ihme, measurements_1, covariates_3,
                                                          covariate_column_3, execution_context):
    """
    covariates have multiple time values, two age groups, and only both_sexes (no female, no male)
    measurements have all three sexes (female, male, both), and multiple age groups and time values
    measurements have one age group which is missing from the middle of the covariate overall age interval
    """
    cov_col = assign_interpolated_covariate_values(measurements_1, covariates_3, False)

    pd.testing.assert_series_equal(covariate_column_3, cov_col, check_exact=False)
