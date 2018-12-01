import pytest

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from cascade.input_data.configuration.construct_country import (
    assign_interpolated_covariate_values,
    compute_covariate_age_interval,
    get_covariate_data_by_sex,
    )


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
    measurements_1["x_sex"] = [0, 0, 0, 0.5, -0.5, -0.5, 0, 0.5, -0.5]

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
    measurements_2["x_sex"] = [0, 0, 0, 0, 0, 0, 0, 0, 0]

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
    covariates_1["x_sex"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0]
    covariates_1["mean_value"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0]

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
    covariates_2["x_sex"] = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
                             -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
                             -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
                             0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                             0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                             0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    covariates_2["mean_value"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0,
                                  2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                  2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                  2, 2, 2, 2, 2, 2, 2, 2]
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
    covariates_3["x_sex"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0]
    covariates_3["mean_value"] = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                  4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                  4, 4, 4, 4, 4, 4, 4, 4,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0]

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
                                  index=np.arange(0,28))

    return mean_value_covs_2


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
    FEMALE = -0.5
    MALE = 0.5
    BOTH = 0

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


def test_assign_interpolated_covariate_values_sex_both_1d(measurements_1, covariates_1, covariate_column_1):
    """
    covariates have multiple time values, only one age group, and only both_sexes (no female, no male)
    measurements have all three sexes (female, male, both), and multiple age groups and time values
    """
    sex = pd.Series()

    cov_col = assign_interpolated_covariate_values(measurements_1, sex, covariates_1)

    pd.testing.assert_series_equal(covariate_column_1, cov_col)


@pytest.mark.skip
def test_assign_interpolated_covariate_values_sex_mf_1d(measurements_2, covariates_2, covariate_column_2):
    """
    covariates have multiple time values, only one age group, and two sexes (female, male)
    measurements have only both_sexes (no female, no male), and multiple age groups and time values
    """
    sex = pd.Series()

    #cov_col = assign_interpolated_covariate_values(measurements_2, sex, covariates_2)
    #pd.testing.assert_series_equal(covariate_column_2, cov_col)

    #print(f"cov_col: {cov_col}")
    print(f"covariate_column_2: {covariate_column_2}")

    covs_f = covariates_2[covariates_2["x_sex"] == -0.5]
    covs_m = covariates_2[covariates_2["x_sex"] == 0.5]
    covs_b = covs_f.merge(covs_m, on=["age_lower", "age_upper", "time_lower", "time_upper"],
            how="inner")
    covs_b["mean_value"] = covs_b[["mean_value_x", "mean_value_y"]].mean(axis=1)

    #print(f"covs_b cols: {covs_b.columns}")
    #print(f"size covs_f: {covs_f.shape}")
    #print(f"size covs_m: {covs_m.shape}")
    #print(f"size covs_b: {covs_b.shape}")
    #print(f" covs_b min mean_value: {covs_b['mean_value'].min(), covs_b['mean_value'].max()}")
    
    z = griddata((covariates_2["avg_time"],), covariates_2["mean_value"], (covariates_2["avg_time"],))    

    #print(f"length z: {len(z)}")

    meas_b = measurements_2[measurements_2["x_sex"] == 0]
    meas_f = measurements_2[measurements_2["x_sex"] == -0.5]
    meas_m = measurements_2[measurements_2["x_sex"] == 0.5]

    #print(f"meas_b size: {meas_b.shape}")
    #print(f"meas_f size: {meas_f.shape}")
    #print(f"meas_m size: {meas_m.shape}")

    z_b = griddata((covs_b["avg_time_x"],), covs_b["mean_value"], (meas_b["avg_time"],))
    #print(f"z_b: {z_b}")

    z_f = griddata((covs_f["avg_time"],), covs_f["mean_value"], (meas_f["avg_time"],))
    #print(f"z_f: {z_f}, {type(z_f)}")
    #print(f"z_f: avg_time: {covs_f['avg_time']}")
    #print(f"z_f: mean_value: {covs_f['mean_value']}")
    #print(f"z_f: meas avg_time: {meas_f['avg_time']}")

    covs_m = covs_m.reset_index()
    #print(f"z_m: avg_time: {covs_m['avg_time']}")
    #print(f"z_m: mean_value: {covs_m['mean_value']}")
    #print(f"z_m: meas avg_time: {meas_m['avg_time']}")

    z_m = griddata((covs_m["avg_time"],), covs_m["mean_value"], (meas_m["avg_time"],))
    #print(f"z_m: {z_m}")


def test_assign_interpolated_covariate_values_sex_both_2d(measurements_1, covariates_3, covariate_column_3):
    """
    covariates have multiple time values, two age groups, and only both_sexes (no female, no male)
    measurements have all three sexes (female, male, both), and multiple age groups and time values
    measurements have one age group which is missing from the middle of the covariate overall age interval
    """

    sex = pd.Series()

    cov_col = assign_interpolated_covariate_values(measurements_1, sex, covariates_3)

    pd.testing.assert_series_equal(covariate_column_3, cov_col, check_exact=False)


