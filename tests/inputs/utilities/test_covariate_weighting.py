import pytest
import numpy as np
import pandas as pd

from cascade_at.inputs.utilities.covariate_weighting import CovariateInterpolator


@pytest.fixture
def age_df():
    return pd.DataFrame(
        [[31, 85, 90],
         [32, 90, 95],
         [235, 95, 125]],
        columns=['age_group_id', 'age_lower', 'age_upper']
    )


@pytest.fixture
def test_cov(age_df):
    dd = pd.DataFrame(
        [[2010, 31, 0.1],
         [2010, 32, 0.2],
         [2010, 235, 0.3],
         [2011, 31, 0.35],
         [2011, 32, 0.4],
         [2011, 235, 0.5]],
        columns=['year_id', 'age_group_id', 'mean_value']
    )
    dd['location_id'] = 100
    dd['sex_id'] = 1
    dd = dd.merge(age_df, how='left').sort_values(by=['year_id', 'age_lower'])
    return dd


@pytest.fixture
def test_pop(age_df):
    dd = pd.DataFrame(
        [[31, 2010, 1000],
         [31, 2011, 2000],
         [32, 2010, 1500],
         [32, 2011, 2500],
         [235, 2010, 3000],
         [235, 2011, 4000]],
        columns=['age_group_id', 'year_id', 'population']
    )
    dd['location_id'] = 100
    dd['sex_id'] = 1
    dd = dd.merge(age_df, how='left').sort_values(by=['year_id', 'age_lower'])
    return dd


@pytest.fixture
def test_data(age_df):
    test_data = pd.DataFrame([
        dict(age_lower=90, age_upper=95, time_lower=2010, time_upper=2011)
    ])
    test_data['location_id'] = 100
    test_data['sex_id'] = 1
    return test_data


@pytest.fixture
def covariate_interpolator(test_cov, test_pop):
    return CovariateInterpolator(test_cov, test_pop)


@pytest.fixture
def y0():
    return 2010.1


@pytest.fixture
def y1():
    return 2011.5


@pytest.fixture
def a0():
    return 87


@pytest.fixture
def a1():
    return 100


@pytest.fixture
def time_wt(y0, y1):
    return


@pytest.fixture
def age_wt(a0, a1):
    return


df = pd.DataFrame(
        [[2010, 31, 0.1],
         [2010, 32, 0.2],
         [2010, 235, 0.3],
         [2011, 31, 0.35],
         [2011, 32, 0.4],
         [2011, 235, 0.5]],
        columns=['year_id', 'age_group_id', 'mean_value']
    )


@pytest.mark.parametrize("age_lower,age_upper,time_lower,time_upper,answer", [
    (85., 85.1, 2010., 2010.1, 0.1),
    (90., 90.1, 2010., 2010.1, 0.2),
    (95., 95.1, 2010., 2010.1, 0.3),
    (85., 85.1, 2011., 2011.1, 0.35),
    (90., 90.1, 2011., 2011.1, 0.4),
    (95., 95.1, 2011., 2011.1, 0.5),
])
def test_exact_values_age_time(covariate_interpolator, age_lower, age_upper, time_lower, time_upper, answer):
    assert np.allclose(
        covariate_interpolator.interpolate(
            [100, 1, age_lower, age_upper, time_lower, time_upper],
        ), answer, atol=1e-10, rtol=1e-10
    )


@pytest.mark.parametrize("y0,y1", [
    (2010.0, 2011.0),
    (2010.0, 2011.5),
    (2010.1, 2011.5),
    (2011.0, 2011.5),
    (2011.0, 2013.0),
    (2012.0, 2018.0),
    (2009.0, 2010.0),
    (2007.0, 2009.0)
])
def test_covariate_interpolation_over_time(covariate_interpolator, y0, y1, test_cov, test_pop, test_data):
    data = test_data.copy()
    data[['time_lower', 'time_upper']] = y0, y1
    cov = test_cov[(test_cov.age_group_id == 32)]
    pop = test_pop[(test_pop.age_group_id == 32)]

    y0 = covariate_interpolator._restrict_time(y0, 2010, 2012)
    y1 = covariate_interpolator._restrict_time(y1, 2010, 2012)

    time_wt = np.asarray([(2011 - y0) / (2012 - 2011), (y1 - 2011) / (2012 - 2011)])
    time_wt[time_wt < 0] = 0.
    pop_wt = time_wt * pop.population.values
    weighted_cov = np.sum(cov.mean_value.values * pop_wt/pop_wt.sum())
    assert np.allclose(
        covariate_interpolator.interpolate(
            [int(data.location_id), int(data.sex_id), float(data.age_lower),
             float(data.age_upper), float(data.time_lower), float(data.time_upper)]
        ), weighted_cov, atol=1e-10, rtol=1e-10
    )


@pytest.mark.parametrize("a0,a1", [
    (87, 100),
    (90, 95),
    (90, 96),
    (80, 100)
])
def test_covariate_interpolation_over_age(covariate_interpolator, a0, a1, test_cov, test_pop, test_data, age_wt):
    data = test_data.copy()
    data[['age_lower', 'age_upper']] = a0, a1
    cov = test_cov[(test_cov.year_id == 2010)]
    pop = test_pop[(test_pop.year_id == 2010)]

    age_wt = np.asarray([(90 - a0) / (90 - 85), 1, (a1 - 95) / (125 - 95)])
    pop_wt = age_wt * pop.population.values
    weighted_cov = np.sum(cov.mean_value.values * pop_wt/pop_wt.sum())
    assert np.allclose(
        covariate_interpolator.interpolate(
            [int(data.location_id), int(data.sex_id), float(data.age_lower),
             float(data.age_upper), float(data.time_lower), float(data.time_upper)]
        ), weighted_cov, atol=1e-10, rtol=1e-10
    )


@pytest.mark.parametrize("y0,y1", [
    (2010.0, 2011.0),
    (2010.0, 2011.5),
    (2010.1, 2011.5),
    (2011.0, 2011.5)
])
@pytest.mark.parametrize("a0,a1", [
    (87, 100),
    (90, 95),
    (90, 96),
    (80, 100)
])
def test_covariate_interpolation_over_age_and_time(covariate_interpolator, y0, y1, a0, a1,
                                                   test_cov, test_pop, test_data):
    data = test_data.copy()
    data[['age_lower', 'age_upper']] = a0, a1
    data[['time_lower', 'time_upper']] = y0, y1
    cov = test_cov.copy()
    pop = test_pop.copy()

    age_wt = np.asarray([(90 - a0) / (90 - 85), 1, (a1 - 95) / (125 - 95)])
    time_wt = np.asarray([(2011 - y0) / (2012 - 2011), (y1 - 2011) / (2012 - 2011)])
    wt = np.outer(time_wt, age_wt)

    pop_wt = wt * pop.population.values.reshape(wt.shape)
    weighted_cov = np.sum(cov.mean_value.values.reshape(wt.shape) * pop_wt/pop_wt.sum())
    assert np.allclose(
        covariate_interpolator.interpolate(
            [int(data.location_id), int(data.sex_id), float(data.age_lower),
             float(data.age_upper), float(data.time_lower), float(data.time_upper)],
        ),
        weighted_cov, atol=1e-10, rtol=1e-10
    )


def test_restrict_year(covariate_interpolator):
    assert covariate_interpolator._restrict_time(1970, time_min=1980, time_max=1990) == 1980
    assert covariate_interpolator._restrict_time(1991, time_min=1980, time_max=1990) == 1990
    assert covariate_interpolator._restrict_time(1985, time_min=1980, time_max=1990) == 1985
