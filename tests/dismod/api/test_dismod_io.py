import pytest
import numpy as np
import pandas as pd
import sys

from cascade_at.dismod.api.dismod_io import DismodIO


@pytest.fixture
def dm(tmp_path):
    return DismodIO(path=tmp_path / 'dismod.db')


@pytest.fixture
def dm_read(tmp_path):
    return DismodIO(path=tmp_path / 'dismod.db')


def test_age(dm, dm_read):
    age = pd.DataFrame({
        'age': [0.0, 0.1, 0.2, 5., 10.]
    })
    dm.age = age
    assert (dm_read.age['age'] == age['age']).all()
    assert len(dm_read.age) == len(age)
    assert all(dm_read.age.columns == ['age_id', 'age'])


def test_time(dm, dm_read):
    time = pd.DataFrame({
        'time': [1990, 2000]
    })
    dm.time = time
    assert (dm_read.time['time'] == time['time']).all()
    assert len(dm_read.time) == len(time)
    assert all(dm_read.time.columns == ['time_id', 'time'])


def test_age_wrong(dm):
    with pytest.raises(Exception):
        dm.time = pd.DataFrame({
            'age': [1, 2, 3]
        })


def test_integrand(dm, dm_read):
    dm.integrand = pd.DataFrame({
        'integrand_id': 1, 'integrand_name': '', 'minimum_meas_cv': 0.
    }, index=[0])
    assert len(dm_read.integrand) == 1
    assert all(dm_read.integrand.columns == ['integrand_id', 'integrand_name', 'minimum_meas_cv'])


def test_density(dm, dm_read):
    dm.density = pd.DataFrame({
        'density_id': 1, 'density_name': ''
    }, index=[0])
    assert len(dm_read.density) == 1
    assert all(dm_read.density.columns == ['density_id', 'density_name'])


def test_covariates(dm, dm_read):
    dm.covariate = pd.DataFrame({
        'covariate_id': 1, 'covariate_name': '', 'reference': np.nan, 'max_difference': np.nan
    }, index=[0])
    assert len(dm_read.covariate) == 1
    assert all(dm_read.covariate.columns == ['covariate_id', 'covariate_name', 'reference', 'max_difference'])


def test_node(dm, dm_read):
    dm.node = pd.DataFrame({
        'node_id': 1, 'node_name': '', 'parent': 0
    }, index=[0])
    assert len(dm_read.node) == 1
    assert all(dm_read.node.columns == ['node_id', 'node_name', 'parent'])


def test_node_error(dm, dm_read):
    dm.node_df = pd.DataFrame({
        'node_id': 1
    }, index=[0])
    with pytest.raises(ValueError):
        df = dm_read.node


def test_prior(dm, dm_read):
    dm.prior = pd.DataFrame({
        'prior_id': 1, 'prior_name': '', 'density_id': 2, 'lower': 1., 'upper': 1.,
        'mean': 1., 'std': 0.1, 'eta': 1e-6, 'nu': np.nan
    }, index=[0])
    assert len(dm_read.prior) == 1
    assert all(dm_read.prior.columns == ['prior_id', 'prior_name', 'density_id',
                                         'lower', 'upper', 'mean', 'std', 'eta', 'nu'])


def test_weight(dm, dm_read):
    dm.weight = pd.DataFrame({
        'weight_id': 1, 'weight_name': '', 'n_age': 1, 'n_time': 1
    }, index=[0])
    assert len(dm_read.weight) == 1
    assert all(dm_read.weight.columns == ['weight_id', 'weight_name', 'n_age', 'n_time'])


def test_weight_grid(dm, dm_read):
    dm.weight_grid = pd.DataFrame({
        'weight_grid_id': 1, 'weight_id': 1, 'age_id': 0, 'time_id': 0, 'weight': 1
    }, index=[0])
    assert len(dm_read.weight_grid) == 1
    assert all(dm_read.weight_grid.columns == ['weight_grid_id', 'weight_id', 'age_id', 'time_id', 'weight'])


def test_smooth(dm, dm_read):
    dm.smooth = pd.DataFrame({
        'smooth_id': 1, 'smooth_name': '', 'n_age': 1, 'n_time': 1,
        'mulstd_value_prior_id': 1, 'mulstd_dage_prior_id': 1, 'mulstd_dtime_prior_id': 1
    }, index=[0])
    assert len(dm_read.smooth) == 1
    assert all(dm_read.smooth.columns == ['smooth_id', 'smooth_name', 'n_age', 'n_time',
                                          'mulstd_value_prior_id', 'mulstd_dage_prior_id', 'mulstd_dtime_prior_id'])


def test_smooth_grid(dm, dm_read):
    dm.smooth_grid = pd.DataFrame({
        'smooth_grid_id': 1, 'smooth_id': 1, 'age_id': 1, 'time_id': 1,
        'value_prior_id': 1, 'dage_prior_id': 1, 'dtime_prior_id': 1, 'const_value': 1.
    }, index=[0])
    assert len(dm_read.smooth_grid) == 1
    assert all(dm_read.smooth_grid.columns == ['smooth_grid_id', 'smooth_id', 'age_id', 'time_id',
                                               'value_prior_id', 'dage_prior_id', 'dtime_prior_id', 'const_value'])


def test_nslist(dm, dm_read):
    dm.nslist = pd.DataFrame({
        'nslist_id': 1, 'nslist_name': ''
    }, index=[0])
    assert len(dm_read.nslist) == 1
    assert all(dm_read.nslist.columns == ['nslist_id', 'nslist_name'])


def test_nslist_pair(dm, dm_read):
    dm.nslist_pair = pd.DataFrame({
        'nslist_pair_id': 1, 'nslist_id': 1, 'node_id': 1, 'smooth_id': 1
    }, index=[0])
    assert len(dm_read.nslist_pair) == 1
    assert all(dm_read.nslist_pair.columns == ['nslist_pair_id', 'nslist_id', 'node_id', 'smooth_id'])


def test_rate(dm, dm_read):
    dm.rate = pd.DataFrame({
        'rate_id': 1, 'rate_name': '', 'parent_smooth_id': 1,
        'child_smooth_id': 1, 'child_nslist_id': 1
    }, index=[0])
    assert len(dm_read.rate) == 1
    assert all(dm_read.rate.columns == ['rate_id', 'rate_name', 'parent_smooth_id',
                                        'child_smooth_id', 'child_nslist_id'])


def test_mulcov(dm, dm_read):
    dm.mulcov = pd.DataFrame({
        'mulcov_id': 1, 'mulcov_type': '', 'rate_id': 1,
        'integrand_id': 1, 'covariate_id': 1, 'group_smooth_id': 1,
        'group_id': 0, 'subgroup_smooth_id': np.nan
    }, index=[0])
    assert len(dm_read.mulcov) == 1
    assert all(dm_read.mulcov.columns == ['mulcov_id', 'mulcov_type', 'rate_id', 'integrand_id',
                                          'covariate_id', 'group_smooth_id', 'group_id', 'subgroup_smooth_id'])


def test_avgint(dm, dm_read):
    dm.avgint = pd.DataFrame({
        'avgint_id': 1, 'integrand_id': 1, 'node_id': 1, 'weight_id': 1, 'subgroup_id': 0,
        'age_lower': 0., 'age_upper': 1., 'time_lower': 0., 'time_upper': 1.
    }, index=[0])
    assert len(dm_read.avgint) == 1
    assert all(dm_read.avgint.columns == ['avgint_id', 'integrand_id', 'node_id', 'weight_id', 'subgroup_id',
                                          'age_lower', 'age_upper', 'time_lower', 'time_upper'])


def test_data(dm, dm_read):
    dm.data = pd.DataFrame({
        'data_id': 1, 'data_name': '', 'integrand_id': 1, 'density_id': 1,
        'node_id': 1, 'weight_id': 1, 'subgroup_id': 0, 'hold_out': 0, 'meas_value': 1., 'meas_std': 1.,
        'eta': 1e-6, 'nu': np.nan, 'age_lower': 0., 'age_upper': 1., 'time_lower': 0., 'time_upper': 1.
    }, index=[0])
    assert len(dm_read.data) == 1
    assert all(dm_read.data.columns == ['data_id', 'data_name', 'integrand_id', 'density_id',
                                        'node_id', 'weight_id', 'subgroup_id', 'hold_out', 'meas_value', 'meas_std',
                                        'eta', 'nu', 'age_lower', 'age_upper', 'time_lower', 'time_upper'])


def test_option(dm, dm_read):
    dm.option = pd.DataFrame({
        'option_id': 0, 'option_name': '', 'option_value': ''
    }, index=[0])
    assert len(dm_read.option) == 1
    assert all(dm_read.option.columns == ['option_id', 'option_name', 'option_value'])


def test_subgroup(dm, dm_read):
    dm.subgroup = pd.DataFrame({
        'subgroup_id': 0,
        'subgroup_name': 'world',
        'group_id': 0,
        'group_name': 'world'
    }, index=[0])
    assert len(dm_read.subgroup) == 1
    assert all(dm_read.subgroup.columns == ['subgroup_id', 'subgroup_name', 'group_id', 'group_name'])
