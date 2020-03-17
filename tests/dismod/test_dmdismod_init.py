"""
This test uses Brad's getting_started_db
example to create database tables and then make sure
that we can run init on it using this code base to fill the tables.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from cascade_at.dismod.api.run_dismod import run_dismod
from cascade_at.dismod.api.dismod_io import DismodIO
from cascade_at.dismod.api.fill_extract_helpers.reference_tables import (
    default_rate_table, construct_integrand_table, construct_density_table
)


@pytest.fixture
def dm(dismod):
    return DismodIO(path=Path('dismod-init.db'))


def test_fill_tables(dm):
    unknown_omega_world = 1e-2
    known_income_multiplier = -1e-3
    adjusted_omega = unknown_omega_world * np.exp(known_income_multiplier * 1000.0)
    dm.integrand = construct_integrand_table()
    rate = default_rate_table()
    rate.loc[rate.rate_name == 'omega', 'parent_smooth_id'] = 0
    dm.rate = rate
    dm.density = construct_density_table()
    dm.age = pd.DataFrame({
        'age_id': [0, 1],
        'age': [0.0, 100.0]
    })
    dm.time = pd.DataFrame({
        'time_id': [0, 1],
        'time': [1995.0, 2015.0]
    })
    dm.node = pd.DataFrame({
        'node_id': [0],
        'node_name': ['world'],
        'parent': [np.nan]
    })
    dm.weight = pd.DataFrame({
        'weight_id': [0],
        'weight_name': ['constant'],
        'n_age': [1],
        'n_time': [1]
    })
    dm.weight_grid = pd.DataFrame({
        'weight_id': [0],
        'weight_grid_id': [0],
        'age_id': [0],
        'time_id': [0],
        'weight': [1]
    })
    dm.covariate = pd.DataFrame({
        'covariate_id': [0],
        'covariate_name': ['income'],
        'reference': [0],
        'max_difference': [0]
    })
    dm.avgint = pd.DataFrame({
        'avgint_id': [0],
        'integrand_id': [0],
        'node_id': [0],
        'weight_id': [0],
        'subgroup_id': [0],
        'age_lower': [0],
        'age_upper': [0],
        'time_lower': [1995.],
        'time_upper': [1995.],
        'x_0': [1.]
    })
    dm.data = pd.DataFrame({
        'data_id': [0],
        'data_name': ['world'],
        'integrand_id': [0],
        'density_id': [1],
        'node_id': [0],
        'weight_id': [0],
        'subgroup_id': [0],
        'hold_out': [0],
        'meas_value': [np.exp(-1 * adjusted_omega * 50.)],
        'meas_std': [np.exp(-1 * adjusted_omega * 50.) / 20.],
        'eta': [np.nan],
        'nu': [np.nan],
        'age_lower': [50.],
        'age_upper': [50.],
        'time_lower': [2000.],
        'time_upper': [2000.],
        'x_0': [1.]
    })
    dm.prior = pd.DataFrame({
        'prior_id': [0, 1, 2],
        'prior_name': ['prior_not_used', 'prior_omega_parent', 'prior_income_multiplier'],
        'density_id': [0, 0, 0],
        'lower': [np.nan, 1e-4, known_income_multiplier],
        'upper': [np.nan, 1.0, known_income_multiplier],
        'mean': [0.0, 1e-1, known_income_multiplier],
        'std': np.nan,
        'eta': np.nan,
        'nu': np.nan
    })
    dm.smooth = pd.DataFrame({
        'smooth_id': [0, 1],
        'smooth_name': ['smooth_omega_parent', 'smooth_income_multiplier'],
        'n_age': [1, 1],
        'n_time': [1, 1],
        'mulstd_value_prior_id': [1, 2]
    })
    dm.smooth_grid = pd.DataFrame({
        'smooth_grid_id': [0, 1],
        'smooth_id': [0, 1],
        'age_id': [0, 0],
        'time_id': [0, 0],
        'const_value': [1e-3, 1e-3]
    })
    dm.mulcov = pd.DataFrame({
        'mulcov_id': [0],
        'mulcov_type': 'rate_value',
        'rate_id': [0],
        'integrand_id': [0],
        'covariate_id': [0],
        'group_id': [0],
        'subgroup_smooth_id': [1]
    })
    dm.option = pd.DataFrame({
        'option_id': [0, 1, 2, 3],
        'option_name': ['parent_node_id', 'ode_step_size', 'age_avg_split', 'rate_case'],
        'option_value': ['0', '10.0', '5.0', 'iota_zero_rho_zero']
    })
    dm.subgroup = pd.DataFrame({
        'subgroup_id': [0],
        'subgroup_name': ['world'],
        'group_id': [0],
        'group_name': ['world']
    })
    dm.nslist = pd.DataFrame({
        'nslist_id': [0],
        'nslist_name': ['parent']
    })
    dm.nslist_pair = pd.DataFrame({
        'nslist_id': [0],
        'nslist_pair_id': [0],
        'node_id': [0],
        'smooth_id': [0]
    })


def test_dmdismod_init(dm):
    run = run_dismod(dm_file=str(dm.path), command='init')
    if run.exit_status:
        print(run.stdout)
        print(run.stderr)
    assert run.exit_status == 0

