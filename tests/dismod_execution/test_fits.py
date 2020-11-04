import os
import pytest
import numpy as np
import sqlite3
from collections import OrderedDict
from sqlalchemy import create_engine
try:
    from . import example_db
    from . import dismod_tests
except:
    import example_db
    import dismod_tests

print ('Case 11: Compare fit both, sample simulate both, and sample asymptotic both.')

def to_sql(filename, df, table_name, sql_type):
    with sqlite3.connect(filename, isolation_level = 'exclusive') as con:
        df.to_sql(table_name, con, if_exists='replace', index=False, dtype=sql_type)

use_group_mulcov = True
file_name = 'example.db'

config = {'sex_effect': True,
          'node_effects': True,
          'group_effects': False,
          'use_group_mulcov': use_group_mulcov,
          'include_group_data': False,
          'zero_sum_mulcov': False}

truth, prior, node_effects, group_effects = dismod_tests.test_setup(use_group_mulcov)

db_kwds = dict(test_config = config,
               truth = truth,
               prior = prior,
               node_effects = node_effects,
               subgroup_effects = group_effects,
               tol_fixed = dismod_tests.tol_fixed,
               tol_random = dismod_tests.tol_random)

def test_1(dismod, assert_correct = True):
    success = True
    db = example_db.example_db(file_name, **db_kwds)
    var_truth = [-.1, +.1, .01, .2]
    data = db.data
    data['x_1'] = [0.0, +1.0, 0.0, +1.0]
    db.data = data

    os.system(f'dmdismod {db.path} init')
    os.system(f'dmdismod {db.path} fit both')
    success &= np.allclose(db.fit_var.fit_var_value, var_truth)

    os.system(f'dmdismod {db.path} set truth_var fit_var')
    os.system(f'dmdismod {db.path} simulate 1')
    sim = db.data_sim
    sim['data_sim_value'] = data['meas_value']
    sim['data_sim_stdcv'] = data['meas_std']
    sim['data_sim_delta'] = data['meas_std']
    data_sim_dtypes = OrderedDict([('data_sim_id', 'integer primary key'),
                                   ('simulate_index', 'integer'),
                                   ('data_subset_id', 'integer'),
                                   ('data_sim_value', 'real'),
                                   ('data_sim_stdcv', 'real'),
                                   ('data_sim_delta', 'real')])
    to_sql(db.path, sim, 'data_sim', data_sim_dtypes)
    os.system(f'dmdismod {db.path} sample simulate both 1')
    success &= np.allclose(db.sample.var_value, var_truth)

    # Use Gaussian priors so asymptotic sample will work
    prior = db.prior
    prior['density_id'] = 1
    prior['std'] = 1
    db.prior = prior
    os.system(f'dmdismod {db.path} sample asymptotic both 100000')
    sample = db.sample.groupby('var_id', as_index = False)
    sample_mean = sample.var_value.mean()
    print (sample_mean)
    success &= np.allclose(sample_mean.var_value, var_truth, atol=.001, rtol=.0001)
    if success: print ('Dismod_AT succeeded -- that is the correct result.')
    if assert_correct:
        assert success, 'Dismod_AT succeeded -- that is the correct result.'

if __name__ == '__main__':
    test_1(None, assert_correct = False)
    
