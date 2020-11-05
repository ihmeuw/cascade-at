import pytest
import numpy as np
import pandas as pd
import os

try:
    from . import example_db
    from . import dismod_tests
except:
    import example_db
    import dismod_tests

print ('Case 10: Check prediction from sample table.')

file_name = 'example.db'

config = {'sex_effect': True,
          'node_effects': False,
          'group_effects': True,
          'use_group_mulcov': False,
          'include_group_data': False,
          'zero_sum_mulcov': False}

truth, prior, node_effects, group_effects = dismod_tests.test_setup(use_group_mulcov=False)

# This problem does not solve unless there is a gaussian prior on the subgroup random effects

group_effects = {'none': 0, 's1': 0, 's2': 0}

def test_1(assert_correct=False):
    db_kwds = dict(test_config = config,
                   truth = truth,
                   prior = prior,
                   node_effects = node_effects,
                   subgroup_effects = group_effects,
                   tol_fixed = dismod_tests.tol_fixed,
                   tol_random = dismod_tests.tol_random)

    db = example_db.example_db(file_name, **db_kwds)
    data = db.data[1:3].reset_index(drop=True)
    data['data_id'] = data.index
    data['subgroup_id'] = 1
    db.data = data
    db.avgint = db.data.rename(columns={'data_id': 'avgint_id'})[db.avgint.columns]

    dismod_tests.system(['dmdismod', db.path, 'init'])
    dismod_tests.system(['dmdismod', db.path, 'fit', 'fixed'])
    dismod_tests.system(['dmdismod', db.path, 'sample', 'asymptotic', 'fixed', '10000'])
    dismod_tests.system(['dmdismod', db.path, 'predict', 'sample'])
    grps = db.predict.groupby('avgint_id')
    mean = grps.avg_integrand.mean()
    std =  grps.avg_integrand.std(ddof=1)
    assert np.allclose(mean, data.meas_value, atol=1e-4, rtol = 1e-4), 'Dismod_AT mean prediction failed -- that is not the correct result.'
    assert np.allclose(std, data.meas_std, atol=1e-4, rtol = 1e-4), 'Dismod_AT std prediction failed -- that is not the correct result.'
    print (f'Dismod_AT succeeded -- that is the correct result')
    print (f'  Predict abs differences from data -- mean: {(mean-data.meas_value).abs().max()}, '
           f'std: {(std-data.meas_std).abs().max()}')
    if assert_correct:
        assert success, 'Dismod_AT succeeded -- that is the correct result.'

if __name__ == '__main__':
    test_1(assert_correct = False)
