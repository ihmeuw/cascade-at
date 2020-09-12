import pytest
from . import example_db
from . import dismod_tests

print ('Case 9: Location and group fixed and random effects with group covariate prior, sex covariate, without group data .')

use_group_mulcov = False
file_name = 'example.db'

config = {'sex_effect': True,
          'node_effects': True,
          'group_effects': True,
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

    db = example_db.example_db(file_name, **db_kwds)
    success, db = dismod_tests.run_test(file_name, config, truth, test_asymptotic=True)

    if not success: print ('Dismod_AT did not succeed Hessian problems -- that is the correct result.')
    if assert_correct: assert not success

    # For the Hessian to be positive, parent rate and subgroup random effect densities must be something other than uniform
    prior['parent_density'] = 'gaussian'
    prior['parent_std'] = 1000
    prior['subgroup_density'] = 'gaussian'
    prior['subgroup_std'] = 1000

    db_kwds.update(dict(prior=prior))
    example_db.example_db(file_name, **db_kwds)
    success, db = dismod_tests.run_test(file_name, config, truth, test_asymptotic=True)

    if success: print ('Dismod_AT asymptotic statistics succeeded -- that is the correct result.')
    if assert_correct: assert success

if __name__ == '__main__':
    test_1(None, assert_correct = False)
