import pytest
from . import example_db
from . import dismod_tests

print ('Case 6: Location and group fixed and random effects with group covariate prior.')

file_name = 'example.db'
use_group_mulcov = False

config = {'sex_effect': False,
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

def test_1(assert_correct = True):

    db = example_db.example_db(file_name, **db_kwds)
    success, db = dismod_tests.run_test(file_name, config, truth)

    assert success
    if success: print ('Dismod_AT succeeded -- that is the correct result.')

def test_2(assert_correct = True):

    # Parent rate and subgroup random effect densities must be something other than uniform for the Hessian to be non-singular
    prior['parent_density'] = 'gaussian'
    prior['parent_std'] = 10000
    prior['subgroup_density'] = 'gaussian'
    prior['subgroup_std'] = 10000
    db_kwds.update(dict(prior=prior))
    example_db.example_db(file_name, **db_kwds)
    success, db = dismod_tests.run_test(file_name, config, truth, test_asymptotic=True)

    assert success
    if success: print ('Dismod_AT asymptotic statistics succeeded -- that is the correct result.')

if __name__ == '__main__':
    test_1(assert_correct = False)
    test_2(assert_correct = False)
