import pytest
from . import example_db
from . import dismod_tests

print ('Case 7: Location and group fixed and random effects with group covariate prior, with group data and sex covariate.')

use_group_mulcov = False
file_name = 'example.db'

config = {'sex_effect': True,
          'node_effects': True,
          'group_effects': True,
          'use_group_mulcov': use_group_mulcov,
          'include_group_data': True,
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
    success, db = dismod_tests.run_test(file_name, config, truth)

    if success: print ('Dismod_AT fit succeeded -- that is the correct result.')
    if assert_correct:
        assert success, 'Dismod_AT fit failed -- that is not the correct result.'

def test_2(dismod, assert_correct = True):

    db = example_db.example_db(file_name, **db_kwds)
    success, db = dismod_tests.run_test(file_name, config, truth, test_asymptotic=True)

    if success: print ('Dismod_AT fit with setup for asymptotic statistics succeeded -- that is the correct result.')
    if assert_correct:
        assert success, 'Dismod_AT fit with asymptotic statistics failed -- that is not the correct result.'

if __name__ == '__main__':
    test_1(None, assert_correct = False)
    test_2(None, assert_correct = False)
