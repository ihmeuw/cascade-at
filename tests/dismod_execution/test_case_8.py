import pytest
if __name__ == '__main__':
    import example_db
    import dismod_tests
else:
    from . import example_db
    from . import dismod_tests

print ('Case 8: Location and group fixed and random effects with group covariate prior, sex covariate, without group data .')

file_name = 'example.db'
use_group_mulcov = False

config = {'sex_effect': True,
          'node_effects': True,
          'group_effects': True,
          'use_group_mulcov': use_group_mulcov,
          'include_group_data': False,
          'zero_sum_mulcov': False}

truth, prior, node_effects, group_effects = dismod_tests.test_setup(use_group_mulcov)

# This problem does not solve unless there is a gaussian prior on the subgroup random effects

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

    if not success: print ('Dismod_AT failed -- that is the correct result.')
    if assert_correct:
        assert not success, 'Dismod_AT was supposed to fail -- that is the correct result.'

def test_2(dismod, assert_correct = True):

    prior['subgroup_density'] = 'gaussian'
    prior['subgroup_std'] = 10000

    db_kwds.update({'prior': prior})
    db = example_db.example_db(file_name, **db_kwds)
    success, db = dismod_tests.run_test(file_name, config, truth)

    if success: print ('Dismod_AT succeeded -- that is the correct result.')
    if assert_correct:
        assert success, 'Dismod_AT succeeded -- that is the correct result.'

def test_3(dismod, assert_correct = True):
    prior['parent_density'] = 'gaussian'
    prior['parent_std'] = 10000
    db_kwds.update({'prior': prior})
    db = example_db.example_db(file_name, **db_kwds)
    success, db = dismod_tests.run_test(file_name, config, truth, test_asymptotic = True)

    if success: print ('Dismod_AT succeeded -- that is the correct result.')
    if assert_correct:
        assert success, 'Dismod_AT succeeded -- that is the correct result.'

if __name__ == '__main__':
    test_1(None, assert_correct = False)
    test_2(None, assert_correct = False)
    test_3(None, assert_correct = False)
