import pytest
if __name__ == '__main__':
    import example_db
    import dismod_tests
else:
    from . import example_db
    from . import dismod_tests

print ('Case 3: Location and group fixed and random effects.')

use_group_mulcov = False
file_name = 'example.db'

config = {'sex_effect': False,
          'node_effects': True,
          'group_effects': True,
          'use_group_mulcov': use_group_mulcov,
          'include_group_data': False,
          'zero_sum_mulcov': True}

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

    if not success: print ('Dismod_AT failed -- that is the correct result.')
    if assert_correct:
        assert not success

if __name__ == '__main__':
    test_1(None, assert_correct = False)
    
