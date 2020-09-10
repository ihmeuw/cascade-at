import pytest
from . import example_db
from . import dismod_tests
from cascade_at.dismod.api.run_dismod import run_dismod
import re

print ('Case 4: Location and group fixed and random effects with group covariate.')

use_group_mulcov = True
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
               node_effects = node_effects,
               subgroup_effects = group_effects,
               tol_fixed = dismod_tests.tol_fixed,
               tol_random = dismod_tests.tol_random)

def test_0(assert_correct = True):
    """
    Parent rate and subgroup random effect densities must be something other than uniform for this problem to solve.
    With gaussian priors, there seems to be a scaling problem in dismod (see the initial objective function), 
    and dismod does not converge well.
    """

    prior['parent_density'] = 'gaussian'
    prior['parent_std'] = 100
    prior['subgroup_density'] = 'gaussian'
    prior['subgroup_std'] = 100

    db_kwds.update({'prior': prior})
    db = example_db.example_db(file_name, **db_kwds)
    run_dismod(db.path, 'init')
    info = run_dismod(db.path, 'fit both')
    print (info.stdout)
    global stdout
    stdout = info.stdout

    lines = stdout.splitlines()
    for i, line in enumerate(lines):
        if 'iter' in line and 'objective' in line:
            break
    col_index = lines[i].split().index('objective')
    objective = float(lines[i+1].split()[col_index])
    assert objective > 1e-4, "Dismod scaled this problem incorrectly."

def test_1(assert_correct = True):
    """
    Parent rate and subgroup random effect densities must be something other than uniform for this problem to solve.
    With gaussian priors, there seems to be a scaling problem in dismod (see the initial objective function), 
    and dismod does not converge well.
    """

    prior['parent_density'] = 'gaussian'
    prior['parent_std'] = 100
    prior['subgroup_density'] = 'gaussian'
    prior['subgroup_std'] = 100

    db_kwds.update({'prior': prior})
    db = example_db.example_db(file_name, **db_kwds)
    # run_dismod(db.path, 'set option print_level_fixed 0')
    success, db = dismod_tests.run_test(file_name, config, truth)

    if not success:
        msg = 'Dismod_AT succeeded, but there is unresolvable ambiguity between the rate and group rate mulcov.'
        var = ((sum((db.fit_var.fit_var_value - db.truth_var.truth_var_value)[:4]**2))**.5 < 1e-7)
        fit = ((sum(db.fit_data_subset.weighted_residual**2))**.5 < 1e-7)
        if not var:
            msg += '\nThe fit_var values do not match the truth.'
        if not fit:
            msg += "\nData weighted residual errors are too large."
        if not (var and fit):
            msg += '\nDismod may be scaling this problem improperly.'
        assert var and fit, msg

def test_2(assert_correct = True):
    """
    Parent rate must be log-gaussian and subgroup random effect densities must be something other than uniform for this problem to solve
    With a log-gaussian prior, dismod converges well
    """

    prior['parent_density'] = 'log_gaussian'
    prior['parent_std'] = 100
    prior['parent_eta'] = 1e-5
    prior['subgroup_density'] = 'gaussian'
    prior['subgroup_std'] = 1000

    db_kwds.update({'prior': prior})
    db = example_db.example_db(file_name, **db_kwds)
    success, db = dismod_tests.run_test(file_name, config, truth)

    if not success:
        msg = 'Dismod_AT succeeded, but there is unresolvable ambiguity between the rate and group rate mulcov.'
        var = ((sum((db.fit_var.fit_var_value - db.truth_var.truth_var_value)[:4]**2))**.5 < 1e-7)
        fit = ((sum(db.fit_data_subset.weighted_residual**2))**.5 < 1e-7)
        if not var:
            msg += '\nThe fit values for the unambiguous variables failed.'
        if not fit:
            msg += "\nData weighted residual errors are too large."
        if not (var and fit):
            msg += '\nDismod may be scaling this problem improperly.'
        assert var and fit, msg

def test_3(assert_correct = True):
    """
    Parent rate must be log-gaussian and subgroup random effect densities must be something other than uniform for this problem to solve
    Parent rate must have guidance for to resolve that ambiguity
    """

    # Give the parent rate guidance
    prior['parent_density'] = 'gaussian'
    prior['parent_mean'] = truth['iota_parent_true']
    prior['parent_std'] = 1e-10
    prior['subgroup_density'] = 'gaussian'
    prior['subgroup_std'] = 10000

    db_kwds.update({'prior': prior})
    db = example_db.example_db(file_name, **db_kwds)
    success, db = dismod_tests.run_test(file_name, config, truth)

    if assert_correct:
        assert success, 'Dismod_AT ran, but there is ambiguity between the rate and group rate mulcov in this test.'

def test_4(assert_correct = True):
    """
    Parent rate must be log-gaussian and subgroup random effect densities must be something other than uniform for this problem to solve
    Parent rate must have guidance for to resolve that ambiguity
    With this configuration, the asymptotic statistics work fine
    """

    success, db = dismod_tests.run_test(file_name, config, truth, test_asymptotic = True)

    if assert_correct:
        assert success, 'Dismod_AT asymptotics failed. The Hessian identifies the presence of the ambiguity.'

if __name__ == '__main__':
    test_0(assert_correct = True)
    test_1(assert_correct = True)
    test_2(assert_correct = True)
    test_3(assert_correct = True)
    test_4(assert_correct = True)
