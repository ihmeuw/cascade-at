import pytest
import os
import numpy as np

if __name__ == '__main__':
    import example_db
    import dismod_tests
    dismod = True
else:
    from . import example_db
    from . import dismod_tests

print('Case 1: New fitting strategy -- first fit the non-ODE integrands, use that solution to initialize the ODE integrand fit.')

use_group_mulcov = False
file_name = 'ode_example.db'

config = {'sex_effect': True,
          'node_effects': True,
          'group_effects': False,
          'use_group_mulcov': use_group_mulcov,
          'include_group_data': False,
          'zero_sum_mulcov': False}

truth, prior, node_effects, group_effects = dismod_tests.test_setup(use_group_mulcov)


db_kwds = dict(test_config=config,
               truth=truth,
               prior=prior,
               node_effects=node_effects,
               subgroup_effects=group_effects,
               tol_fixed=dismod_tests.tol_fixed,
               tol_random=dismod_tests.tol_random)

db = example_db.example_db(file_name, **db_kwds)
truth = dismod_tests.get_truth(db_kwds['test_config'], db_kwds['truth']) 

def test_setup_covariate_names(dismod, assert_correct=True):
    # The covariate naming is funky. Fix it for the test.
    try:
        cov = db.covariate
        cov.loc[cov.covariate_name == 'one', 'c_covariate_name'] = 's_one'
        cov.loc[cov.covariate_name == 'sex', 'c_covariate_name'] = 's_sex'
        cov.loc[cov.covariate_name == 'one', 'covariate_name'] = 'x_0'
        cov.loc[cov.covariate_name == 'sex', 'covariate_name'] = 'x_1'
        db.covariate = cov
        success = True
    except:
        success = False
    if assert_correct:
        assert success

def test_help(dismod, assert_correct=True):
    try:
        # Make sure dismod works
        import subprocess
        expect = ("usage:    dmdismod [-h | --help]                      # Print detailed help.\n"
                  "usage:    dmdismod database [ODE] command [arguments] # Run dmdismod commands.\n"
                  "Omitting 'ODE' calls the standard dismod_at executable.\n"
                  "Specifying 'ODE' dispatches to the ODE fitting strategy code.\n").replace(' ', '').replace('\n','')
        rtn = subprocess.check_output('dmdismod').decode().replace(' ', '').replace('\n','')
        assert rtn == expect, "dmdismod without arguments return was not correct."

        expect = ("/opt/local/bin/dmdismod --help\n"
                  "usage: dmdismod [-h] [-m [MAX_COVARIATE_EFFECT]] [-c MULCOV_VALUES [MULCOV_VALUES ...]] [-o [ODE_HOLD_OUT_LIST]] [-s [RANDOM_SEED]] [-f [SUBSET]] [-d [RANDOM_SUBSAMPLE]] [-p [SAVE_TO_PATH]]\n"
                  "[-t [REFERENCE_DB]]\n"
                  "path dispatch option\n"
                  "\n"
                  "positional arguments:\n"
                  "path                  Path to the Dismod_AT sqlite database\n"
                  "dispatch              If dispatch == 'ODE', use ODE fitting strategy.If missing, use standard dismod_at commands.\n"
                  "option                For the ODE fitting strategy, one of ('init', 'fit' or 'students').\n"
                  "\n"
                  "optional arguments:\n"
                  "-h, --help            show this help message and exit\n"
                  "-m [MAX_COVARIATE_EFFECT], --max-covariate-effect [MAX_COVARIATE_EFFECT]\n"
                  "Maximum absolute covariate effect = multiplier * (covariate - referece). Note that exp(effect) multiplies a model value to get the model value for this covariate value. (Noise\n"
                  "covariate multipliers are not included.)\n"
                  "-c MULCOV_VALUES [MULCOV_VALUES ...], --mulcov-values MULCOV_VALUES [MULCOV_VALUES ...]\n"
                  "Constrain covariate multipliers to the specified value\n"
                  "-o [ODE_HOLD_OUT_LIST], --ode-hold-out-list [ODE_HOLD_OUT_LIST]\n"
                  "Integrands to hold out during the ODE fit\n"
                  "-s [RANDOM_SEED], --random-seed [RANDOM_SEED]\n"
                  "Random seed for the random_subsampling\n"
                  "-f [SUBSET], --subset [SUBSET]\n"
                  "Filter out all hold out and covariate out-of-range data prior to fit.\n"
                  "-d [RANDOM_SUBSAMPLE], --random-subsample [RANDOM_SUBSAMPLE]\n"
                  "Number of random subsamples to fit.\n"
                  "-p [SAVE_TO_PATH], --save-to-path [SAVE_TO_PATH]\n"
                  "Path to directory where to store the results\n"
                  "-t [REFERENCE_DB], --reference_db [REFERENCE_DB]\n"
                  "Path to the reference databases. Fit results are compared to these databases for testing purposes.\n").replace(' ', '').replace('\n','')
        rtn = subprocess.check_output(['dmdismod', '--help']).decode().replace(' ', '').replace('\n','')
        assert rtn == expect, "Help return was not correct."
        success = True
    except:
        success = False
    if assert_correct:
        assert success

def test_ode_init(dismod, assert_correct=True):
    try:
        os.system(f'dmdismod {db.path} ODE init --random-seed 123')

        # Remove the meas_noise the ODE fitting strategy adds and check the fit value
        fit_var = db.var.merge(db.fit_var, left_on = 'var_id', right_on = 'fit_var_id')
        fit = fit_var.loc[fit_var.var_type != 'mulcov_meas_noise', 'fit_var_value'].values
        success = np.allclose(truth, fit, atol=1e-8, rtol=1e-8)
    except:
        success = False
    if assert_correct:
        assert success

def test_ode_fit(dismod, assert_correct=True):
    try:
        os.system(f'dmdismod {db.path} ODE fit')

        # Remove the meas_noise the ODE fitting strategy adds
        fit_var = db.var.merge(db.fit_var, left_on = 'var_id', right_on = 'fit_var_id')
        fit = fit_var.loc[fit_var.var_type != 'mulcov_meas_noise', 'fit_var_value'].values
        success = np.allclose(truth, fit, atol=1e-8, rtol=1e-4)
    except:
        success = False
    if assert_correct:
        assert success

def test_ode_students(dismod, assert_correct=True):
    try:
        os.system(f'dmdismod {db.path} ODE students')
        # Remove the meas_noise the ODE fitting strategy adds
        fit_var = db.var.merge(db.fit_var, left_on = 'var_id', right_on = 'fit_var_id')
        fit = fit_var.loc[fit_var.var_type != 'mulcov_meas_noise', 'fit_var_value'].values
        success = np.allclose(truth, fit, atol=1e-8, rtol=1e-4)
    except:
        success = False
    if assert_correct:
        assert success

if __name__ == '__main__':
    test_setup_covariate_names(dismod, assert_correct=True)
    test_help(dismod, assert_correct=True)
    test_ode_init(dismod, assert_correct=True)
    test_ode_fit(dismod, assert_correct=True)
    test_ode_students(dismod, assert_correct=True)
