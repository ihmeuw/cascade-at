import pytest
import os
import numpy as np

if __name__ == '__main__':
    import example_db
    import dismod_tests
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

def test_1(dismod, assert_correct=True):
    db = example_db.example_db(file_name, **db_kwds)
    # FIXME -- hacks
    cov = db.covariate
    cov.loc[cov.covariate_name == 'one', 'c_covariate_name'] = 's_one'
    cov.loc[cov.covariate_name == 'sex', 'c_covariate_name'] = 's_sex'
    cov.loc[cov.covariate_name == 'one', 'covariate_name'] = 'x_0'
    cov.loc[cov.covariate_name == 'sex', 'covariate_name'] = 'x_1'
    db.covariate = cov
    truth = dismod_tests.get_truth(db_kwds['test_config'], db_kwds['truth']) 
    try:
        os.system('dmdismod')
        os.system('dmdismod --help')
        os.system(f'dmdismod {db.path} ODE init')

        # Remove the meas_noise the ODE fitting strategy adds
        fit_var = db.var.merge(db.fit_var, left_on = 'var_id', right_on = 'fit_var_id')
        fit = fit_var.loc[fit_var.var_type != 'mulcov_meas_noise', 'fit_var_value'].values
        success = np.allclose(truth, fit, atol=1e-8, rtol=1e-8)
        os.system(f'dmdismod {db.path} ODE fit')

        # Remove the meas_noise the ODE fitting strategy adds
        fit_var = db.var.merge(db.fit_var, left_on = 'var_id', right_on = 'fit_var_id')
        fit = fit_var.loc[fit_var.var_type != 'mulcov_meas_noise', 'fit_var_value'].values
        success &= np.allclose(truth, fit, atol=1e-8, rtol=1e-8)

        os.system(f'dmdismod {db.path} ODE students')
        # Remove the meas_noise the ODE fitting strategy adds
        fit_var = db.var.merge(db.fit_var, left_on = 'var_id', right_on = 'fit_var_id')
        fit = fit_var.loc[fit_var.var_type != 'mulcov_meas_noise', 'fit_var_value'].values
        np.allclose(truth, fit, atol=1e-8, rtol=1e-4)
    except:
        success = False
    if assert_correct:
        assert success

if __name__ == '__main__':
    test_1(None, assert_correct=True)
