import sys
import os
import numpy as np
from pdb import set_trace
from collections import OrderedDict
from cascade_at.dismod.api.dismod_io import DismodIO
from cascade_at.dismod.api.run_dismod import run_dismod

tol_fixed = 1e-10
tol_random = 1e-10

if 0:
    def system(args):
        cmd = ' '.join(args)
        rtn = os.system(cmd)
        if rtn != 0:
            raise Exception (f"ERROR -- return status {rtn} from: '{cmd}' ")
        return rtn
else:
    def system(args):
        # run_dismod assumes that the execitable name is dmdismod
        # To make this code work with the dismod_at executable, make an executable simlink: dmdismod -> dismod_at
        program = args[0]
        db = args[1]
        cmd = ' '.join(args[2:])
        info = run_dismod(db, cmd)
        if info.stdout or info.stderr: print ('-'*30)
        if info.stdout: print (info.stdout)
        if info.stderr: print (info.stderr)
        # if info.stderr:
        #     raise Exception(f'ERROR: {info.stderr}')
        if ' step ' in info.stdout and ' f ' in info.stdout and ' grad ' in info.stdout:
            raise Exception (f'GRADIENT ERROR')
        if 'error' in info.stderr.lower() or info.exit_status:
            raise Exception(f'ERROR: {info.stderr}')
        if ' invalid number ' in info.stderr.lower():
            raise Exception(f'ERROR: {info.stderr}')

def test_setup (use_group_mulcov = False, test_asymptotic = False):
    iota_child_true = .1
    iota_group_true = .3 if use_group_mulcov else 0
    iota_subgroup_true = 0.3
    truth = dict(iota_sex_true = {'none': 0, 'g1': 0.2}, # sex effect
                 iota_parent_true = .01,                 # value corresponding to parent with no effects
                 iota_child_true = iota_child_true,      # absolute value of child random effects 
                 iota_group_true = iota_group_true,      # value corresponding to group with no effects
                 iota_subgroup_true = iota_subgroup_true) # absolute value of group random effects 

    node_effects = {'p1': 0,
                    'c1': -iota_child_true,
                    'c2': +iota_child_true}

    group_effects = {'none': truth['iota_group_true'],
                     's1': -truth['iota_subgroup_true'],
                     's2': +truth['iota_subgroup_true']}
                     

    # These settings affect whether or not the asymptotic statistics work
    iota_subgroup_density = 'uniform'
    iota_subgroup_std = None
    if 0 and test_asymptotic:
        iota_subgroup_density = 'gaussian'
        iota_subgroup_std = 100
        iota_child_density = 'gaussian'
        iota_child_std = 100

    priors = dict(parent_density = 'uniform',
                  parent_std = None,
                  child_density = 'uniform',
                  child_std = None,
                  subgroup_density = iota_subgroup_density,
                  subgroup_std = iota_subgroup_std)

    return (truth, priors, node_effects, group_effects)

def run_test(file_name, test_config, truth_in,
             start_from_truth = False, test_asymptotic = False):
    program = 'dismod_at'
    #

    gradient_error = False
    try:
        db = DismodIO(file_name)

        # from dismod_db_api import DismodDbAPI as API
        # db = API(file_name)

        if not test_config['node_effects'] and not test_config['group_effects']:
            truth = truth[truth_in['iota_parent_true']]
        if test_config['node_effects'] and not test_config['group_effects']:
            truth = [-truth_in['iota_child_true'], +truth_in['iota_child_true'], truth_in['iota_parent_true']]
        if not test_config['node_effects'] and test_config['group_effects']:
            truth = [-truth_in['iota_subgroup_true'], +truth_in['iota_subgroup_true'], truth_in['iota_parent_true']]
            if test_config['use_group_mulcov']: truth += [truth_in['iota_group_true']]
        if test_config['node_effects'] and test_config['group_effects']:
            truth = [-truth_in['iota_child_true'], +truth_in['iota_child_true'], -truth_in['iota_subgroup_true'], 
                     truth_in['iota_subgroup_true'], truth_in['iota_parent_true']]
            if test_config['use_group_mulcov']: truth += [truth_in['iota_group_true']]
        if test_config['sex_effect']:
            # print ('There is ambiguity between the sex and the group effects -- infinite number of solutions.')
            truth.append(truth_in['iota_sex_true']['g1'])

        system([ program, file_name, 'init' ])

        # Initialize the truth_var table to the correct answer
        if True:
            # Need to create the truth_var table before setting it.
            # Can't seem to get db.create_tables to work, so use dismod_at to do it
            system([ program, file_name, 'set', 'truth_var', 'prior_mean'])
        truth_var = db.truth_var
        truth_var['truth_var_value'] = truth
        db.truth_var = truth_var

        if 0:
            try:
                # Check dismod gradients
                gradient_error = None
                option = db.option
                system([ program, file_name, 'set', 'option', 'derivative_test_fixed', 'adaptive'])
                system([ program, file_name, 'set', 'option', 'derivative_test_random', 'second-order'])
                system([ program, file_name, 'set', 'option', 'max_num_iter_fixed', '-1'])
                system([ program, file_name, 'set', 'option', 'max_num_iter_random', '100'])
                # Start from the truth
                if 0:
                    system([ program, file_name, 'set', 'start_var', 'truth_var'])
                    system([ program, file_name, 'set', 'scale_var', 'truth_var'])
                system([ program, file_name, 'fit', 'fixed'])
                system([ program, file_name, 'set', 'start_var', 'fit_var'])
                system([ program, file_name, 'fit', 'both'])
            except Exception as ex:
                print (ex)
                gradient_error = ex
                raise ex
            finally:
                db.option = option

        if start_from_truth:
            system([ program, file_name, 'set', 'start_var', 'truth_var'])
            system([ program, file_name, 'set', 'scale_var', 'truth_var'])

        # Check that prediction matches the measured data
        cols = db.avgint.columns.tolist()
        db.avgint = db.data.rename(columns={'data_id':'avgint_id'})[cols]
        system([ program, file_name, 'predict', 'truth_var'])
        check = np.allclose(db.data.meas_value, db.predict.avg_integrand, atol=1e-10, rtol=1e-10)
        assert check, 'ERROR: Predict from truth does not match the data'

        #
        # Fit fixed effects
        system([ program, file_name, 'fit', 'fixed'])
        os.system(f'dismodat.py {file_name} db2csv')
        if test_asymptotic:
            system([ program, file_name, 'sample', 'asymptotic', 'fixed', '10'])
        #
        # Fit both fixed and random effects
        system([ program, file_name, 'set', 'start_var', 'fit_var'])
        system([ program, file_name, 'set', 'scale_var', 'fit_var'])
        if (test_config['group_effects'] or test_config['node_effects']):
            system([ program, file_name, 'fit', 'both'])
            os.system(f'dismodat.py {file_name} db2csv')
        else:
            print ('Skipping fit both because there are no random effects.')

        check = np.allclose(db.fit_data_subset.weighted_residual, [0]*len(db.fit_data_subset),
                            atol=1e-8, rtol=1e-8)
        assert check, 'ERROR: Measured values do not match the fit result integrand values.'

        print ('Tests OK -- fit both fit_data_subset and measured_data agree.')

        if test_asymptotic:
            system([ program, file_name, 'sample', 'asymptotic', 'both', '10'])

        # -----------------------------------------------------------------------

        if gradient_error:
            print ('ERROR: Gradient check failed.')
            print (gradient_error)
            return False, db
        else:
            print ('Test OK')
            return True, db

    except Exception as ex:
        print (ex)
        print ('Test FAILED')
        return False, db
    finally:
        print (f'fit_var_value: {db.fit_var.fit_var_value.tolist()}')
        print (f'RMS(fit_var_value - truth): {np.sqrt(np.sum((db.fit_var.fit_var_value - db.truth_var.truth_var_value)**2))}')
        print (db.var.merge(db.fit_var, left_on = 'var_id', right_on = 'fit_var_id')
               .drop(columns = ['integrand_id', 'fit_var_id', 'residual_value', 'residual_dage',
                                'residual_dtime', 'lagrange_value', 'lagrange_dage', 'lagrange_dtime']))
        print (f'RMS(weighted_residual): {np.sum(np.sqrt((db.fit_data_subset.weighted_residual)**2))}')
        print (db.data.merge(db.fit_data_subset, left_on = 'data_id', right_on = 'fit_data_subset_id')
               .drop(columns = ['fit_data_subset_id', 'integrand_id', 'weight_id', 'eta', 'nu', 'meas_std', 'avg_integrand', 'hold_out']))

