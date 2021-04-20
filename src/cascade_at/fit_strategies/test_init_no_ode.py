import numpy as np
import shutil
from cascade_at.fit_strategies.init_no_ode import system, setup_db, FitNoODE
from cascade_at.fit_strategies.init_no_ode import init_ode_command, fit_ode_command, fit_students_command
from cascade_at.fit_strategies.init_no_ode import _dismod_cmd_, _fit_ihme_py_, _max_iters_
from cascade_at.dismod.api.dismod_io import DismodIO

db = None

def test_cases(case, specific_name = 'fitODE'):
    crohns = '/Users/gma/ihme/epi/at_cascade/data/475533/dbs/1/2/dismod.db'
    # dialysis = '/Users/gma/ihme/epi/at_cascade/data/475527/dbs/96/2/dismod.db' # S Latin America
    dialysis = '/Users/gma/ihme/epi/at_cascade/data/475527/dbs/1/2/dismod.db'  # Global
    kidney = '/Users/gma/ihme/epi/at_cascade/data/475718/dbs/70/1/dismod.db'
    osteo_hip =  '/Users/gma/ihme/epi/at_cascade/data/475526/dbs/1/2/dismod.db'
    osteo_hip_world = '/Users/gma/ihme/epi/at_cascade/data/475745/dbs/1/2/dismod.db'
    osteo_knee = '/Users/gma/ihme/epi/at_cascade/data/475746/dbs/64/2/dismod.db'
    # t1_diabetes =  '/Users/gma/ihme/epi/at_cascade/data/475882/dbs/1/2/dismod.db' # world
    t1_diabetes =  '/Users/gma/ihme/epi/at_cascade/data/475882/dbs/100/2/dismod.db' # HI N America female
    t1_diabetes = '/Users/gma/ihme/epi/at_cascade/data/475588/dbs/100/3/dismod.db' # HI N America both

    if case == 't1_diabetes':
        file_in = t1_diabetes
        max_covariate_effect = 2
        ode_hold_out_list = ['mtexcess']
        mulcov_values = []
    elif case == 'crohns':
        file_in = crohns
        max_covariate_effect = 2
        ode_hold_out_list = []
        mulcov_values = [[ 'x_0', 'iota', 3.8661 ]]
    elif case == 'dialysis':
        file_in = dialysis
        max_covariate_effect = 4
        ode_hold_out_list = []
        mulcov_values = []
    elif case == 'kidney':
        file_in = kidney
        max_covariate_effect = 2
        ode_hold_out_list = []
        mulcov_values = []
    elif case == 'osteo_hip':
        file_in = osteo_hip
        max_covariate_effect = 2
        ode_hold_out_list = []
        mulcov_values = []
    elif case == 'osteo_hip_world':
        file_in = osteo_hip_world
        max_covariate_effect = 2
        ode_hold_out_list = []
        mulcov_values = []
    elif case == 'osteo_knee':
        file_in = osteo_knee
        max_covariate_effect = 2
        ode_hold_out_list = []
        mulcov_values = []
    else:
        raise Exception(f'Disease {case} not found')

    # Keep the original database unmodified
    path, ext = os.path.splitext(file_in)
    db_path = f'{path}_{specific_name}{ext}'
    shutil.copy2(file_in, db_path)

    return db_path, max_covariate_effect, ode_hold_out_list, mulcov_values

def test_commands(case, step, db_path, max_covariate_effect=2, ode_hold_out_list=[], mulcov_values=[],
                  subset=True, random_seed=None, random_subsample=None, reference_db = None):

    global path_ode_cmds, path_students_cmds
    path_no_ode_cmds = f'{fit_ihme_path}/cascade/no_ode_cmds.db'
    path_yes_ode_cmds = f'{fit_ihme_path}/cascade/yes_ode_cmds.db'
    path_students_cmds = f'{fit_ihme_path}/cascade/students_cmds.db'

    cmd = f'{_dismod_cmd_} {db_path} fit ode'

    args = cmd.split()
    path = args[1]

    global db               # For debugging
    if step == 'no_ode':
        db = init_ode_command(args, max_covariate_effect = max_covariate_effect,
                              mulcov_values = mulcov_values,
                              ode_hold_out_list = ode_hold_out_list,
                              subset = subset, random_seed = random_seed, random_subsample = random_subsample,
                              save_to_path = path_no_ode_cmds, reference_db = reference_db)

    if step == 'yes_ode':
        fit_ode_command(args, ode_hold_out_list = ode_hold_out_list,
                        subset = subset, random_seed = random_seed, random_subsample = random_subsample,
                        save_to_path = path_yes_ode_cmds, reference_db = reference_db)

    if step == 'students':
        cmd = f'{_dismod_cmd_} {db_path} fit students'
        args = cmd.split()
        fit_students_command(args, ode_hold_out_list = ode_hold_out_list, 
                             subset = subset, random_seed = random_seed, random_subsample = random_subsample,
                             save_to_path = path_students_cmds, reference_db = reference_db)
    return db

def test(case, step, db_path, max_covariate_effect=2, ode_hold_out_list=[], mulcov_values=[],
         subset=True, random_seed=None, random_subsample=None, reference_db = None): 

    def fix_data_table(db, dm, bypass_hold_out = False):
        # For some reason, the fit_ihme.py data table is sometimes slightly different than the original
        # This causes divergence in the fit results
        if bypass_hold_out:
            hold_out = db.data.hold_out
        cols = db.data.columns.drop('hold_out')
        mask = (dm.data.fillna(-1) != db.data.fillna(-1))
        if bypass_hold_out:
            mask['hold_out'] = False
        mask0 = mask.any(1)
        data = db.data
        if len(data.values[mask]) > 0:
            diff = np.max(np.abs(dm.data.values[mask] - data.values[mask]))
            assert diff < 1e-10, 'Error was too large'
            if np.any(mask):
                print (f'WARNING -- fixed {np.sum(mask.values)} slight differences max ({diff}) between fit_ihme and this data table.')
            data[mask0] = dm.data[mask0]
            if bypass_hold_out:
                data['hold_out'] = hold_out
            db.data = data
            assert compare_dataframes(db.data, dm.data), 'Assignment in fix_data_table  failed'


    fit_ihme_path = f'/Users/gma/ihme/epi/at_cascade/{case}'
    global path_no_ode, path_yes_ode, path_students
    path_no_ode = f'{fit_ihme_path}/cascade/no_ode.db'
    path_yes_ode = f'{fit_ihme_path}/cascade/yes_ode.db'
    path_students = f'{fit_ihme_path}/cascade/students.db'

    cascade_path = f'/Users/gma/ihme/epi/at_cascade/{case}/cascade'
    if not os.path.isdir(cascade_path):
        os.makedirs(cascade_path, exist_ok=True)


    kwds = dict(mulcov_values = mulcov_values, ode_hold_out_list = ode_hold_out_list)

    global db               # For debugging

    if step == 'no_yes_ode':
        print ('--- no_yes_ode ---')
        db = setup_db(db_path, ode_hold_out_list = ode_hold_out_list)
        try:
            # -- no_ode portion --
            system(f'{db.dismod} {db.path} init')
            db.simplify_data(subset = True, random_seed = random_seed, random_subsample = random_subsample)
            db.setup_ode_fit(max_covariate_effect, **kwds)
            db.hold_out_data(integrand_names = db.yes_ode_integrands, hold_out=1)

            if reference_db and case == 'crohns':
                fix_data_table(db, reference_db)

            system(f'{db.dismod} {db.path} init')
            if _max_iters_ is not None: 
                db.set_option('max_num_iter_fixed', _max_iters_)

            if reference_db:
                db.check_input_tables(reference_db, check_hold_out = True)
            db.fit(msg = 'fit_no_ode')
            if reference_db:
                db.check_output_tables(reference_db)
            db.save_database(path_no_ode)

            # -- yes_ode portion --
            db.data = db.input_data
            db.simplify_data(subset = True, random_seed = random_seed, random_subsample = random_subsample)
            db.hold_out_data(integrand_names = db.ode_hold_out_list, hold_out=1)

            if reference_db and case == 'crohns':
                fix_data_table(db, reference_db)

            # use previous fit as starting point
            system(f'{db.dismod} {db.path} set start_var fit_var')

            if reference_db:
                db.check_input_tables(reference_db, check_hold_out = True)
            db.fit(msg='fit_with_ode')
            db.save_database(path_yes_ode)
            if reference_db:
                db.check_output_tables(reference_db)
        except: raise
        finally:
            db.data = db.input_data

    if step == 'no_ode':
        print ('--- no_ode ---')

        db = setup_db(db_path, ode_hold_out_list = ode_hold_out_list)

        try:
            db.simplify_data(subset = subset, random_seed = random_seed, random_subsample = random_subsample)
            db.setup_ode_fit(max_covariate_effect, **kwds)
            db.hold_out_data(integrand_names = db.yes_ode_integrands, hold_out=1)

            if reference_db and case == 'crohns':
                fix_data_table(db, reference_db)

            system(f'{db.dismod} {db.path} init')

            if _max_iters_ is not None:
                db.set_option('max_num_iter_fixed', _max_iters_)

            if reference_db:
                db.check_input_tables(reference_db, check_hold_out = True)
            db.fit(msg = 'fit_no_ode')
            db.save_database(path_no_ode)
            if reference_db:
                db.check_output_tables(reference_db)

        except: raise
        finally:
            db.data = db.input_data

    if step == 'yes_ode':
        print ('--- yes_ode ---')

        db = setup_db(db_path, ode_hold_out_list = ode_hold_out_list)

        try:
            db.simplify_data(subset = subset, random_seed = random_seed, random_subsample = random_subsample)
            db.hold_out_data(integrand_names = db.ode_hold_out_list, hold_out=1)

            if reference_db and case == 'crohns':
                fix_data_table(db, reference_db)

            # use previous fit as starting point
            system(f'{db.dismod} {db.path} set start_var fit_var')

            if _max_iters_ is not None:
                db.set_option('max_num_iter_fixed', _max_iters_)

            if reference_db:
                db.check_input_tables(reference_db, check_hold_out = True)
            db.fit(msg='fit_with_ode')
            db.save_database(path_yes_ode)
            if reference_db:
                db.check_output_tables(reference_db)

        except: raise
        finally:
            db.data = db.input_data

    if step == 'students':
        print ('--- students ---')

        db = setup_db(db_path, ode_hold_out_list = ode_hold_out_list)

        try:
            db.simplify_data(subset = subset, random_seed = random_seed, random_subsample = random_subsample)
            db.hold_out_data(integrand_names = db.ode_hold_out_list, hold_out=1)
            db.set_student_likelihoods(factor_eta = 1e-2, nu = 5)

            if reference_db and case == 'crohns':
                fix_data_table(db, reference_db)

            system(f'{db.dismod} {db.path} set start_var fit_var')

            if _max_iters_ is not None: 
                db.set_option('max_num_iter_fixed', _max_iters_)

            if reference_db:
                db.check_input_tables(reference_db, check_hold_out = True)
            db.fit(msg = 'fit_students')
            db.save_database(path_students)
            if reference_db:
                db.check_output_tables(reference_db)
        except: raise
        finally:
            db.data = db.input_data
    return db

class disable_disease_smoothings:
    """
    Disease smoothings should be included in the disease settings.json file.
    If they are, fit_ihme.py, which uses the disease module, will duplicate those smoothings, causing
    the reference and the newly created smooth tables to differ, causing these tests to fail.

    This class disables the custom smoothings in the disease module for testing.
    """
    def __init__(self, case):
        import importlib
        import dismod_at
        module_name = f'dismod_at.ihme.{case}'
        specific = importlib.import_module(module_name, package = dismod_at)
        self.filename = specific.__file__
        with open (self.filename, 'r') as fn:
            self.file_text = fn.read()
    def disable(self):
        print (f"Disable custom smoothings in {self.filename}")
        txt = self.file_text[:]
        txt = txt.replace('parent_smoothing[', '#disable-parent_smoothing[')
        txt = txt.replace('child_smoothing[', '#disable-child_smoothing[')
        with open (self.filename, 'w') as fn:
            fn.write(txt)
    def restore(self):
        print (f"Restore custom smoothings in {self.filename}")
        with open (self.filename, 'w') as fn:
            fn.write(self.file_text)

def reference_dbs(case):
    fit_ihme_path = f'/Users/gma/ihme/epi/at_cascade/{case}'
    return fit_ihme_path, dict(no_ode = DismodIO(f'{fit_ihme_path}/no_ode/no_ode.db'),
                               yes_ode = DismodIO(f'{fit_ihme_path}/yes_ode/yes_ode.db'),
                               students = DismodIO(f'{fit_ihme_path}/students/students.db'))

def _fit_ihme(case, step, random_seed):
    if step == 'no_yes_ode':
        shutil.rmtree(os.path.join('~/ihme/epi/at_cascade'), case)
        cmd = f'{_fit_ihme_py_} ~/ihme/epi/at_cascade {case} no_ode {random_seed}'
        print (cmd); os.system(cmd)
        cmd = f'{_fit_ihme_py_} ~/ihme/epi/at_cascade {case} yes_ode'
        print (cmd); os.system(cmd)
    if step == 'no_ode':
        shutil.rmtree(os.path.join('~/ihme/epi/at_cascade'), case)
        cmd = f'{_fit_ihme_py_} ~/ihme/epi/at_cascade {case} no_ode {random_seed}'
        print (cmd); os.system(cmd)
    if step == 'yes_ode':
        cmd = f'{_fit_ihme_py_} ~/ihme/epi/at_cascade {case} yes_ode'
        print (cmd); os.system(cmd)
    if step == 'students':
        cmd = f'{_fit_ihme_py_} ~/ihme/epi/at_cascade {case} students'
        print (cmd); os.system(cmd)

if __name__ == '__main__':
    
    if 1:
        # This option closest to the enhanced dismod command version
        # The ODE 'init' command runs dismod init, then a no_ode fit -- the no_ode option
        # The ODE 'fit' command runs a yes_ode fit -- the yes_ode option
        # The ODE 'students' command runs a students fit -- the students option
        ode_option = dict(no_yes_ode = False, no_ode = True, yes_ode = True, students = True)
    else:
        # This option runs dismod init, then no_ode and yes_ode fits in a single step
        ode_option = dict(no_yes_ode = True, no_ode = False, yes_ode = False, students = True)

    common_kwds = dict(subset = True, random_seed = 1234, random_subsample = 1000)

    cases_with_json_smoothings_set_to_brads_values = ['osteo_hip','osteo_knee', 'dialysis', 'kidney', 't1_diabetes', 'crohns']

    # cases = ['osteo_hip']
    # cases = ['osteo_knee']
    # cases = ['t1_diabetes']
    # cases = ['kidney']
    # cases = ['crohns']
    # cases = ['dialysis']
    cases = ['dialysis', 't1_diabetes', 'crohns', 'osteo_hip'] # These cover the range of test options
    # cases = ['osteo_hip','osteo_knee', 'dialysis', 'kidney', 't1_diabetes', 'crohns']

    steps = ['no_ode', 'yes_ode', 'students']


    make_reference = True
    test_funs = True
    test_cmds = True
    test_sh = True

    cases = ['t1_diabetes']

    for case in cases:
        fit_ihme_path, ref_dbs = reference_dbs(case)
        for step in steps:
            ode_option['steps'] = steps
            print ()
            reference_db = ref_dbs[step]
            print ('='*200)
            if make_reference:
                print ('='*200)
                print (f">>> Making reference by running fit_ihme.py for {case} {step} <<<")
                try:
                    if case in cases_with_json_smoothings_set_to_brads_values:
                        disease_smoothings = disable_disease_smoothings(case)
                        disease_smoothings.disable()
                        _fit_ihme(case, step, common_kwds['random_seed'])
                except:
                    raise
                finally:
                    if case in cases_with_json_smoothings_set_to_brads_values:
                        disease_smoothings.restore()

            if test_funs:
                if step == 'no_ode': # The steps [no_ode, yes_ode, students] reuse the previous database 
                    db0_path, max_covariate_effect, ode_hold_out_list, mulcov_values = test_cases(case, 'FitODE_test_functions')

                print ('-'*200)
                print (f">>> Running test functions for {case} {step} on {db0_path} <<<")
                db0 = test(case, step, db0_path, max_covariate_effect = max_covariate_effect, ode_hold_out_list = ode_hold_out_list,
                           mulcov_values = mulcov_values, reference_db = reference_db, **common_kwds)

            if test_cmds:
                if step == 'no_ode': # The steps [no_ode, yes_ode, students] reuse the previous database 
                    db1_path, max_covariate_effect, ode_hold_out_list, mulcov_values = test_cases(case, 'FitODE_test_commands')

                print ('-'*200)
                print (f">>> Running test commands for {case} {step} on {db1_path} <<<")
                db1 = test_commands(case, step, db1_path, max_covariate_effect = max_covariate_effect, ode_hold_out_list = ode_hold_out_list,
                                    mulcov_values = mulcov_values, reference_db = reference_db, **common_kwds)

            if test_sh:
                if step == 'no_ode': # The steps [no_ode, yes_ode, students] reuse the previous database 
                    db2_path, max_covariate_effect, ode_hold_out_list, mulcov_values = test_cases(case, 'FitODE_test_scripts')

                print ('-'*200)
                print (f">>> Running test shell scripts for {case} {step} on {db2_path} <<<")
                kwd_str = (f"--random-seed {common_kwds['random_seed']} "
                           f"--subset {common_kwds['subset']} "
                           f"--random-subsample {common_kwds['random_subsample']} "
                           f"--ode-hold-out-list {' '.join(ode_hold_out_list)} "
                           f"--max-covariate-effect {max_covariate_effect}")
                if mulcov_values:
                     kwd_str += f' --mulcov-values {" ".join([str(b) for a in mulcov_values for b in a])}'

                tol = dict(rtol = 1e-10, atol=1e-8)

                db2 = FitNoODE(db2_path)
        
                if step == 'no_ode':
                    cmd = f'dmdismod {db2.path} ODE init {kwd_str}'
                    os.system(cmd)
                    assert (np.allclose(db2.fit_var.fit_var_value, reference_db.fit_var.fit_var_value, **tol))

                if step == 'yes_ode':
                    cmd = f'dmdismod {db2.path} ODE fit {kwd_str}'
                    os.system(cmd)
                    assert (np.allclose(db2.fit_var.fit_var_value, reference_db.fit_var.fit_var_value, **tol))

                if step == 'students':
                    cmd = f'dmdismod {db2.path} ODE students {kwd_str}'
                    os.system(cmd)
                    assert (np.allclose(db2.fit_var.fit_var_value, reference_db.fit_var.fit_var_value, **tol))


print (f'Test {ode_option} OK')
