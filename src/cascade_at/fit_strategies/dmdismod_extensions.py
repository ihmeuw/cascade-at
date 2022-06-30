#!/usr/bin/env python

"""
Example shell command sequence:

cp /Users/gma/ihme/epi/at_cascade/data/475588/dbs/100/3/dismod.db /tmp/t1_diabetes.db

./dmdismod_extensions.py /tmp/t1_diabetes.db ODE init  --random-seed 1234 --random-subsample 1000 --save-to-path /tmp/t1_diabetes_no_ode.db --reference /Users/gma/ihme/epi/at_cascade/t1_diabetes/no_ode/no_ode.db

./dmdismod_extensions.py /tmp/t1_diabetes.db ODE fit --ode-hold-out-list mtexcess  --random-seed 1234 --random-subsample 1000 --save-to-path /tmp/t1_diabetes_yes_ode.db --reference /Users/gma/ihme/epi/at_cascade/t1_diabetes/yes_ode/yes_ode.db

./dmdismod_extensions.py /tmp/t1_diabetes.db ODE students --ode-hold-out-list mtexcess  --random-seed 1234 --random-subsample 1000 --save-to-path /tmp/t1_diabetes_students.db --reference /Users/gma/ihme/epi/at_cascade/t1_diabetes/students/students.db

"""

import sys
import os
import pandas as pd
import numpy as np
import shutil

from collections import OrderedDict
from cascade_at.fit_strategies.init_no_ode import init_ode_command, fit_ode_command, fit_students_command
from cascade_at.dismod.api.dismod_io import DismodIO
from cascade_at.core.log import logging, get_loggers, LEVELS


LOG = get_loggers(__name__)
logging.basicConfig(level=LEVELS['info'])

from cascade_at.dismod.constants import _dismod_cmd_
def dmdismod(cmd):
    """
    Example calling sequence:
    os.system('cp /Users/gma/ihme/epi/at_cascade/data/475588/dbs/100/3/dismod.db /tmp/t1_diabetes.db')
    dmdismod(f'{_dismod_cmd_} /tmp/t1_diabetes.db ODE init')
    dmdismod(f'{_dismod_cmd_} /tmp/t1_diabetes.db ODE fit --ode-hold-out-list mtexcess')
    dmdismod(f'{_dismod_cmd_} /tmp/t1_diabetes.db ODE students --ode-hold-out-list mtexcess')
    """

    help=("An extended dmdismod command, to handle Brad's strategy of:\n"
          "  1) fit the non-ODE integrands to initialize an ODE fit,\n"
          "  2) fit the non-ODE and ODE integrands\n"
          "  3) fit to log-student data densities.")

    def parse_args(args):
        import argparse
        from distutils.util import strtobool as str2bool
        parser = argparse.ArgumentParser()

        parser.add_argument('path', type=str, help = 'Path to the Dismod_AT sqlite database')
        parser.add_argument('dispatch', type=str,
                            help = ("If dispatch == 'ODE', use ODE fitting strategy."
                                    "If missing, use standard dismod_at commands."))
        parser.add_argument('option', type=str,
                            help = "For the ODE fitting strategy, one of ('init', 'fit' or 'students').")
        parser.add_argument("-m", "--max-covariate-effect", nargs='?', type=float, default = 2,
                            help = ("Maximum absolute covariate effect = multiplier * (covariate - referece). "
                                    "Note that exp(effect) multiplies a model value to get the model value for "
                                    "this covariate value. (Noise covariate multipliers are not included.)"))
        parser.add_argument("-c", '--mulcov-values', nargs='+', type=str, default = None,
                            help = "Constrain covariate multipliers to the specified value")
        parser.add_argument("-o", "--ode-hold-out-list", nargs='?', type=str, default = None, const = None,
                            help = "Integrands to hold out during the ODE fit") 
        parser.add_argument("-s", "--random-seed", nargs='?', type=int, default = None,
                            help = "Random seed for the random_subsampling") 
        parser.add_argument("-d", "--random-subsample", nargs='?', type=int, default = 1000, const = None,
                            help = "Number of random subsamples to fit.")
        parser.add_argument("-p", "--save-to-path", nargs='?', type=str, default = None, const = None,
                            help = "Path to directory where to store the results") 
        parser.add_argument("-t", "--reference_db", nargs='?', type=str, default = "", const = "",
                            help = "Path to the reference databases. Fit results are compared to these databases for testing purposes.")

        get_help = len(args) > 1 and any(a.startswith('-h') for a in args[1:])
        args = parser.parse_args(args[1:])
        args.cmd = sys.argv[0]
        if args.mulcov_values is None:
            args.mulcov_values = []
        else:
            args.mulcov_values = [[a,b,float(c)] for a,b,c in np.asarray(args.mulcov_values).reshape(-1, 3)]
        return args

    def get_random_seed(path, random_seed):
        db = DismodIO(path)
        option = db.option
        random_seed = option.loc[option.option_name == 'random_seed', 'option_value']
        if not random_seed.empty:
            random_seed = int(random_seed)
            LOG.info(f"Setting the subsampling random_seed to the database option table value = {random_seed}")
        else:
            random_seed = None
            LOG.info(f"The subsampling random_seed not set.")


    args = cmd.split()
    p_args = parse_args(cmd.split())
    print ('-'*10)
    LOG.info(cmd)
    print ('-'*10)

    random_seed = None
    if p_args.random_seed:
        random_seed = p_args.random_seed
        LOG.info(f"Setting the subsampling random_seed to the dmdismod argument value = {random_seed}")
    
    if p_args.dispatch != 'BB':
        if p_args.option == "init":
            if random_seed:
                random_seed = get_random_seed(p_args.path, random_seed) 
            db = init_ode_command([_dismod_cmd_] + args[1:], 
                                  max_covariate_effect = p_args.max_covariate_effect,
                                  mulcov_values = p_args.mulcov_values,
                                  ode_hold_out_list = p_args.ode_hold_out_list,
                                  random_seed = random_seed,
                                  random_subsample = p_args.random_subsample,
                                  save_to_path = p_args.save_to_path,
                                  reference_db = p_args.reference_db)
            set_random_seed(db, random_seed)
        elif p_args.option == "fit":
            db = fit_ode_command([_dismod_cmd_] + args[1:],
                                 ode_hold_out_list = p_args.ode_hold_out_list,
                                 random_seed = random_seed,
                                 random_subsample = p_args.random_subsample,
                                 save_to_path = p_args.save_to_path,
                                 reference_db = p_args.reference_db)
        elif p_args.option == "students":
            fit_students_command([_dismod_cmd_] + args[1:],
                                 ode_hold_out_list = p_args.ode_hold_out_list,
                                 random_seed = random_seed,
                                 random_subsample = p_args.random_subsample,
                                 save_to_path = p_args.save_to_path,
                                 reference_db = p_args.reference_db)
    else:

        DATA_DIR = p_args.path
        if p_args.option in ('all_node', 'shared', 'setup', 'drill', 'predict', 'summary'):
            ALL_NODE_CMD='python /Users/gma/Projects/IHME/GIT/cascade-at/src/cascade_at/inputs/all_node_database.py'
            CASCADE_CMD='python /Users/gma/Projects/IHME/GIT/cascade-at/src/cascade_at/executor/dismod_cascade_brad.py'
            if p_args.option == 'all_node':
                I = os.path.join(DATA_DIR, 'dbs/100/3/dismod.db')
                O = os.path.join(DATA_DIR, 'outputs/root_node.db')
                os.makedirs(os.path.dirname(O), exist_ok=True)
                print (f'Copy {I} to {O}')
                shutil.copy2(I, O)
                json_file = f'{DATA_DIR}/inputs/settings.json'
                cmd = (f'{ALL_NODE_CMD} --root-node-path {DATA_DIR}/outputs/root_node.db --model-version-id 475873 '
                       f'--inputs-file {DATA_DIR}/inputs/inputs.p --json-file {json_file} --cause-id 587 --age-group-set-id 12')
            elif p_args.option == 'shared':
                cmd = f"{CASCADE_CMD} {p_args.path}/outputs {p_args.option} {p_args.path}/outputs/root_node.db"
            elif p_args.option == 'setup':
                cmd = f"{CASCADE_CMD} {p_args.path}/outputs {p_args.option}"
                if random_seed:
                    cmd += ' --random-seed {random_seed}'
            else:
                cmd = f"{CASCADE_CMD} {p_args.path}/outputs {p_args.option}"
            print (cmd)
            os.system(cmd)

if __name__ == '__main__':

    _random_seed_ = 1234
    __check__ = True

    def test():
        def test_args(cmd, disease):
            dispatch = {'init': 'no_ode',
                        'fit': 'yes_ode',
                        'students': 'students'} 
            path = cmd.split()[1]
            type = dispatch[cmd.split()[3]]
            save_path = path.replace('.db', f'_{type}.db')
            arg_str = (f" --random-seed {_random_seed_} --random-subsample 1000"
                       f" --save-to-path {save_path}"
                       f" --reference /Users/gma/ihme/epi/at_cascade/{disease}/{type}/{type}.db")
            return arg_str

        paths = dict( crohns = '/Users/gma/ihme/epi/at_cascade/data/475533/dbs/1/2/dismod.db',
                      dialysis = '/Users/gma/ihme/epi/at_cascade/data/475527/dbs/96/2/dismod.db', # S Latin America
                      # dialysis = '/Users/gma/ihme/epi/at_cascade/data/475527/dbs/1/2/dismod.db',  # Global
                      kidney = '/Users/gma/ihme/epi/at_cascade/data/475718/dbs/70/1/dismod.db',
                      osteo_hip =  '/Users/gma/ihme/epi/at_cascade/data/475526/dbs/1/2/dismod.db',
                      # osteo_hip_world = '/Users/gma/ihme/epi/at_cascade/data/475745/dbs/1/2/dismod.db',
                      osteo_knee = '/Users/gma/ihme/epi/at_cascade/data/475746/dbs/64/2/dismod.db',
                      # t1_diabetes =  '/Users/gma/ihme/epi/at_cascade/data/475882/dbs/1/2/dismod.db', # world
                      # t1_diabetes =  '/Users/gma/ihme/epi/at_cascade/data/475882/dbs/100/2/dismod.db', # HI N America female
                      t1_diabetes = '/Users/gma/ihme/epi/at_cascade/data/475588/dbs/100/3/dismod.db', # HI N America both
                      )

        cmds = OrderedDict(t1_diabetes = [f'{_dismod_cmd_} /tmp/t1_diabetes.db ODE init',
                                          f'{_dismod_cmd_} /tmp/t1_diabetes.db ODE fit --ode-hold-out-list mtexcess',
                                          f'{_dismod_cmd_} /tmp/t1_diabetes.db ODE students --ode-hold-out-list mtexcess'],
                           dialysis = [f'{_dismod_cmd_} /tmp/dialysis.db ODE init --max-covariate-effect 4',
                                       f'{_dismod_cmd_} /tmp/dialysis.db ODE fit --max-covariate-effect 4',
                                       f'{_dismod_cmd_} /tmp/dialysis.db ODE students --max-covariate-effect 4'],
                           kidney = [f'{_dismod_cmd_} /tmp/kidney.db ODE init',
                                     f'{_dismod_cmd_} /tmp/kidney.db ODE fit',
                                     f'{_dismod_cmd_} /tmp/kidney.db ODE students'],
                           osteo_hip = [f'{_dismod_cmd_} /tmp/osteo_hip.db ODE init',
                                        f'{_dismod_cmd_} /tmp/osteo_hip.db ODE fit',
                                        f'{_dismod_cmd_} /tmp/osteo_hip.db ODE students'],
                           osteo_knee = [f'{_dismod_cmd_} /tmp/osteo_knee.db ODE init',
                                         f'{_dismod_cmd_} /tmp/osteo_knee.db ODE fit',
                                         f'{_dismod_cmd_} /tmp/osteo_knee.db ODE students'],
                           crohns = [f'{_dismod_cmd_} /tmp/crohns.db ODE init --mulcov-values x_0 iota 3.8661',
                                     f'{_dismod_cmd_} /tmp/crohns.db ODE fit --mulcov-values x_0 iota 3.8661',
                                     f'{_dismod_cmd_} /tmp/crohns.db ODE students --mulcov-values x_0 iota 3.8661'],
                    )

        diseases = ['osteo_hip','osteo_knee', 'dialysis', 'kidney', 't1_diabetes', 'crohns']
        for disease in diseases:
            _cmds = cmds[disease]
            path_in = paths[disease]
            path_fit = f'/tmp/{disease}.db'
            print (f'Copying {path_in} to {path_fit} for testing.')
            shutil.copy2(path_in, path_fit)
            for cmd in _cmds:
                dispatch = {'init': 'no_ode',
                            'fit': 'yes_ode',
                            'students': 'students'} 
                if __check__:
                    cmd += test_args(cmd, disease)

                print (cmd)

                dmdismod(cmd)

if __name__ == '__main__':
    if sys.argv[0]:
        cmd = ' '.join(sys.argv)
        print (cmd)
        dmdismod(cmd)
    else:
        test()

