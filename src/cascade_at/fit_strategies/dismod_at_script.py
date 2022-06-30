#!/usr/bin/env python
import sys
import os
import shutil

from cascade_at.dismod.constants import _dismod_cmd_

"""
# An example for debugging this script
# Shell commands for running the diabetes case, and checking the result to the correct answer given in the reference

dmdismod
dmdismod --help

cp /Users/gma/ihme/epi/at_cascade/data/475588/dbs/100/3/dismod.db /tmp/t1_diabetes.db

dmdismod /tmp/t1_diabetes.db init
dmdismod /tmp/t1_diabetes.db set option max_num_iter_fixed 200

dmdismod /tmp/t1_diabetes.db ODE init  --random-seed 1234 --subset True --random-subsample 1000 --save-to-path /tmp/t1_diabetes_no_ode.db --reference /Users/gma/ihme/epi/at_cascade/t1_diabetes/no_ode/no_ode.db

dmdismod /tmp/t1_diabetes.db ODE fit --random-seed 1234 --subset True --random-subsample 1000 --save-to-path /tmp/t1_diabetes_yes_ode.db --reference /Users/gma/ihme/epi/at_cascade/t1_diabetes/yes_ode/yes_ode.db --ode-hold-out-list mtexcess

dmdismod /tmp/t1_diabetes.db ODE students --random-seed 1234 --subset True --random-subsample 1000 --save-to-path /tmp/t1_diabetes_students.db --reference /Users/gma/ihme/epi/at_cascade/t1_diabetes/students/students.db --ode-hold-out-list mtexcess
    """

def main():
    cmd_str = ' '.join([k if  ' ' not in k else f"'{k}'" for k in sys.argv[1:]])

    # Apple Darwin does not forward library_path variables to subprocesses for security reasons
    # so set it explicitly for the subprocess.
    lib_path = 'LD_LIBRARY_PATH=' + os.getenv('DISMOD_LIBRARY_PATH', '').strip(':')
    lib_path = ''

    if len(sys.argv) == 1:
        print (f'usage:    {_dismod_cmd_} [-h | --help]                      # Print detailed help.')
        print (f'usage:    {_dismod_cmd_} database [ODE] command [arguments] # Run dmdismod commands.')
        print ("          Omitting 'ODE' calls the standard dismod_at executable.")
        print ("          Specifying 'ODE' dispatches to the ODE fitting strategy code.")
        # os.system (f"{lib_path} dismod_at {cmd_str}")
        cmd = f"{lib_path} dmdismod {cmd_str}"
        print (f'---- cmd')
        os.system (cmd)
    else:
        if '-h' in sys.argv or '--help' in sys.argv or sys.argv[2].upper() == 'ODE' or sys.argv[2].upper() == 'BB':
            from cascade_at.fit_strategies.dmdismod_extensions import dmdismod
            sys.argv[0] = sys.argv[0].replace('_script', '')
            cmd_str = ' '.join(sys.argv)
            print (f'---- {cmd_str}')
            dmdismod(cmd_str)
        else:
            cmd_str = ' '.join([k if  ' ' not in k else f"'{k}'" for k in sys.argv[1:]])
            print (f'---- {cmd_str}')
            os.system (f"{lib_path} dmdismod {cmd_str}")

if __name__ == '__main__':
    if not sys.argv[0]:
        filename = '/tmp/t1_diabetes.db'
        if 0:
            sys.argv = (f'dmdismod {filename} ODE init  --random-seed 1234 --random-subsample 1000 '
                        '--save-to-path /tmp/t1_diabetes_no_ode.db --reference /Users/gma/ihme/epi/at_cascade/t1_diabetes/no_ode/no_ode.db').split()
        if 1:
            DATA_DIR = '/tmp/cascade/cascade_dir/data/475873'
            I = os.path.join(DATA_DIR, 'dbs/100/3/dismod.db')
            O = os.path.join(DATA_DIR, 'outputs/root_node.db')
            os.makedirs(os.path.dirname(O), exist_ok=True)
            shutil.copy2(I, O)
            sys.argv = f'dmdismod {O} BB init'.split()
            
    main()
