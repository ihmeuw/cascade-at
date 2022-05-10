#! /usr/bin/env python3
# -----------------------------------------------------------------------------
# at_cascade: Cascading Dismod_at Analysis From Parent To Child Regions
#           Copyright (C) 2021-22 University of Washington
#              (Bradley M. Bell bradbell@uw.edu)
#
# This program is distributed under the terms of the
#     GNU Affero General Public License version 3.0 or later
# see http://www.gnu.org/licenses/agpl.txt
# ----------------------------------------------------------------------------

import os
import sys
import statistics
import math
import copy
import time
import multiprocessing
import shutil
import pandas as pd
from sqlalchemy import create_engine

_start_at_global_ = not True
_clean_run_ = not True

sys.path.append('/opt/prefix/dismod_at/lib/python3.9/site-packages')
import dismod_at
#

_gma_dir_ = f'/Users/gma/Projects/IHME/GIT/at_cascade.git'
_mvid_ = 475873

sys.path.append(_gma_dir_)
import at_cascade.ihme
if __name__ == '__main__':
    os.chdir(_gma_dir_)
sys.path.append(_gma_dir_)
sys.path.append('/Users/gma/Projects/IHME/GIT/at_cascade.git')
from at_cascade.ihme.dismod_db_api import DismodDbAPI
# ----------------------------------------------------------------------------
# Begin settings that can be changed without understanding this program
# ----------------------------------------------------------------------------
#
# input files
# Use None for csmr_inp_file if you do not want to include it in fit
data_inp_file   = ""
csmr_inp_file   = None

data_path = f'/Users/gma/ihme/at_cascade.ihme_db/{_mvid_}'
root_node_db = os.path.join(data_path, 'root_node.db')
all_node_db = os.path.join(data_path, 'all_node.db')

result_dir = f'{_gma_dir_}/ihme_db/DisMod_AT/results/{_mvid_}'

root_node_database = f'{result_dir}/root_node.db'
all_node_database = f'{result_dir}/all_node.db'

#
# root_node_name
# name of the node where the cascade will start
# root_node_name      = '1_Global'
if not _start_at_global_:
    root_node_name      = '100_High-income_North_America'
    _parent_location_ = 100
else:
    root_node_name      = '1_Global'
    _parent_location_ = 1
#
# random_seed
# If this seed is zero, the clock is used for the random seed.
# Otherwise this value is used. In either case the actual seed is reported.
random_seed = 1234
#
# perturb_optimization_scale, perturb_optimization_start
# Amount to randomly move, in log space, the optimization scaling point
# starting points.
perturb_optimization_scale = 0.2
perturb_optimization_start = 0.2
#
# shift_prior_std_factor
# Factor that multipliers standard deviation that is passed down the cascade.
shift_prior_std_factor = 2.0
#
# max_number_cpu
# maximum number of processors, if one, run sequentally, otherwise
# run at most max_number_cpu jobs at at time.
max_number_cpu = max(1, multiprocessing.cpu_count() - 1)
#
# max_fit
# Maximum number of data rows per integrand to include in a f
max_fit             = 250
#
# max_abs_effect
# Maximum absolute effect for any covriate multiplier.
max_abs_effect      = 3.0
#
# max_plot
# Maximum number of data points to plot per integrand.
max_plot            = 2000
#
# node_split_name_set
# Name of the nodes where we are splitting from Both to Female, Male
node_split_name_set = { root_node_name }

# mulcov_freeze_list
# Freeze the covariate multipliers at the Global level after the sex split
mulcov_freeze_list = [
    {   'node'      : root_node_name,
        'sex'       : 'Male',
    },
    {   'node'      : root_node_name,
        'sex'       : 'Female',
    },
]
#
# fit_goal_set
# Name of the nodes that we are drilling to (must be below root_node).
# You can change this setting and then run
#   bin/ihme/{_mvid_}.py continue database
# fit_goal_set = { '1_Global' }
# fit_goal_set = { '64_High-income' }
# fit_goal_set = { '81_Germany' }
# fit_goal_set = { '161_Bangladesh' }
# fit_goal_set = { '44758_Tower_Hamlets', '527_California' }
# fit_goal_set = {
#     '527_California', '547_Mississippi', '81_Germany', '84_Ireland'
# }
if not _start_at_global_:
    fit_goal_set = {
        '527_California', '547_Mississippi', '101_Canada',
    }
else:
    fit_goal_set = {
        '8_Taiwan_(Province_of_China)',
        '514_Shanghai',
        '18_Thailand',
        '16_Philippines',
        '22_Fiji',
        '26_Papua_New_Guinea',
        '41_Uzbekistan',
        '38_Mongolia',
        '505_Inner_Mongolia',
        '61_Republic_of_Moldova',
        '44850_New_Zealand_Maori_population',
        '44851_New_Zealand_non-Maori_population',
        '35469_Kagoshima',
        '68_Republic_of_Korea',
        "7_Democratic_Peoples_Republic_of_Korea",
        '349_Greenland',
        '527_California',
        '4644_Baja_California',
        '4645_Baja_California_Sur',
        '547_Mississippi',
        '97_Argentina',
        '99_Uruguay',
        '81_Germany',
        '84_Ireland',
        '433_Northern_Ireland',
        '44758_Tower_Hamlets',
        '123_Peru',
        '121_Bolivia_(Plurinational_State_of)',
        '107_Barbados',
        '116_Saint_Lucia',
        '129_Honduras',
        '4670_Tamaulipas',
        '136_Paraguay',
        '150_Oman',
        '44872_Golestan',
        '161_Bangladesh',
        '171_Democratic_Republic_of_the_Congo',
        '168_Angola',
        '185_Rwanda',
        '179_Ethiopia',
        '194_Lesotho',
        '482_Eastern_Cape',
        '218_Togo',
        '25329_Edo',
    }
# ----------------------------------------------------------------------------
# End settings that can be changed without understanding this program
# ----------------------------------------------------------------------------
#
# random.seed
if __name__ == '__main__' :
    if random_seed == 0 :
        random_seed = int( time.time() )
# ----------------------------------------------------------------------------
def write_table_sql(conn, table_name, df, dtypes=None):
    id_column = f"{table_name}_id"
    if id_column not in df:
        df[id_column] = df.reset_index(drop=True).index
    if dtypes:
        keys = ', '.join([f'{k} {v}' for k,v in dtypes.items() if k != id_column])
        conn.execute(f"DROP TABLE IF EXISTS '{table_name}'")
        conn.execute(f"CREATE TABLE '{table_name}' ({id_column} integer primary key, {keys})")
        cols = [k for k in dtypes if k != id_column]
        df[cols].to_sql(table_name, conn, index_label = id_column, if_exists="append")
    else:
        try:
            conn.execute(f"DELETE FROM '{table_name}'")
        except:
            pass
        df.to_sql(table_name, conn, index_label = id_column, if_exists="append")

def setup_function():
    root_engine = create_engine("sqlite:///"+root_node_database)
    all_engine = create_engine("sqlite:///"+all_node_database)

    def clear_tables():
        db.avgint = pd.DataFrame()
        db.nslist = pd.DataFrame()
        db.nslist_pair = pd.DataFrame()
        rate = db.rate
        rate.loc[rate.rate_name == 'omega', ['parent_smooth_id', 'child_smooth_id', 'child_nslist_id']] = [None]*3
        db.rate = rate

    def switch_to_Brads_naming():
        # Change the covariate from bmi to obesity
        covariate = db.covariate

        covariate_name = [n for n in covariate.columns if 'name' in n and 'c_' in n]
        if covariate_name:
            covariate_name = covariate_name[0]

        covariate.loc[covariate[covariate_name] == 'c_mean_BMI', 'c_covariate_name'] = 'c_mean_BMI'
        covariate.loc[covariate[covariate_name] == 'c_LDI_pc_log', 'c_covariate_name'] = 'c_LDI_pc_log'

        data_rename = dict(covariate[['covariate_name', 'c_covariate_name']].values)
        for k,v in data_rename.items():
            if v.startswith('c_') or v.startswith('s_'):
                v = v[2:]
                data_rename[k] = v

            covariate['covariate_name'] = data_rename.values()
        db.covariate = covariate

        node = db.node
        node['node_name'] = [v.replace(' ', '_') for k,v in node[['node_id', 'node_name']].values]
        node['node_name'] = [v.replace("'", "") for k,v in node[['node_id', 'node_name']].values]
        breakpoint()
        name_update = [f"{k}_{v}" for k,v in node[['c_location_id', 'node_name']].values if not v.startswith(f"{k}_")]
        if name_update: node['node_name'] = name_update
        db.node = node

        option = pd.read_sql('option', root_engine)
        option = option[option.option_value != 'data_extra_columns'].reset_index(drop=True)
        if not 'parent_node_name' in option.option_name.values:
            parent_node_id = int(option.loc[option.option_name == 'parent_node_id', 'option_value'])
            parent_node_name = str(node.loc[node.node_id == parent_node_id, 'node_name'].squeeze())
            option.loc[option.option_name == 'parent_node_id',
                       ['option_name', 'option_value']] = ['parent_node_name', parent_node_name]
            option = option[option.option_value != 'data_extra_columns'].reset_index(drop=True)
            option['option_id'] = option.index
        option.loc[option.option_name == 'parent_node_name', 'option_value'] = root_node_name
        option['option_id'] = option.index
        write_table_sql(root_engine, 'option', option,
                        dtypes = {'option_id': 'integer primary key',
                                  'option_name': 'text', 
                                  'option_value': 'text'})

    def correct_root_node_database():
        # ---------------------------------------------------------------------------
        # Corrections to root_node_database
        #
        def sql_types(dtypes):
            if not isinstance(dtypes, dict):
                dtypes = dict(dtypes)
            for k,v in dtypes.items():
                if 'object' in str(v): dtypes[k] = 'text'
                if 'int' in str(v): dtypes[k] = 'integer'
                if 'float' in str(v): dtypes[k] = 'real'
            return dtypes

        # integrand_table
        # All the covariate multipliers must be in integrand table

        covariate = db.covariate
        map = list(zip(*[(k, n, n[2:]) if n[:2] in ['c_', 's_'] else n for k,n in covariate[['covariate_name', 'c_covariate_name']].values]))

        covariate['covariate_name'] = map[2]
        db.covariate = covariate

        integrand_table = db.integrand
        mulcov_table    = db.mulcov
        mulcov_table['integrand_name'] = [f"mulcov_{name}" for name in mulcov_table.mulcov_id]
        mulcov_table['minimum_meas_cv'] = 0
        mask = mulcov_table.integrand_name.isin(integrand_table.integrand_name)
        integrand_table = pd.concat([integrand_table, mulcov_table.loc[~mask, ['integrand_name', 'minimum_meas_cv']]]).reset_index(drop=True)
        integrand_table['integrand_id'] = integrand_table.index
        db.integrand = integrand_table
        #
        # option table, parent_node_id
        # at_cascade requires one to use parent_node_name (not parent_node_id)
        # (turn on ipopt_trace)
        option = db.option
        node = db.node
        if 'parent_node_name' not in option.option_name.values:
            parent_node_id = int(option.loc[option.option_name == 'parent_node_id', 'option_value'])
            parent_node_name, parent_loc_id = node.loc[node.node_id == parent_node_id, ['node_name', 'c_location_id']].squeeze()
            parent_node_name = (parent_node_name if parent_node_name.startswith(str(parent_loc_id)) else f'{parent_loc_id}_{parent_node_name}')
            parent_node_name = parent_node_name.replace(' ', '_')
            mask = option.option_name == 'parent_node_id'
            option.loc[mask, ['option_name', 'option_value']] = ['parent_node_name', parent_node_name]
        brads_options = {# 'data_extra_columns'          :'c_seq c_nid',
                         'meas_noise_effect'           :'add_std_scale_none',
                         'quasi_fixed'                 :'false' ,
                         'tolerance_fixed'             :'1e-8',
                         'max_num_iter_fixed'          :'40',
                         'print_level_fixed'           :'5',
                         'accept_after_max_steps_fixed':'10'}

        for k,v in brads_options.items():
            if k in option.option_name.values:
                option.loc[option.option_name == k, 'option_value'] = v
            else:
                option = pd.concat([option, pd.DataFrame([{'option_name': k, 'option_value': v}])])
        option = option.reset_index(drop=True)
        option.option_id = option.index
        db.option = option

        node['node_name'] = [n.node_name if n.node_name.startswith(str(n.c_location_id))
                                  else f'{n.c_location_id}_{n.node_name}' for i,n in node.iterrows()]
        db.node = node
        #
        # rate table
        # all omega rates must be null
        rate    = db.rate
        omega_rate_id = rate.loc[rate.rate_name == 'omega', 'rate_id']
        rate.loc[omega_rate_id, ['parent_smooth_id', 'child_smooth_id', 'child_nslist_id']] = None, None, None
        db.rate = rate

        #
        # nslist and nslist_pair tables
        db.nslist = pd.DataFrame()
        db.nslist_pair = pd.DataFrame()
        print (f'*** Modified {db.filename}')
        #

    def write_all_option():
        all_option = {
        'absolute_covariates'          : 'one',
        'split_covariate_name'         : 'sex',
        'root_split_reference_name'    : 'Both',
        'result_dir'                   : result_dir,
        'root_node_name'               : root_node_name,
        'max_abs_effect'               : max_abs_effect,
        'max_fit'                      : str(max_fit),
        'max_number_cpu'               : max_number_cpu,
        'shift_prior_std_factor'       : shift_prior_std_factor,
        'perturb_optimization_scale'   : perturb_optimization_scale,
        'perturb_optimization_start'   : perturb_optimization_start,
        }
        #
        # all_option_table
        all_option_table = pd.DataFrame(
            [{ 'all_option_id' : i, 'option_name' : k , 'option_value' : v}
             for i,(k,v) in enumerate(all_option.items())])
        write_table_sql(all_engine, 'all_option', all_option_table,
                        dtypes = {'all_option_id': 'integer primary key',
                                  'option_name': 'text',
                                  'option_value': 'text'})

    def write_split_reference_table():
        sex_info_dict      = at_cascade.ihme.sex_info_dict
        split_reference = pd.DataFrame([{'split_reference_id': v['split_reference_id'],
                                         'split_reference_name': k,
                                         'split_reference_value': v['covariate_value']}
                                        for k,v in sex_info_dict.items()])

        write_table_sql(all_engine, 'split_reference', split_reference, 
                        dtypes = {'split_reference_id': 'INTEGER PRIMARY KEY',
                                  'split_reference_name': 'INTEGER',
                                  'split_reference_value': 'REAL'})

    def write_split_goal_table():
        sex_info_dict      = at_cascade.ihme.sex_info_dict
        split_goal = pd.DataFrame([{'split_goal_id': v['split_reference_id'],
                                         'split_goal_name': k,
                                         'split_goal_value': v['covariate_value']}
                                        for k,v in sex_info_dict.items()])

        write_table_sql(all_engine, 'split_goal', split_goal, 
                        dtypes = {'split_goal_id': 'INTEGER PRIMARY KEY',
                                  'split_goal_name': 'INTEGER',
                                  'split_goal_value': 'REAL'})

    def write_mulcov_freeze_table(mulcov_freeze_list = None):
        assert type(result_dir) == str
        assert type(root_node_database) == str
        assert type(mulcov_freeze_list) == list

        # root_table
        root_table = {tbl_name: pd.read_sql(tbl_name, root_engine)
                      for tbl_name in [ 'covariate', 'mulcov', 'node', 'rate' ]}
        node = root_table['node']
        mulcov = root_table['mulcov']

        sex_info_dict      = at_cascade.ihme.sex_info_dict

        # mulcov_freeze_table_file
        mulcov_freeze_table = pd.DataFrame([{'fit_node_id': int(node.loc[node.node_name == row_freeze['node'], 'node_id']),
                                             'split_reference_id': int(sex_info_dict[row_freeze['sex']]['split_reference_id']),
                                             'mulcov_id': mulcov_id}
                                            for mulcov_id in mulcov.mulcov_id.values
                                            for row_freeze in mulcov_freeze_list])
        write_table_sql(all_engine, 'mulcov_freeze', mulcov_freeze_table,
                        dtypes = {'mulcov_freeze_id':  'INTEGER',
                                  'fit_node_id': 'INTEGER',
                                  'split_reference_id': 'INTEGER',
                                  'mulcov_id': 'INTEGER'})


    def write_node_split_table(results_dir, node_split_name_set, root_node_database):
        #
        # root_table
        new        = False
        connection = dismod_at.create_connection(root_node_database, new)
        node_table = dismod_at.get_table_dict(connection, 'node')
        connection.close()
        #
        # node_split_table
        node_split_table = list()
        for node_name in node_split_name_set :
            node_id = at_cascade.table_name2id(node_table, 'node', node_name)
            #
            # row_out
            row_out = {
                'node_name'    : node_name ,
                'node_id'      : node_id,
            }
            node_split_table.append( row_out )
        #
        #
        # node_split_table_file
        fieldnames = [ 'node_name', 'node_id' ]
        write_table_sql(all_engine, 'node_split', pd.DataFrame(node_split_table),
                        dtypes = {'node_split_id':  'INTEGER',
                                  'node_id': 'INTEGER'})




    correct_root_node_database()
    switch_to_Brads_naming()
    clear_tables()

    mulcov = db.mulcov
    integrand = db.integrand
    for j in range( len(mulcov) ) :
        if f'mulcov_{j}' not in integrand.integrand_name.values:
            integrand = integrand.append( { 'integrand_name' : f'mulcov_{j}', 'minimum_meas_cv': 0.1  }, ignore_index=True )
    integrand = integrand.reset_index(drop=True)
    integrand['integrand_id'] = integrand.index
    db.integrand = integrand

    #
    # write_all_option_table
    write_all_option()

    #
    # write_split_reference_table
    write_split_reference_table()

    #
    # write_split_goal_table
    write_split_goal_table()

    #
    # write_mulcov_freeze_table
    write_mulcov_freeze_table(mulcov_freeze_list = mulcov_freeze_list)
    #
    # write_node_split_table
    write_node_split_table(result_dir, node_split_name_set, root_node_database)
    #
    # write_all_node_database

    print ('FIXME -- why do I have to do this?')
    data = db.data
    for col in data.columns:
        if col.startswith('x_'):
            data[col] = data[col].astype('float')
    db.data = data

# ----------------------------------------------------------------------------
# Without __name__ == '__main__', the mac will try to execute main on each processor.

def run(gbd_round_id = None, fit_goal_set = fit_goal_set, cov_dict = {}):
    if 'drill' in sys.argv:
        root_path = '/Users/gma/Projects/IHME/GIT/at_cascade.git/ihme_db/DisMod_AT/results/475873/{root_node_name}'
        if os.path.exists(root_path): shutil.rmtree(root_path)
        db.avgint = pd.DataFrame()

    at_cascade.ihme.main(
    result_dir              = result_dir,
    root_node_name          = root_node_name,
    fit_goal_set            = fit_goal_set,
    setup_function          = lambda: None,
    max_plot                = max_plot,
    covariate_csv_file_dict = {},
    scale_covariate_dict    = {},
    root_node_database      = root_node_database,
    no_ode_fit              = False,
    fit_type_list = [ 'both', 'fixed' ],
    random_seed = random_seed,
    use_csv_files = False,
    gbd_round_id = gbd_round_id,
)

if __name__ == '__main__':
    import db_queries
    if _clean_run_:
        shutil.rmtree(result_dir)
    try:
        display_cmd = f'display {root_node_name}/dismod.db'
        _cmds_ = ['setup', 'drill',
                  'predict', 'summary', display_cmd]
        _cmds_ = [f'shared {result_dir}/all_node.db',
                  'setup', 'drill',
                  'predict', 'summary', display_cmd]
        # _cmds_ = ['predict', 'summary', f'display {root_node_name}/dismod.db']
        # _cmds_ = [f'shared {result_dir}/all_node.db', 'drill']
        _sex_id_ = 3
        json_file = f'/Users/gma/ihme/epi/at_cascade/data/{_mvid_}/inputs/settings-{root_node_name}.json'
        _json_ = f'--json-file {json_file}'
        import json
        with open(json_file, 'r') as stream:
            settings = json.load(stream)
        gbd_round_id = settings['gbd_round_id']
        
        if len(sys.argv) <= 1:
            if not os.path.exists(root_node_database):
                if 1:
                    print ('Running configure_inputs')
                    os.system(f'configure_inputs --model-version-id {_mvid_} --make --configure {_json_}')
                if 1:
                    # Build the covariate values for all locations?
                    print ('Running dismod_db')
                    os.system(f'dismod_db --model-version-id {_mvid_} --parent-location-id {_parent_location_} --sex-id {_sex_id_} --fill')

                    os.makedirs(result_dir, exist_ok=True)
                    shutil.copy2(f'/Users/gma/ihme/epi/at_cascade/data/{_mvid_}/dbs/100/3/dismod.db',
                                 root_node_database)
        db = DismodDbAPI(root_node_database)
        db.node['node_name'] = [n.replace(' ', '_') for n in db.node.node_name.values]

        if len(sys.argv) <= 1:
            if 'setup' in _cmds_:
                os.system (f'rm -rf {result_dir}/{root_node_name}')
                if not os.path.exists(all_node_database):
                    # The all_node_database command
                    print ('Running all_node_database')
                    if 0:
                        os.system((f'python /Users/gma/Projects/IHME/GIT/cascade-at/src/cascade_at/inputs/all_node_database.py'
                                   f' --root-node-path {result_dir}/root_node.db -m {_mvid_} -c 587 -a 12 {_json_}'))
                    else:
                        # An all_node_database call
                        from cascade_at.inputs.all_node_database import main as all_node_main
                        all_node_obj = all_node_main(root_node_path = os.path.join(result_dir, 'root_node.db'),
                                                     mvid = _mvid_,
                                                     cause_id = 587,
                                                     age_group_set_id = 12,
                                                     json_file = _json_.split()[-1])
                        import sqlite3
                        conn = sqlite3.connect(all_node_obj.all_node_db)
                        for k in all_node_obj.covariate.c_covariate_name:
                            if not k.startswith('c_'): continue
                            v = getattr(all_node_obj, k, None)
                            all_node_obj.write_table_sql(conn, k, v.dtypes)
                        print ('Running setup_function')
                        setup_function()
        
            for cmd in _cmds_:
                tmp = f'{_mvid_}.py {cmd}'
                print (tmp)
                sys.argv = tmp.split()
                run(gbd_round_id = gbd_round_id, fit_goal_set = fit_goal_set)
        else:
            run(gbd_round_id = gbd_round_id, fit_goal_set = fit_goal_set)
    except:
        raise
    finally:
        sys.argv = [""]
        
print('random_seed = ', random_seed)
print(f'{_mvid_}.py: OK')
