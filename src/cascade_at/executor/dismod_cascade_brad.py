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
import db_queries

_start_at_global_ = not True
_clean_run_ = True
_include_all_leaves_ = not True

if 1:
    #
    # root_node_name
    # name of the node where the cascade will start
    # root_node_name      = '1_Global'
    if not _start_at_global_:
        root_node_id = 100
        root_node_name = f'{root_node_id}_High-income_North_America'
    else:
        root_node_id = 1
        root_node_name = f'{root_node_id}_Global'

    sys.path.append('/opt/prefix/dismod_at/lib/python3.9/site-packages')
    import dismod_at
    #

    _mvid_ = 475873

    _gma_dir_ = f'/Users/gma/Projects/IHME/GIT/at_cascade.git'
    sys.path.append(_gma_dir_)
    import at_cascade.ihme
    if __name__ == '__main__':
        os.chdir(_gma_dir_)

    from at_cascade.ihme.dismod_db_api import DismodDbAPI
    from at_cascade.ihme.dag import DAG
    # ----------------------------------------------------------------------------
    # Begin settings that can be changed without understanding this program
    # ----------------------------------------------------------------------------

    # data locations
    from cascade_at.context.model_context import Context
    context = Context( model_version_id=_mvid_, root_directory = None )
    result_dir = str(context.outputs_dir)
    print ('result_dir', result_dir)
    root_node_database = os.path.join(result_dir, 'root_node.db')
    all_node_database = os.path.join(result_dir, 'all_node.db')
    if __debug__:
        db = DismodDbAPI(root_node_database)
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
    # ----------------------------------------------------------------------------
    # End settings that can be changed without understanding this program
    # ----------------------------------------------------------------------------

if __debug__:
    def fit_goal_subset(root_node_database, root_node_name, reduced_subset=True):
        
        db = DismodDbAPI(root_node_database)

        if not reduced_subset:
            node = db.node
            dag = DAG(node)
            root_location_id = int(root_node_name.split('_')[0])
            root_node_id = int(node[node.c_location_id == root_location_id].node_id)
            leaf_node_ids = dag.leaves(root_node_id)
            fit_goal_set = set(node[node.node_id.isin(leaf_node_ids)].node_name)
            return fit_goal_set

        # fit_goal_set
        # Name of the nodes that we are drilling to (must be below root_node).
        # You can change this setting and then run
        #   bin/ihme/{_mvid_}.py continue database

        fit_goal_set = { '1_Global' }
        fit_goal_set = { '64_High-income' }
        fit_goal_set = { '81_Germany' }
        fit_goal_set = { '161_Bangladesh' }
        fit_goal_set = { '44758_Tower_Hamlets', '527_California' }
        fit_goal_set = {
            '527_California', '547_Mississippi', '81_Germany', '84_Ireland'
        }
        if '100_High-income_North_America' in root_node_name :
            fit_goal_set = {
                '527_California', '547_Mississippi', '101_Canada',
            }
        if '1_global' in root_node_name:
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
        return fit_goal_set

# ----------------------------------------------------------------------------
def sql_types(dtypes):
    if not isinstance(dtypes, dict):
        dtypes = dict(dtypes)
    for k,v in dtypes.items():
        if 'object' in str(v): dtypes[k] = 'text'
        if 'int' in str(v): dtypes[k] = 'integer'
        if 'float' in str(v): dtypes[k] = 'real'
    return dtypes

def write_table_sql(conn, table_name, df, dtypes=None):
    id_column = f"{table_name}_id"
    if id_column not in df:
        df[id_column] = df.reset_index(drop=True).index
    if not dtypes:
        types = df.dtypes
        dtypes = sql_types(types)
    keys = ', '.join([f'{k} {v}' for k,v in dtypes.items() if k != id_column])
    cols = [k for k in dtypes.keys() if k != id_column]
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.execute(f"CREATE TABLE {table_name} ({id_column} integer primary key, {keys})")
    df[cols].to_sql(table_name, conn, index_label = id_column, if_exists="append")

def setup_function(root_node_database = None, all_node_database = None):
    print ('Running setup_function')

    db = DismodDbAPI(root_node_database)

    root_engine = create_engine("sqlite:///"+root_node_database)
    all_engine = create_engine("sqlite:///"+all_node_database)

    def clear_tables():
        db.avgint = pd.DataFrame()
        db.nslist = pd.DataFrame()
        db.nslist_pair = pd.DataFrame()

    def switch_to_Brads_naming(db, root_node_name):
        # Change the covariate from bmi to obesity
        covariate = db.covariate

        covariate_name = [n for n in covariate.columns if 'name' in n and 'c_' in n]
        if covariate_name:
            covariate_name = covariate_name[0]

        data_rename = dict(covariate[['covariate_name', 'c_covariate_name']].values)
        for k,v in data_rename.items():
            if v.startswith('c_') or v.startswith('s_'):
                v = v[2:]
                data_rename[k] = v

            covariate['covariate_name'] = data_rename.values()
        db.covariate = covariate

        if 1:
            # This renaming is done by all_node_database.py
            # Brad's node naming scheme
            node = db.node
            def node_name_change(row):
                name = row.node_name.replace(' ', '_').replace("'", "")
                if not name.startswith(f'{row.c_location_id}_'):
                    name = f'{row.c_location_id}_{row.node_name}'
                return name
            node['node_name'] = node.apply(node_name_change, axis=1)
            db.node = node

        option = db.option
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
        db.option = option

    def correct_root_node_database(db):
        # ---------------------------------------------------------------------------
        # Corrections to root_node_database
        #
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
            parent_node_name = parent_node_name.replace(' ', '_').replace("'", "")
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

    def write_all_option(root_node_name = None):
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

    correct_root_node_database(db)
    switch_to_Brads_naming(db, root_node_name)
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
    write_all_option(root_node_name = root_node_name)

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
    #
    write_node_split_table(result_dir, node_split_name_set, root_node_database)

    #
    # write_all_node_database
    print ('FIXME -- why do I have to do this?')
    data = db.data
    for col in data.columns:
        if col.startswith('x_'):
            data[col] = data[col].astype('float')
    db.data = data

    # node = db.node
    # node['node_name'] = [n.replace(' ', '_') for n in db.node.node_name.values]
    # db.node = node


# ----------------------------------------------------------------------------
# Without __name__ == '__main__', the mac will try to execute main on each processor.

if __name__ == '__main__':

    #
    # root_node_name
    # name of the node where the cascade will start
    # root_node_name      = '1_Global'
    if not _start_at_global_:
        root_node_name      = '100_High-income_North_America'
    else:
        root_node_name      = '1_Global'

    # mulcov_freeze_list
    # Freeze the covariate multipliers at the Global level after the sex split
    mulcov_freeze_list = [ { 'node' : root_node_name, 'sex' : 'Male'},
                           { 'node' : root_node_name, 'sex' : 'Female'} ]
    # node_split_name_set
    # Name of the nodes where we are splitting from Both to Female, Male
    node_split_name_set = { root_node_name }


    json_file = f'/Users/gma/ihme/epi/at_cascade/data/{_mvid_}/inputs/settings-{root_node_name}.json'
    _json_ = f'--json-file {json_file}'
    import json
    with open(json_file, 'r') as stream:
        settings = json.load(stream)
    gbd_round_id = settings['gbd_round_id']
    parent_location_id = settings['model']['drill_location_start']

    if random_seed == 0 :
        random_seed = int( time.time() )

    kwds = dict(result_dir              = result_dir,
                root_node_name          = root_node_name,
                setup_function          = lambda: None,
                max_plot                = max_plot,
                covariate_csv_file_dict = {},
                scale_covariate_dict    = {},
                root_node_database      = root_node_database,
                all_node_database       = all_node_database,
                no_ode_fit              = False,
                fit_type_list           = [ 'both', 'fixed' ],
                random_seed             = random_seed,
                use_csv_files           = False,
                gbd_round_id            = gbd_round_id)


    if _clean_run_:
        shutil.rmtree(result_dir, ignore_errors=True)
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
        
        run_setup = False

        if len(sys.argv) <= 1:
            if not os.path.exists(root_node_database):
                run_setup = True
                print ('FIXME -- change configure_inputs and/or dismod_db so I can control where the files are written.')
                if 1:
                    cmd = f'configure_inputs --model-version-id {_mvid_} --make --configure {_json_}'
                    print (f'INFO: {cmd}')
                    os.system(cmd)
                if 1:
                    # Build the covariate values for all locations?
                    print ('Running dismod_db')
                    os.system(f'dismod_db --model-version-id {_mvid_} --parent-location-id {parent_location_id} --sex-id {_sex_id_} --fill')

                    configure_inputs_path = f'/Users/gma/ihme/epi/at_cascade/data/475873/dbs/{root_node_id}/{_sex_id_}/dismod.db' 
                    print (f'Copying {configure_inputs_path} to {root_node_database}')
                    shutil.copy2(configure_inputs_path, root_node_database)
        db = DismodDbAPI(root_node_database)
            
        if len(sys.argv) <= 1:
            if not os.path.exists(all_node_database):
                # The all_node_database command
                run_setup = True
                print ('Running all_node_database')
                if 0:
                    os.system((f'python /Users/gma/Projects/IHME/GIT/cascade-at/src/cascade_at/inputs/all_node_database.py'
                               f' --root-node-path {result_dir}/root_node.db -m {_mvid_} -c 587 -a 12 {_json_}'))
                else:
                    # An all_node_database call
                    from cascade_at.inputs.all_node_database import main as all_node_main
                    json_file = _json_.split()[-1]
                    inputs_file = os.path.join(os.path.dirname(json_file), 'inputs.p')
                    if not os.path.isfile(inputs_file):
                        inputs_file = None
                    kwds2 = dict(root_node_path = os.path.join(result_dir, 'root_node.db'),
                                 mvid = _mvid_,
                                 cause_id = 587,
                                 age_group_set_id = 12,
                                 json_file = json_file,
                                 inputs_file = inputs_file)
                    all_node_obj = all_node_main(**kwds2)
                    import sqlite3
                    conn = sqlite3.connect(all_node_obj.all_node_db)
                    for k in all_node_obj.covariate.c_covariate_name:
                        if not k.startswith('c_'): continue
                        v = getattr(all_node_obj, k, None)
                        write_table_sql(conn, k, v)
        
            if run_setup:
                setup_function(root_node_database = root_node_database, all_node_database = all_node_database)
            fit_goal_set = fit_goal_subset(root_node_database, root_node_name, reduced_subset = not _include_all_leaves_)

            for cmd in _cmds_:
                if cmd == 'setup':
                    os.system (f'rm -rf {result_dir}/{root_node_name}')
                tmp = f'{_mvid_}.py {cmd}'
                print (tmp)
                sys.argv = tmp.split()
                at_cascade.ihme.main(fit_goal_set = fit_goal_set, **kwds)
        else:
            at_cascade.ihme.main(fit_goal_set = fit_goal_set, **kwds)
    except:
        raise
    finally:
        sys.argv = [""]
        
print('random_seed = ', random_seed)
print(f'{_mvid_}.py: OK')
