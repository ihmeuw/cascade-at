# This is Brad's example_db function used in his testing, with slight modifications for the cascade test suite.
import pandas as pd
import numpy as np
import copy
import os

def example_db (file_name,
                test_config = {'node_effects': False,
                               'group_effects': False,
                               'sex_effect': False,
                               'use_group_mulcov': False,
                               'include_group_data': False,
                               'zero_sum_mulcov': False},
                truth = {},
                prior = dict(subgroup_effects = None,
                             parent_density = 'uniform',
                             parent_std = None,
                             child_density = 'uniform',
                             child_std = None,
                             subgroup_density = 'uniform',
                             subgroup_std = 1),
                node_effects = None,
                subgroup_effects = None,

                tol_fixed = 1e-10,
                tol_random = 1e-10,

                ):

    if os.path.exists(file_name):
        os.remove(file_name)
    
    # Note that the a, t values are not used for this example

    def fun_iota_parent(a, t) :
        return ('prior_iota_parent', None, None)
    if test_config['node_effects']:
        def fun_iota_child(a, t) :
            return ('prior_iota_child', None, None)
    if test_config['group_effects']:
        def fun_iota_group(a, t) :
            return ('prior_iota_group', None, None)
        def fun_iota_subgroup(a, t) :
            return ('prior_iota_subgroup', None, None)
    if test_config['sex_effect']:
        def fun_iota_sex(a, t) :
            return ('prior_iota_sex', None, None)
    # TODO: Delete dependency with dismod_at
    # ----------------------------------------------------------------------
    # age table
    age_list    = [    0.0, 100.0 ]
    #
    # time table
    time_list   = [ 1990.0, 2020.0 ]
    #
    # integrand table
    integrand_table = [
        { 'name':'Sincidence' }
    ]
    #
    # node table: world -> north_america
    #             north_america -> (united_states, canada)
    if test_config['node_effects']:
        node_table = [
            { 'name':'p1', 'parent':'' },
            { 'name':'c1', 'parent':'p1' },
            { 'name':'c2', 'parent':'p1' },
        ]
    else:
        node_table = [
            { 'name':'p1', 'parent':'' },
        ]
    
    #
    # weight table:
    weight_table = list()
    #
    # covariate table
    covariate_table = [{ 'name' : 'one','reference' : 0.0, 'max_difference': None }]
    if test_config['sex_effect']:
        covariate_table.append({ 'name' : 'sex','reference' : 0.0, 'max_difference': None })
    #
    # mulcov table
    mulcov_table = []
    if test_config['group_effects']:
        mulcov_table.append({'covariate': 'one',
                             'type':      'rate_value',
                             'effected':  'iota',
                             'group':     'g1',
                             'smooth' :   'smooth_iota_group' if test_config['use_group_mulcov'] else None,
                             # 'smooth' :   None,
                             'subsmooth': 'smooth_iota_subgroup'
        })
    if test_config['sex_effect']:
        mulcov_table.append({'covariate': 'sex',
                             'type':      'rate_value',
                             'effected':  'iota',
                             'group':     'g1' if test_config['group_effects'] else 'none',
                             'smooth' :   'smooth_iota_sex'})
    #
    # avgint table:
    avgint_table = list()
    #
    # nslist_table:
    nslist_table = dict()
    # ----------------------------------------------------------------------
    # subgroup_table
    subgroup_table = [
        { 'subgroup':'none', 'group':'none'},
        { 'subgroup':'s1',   'group':'g1'  },
        { 'subgroup':'s2',   'group':'g1'  },
    ]
    # ----------------------------------------------------------------------
    # data table:
    data_table = list()
    # write out data
    row = {
        'density':     'gaussian',
        'weight':      '',
        'hold_out':     False,
        'time_lower':   2000.0,
        'time_upper':   2000.0,
        'age_lower':    50.0,
        'age_upper':    50.0,
        'integrand':    'Sincidence',
        'one' :         1,
        # 'node':         'north_america',
        'node':         'p1',
        # 'eta':          1e-4,
    }
    sexs = [0,1] if test_config['sex_effect'] else [0]
    for node, node_effect in node_effects.items():
        if (not test_config['node_effects'] and node != 'p1'): continue
        # Exclude data for the parent node
        if (test_config['node_effects'] and node == 'p1'): continue
        for sex in sexs:
            for sg, sge in subgroup_effects.items():
                if (not test_config['group_effects'] and sg != 'none'): continue
                # Exclude data for the group -- if fitting both nodes and groups, omitting sg none creates Hessian errors
                if (test_config['group_effects'] and not test_config['include_group_data'] and sg == 'none'): continue
                total_effect = 0
                if test_config['sex_effect']:
                    row['sex'] = sex if sg != 'none' else -1
                    subgroups = pd.DataFrame(subgroup_table)
                    group = subgroups.loc[subgroups.subgroup == sg, 'group'].squeeze() if test_config['group_effects'] else 'g1'
                    sex_effect = sex*truth['iota_sex_true'][group]
                    total_effect += sex_effect
                if test_config['node_effects']:
                    row['node'] = node
                    total_effect += node_effect
                row['subgroup'] = sg
                sg_effect = 0
                if test_config['group_effects']:
                    if sg in ('s1', 's2'):
                        sg_effect = truth['iota_group_true'] + sge
                total_effect += sg_effect
                # print ({'sex_effect': (sex, sex_effect), 'node_effect': (node, node_effect), 'sg_effect': (sg, sg_effect), 'total_effect': total_effect})
                row['meas_value']  = truth['iota_parent_true'] * np.exp(total_effect)
                row['meas_std']    = row['meas_value'] * 1e-1
                data_table.append( copy.copy(row) )
    # ----------------------------------------------------------------------
    # prior_table
    prior_table = [
        { # prior_iota_parent
            'name':     'prior_iota_parent',
            'density':  prior.get('parent_density', 'iniform'),
            'mean':     prior.get('parent_mean', truth['iota_parent_true'] * .5),
            'std':      prior.get('parent_std', 0),
            'eta':      prior.get('parent_eta', None),
            'lower':    truth['iota_parent_true'] * 1e-2,
            'upper':    truth['iota_parent_true'] * 1e+2,
        },{ # prior_iota_child
            'name':     'prior_iota_child',
            'density':  prior.get('child_density', 'uniform'),
            'mean':     prior.get('child_mean', .001),
            'std':      prior.get('child_std', 0),
            'eta':      prior.get('child_eta', None),
            'lower':    -np.inf,
            'upper':    +np.inf,
        },
        { # prior_iota_group
            'name': 'prior_iota_group',
            'density': prior.get('group_density', 'uniform'),
            'mean':    prior.get('group_mean', 0.0),
            'std': prior.get('group_std', 0),
            # 'density': 'gaussian',
            # 'mean': 0.0,
            # 'std': 10.0,
        },
        { # prior_iota_subgroup
            'name': 'prior_iota_subgroup',
            'density': prior.get('subgroup_density', 'uniform'),
            'mean':    prior.get('subgroup_mean', 0.0),
            'std':     prior.get('subgroup_std', 0),
        }
    ]
    if test_config['sex_effect']:
        prior_table.append({ # prior_iota_sex
            'name': 'prior_iota_sex',
            'density': 'uniform', 
            'mean': 0.0,
            'lower': -100, 'upper': 100})
    # ----------------------------------------------------------------------
    # smooth table
    smooth_table = [
        { # smooth_iota_parent
            'name':                     'smooth_iota_parent',
            'age_id':                   [ 0 ],
            'time_id':                  [ 0 ],
            'fun':                       fun_iota_parent
        }]
    if test_config['node_effects']:
        smooth_table += [{ # smooth_iota_child
            'name':                     'smooth_iota_child',
            'age_id':                   [ 0 ],
            'time_id':                  [ 0 ],
            'fun':                       fun_iota_child
        }]
    if test_config['group_effects']:
        if test_config['use_group_mulcov']:
            smooth_table += [
                { # smooth_iota_group
                    'name':                     'smooth_iota_group',
                    'age_id':                   [ 0 ],
                    'time_id':                  [ 0 ],
                    'fun':                      fun_iota_group
                }]
        smooth_table += [
            { # smooth_iota_subgroup
                'name':                     'smooth_iota_subgroup',
                'age_id':                   [ 0 ],
                'time_id':                  [ 0 ],
                'fun':                      fun_iota_subgroup
            }]
    if test_config['sex_effect']:
        smooth_table.append( { # smooth_iota_sex
            'name':                     'smooth_iota_sex',
            'age_id':                   [ 0 ],
            'time_id':                  [ 0 ],
            'fun':                      fun_iota_sex})
    # ----------------------------------------------------------------------
    # rate table
    rate_table = [
        {
            'name':          'iota',
            'parent_smooth': 'smooth_iota_parent',
            'child_smooth':  'smooth_iota_child' if test_config['node_effects'] else None,
        }
    ]
    
    # ----------------------------------------------------------------------
    # option_table
    option_table = [
        # { 'name':'parent_node_name',       'value':'north_america' },
        { 'name':'parent_node_name',       'value':'p1' },
        { 'name':'print_level_fixed',      'value':5               },
        # { 'name':'print_level_fixed',      'value':0               },
        { 'name':'quasi_fixed',            'value':'false'         },
        # { 'name':'derivative_test_fixed',  'value':'second-order'   },
        # { 'name':'derivative_test_fixed',  'value':'trace-adaptive'   },
        { 'name':'tolerance_fixed',        'value':tol_fixed       },
        { 'name':'bound_frac_fixed',       'value':'1e-10'         },
        { 'name':'derivative_test_random', 'value':'second-order'  },
        { 'name':'tolerance_random',       'value':tol_random      },
        { 'name':'zero_sum_mulcov_group',  'value':'g1' if test_config['group_effects'] and test_config['zero_sum_mulcov'] else None},
        { 'name':'zero_sum_child_rate',    'value':'iota' if test_config['node_effects'] else None},
        { 'name':'rate_case',              'value':'iota_pos_rho_zero'},
        { 'name':'max_num_iter_fixed',     'value':'1000'           },
        { 'name':'max_num_iter_random',    'value':'100'           }
    ]
    # ----------------------------------------------------------------------

    # TODO: Change to using DismodIO instead of dismod_at.create_database
    from cascade_at.dismod.api.dismod_io import DismodIO
    db = DismodIO(file_name)
    from .create_database import create_database

    # create database
    #dismod_at.create_database(
    create_database(
        file_name,
        age_list,
        time_list,
        integrand_table,
        node_table,
        subgroup_table,
        weight_table,
        covariate_table,
        avgint_table,
        data_table,
        prior_table,
        smooth_table,
        nslist_table,
        rate_table,
        mulcov_table,
        option_table
    )
    # ----------------------------------------------------------------------
    from cascade_at.dismod.api.dismod_io import DismodIO
    db = DismodIO(file_name)
    return db
