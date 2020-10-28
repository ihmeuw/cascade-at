"""
This is Brad's create_database.py module used in his testing, 
found in https://github.com/bradbell/dismod_at.git/build/lib/example_db.py
modified to use the cascade_at database API
"""
import pandas as pd

def create_database(
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
) :
    import sys
    #*# import dismod_at
    from cascade_at.dismod.api.dismod_io import DismodIO
    db = DismodIO(file_name)

    # ----------------------------------------------------------------------
    # avgint_extra_columns, data_extra_columns
    avgint_extra_columns = list()
    data_extra_columns   = list()
    for row in option_table :
        if row['name'] == 'avgint_extra_columns' :
            avgint_extra_columns = row['value'].split()
        if row['name'] == 'data_extra_columns' :
            data_extra_columns = row['value'].split()
    # ----------------------------------------------------------------------
    # create database
    new            = True
    #*# connection     = dismod_at.create_connection(file_name, new)
    # ----------------------------------------------------------------------
    # create age table
    col_name = [ 'age' ]
    col_type = [ 'real' ]
    row_list = []
    for age in age_list :
        row_list.append( [age] )
    tbl_name = 'age'
    #*# dismod_at.create_table(connection, tbl_name, col_name, col_type, row_list)
    db.age = pd.DataFrame(row_list, columns = col_name)
    # ----------------------------------------------------------------------
    # create time table
    col_name = [ 'time' ]
    col_type = [ 'real' ]
    row_list = []
    for time in time_list :
        row_list.append( [time] )
    tbl_name = 'time'
    #*# dismod_at.create_table(connection, tbl_name, col_name, col_type, row_list)
    db.time = pd.DataFrame(row_list, columns = col_name)
    # ----------------------------------------------------------------------
    # create integrand table
    col_name = [ 'integrand_name', 'minimum_meas_cv' ]
    col_type = [ 'text',           'real' ]
    row_list = []
    for i in range( len(integrand_table) ) :
        minimum_meas_cv = 0.0
        if 'minimum_meas_cv' in integrand_table[i] :
            minimum_meas_cv = integrand_table[i]['minimum_meas_cv']
        row = [ integrand_table[i]['name'], minimum_meas_cv ]
        row_list.append( row )
    tbl_name = 'integrand'
    #*# dismod_at.create_table(connection, tbl_name, col_name, col_type, row_list)
    db.integrand = pd.DataFrame(row_list, columns = col_name)
    #
    global_integrand_name2id = {}
    for i in range( len(row_list) ) :
        global_integrand_name2id[ row_list[i][0] ] = i
    # ----------------------------------------------------------------------
    # create density table
    col_name = [  'density_name'   ]
    col_type = [  'text'        ]
    row_list = [
        ['uniform'],
        ['gaussian'],
        ['laplace'],
        ['students'],
        ['log_gaussian'],
        ['log_laplace'],
        ['log_students'],
        ['cen_gaussian'],
        ['cen_laplace'],
        ['cen_log_gaussian'],
        ['cen_log_laplace'],
    ]
    tbl_name = 'density'
    #*# dismod_at.create_table(connection, tbl_name, col_name, col_type, row_list)
    db.density = pd.DataFrame(row_list, columns = col_name)
    #
    global_density_name2id = {}
    for i in range( len(row_list) ) :
        global_density_name2id[ row_list[i][0] ] = i
    # ----------------------------------------------------------------------
    # create covariate table
    col_name = [ 'covariate_name',    'reference', 'max_difference' ]
    col_type = [ 'text',             'real',     'real'           ]
    row_list = [ ]
    for i in range( len(covariate_table) ) :
        max_difference = None
        if 'max_difference' in covariate_table[i] :
            max_difference = covariate_table[i]['max_difference']
        row       = [
            covariate_table[i]['name'],
            covariate_table[i]['reference'],
            max_difference
        ]
        row_list.append(row)
    tbl_name = 'covariate'
    #*# dismod_at.create_table(connection, tbl_name, col_name, col_type, row_list)
    db.covariate = pd.DataFrame(row_list, columns = col_name)
    #
    global_covariate_name2id = {}
    for i in range( len(covariate_table) ) :
        global_covariate_name2id[ covariate_table[i]['name'] ] = i
    # ----------------------------------------------------------------------
    # create node table
    global_node_name2id = {}
    for i in range( len(node_table) ) :
        global_node_name2id[ node_table[i]['name'] ] = i
    #
    col_name = [ 'node_name', 'parent' ]
    col_type = [ 'text',      'integer'   ]
    row_list = []
    for i in range( len(node_table) ) :
        node   = node_table[i]
        name   = node['name']
        parent = node['parent']
        if parent == '' :
            parent = None
        else :
            parent = global_node_name2id[parent]
        row_list.append( [ name, parent ] )
    tbl_name = 'node'
    #*# dismod_at.create_table(connection, tbl_name, col_name, col_type, row_list)
    db.node = pd.DataFrame(row_list, columns = col_name)

    # create subgroup table
    global_subgroup_name2id = {}
    global_group_name2id = {}
    group_id   = 0
    group_name = subgroup_table[0]['group']
    global_group_name2id[group_name] = group_id
    for i in range( len(subgroup_table) ) :
        global_subgroup_name2id[ subgroup_table[i]['subgroup'] ] = i
        if subgroup_table[i]['group'] != group_name :
            group_id   = group_id + 1
            group_name = subgroup_table[i]['group']
            global_group_name2id[group_name] = group_id
    #
    col_name      = [ 'subgroup_name', 'group_id', 'group_name' ]
    col_type      = [ 'text',          'integer',  'text'       ]
    row_list      = []
    for i in range( len(subgroup_table) ) :
        if i == 0 :
            group_id   = 0
            group_name = subgroup_table[0]['group']
        elif subgroup_table[i]['group'] != group_name :
            group_id   = group_id + 1
            group_name = subgroup_table[i]['group']
        subgroup_name   = subgroup_table[i]['subgroup']
        row_list.append( [ subgroup_name, group_id, group_name ] )
    tbl_name = 'subgroup'
    #*# dismod_at.create_table(connection, tbl_name, col_name, col_type, row_list)
    db.subgroup = pd.DataFrame(row_list, columns = col_name)

    # ----------------------------------------------------------------------
    # create prior table
    col_name = [
    'prior_name', 'lower', 'upper', 'mean', 'std',  'density_id', 'eta', 'nu'
    ]
    col_type = [
    'text', 'real',  'real',  'real', 'real', 'integer',  'real', 'real'
    ]
    row_list = [ ]
    for i in range( len( prior_table ) ) :
        prior         = prior_table[i]
        density_id   = global_density_name2id[ prior['density'] ]
        #
        # columns that have null for default value
        for key in [ 'lower', 'upper', 'std', 'eta', 'nu' ] :
            if not key in prior :
                prior[key] = None
        #
        row  = [
            prior['name'],
            prior['lower'],
            prior['upper'],
            prior['mean'],
            prior['std'],
            density_id,
            prior['eta'],
            prior['nu'],
        ]
        row_list.append( row )
    tbl_name = 'prior'
    #*# dismod_at.create_table(connection, tbl_name, col_name, col_type, row_list)
    db.prior = pd.DataFrame(row_list, columns = col_name)
    #
    global_prior_name2id = {}
    for i in range( len(row_list) ) :
        global_prior_name2id[ row_list[i][0] ] = i
    # ----------------------------------------------------------------------
    # create weight table
    col_name = [ 'weight_name', 'n_age',   'n_time'   ]
    col_type = [ 'text',        'integer', 'integer'  ]
    row_list = [ ]
    for i in range( len(weight_table) ) :
        weight = weight_table[i]
        name   = weight['name']
        n_age  = len( weight['age_id'] )
        n_time = len( weight['time_id'] )
        row_list.append( [ name, n_age, n_time ] )
    tbl_name = 'weight'
    #*# dismod_at.create_table(connection, tbl_name, col_name, col_type, row_list)
    db.weight = pd.DataFrame(row_list, columns = col_name)
    #
    global_weight_name2id = {}
    for i in range( len(weight_table) ) :
        global_weight_name2id[ weight_table[i]['name'] ] = i
    # null is used for constant weighting
    global_weight_name2id[ '' ] = None
    # ----------------------------------------------------------------------
    # create weight_grid table
    col_name = [  'weight_id', 'age_id',   'time_id',  'weight' ]
    col_type = [  'integer',   'integer',  'integer',  'real'   ]
    row_list = [ ]
    for i in range( len(weight_table) ) :
        weight  = weight_table[i]
        age_id  = weight['age_id']
        time_id = weight['time_id']
        fun     = weight['fun']
        for j in age_id :
            for k in time_id :
                w = fun(age_list[j], time_list[k])
                row_list.append( [ i, j, k, w] )
    tbl_name = 'weight_grid'
    #*# dismod_at.create_table(connection, tbl_name, col_name, col_type, row_list)
    db.weight_grid = pd.DataFrame(row_list, columns = col_name)
    # ----------------------------------------------------------------------
    # create smooth table
    col_name = [
        'smooth_name',
        'n_age',
        'n_time',
        'mulstd_value_prior_id',
        'mulstd_dage_prior_id',
        'mulstd_dtime_prior_id'
    ]
    col_type = [
        'text',
        'integer',
        'integer',
        'integer',
        'integer',
        'integer'
    ]
    row_list = [ ]
    for i in range( len(smooth_table) ) :
        smooth        = smooth_table[i]
        name          = smooth['name']
        n_age         = len( smooth['age_id'] )
        n_time        = len( smooth['time_id'] )
        #
        prior_id = dict()
        for key in [ 'value', 'dage', 'dtime' ] :
            prior_id[key] = None
            mulstd_key    = 'mulstd_' + key + '_prior_name'
            if mulstd_key in smooth :
                prior_name    = smooth[mulstd_key]
                if prior_name != None :
                    prior_id[key] = global_prior_name2id[prior_name]
        #
        row_list.append( [
            name,
            n_age,
            n_time,
            prior_id['value'],
            prior_id['dage'],
            prior_id['dtime'],
        ] )
    tbl_name = 'smooth'
    #*# dismod_at.create_table(connection, tbl_name, col_name, col_type, row_list)
    db.smooth = pd.DataFrame(row_list, columns = col_name)
    #
    global_smooth_name2id = {}
    for i in range( len(smooth_table) ) :
        global_smooth_name2id[ smooth_table[i]['name'] ] = i
    # ----------------------------------------------------------------------
    # create smooth_grid table
    col_name = [
        'smooth_id',
        'age_id',
        'time_id',
        'value_prior_id',
        'dage_prior_id',
        'dtime_prior_id',
        'const_value',
    ]
    col_type = [
        'integer',  # smooth_id
        'integer',  # age_id
        'integer',  # time_id
        'integer',  # value_prior_id
        'integer',  # dage_prior_id
        'integer',  # dtime_prior_id
        'real',     # const_value
    ]
    row_list = [ ]
    for i in range( len(smooth_table) ) :
        smooth  = smooth_table[i]
        age_id  = smooth['age_id']
        time_id = smooth['time_id']
        fun     = smooth['fun']
        max_j   = 0
        for j in age_id :
            if age_list[j] > age_list[max_j] :
                max_j = j
        max_k   = 0
        for k in time_id :
            if time_list[k] > time_list[max_k] :
                max_k = k
        for j in age_id :
            for k in time_id :
                (v,da,dt) = fun(age_list[j], time_list[k])
                #
                if j == max_j :
                    da = None
                elif da != None :
                    da = global_prior_name2id[da]
                #
                if k == max_k :
                    dt = None
                elif dt != None :
                    dt = global_prior_name2id[dt]
                #
                const_value = None
                if isinstance(v, float) :
                    const_value = v
                    v = None
                elif v != None :
                    v = global_prior_name2id[v]
                row_list.append( [ i, j, k, v, da, dt, const_value] )
    tbl_name = 'smooth_grid'
    #*# dismod_at.create_table(connection, tbl_name, col_name, col_type, row_list)
    db.smooth_grid = pd.DataFrame(row_list, columns = col_name)
    # ----------------------------------------------------------------------
    # create nslist table
    col_name = [ 'nslist_name' ]
    col_type = [ 'text' ]
    row_list = list()
    for nslist_name in nslist_table :
        row_list.append( [ nslist_name ] )
    tbl_name = 'nslist'
    #*# dismod_at.create_table(connection, tbl_name, col_name, col_type, row_list)
    db.nslist = pd.DataFrame(row_list, columns = col_name)
    #
    global_nslist_name2id = dict()
    for i in range( len( row_list ) ) :
        global_nslist_name2id[ row_list[i][0] ] = i
    # ----------------------------------------------------------------------
    # create nslist_pair table
    col_name = [ 'nslist_id', 'node_id', 'smooth_id' ]
    col_type = [ 'integer',   'integer', 'integer'   ]
    row_list = list()
    tbl_name = 'nslist_pair'
    for key in nslist_table :
        pair_list = nslist_table[key]
        nslist_id = global_nslist_name2id[key]
        for pair in pair_list :
            node_id   = global_node_name2id[ pair[0] ]
            smooth_id = global_smooth_name2id[ pair[1] ]
            row_list.append( [ nslist_id, node_id, smooth_id ] )
    #*# dismod_at.create_table(connection, tbl_name, col_name, col_type, row_list)
    db.nslist_pair = pd.DataFrame(row_list, columns = col_name)
    # ----------------------------------------------------------------------
    # create rate table
    col_name = [
        'rate_name', 'parent_smooth_id', 'child_smooth_id', 'child_nslist_id'
    ]
    col_type = [
        'text',      'integer',         'integer',           'integer'
    ]
    row_list = list()
    for rate_name in [ 'pini', 'iota', 'rho', 'chi', 'omega' ] :
        row = [ rate_name, None, None, None ]
        for i in range( len(rate_table) ) :
            rate = rate_table[i]
            if rate['name'] == rate_name :
                row = [ rate_name ]
                for key in ['parent_smooth', 'child_smooth', 'child_nslist'] :
                    entry  = None
                    if key in rate :
                        entry = rate[key]
                    if entry != None :
                        if key == 'child_nslist' :
                            entry = global_nslist_name2id[ entry ]
                        else :
                            entry = global_smooth_name2id[ entry ]
                    row.append( entry )
        row_list.append( row )
    tbl_name = 'rate'
    #*# dismod_at.create_table(connection, tbl_name, col_name, col_type, row_list)
    db.rate = pd.DataFrame(row_list, columns = col_name)
    global_rate_name2id = {}
    for i in range( len(row_list) ) :
        global_rate_name2id[ row_list[i][0] ] = i
    # ----------------------------------------------------------------------
    # create mulcov table
    col_name = [
        'mulcov_type',
        'rate_id',
        'integrand_id',
        'covariate_id',
        'group_id',
        'group_smooth_id',
        'subgroup_smooth_id',
    ]
    col_type = [
        'text',    # mulcov_type
        'integer', # rate_id
        'integer', # integrand_id
        'integer', # covariate_id
        'integer', # group_id
        'integer', # group_smooth_id
        'integer', # subgroup_smooth_id
    ]
    row_list = []
    warning_printed = False
    for i in range( len(mulcov_table) ) :
        mulcov       = mulcov_table[i]
        mulcov_type  = mulcov['type']
        effected     = mulcov['effected']
        covariate_id = global_covariate_name2id[ mulcov['covariate'] ]
        #
        # rate_id and integrand_id
        if mulcov_type == 'rate_value' :
            rate_id      = global_rate_name2id[ effected ]
            integrand_id = None
        else :
            integrand_id = global_integrand_name2id[ effected ]
            rate_id      = None
        #
        # group_id
        if 'group' in mulcov :
            group_id = global_group_name2id[ mulcov['group'] ]
        else :
            group_id = 0
            if not warning_printed :
                msg  = 'create_database Warning: '
                msg += 'group key missing in mulcov table,\n'
                msg += 'using default value; i.e., first group '
                msg += '(you should fix this).'
                print(msg)
                warning_printed = True
        #
        # group_smooth_id
        if mulcov['smooth'] == None :
            group_smooth_id = None
        else :
            group_smooth_id    = global_smooth_name2id[ mulcov['smooth'] ]
        #
        # subgroup_smooth_id
        if not 'subsmooth' in mulcov :
            subgroup_smooth_id = None
        elif mulcov['subsmooth'] == None :
            subgroup_smooth_id = None
        else :
            subgroup_smooth_id = global_smooth_name2id[ mulcov['subsmooth'] ]
        #
        row_list.append( [
            mulcov_type,
            rate_id,
            integrand_id,
            covariate_id,
            group_id,
            group_smooth_id,
            subgroup_smooth_id,
        ] )
    tbl_name = 'mulcov'
    #*# dismod_at.create_table(connection, tbl_name, col_name, col_type, row_list)
    db.mulcov = pd.DataFrame(row_list, columns = col_name)
    # ----------------------------------------------------------------------
    # avgint table
    #
    # extra_name, extra_type
    extra_name = []
    extra_type = []
    if( len( avgint_table ) > 0 ) :
        extra_name = avgint_extra_columns
        row        = avgint_table[0]
        for key in extra_name :
            if isinstance(row[key], str) :
                extra_type.append('text')
            elif isinstance(row[key], int) :
                extra_type.append('integer')
            elif isinstance(row[key], float) :
                extra_type.append('real')
            else :
                assert False
    #
    # col_name
    col_name = extra_name + [
        'integrand_id',
        'node_id',
        'subgroup_id',
        'weight_id',
        'age_lower',
        'age_upper',
        'time_lower',
        'time_upper'
    ]
    for j in range( len(covariate_table) ) :
        col_name.append( 'x_%s' % j )
    #
    # col_type
    col_type = extra_type + [
        'integer',              # integrand_id
        'integer',              # node_id
        'integer',              # subgroup_id
        'integer',              # weight_id
        'real',                 # age_lower
        'real',                 # age_upper
        'real',                 # time_lower
        'real'                  # time_upper
    ]
    for j in range( len(covariate_table) )  :
        col_type.append( 'real' )
    #
    # row_list
    row_list = [ ]
    warning_printed = False;
    for i in range( len(avgint_table) ) :
        avgint = avgint_table[i]
        #
        # subgroup column has a default value
        if 'subgroup' not in avgint :
            avgint['subgroup'] = subgroup_table[0]['subgroup']
            if not warning_printed :
                msg  = 'create_database Warning: '
                msg += 'subgroup key missing in avgint table,\n'
                msg += 'using default value; i.e., first subgroup '
                msg += '(you should fix this).'
                print(msg)
                warning_printed = True
        #
        # extra columns first
        row = list()
        for name in extra_name :
            row.append( avgint[ name ] )
        #
        avgint_id      = i
        integrand_id = global_integrand_name2id[ avgint['integrand'] ]
        node_id      = global_node_name2id[ avgint['node'] ]
        subgroup_id  = global_subgroup_name2id[ avgint['subgroup'] ]
        weight_id    = global_weight_name2id[ avgint['weight'] ]
        row = row + [
            integrand_id,
            node_id,
            subgroup_id,
            weight_id,
            avgint['age_lower'],
            avgint['age_upper'],
            avgint['time_lower'],
            avgint['time_upper']
        ]
        for j in range( len(covariate_table) ) :
            row.append( avgint[ float(covariate_table[j]['name']) ] )
        row_list.append(row)

    tbl_name = 'avgint'
    #*# dismod_at.create_table(connection, tbl_name, col_name, col_type, row_list)
    db.avgint = (pd.DataFrame(row_list, columns = col_name)
                 .astype(dict(zip(col_name, pd.Series(col_type).replace({'integer': 'int', 'real': 'float'})))))
    # ----------------------------------------------------------------------
    # create data table
    #
    #
    # extra_name, extra_type
    extra_name = []
    extra_type = []
    if( len( data_table ) > 0 ) :
        extra_name = data_extra_columns
        row        = data_table[0]
        for key in extra_name :
            if isinstance(row[key], str) :
                extra_type.append('text')
            elif isinstance(row[key], int) :
                extra_type.append('integer')
            elif isinstance(row[key], float) :
                extra_type.append('real')
            else :
                assert False
    #
    # col_name
    col_name = extra_name + [
        'integrand_id',
        'node_id',
        'subgroup_id',
        'weight_id',
        'age_lower',
        'age_upper',
        'time_lower',
        'time_upper',
        'hold_out',
        'density_id',
        'meas_value',
        'meas_std',
        'eta',
        'nu',
    ]
    for j in range( len(covariate_table) ) :
        col_name.append( 'x_%s' % j )
    #
    # col_type
    col_type = extra_type + [
        'integer',              # integrand_id
        'integer',              # node_id
        'integer',              # subgroup_id
        'integer',              # weight_id
        'real',                 # age_lower
        'real',                 # age_upper
        'real',                 # time_lower
        'real',                 # time_upper
        'integer',              # hold_out
        'integer',              # density_id
        'real',                 # meas_value
        'real',                 # meas_std
        'real',                 # eta
        'real',                 # nu
    ]
    for j in range( len(covariate_table) )  :
        col_type.append( 'real' )
    row_list = [ ]
    warning_printed = False
    for i in range( len(data_table) ) :
        data         = data_table[i]
        #
        # extra columns first
        row = list()
        for name in extra_name :
            row.append( data[name] )
        #
        # columns that have null for default value
        for key in [ 'meas_std', 'eta', 'nu' ] :
            if not key in data :
                data[key] = None
        #
        # subgroup column has a default value
        if not 'subgroup' in data :
            data['subgroup'] = subgroup_table[0]['subgroup']
            if not warning_printed :
                msg  = 'create_database Warning: '
                msg += 'subgroup key missing in data table,\n'
                msg += 'using default value; i.e., first subgroup '
                msg += '(you should fix this).'
                print(msg)
                warning_printed = True
        #
        integrand_id = global_integrand_name2id[ data['integrand'] ]
        density_id   = global_density_name2id[ data['density'] ]
        node_id      = global_node_name2id[ data['node'] ]
        subgroup_id  = global_subgroup_name2id[ data['subgroup'] ]
        weight_id    = global_weight_name2id[ data['weight'] ]
        hold_out     = int( data['hold_out'] )
        row = row + [
            integrand_id,
            node_id,
            subgroup_id,
            weight_id,
            data['age_lower'],
            data['age_upper'],
            data['time_lower'],
            data['time_upper'],
            hold_out,
            density_id,
            data['meas_value'],
            data['meas_std'],
            data['eta'],
            data['nu']
        ]
        for j in range( len(covariate_table) ) :
            row.append( float(data[ covariate_table[j]['name'] ]) )
        row_list.append(row)

    tbl_name = 'data'
    #*# dismod_at.create_table(connection, tbl_name, col_name, col_type, row_list)
    data = pd.DataFrame(row_list, columns = col_name)
    data['data_name'] = ''
    db.data = data
    # ----------------------------------------------------------------------
    # create option table
    col_name = [ 'option_name', 'option_value' ]
    col_type = [ 'text unique', 'text' ]
    row_list = []
    for row in option_table :
        name  = row['name']
        value = row['value']
        row_list.append( [ name, value ] )
    tbl_name = 'option'
    #*# dismod_at.create_table(connection, tbl_name, col_name, col_type, row_list)
    db.option = pd.DataFrame(row_list, columns = col_name)
    # ----------------------------------------------------------------------
    # close the connection
    #*# connection.close()
    return
