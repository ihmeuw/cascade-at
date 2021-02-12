import sys
import shutil
import subprocess
import copy
import numpy as np
import random
import time
from pathlib import Path

"""
If there is mtspecific, hold out mtexcess on the ode fit.
Set the mulcov bounds
Check convergence
Check prediction
"""

enough_mtspecific = 100

sys.path.append('/Users/gma/Projects/IHME/GIT/cascade-at/src')
from cascade_at.dismod.api.dismod_io import DismodIO

def system (command) :
    # flush python's pending standard output in case this command generates more standard output
    sys.stdout.flush()
    print (command)
    if isinstance(command, str):
        command = command.split()
    run = subprocess.run(command)
    if run.returncode != 0 :
        raise Exception(f'"{command}" failed.')

def check_last_command(which_fit) :
    log = db.log
    fit_start = [i for i,l in enumerate(log.message) if l.startswith('begin fit')][-1]
    fit_end = [fit_start+i for i,l in enumerate(log.loc[fit_start:, 'message']) if l.startswith('end fit')]
    fit_end = fit_end[0] if fit_end else None
    which_fit = log.loc[fit_start, 'message'].split()[-1]
    assert fit_end and fit_end > fit_start, f"Fit {which_fit} failed."
    fit_messages = log.loc[fit_start:fit_end]
    msg = '\n'.join(fit_messages.loc[fit_messages.message_type.isin(['warning', 'error' ]), 'message'].tolist())
    if msg == '' :
        msg = f'\nfit_{which_fit} OK\n'
    else :
        msg = f'\nfit_{which_fit} Errors and or Warnings::\n' + msg
    print (msg)

# ============================================================================
# Utilities that use global data tables but do not modify them
# ============================================================================

def relative_covariate(db, covariate_id) :
    column_name = 'x_{}'.format(covariate_id)
    # sex is an absolute covariate and has 3 values, -0.5, 0.0, +0.5
    return len(set(db.data[column_name])) > 3

def useless_covariate(db, covariate_id):
    cov = db.covariate[db.covariate.covariate_id == covariate_id]
    covariate_name = cov.covariate_name.squeeze()
    reference = cov.reference.squeeze()
    return (db.data[covariate_name].unique() == reference).all()

# =============================================================================
# Routines that Only Change Data Table
# =============================================================================

def subset_data (db) :
    # remove data that are held out or have out of bound covariates
    msg  = '\nsubset_data\n'
    msg += 'removing hold out and covariate out of bounds data'
    print(msg)

    data = db.data
    data = data[data.hold_out == 0]
    for i,cov in db.covariate.iterrows():
        if np.isfinite(cov.max_difference):
            difference = abs(data[cov.covariate_name] - cov.reference)
            data = data[difference <= cov.max_difference]
    data = data.reset_index(drop=True)
    data['data_id'] = data.index
    return data

def random_subsample_data(db, integrand_name, max_sample) :
    # for a specified integrand, sample at most max_sample entries.
    # This does random sampling that can be seeded by calling random.seed.
    # The origianl order of the data is preserved (in index plots)
    # by sorting the subsample.
    data = db.data
    integrand_id = int(db.integrand.loc[db.integrand.integrand_name == integrand_name, 'integrand_id'])
    mask = data.integrand_id == integrand_id
    data_ids = data[mask].data_id.tolist()
    if len(data_ids) > max_sample:
        keep_ids = random.sample(data_ids, max_sample)
        data.loc[mask, 'hold_out'] = 1
        data.loc[data.data_id.isin(keep_ids), 'hold_out'] = 0
        print (f'Subsampled {integrand_name} to {sum(data[mask].hold_out == 0)} samples.')
    return data

def hold_out_data (db, integrand_names=(), node_names=(), hold_out=False) :
    if isinstance(integrand_names, str):
        integrand_names = [integrand_names]
    if isinstance(node_names, str):
        node_names = [node_names]
    data = db.data
    cols = data.columns
    data = data.merge(db.integrand).merge(db.node)
    mask = [True]*len(data)
    if integrand_names:
        mask &= data.integrand_name.isin(integrand_names)
    if node_names:
        mask &= data.node_name.isin(node_names)
    print (f"Setting hold_out = {hold_out} for integrand {integrand_names}, node {node_names}")
    data.loc[mask, 'hold_out'] = hold_out
    return data[cols]

"""

# -----------------------------------------------------------------------------
def set_data_likelihood (
		integrand_data, integrand_name, density_name, factor_eta=None, nu=None
) :
	# For a specified integrand, set its density, eta, and nu.
	# The default value for eta and nu is None.
	# If factor_eta is not None, eta is set to the factor times the median
	# value for the integrand.
	assert (factor_eta is None) or 0.0 <= factor_eta
	#
	msg  = '\nset_data_likelihood\n'
	msg += 'integrand = {}'.format(integrand_name)
	msg += ', density = {}'.format(density_name)
	if factor_eta is not None :
		msg += ', eta = m*{}'.format(factor_eta)
	if nu is not None :
		msg += ', nu = {}'.format(nu)
	if factor_eta is not None :
		msg += '\n' + 12 * ' ' + 'where m is the meadian of the'
		msg += ' {} data'.format(integrand_name)
	trace( msg )
	#
	# integrand_id
	integrand_id =integrand_name2id[integrand_name]
	#
	# density_id
	density_id = density_name2id[density_name]
	#
	if factor_eta is None :
		eta = None
	else :
		median = numpy.median( integrand_data[integrand_name] )
		eta = factor_eta * median
	for row in data_table :
		if row['integrand_id'] == integrand_id :
			row['density_id'] = density_id
			row['eta']        = eta
			row['nu']         = nu
	#
	table_name = 'data'
	put_table(table_name, data_table, data_col_name, data_col_type)
"""

def compress_age_time_intervals(db, age_size = 10.0, time_size = 10.0):
    data = db.data
    mask = (data.age_upper - data.age_lower) < age_size
    mean = data[['age_lower', 'age_upper']].mean(axis=1)
    data.loc[mask, 'age_lower'] = data.loc[mask, 'age_upper'] = mean[mask]
    mask = (data.time_upper - data.time_lower) < time_size
    mean = data[['time_lower', 'time_upper']].mean(axis=1)
    data.loc[mask, 'time_lower'] = data.loc[mask, 'time_upper'] = mean[mask]
    return data[db.data.columns]

def new_smoothing(integrand_name, age_grid, time_grid, value_prior, dage_prior, dtime_prior):
    # Add a new smoothing that has one prior that is used for all age and
    # time grid points. The smooth, smooth_grid, age, and time tables are
    # modified, but the new versions are not written by this routine.
    # The arguments value_prior, dage_prior, dtime_prior,
    # contain the priors used in the smothing.
    #

    smooth = db.smooth
    smooth_grid = db.smooth_grid
    prior = db.prior

    n_age = len(age_grid)
    n_time = len(time_grid)
    age_id_list = db.age.loc[db.age.age.isin(age_grid), 'age_id'].tolist()
    time_id_list = db.time.loc[db.time.time.isin(time_grid), 'time_id'].tolist()
    new_smooth_id = len(smooth)
    #
    # add value_prior to prior_table
    new_value_prior_id = len(prior)
    prior = prior.append(copy.copy(value_prior), ignore_index=True)
    #
    # add dage_prior to prior table
    new_dage_prior_id = len(prior)
    prior = prior.append(copy.copy(dage_prior), ignore_index=True)
    #
    # add dtime_prior to prior table
    new_dtime_prior_id = len(prior)
    prior = prior.append(copy.copy(dtime_prior), ignore_index=True)
    #
    # add row to smooth_table
    smooth_name    = f'{integrand_name}_noise_smoothing_{new_smooth_id}'
    row =  {'smooth_name'           : smooth_name    ,
            'n_age'                 : n_age          ,
            'n_time'                : n_time         ,
            'mulstd_value_prior_id' : None           ,
            'mulstd_dage_prior_id'  : None           ,
            'mulstd_dtime_prior_id' : None           ,
            }
    smooth = smooth.append(row, ignore_index=True)
    #
    # add rows to smooth_grid_table
    for i in range(n_age) :
        for j in range(n_time) :
            row = {'smooth_id'      : new_smooth_id                   ,
                   'age_id'         : age_id_list[i]                  ,
                   'time_id'        : time_id_list[j]                 ,
                   'value_prior_id' : new_value_prior_id              ,
                   'dage_prior_id'  : new_dage_prior_id               ,
                   'dtime_prior_id' : new_dtime_prior_id              ,
                   'const_value'    : None                            ,
                   }
            smooth_grid = smooth_grid.append(row, ignore_index=True)
    #
    # return the new smoothing
    smooth = smooth.reset_index(drop=True); smooth['smooth_id'] = smooth.index
    smooth_grid = smooth_grid.reset_index(drop=True); smooth_grid['smooth_grid_id'] = smooth_grid.index
    prior = prior.reset_index(drop=True); prior['prior_id'] = prior.index

    return new_smooth_id, smooth, smooth_grid, prior

def new_zero_smooth_id (db, smooth_id) :
# ----------------------------------------------------------------------------
    # FIXME: Remove this when bounds on mulcov work
    # add a new smoothing that has the same grid as smooth_id smoothing
    # and that constrains to zero. The smooth and smooth_grid tables are
    # modified by this routine but they are not written out.
    if smooth_id is None :
        return None
    #
    smooth = db.smooth
    smooth_grid = db.smooth_grid
    new_smooth_id = len(smooth)
    new_row = copy.copy( smooth[smooth_id] )
    new_row['smooth_name'] = f'zero_smoothing #{new_smooth_id}'
    smooth = smooth.append( new_row )
    #
    mask = smooth_grid.smooth_id == smooth_id
    for i, old_row in smooth_grid.iterrows() :
        if old_row['smooth_id'] == smooth_id :
            new_row = copy.copy( old_row )
            new_row['smooth_id']      = new_smooth_id
            new_row['value_prior_id'] = None
            new_row['dage_prior_id']  = None
            new_row['dtime_prior_id'] = None
            new_row['const_value']    = 0.0
            smooth_grid = smooth_grid.append( new_row )
    db.smooth = smooth
    db.smooth_grid = smooth_grid

def new_bounded_smooth_id (db, smooth_id, lower, upper, smooth_name = '') :
    # add a new smoothing that has the same grid as smooth_id smoothing
    # and that constrains value to be within the specified lower and upper
    # bounds.The prior, smooth and smooth_grid tables are modified but
    # they are not written out. The lower and upper bounds can be None.
    smooth_table = db.smooth
    smooth_grid_table = db.smooth_grid
    prior_table = db.prior
    if smooth_id is None :
        return None
    #
    if lower is None and upper is None :
        mean = 0.0
    elif lower is not None and upper is not None :
        mean = (lower + upper) / 2.0
    elif lower is None :
        if upper >= 0.0 :
            mean = 0.0
        else :
            mean = upper
    else :
        assert upper is None
        if lower <= 0.0 :
            mean = 0.0
        else :
            mean = lower
    #
    # smooth_table
    new_smooth_id = len(smooth_table)
    new_row                = copy.copy( smooth_table.loc[smooth_id] )
    new_row['smooth_id'] = new_smooth_id
    new_row['smooth_name'] = f'{smooth_name}_bound_smoothing_' + str( new_smooth_id )
    smooth_table = smooth_table.append( new_row )
    #
    new_prior_id  = len(prior_table)
    density_id    = int(db.density.loc[db.density.density_name == 'uniform', 'density_id'])
    value_prior  = {
        'prior_name' : 'smoothing_{}_centerd_prior'.format(new_smooth_id),
        'prior_id'   : new_prior_id,
        'density_id' : density_id,
        'lower'      : lower,
        'upper'      : upper,
        'mean'       : mean,
        'std'        : np.nan,
        'eta'        : np.nan,
        'nu'         : np.nan,
    }
    prior_table = prior_table.append( [value_prior] )
    #
    for i, old_row in smooth_grid_table.iterrows() :
        if old_row['smooth_id'] == smooth_id :
            new_row = copy.copy( old_row )
            new_row['smooth_id']      = new_smooth_id
            new_row['smooth_grid_id'] = len(smooth_grid_table)
            new_row['value_prior_id'] = new_prior_id
            new_row['dage_prior_id']  = None
            new_row['dtime_prior_id'] = None
            new_row['const_value']    = None
            smooth_grid_table = smooth_grid_table.append( new_row )
    smooth_table = smooth_table.reset_index(drop=True)
    smooth_grid_table = smooth_grid_table.reset_index(drop=True)
    db.smooth = smooth_table
    db.smooth_grid = smooth_grid_table
    db.prior = prior_table
    
    return new_smooth_id

def set_covariate_reference (db, covariate_id, reference_name = 'median') :
    # set the reference value for a specified covariate where reference_name
    # is 'mean' or 'median'
    #
    # covariate_value
    data = db.data
    covariate = db.covariate
    covariate_name = covariate.loc[covariate_id, 'covariate_name']
    covariate_value = data[covariate_name].tolist()
    new_reference = getattr(np, reference_name)(covariate_value)
    old_reference  = float(covariate.loc[covariate_id, 'reference'])
    #
    covariate.loc[covariate_id, 'reference'] = new_reference
    #
    msg  = '\nset_covariate_reference\n'
    msg += f'for covariate = {covariate_name}'
    msg += f', covariate_id = {covariate_id}'
    msg += f', reference_name = {reference_name}'
    msg += f'\nold_reference = {old_reference:.5g}'
    msg += f', new_reference = {new_reference:.5g}'
    print( msg )
    #
    db.covariate = covariate

def set_mulcov_bound(db, covariate_id, max_covariate_effect = 2) :
    # Set bounds for all of the multipliers for a specified covariate so
    # corresponding effect is bounded by disease_specific_max_covariate_effect.
    # Noise covariate multipliers are not included.
    #
    # reference for this covariate
    data = db.data
    covariate = db.covariate
    mulcov = db.mulcov
    #
    # covariate minus reference
    difference  = list()
    covariate_name = 'x_{}'.format(covariate_id)
    reference = covariate.loc[covariate_id, 'reference']
    difference = data[covariate_name].values - reference
    min_difference = min(difference)
    max_difference = max(difference)
    #
    # bounds on covariate multiplier
    # initialize
    lower = - float("inf")
    upper = + float("inf")
    if max_difference > 0 :
        upper = min(upper,    max_covariate_effect / max_difference)
        lower = max(lower,  - max_covariate_effect / max_difference)
    if min_difference < 0 :
        upper = min(upper,  - max_covariate_effect / min_difference)
        lower = max(lower,    max_covariate_effect / min_difference)
    if upper == float("inf") :
        lower = 0.0
        upper = 0.0
    #
    for i, row in mulcov.iterrows() :
        if row['covariate_id'] == covariate_id :
            if row['mulcov_type'] != 'meas_noise' :
                group_smooth_id = row['group_smooth_id']
                group_smooth_id = new_bounded_smooth_id(
                    db, group_smooth_id, lower, upper, smooth_name = covariate_name)
                row['group_smooth_id'] = group_smooth_id
                #
                subgroup_smooth_id = row['subgroup_smooth_id']
                subgroup_smooth_id = new_bounded_smooth_id(
                    db, subgroup_smooth_id, lower, upper, smooth_name = covariate_name)
                row['subgroup_smooth_id'] = subgroup_smooth_id
                mulcov.loc[i] = row
    db.mulcov = mulcov
    #
    if lower is None :
        lower = - float('inf')
    if upper is None :
        upper = + float('inf')
    msg  = '\nset_mulcov_bound\n'
    msg += 'covariate = x_{}, max_covariate_effect = {}, '
    msg += 'lower = {:.5g}, upper = {:.5g}'
    msg  = msg.format(covariate_id, max_covariate_effect, lower, upper)
    print( msg )
    return

def set_option (db, name, value) :
    # Set option specified by name to its value where name and value are
    # strings. The routine system_command to prints the processing message
    # for this operation.
    system(f'dismod_at {db.path} set option {name} {value}')

def add_meas_noise_mulcov(db, integrand_name, group_id, factor) :
    # Add a meas_noise covariate multiplier for a specified integrand.
    # integrand_data: is the current result of get_integrand_data.
    # group_id: specifies the group for the covariate multiplier.
    #
    # factor: is a dictionary with following keys: mean, lower, upper.
    # For each key the factor multipliers the absolute value of the
    # median of the data for this integrand to get the corresponding value
    # in the uniform prior for the square root of the covariate multiplier.
    # In other words, the factor is times a value is in standard deviation
    # units, while the prior values are in variance units.
    #
    # Note that meas_noise multipliers can't have
    # ramdom effect (so the subgroup id is null in the mulcov table).

    integrand = db.integrand
    data = db.data.merge(integrand, how='left')
    subgroup = db.subgroup
    covariate = db.covariate
    mulcov = db.mulcov
    smooth = db.smooth
    smooth_grid = db.smooth_grid
    prior = db.prior

    tst = (0.0 <= factor['lower'] <= factor['mean'] <= factor['upper'])
    assert tst, 'Factor is not monotonically increasing.'

    mask = (subgroup.group_id == group_id).values
    group_name = subgroup.loc[mask, 'group_name'].squeeze()
    assert group_name or not group_name.empty, 'Group name error for group_id = {group_id}' 

    covariate_id = int(db.covariate.loc[db.covariate.c_covariate_name == 's_one', 'covariate_id'])
    integrand_id = int(db.integrand.loc[db.integrand.integrand_name == integrand_name, 'integrand_id'])

    mulcov_values = mulcov[['mulcov_type', 'integrand_id', 'covariate_id']].values.tolist()
    if ['meas_noise', integrand_id, covariate_id] in mulcov_values:
        return
    #
    median = abs( data.loc[data.integrand_name == integrand_name, 'meas_value'].median() )
    lower  = ( median * factor['lower'] )**2
    mean   = ( median * factor['mean']  )**2
    upper  = ( median * factor['upper'] )**2
    #
    msg  = '\nadd_meas_noise_mulcov\n'
    msg += f'integrand = {integrand_name}, group = {group_name}, uniform value prior\n'
    msg += f'lower = (|median|*{factor["lower"]})^2 = {lower:.5g}\n'
    msg += f'mean  = (|median|*{factor["mean"]})^2 = {mean:.5g}\n'
    msg += f'upper = (|median|*{factor["upper"]})^2 = {upper:.5g}\n'
    msg += 'where median is the median of the {} data'.format(integrand_name)
    print( msg )
    #
    mulcov_id = len(mulcov)
    #
    # prior used in one point smoothing
    density_id = int(db.density.loc[db.density.density_name == 'uniform', 'density_id'])
    value_prior = {
        'prior_name' : integrand_name + '_meas_noise_value_prior'    ,
        'density_id' : density_id     ,
        'lower'      : lower          ,
        'upper'      : upper          ,
        'mean'       : mean           ,
        'std'        : np.nan         ,
        'eta'        : None           ,
        'nu'         : None           ,
    }
    dage_prior  = copy.copy( value_prior )
    dtime_prior = copy.copy( value_prior )
    dage_prior['prior_name']  =  integrand_name + '_meas_noise_dage_prior'
    dtime_prior['prior_name'] =  integrand_name + '_meas_noise_dtime_prior'
    #
    # new one point smoothing
    age_grid  = db.age.loc[:0, 'age'].tolist()
    time_grid = db.time.loc[:0, 'time'].tolist()

    smooth_id, smooth, smooth_grid, prior = new_smoothing(integrand_name,
        age_grid, time_grid, value_prior, dage_prior, dtime_prior )

    #
    # new row in mulcov_table
    row = dict(mulcov_id = None,
               mulcov_type      = 'meas_noise',
               covariate_id     = covariate_id,
               integrand_id     = integrand_id,
               group_id         = group_id,
               group_smooth_id  = smooth_id)
    mulcov = mulcov.append(row, ignore_index=True)
    #
    # write out the tables that changed
    mulcov = mulcov.reset_index(drop=True); mulcov['mulcov_id'] = mulcov.index
    smooth = smooth.reset_index(drop=True); smooth['smooth_id'] = smooth.index
    smooth_grid = smooth_grid.reset_index(drop=True); smooth_grid['smooth_grid_id'] = smooth_grid.index
    prior = prior.reset_index(drop=True); prior['prior_id'] = prior.index

    db.mulcov = mulcov
    db.smooth = smooth
    db.smooth_grid = smooth_grid
    db.prior = prior

def fix_ones_covariate_reference(covariate):
    # Make sure the ones covariate reference is correct
    mask = covariate.c_covariate_name == 's_one'
    if covariate.loc[mask, 'reference'].values == 1:
        covariate.loc[mask, 'reference'] = 0
        print (f"Fixed the 'one' covariate reference:\n{covariate[mask]}")
    return covariate

def check_ones_covariate(db):
    data = db.data
    integrand = db.integrand
    covariate = db.covariate
    mask = covariate.c_covariate_name == 's_one'
    one_covariate_id, one_covariate_name  = covariate.loc[mask, ['covariate_id', 'covariate_name']].squeeze()
    assert (data[one_covariate_name] == 1).all(), f"Error in data table values for covariate:\n {covariate[mask]}"

def drop_holdouts(db):
    data = db.data
    keep_mask = data.hold_out == 0
    data = data[keep_mask].reset_index(drop=True)
    data.data_id = data.index
    return data

def integrand_count(db):
    data = db.data.merge(db.integrand, how='left')
    return{n: len(data[(data.hold_out == 0) & (data.integrand_id == i)])
           for i,n in db.integrand[['integrand_id', 'integrand_name']].values
           if i in data.integrand_id.values}

def set_mulcov_zero (db, covariate_id, restore= None) :
    # set all of the multipliers for a specified covariate to zero without
    # changing the order or size of the var table
    covariate = db.covariate
    mulcov = db.mulcov
    covariate_name = covariate.loc[covariate_id, 'covariate_name']
    msg            = 'covariate = {}'.format(covariate_name)
    msg           += ', covariate_id = {}'.format(covariate_id)
    if restore is None :
        msg  = '\nset_mulcov_zero\n' + msg
    else :
        msg  = '\nrestore_mulcov\n' + msg
    print (msg)
    #
    # -------------------------------------------------------------------------
    if restore is not None :
        for (mulcov_id, group_smooth_id, subgroup_smooth_id) in restore :
            mulcov.loc[mulcov.mulcov_id == mulcov_id, 'group_smooth_id'] = group_smooth_id
            mulcov.loc[mulcov.mulcov_id == mulcov_id, 'subgroup_smooth_id'] = subgroup_smooth_id
        #
        db.mulcov = mulcov
        return None
    # -------------------------------------------------------------------------
    restore = list()
    for (mulcov_id, row)  in  mulcov.iterrows():
        if row['covariate_id'] == covariate_id :
            group_smooth_id, subgroup_smooth_id = row['group_smooth_id', 'subgroup_smooth_id']
            row['group_smooth_id']    = new_zero_smooth_id(group_smooth_id)
            row['subgroup_smooth_id'] = new_zero_smooth_id(subgroup_smooth_id)
            restore.append( (mulcov_id, group_smooth_id, subgroup_smooth_id) )
    #
    db.mulcov = mulcov
    db.smooth = smooth
    db.smooth_grid = smooth_grid
    return restore


original_file =  '/Users/gma/ihme/epi/at_cascade/data/475645/dbs/100/2/dismod.db'
original_file =  '/Users/gma/ihme/epi/at_cascade/data/475647/dbs/100/2/dismod.db'
original_file =  '/Users/gma/ihme/epi/at_cascade/data/475588/dbs/100/3/dismod.db'
# original_file =  '/Users/gma/ihme/epi/at_cascade/data/475882/dbs/100/2/dismod.db'
original_file =  '/Users/gma/ihme/epi/at_cascade/t1_diabetes/temp.db'
temp_file = '/tmp/temp.db'
shutil.copy2(original_file, temp_file)

# seed used to randomly subsample data
random_seed = 123
if random_seed == 0 :
    random_seed = int( time.time() )
random.seed(random_seed)
msg = '\nrandom_seed  = ' + str( random_seed )
print(msg)

db = DismodIO(Path(temp_file))
mask = db.integrand.integrand_id.isin(db.data.integrand_id.unique())
data_integrands = sorted(set(db.integrand.loc[mask, 'integrand_name']) - set(['mtall', 'mtother']))
no_ode_integrands = sorted(set(['Sincidence', 'mtexcess', 'mtother', 'remission']).intersection(data_integrands))
yes_ode_integrands = sorted(set(data_integrands) - set(no_ode_integrands))
integrands = yes_ode_integrands + no_ode_integrands

check_ones_covariate(db)
db.covariate = fix_ones_covariate_reference(db.covariate)
db.data = compress_age_time_intervals(db)

# add measurement noise covariates
group_id = 0
factor   = { 'lower':1e-1, 'mean':1e-1, 'upper':1e-1 }
for integrand in integrands:
    add_meas_noise_mulcov(db, integrand, group_id, factor)

# set bounds for all the covariates
for covariate_id in sorted(db.covariate.covariate_id.unique()):
    if useless_covariate(db, covariate_id ) :
        set_mulcov_zero( covariate_id )
    else :
        # Fixme specific.max_covariate_effect
        # set_mulcov_bound(db, , covariate_id)
        set_mulcov_bound(db, covariate_id, max_covariate_effect = 2)

max_num_iter_fixed = 50
dm_no_ode = DismodIO('/Users/gma/ihme/epi/at_cascade/t1_diabetes/no_ode/no_ode.db')
dm_yes_ode = DismodIO('/Users/gma/ihme/epi/at_cascade/t1_diabetes/yes_ode/yes_ode.db')
dmn = dm_no_ode
dmy = dm_yes_ode


rate = db.rate
iota_zero = not np.isfinite(rate.loc[rate.rate_name == 'iota', 'parent_smooth_id']).squeeze()
rho_zero = not np.isfinite(rate.loc[rate.rate_name == 'rho', 'parent_smooth_id']).squeeze()
chi_zero = not np.isfinite(rate.loc[rate.rate_name == 'chi', 'parent_smooth_id']).squeeze()
rate_case = ('iota_zero' if iota_zero else 'iota_pos') + '_' + ('rho_zero' if rho_zero else 'rho_pos')

set_option(db, 'tolerance_fixed', '1e-8')
set_option(db, 'max_num_iter_fixed', max_num_iter_fixed)
set_option(db, 'quasi_fixed', 'false')
set_option(db, 'zero_sum_child_rate', 'iota rho chi')
set_option(db, 'bound_random', '3')
set_option(db, 'meas_noise_effect', 'add_var_scale_none')
set_option(db, 'rate_case', rate_case)

reference_name  = 'median'
for covariate_id in range( len(db.covariate) ) :
    if relative_covariate(db, covariate_id):
        set_covariate_reference(db, covariate_id)

data_original = db.data


if 1:
    for integrand in data_integrands:
        db.data = random_subsample_data(db, integrand, max_sample = 1000)
db.data = subset_data(db)

db.data = hold_out_data(db, integrand_names = yes_ode_integrands, hold_out=1)

# db.data = drop_holdouts(db)


if 0:
    import pandas as pd
    dmv = pd.read_csv('/Users/gma/ihme/epi/at_cascade/t1_diabetes/no_ode/variable.csv')
    system(f'dismodat.py {temp_file} db2csv')
    dbv = pd.read_csv('/tmp/variable.csv')

if 0:
    for r in ['pini', 'iota', 'chi']:
        print (r)
        print (dmv.loc[(dmv.rate == r) & (dmv.time == 2010) & (dmv.age == 0), ['rate', 'var_type', 'age', 'time', 'density_v', 'lower_v', 'mean_v', 'upper_v', 'std_v', 'eta_v']])
        print (dbv.loc[(dbv.rate == r) & (dmv.time == 2010) & (dbv.age == 0), ['rate', 'var_type', 'age', 'time', 'density_v', 'lower_v', 'mean_v', 'upper_v', 'std_v', 'eta_v']])

if 0:
    print ((dmv.loc[(dmv.var_type != 'rate')].fillna(-1) == dbv.loc[(dbv.var_type != 'rate')].fillna(-1)).all())
    print (dmv.loc[(dmv.var_type != 'rate'), ['rate', 'var_type', 'age', 'time', 'covariate', 'density_v', 'lower_v', 'mean_v', 'upper_v', 'std_v']])
    print (dbv.loc[(dbv.var_type != 'rate'), ['rate', 'var_type', 'age', 'time', 'covariate', 'density_v', 'lower_v', 'mean_v', 'upper_v', 'std_v']])

    print (dmv.loc[(dmv.rate == 'iota'), ['rate', 'var_type', 'age', 'time', 'covariate', 'density_v', 'lower_v', 'mean_v', 'upper_v', 'std_v']])
    print (dbv.loc[(dbv.rate == 'iota'), ['rate', 'var_type', 'age', 'time', 'covariate', 'density_v', 'lower_v', 'mean_v', 'upper_v', 'std_v']])

    print (dmv.loc[(dmv.rate == 'chi'), ['rate', 'var_type', 'age', 'time', 'covariate', 'density_v', 'lower_v', 'mean_v', 'upper_v', 'std_v']])
    print (dbv.loc[(dbv.rate == 'chi'), ['rate', 'var_type', 'age', 'time', 'covariate', 'density_v', 'lower_v', 'mean_v', 'upper_v', 'std_v']])


    for rate_id, rate_name in db.rate[['rate_id', 'rate_name']].values:
        print ('>>>>>>>', rate_name)
        print ((dm.var[dm.var.rate_id==rate_id].drop(columns='var_id').fillna(-1).merge(dm.age).merge(dm.time) == db.var[db.var.rate_id==rate_id].drop(columns='var_id').fillna(-1).merge(db.age).merge(db.time)).all())

if 0:
    print ('Check the smooth')
    omega_id = 4
    cols = ['density_id', 'lower', 'upper', 'mean', 'std', 'eta']
    for name in ['value_prior_id', 'dage_prior_id', 'dtime_prior_id']:
        print (f'\n\n\nChecking {name} priors')
        dmprior_ids = dm.var[dm.var.rate_id != omega_id].merge(dm.smooth_grid, how='left')[name]
        dmprior_ids = dmprior_ids[~dmprior_ids.isna()].unique()
        dmgrps = dict(list(dm.prior.loc[dmprior_ids].fillna(-1).groupby(cols, dropna=False)))
        dbprior_ids = db.var[db.var.rate_id != omega_id].merge(db.smooth_grid, how='left')[name]
        dbprior_ids = dbprior_ids[~dbprior_ids.isna()].unique()
        dbgrps = dict(list(db.prior.loc[dbprior_ids].fillna(-1).groupby(cols, dropna=False)))
        keys = sorted(set(dmgrps.keys()).union(set(dbgrps.keys())))
        for k in keys:
            m = dmgrps.get(k, None)
            b = dbgrps.get(k, None)
            if m is None or b is None:
                print ('dm', m)
                print ('db', b)

print ('Active integrands', integrand_count(db))

system(f'dismod_at {temp_file} init')
system(f'dismod_at {temp_file} fit both')

check_last_command(db)

db.data = data_original

if 1:
    for integrand in data_integrands:
        db.data = random_subsample_data(db, integrand, max_sample = 1000)

data = db.data.merge(db.integrand, how='left')
hold_outs = []
if iota_zero: hold_outs.append('Sincidence')
if rho_zero: hold_outs.append('remission')
if chi_zero: hold_outs.append('mtexcess')
# Consider dropping mtexcess if there is sufficient mtspecific
count = sum(data.integrand_name == 'mtspecific')
if count > enough_mtspecific:
    print (f'Holding out mtexcess because there are sufficient mtspecific data ({count} rows).')
    hold_outs.append('mtexcess')
db.data = hold_out_data(db, integrand_names = hold_outs, hold_out=1)

db.data = drop_holdouts(db)

db.data = compress_age_time_intervals(db)

print ('Active integrands', integrand_count(db))


# for t in ['option', 'data', 'rate', 'var', 'fit_var', 'scale_var', 'start_var', 'covariate', 'mulcov', 'nslist', 'nslist_pair']:
#     if not np.all(getattr(db,t).fillna(-1) == getattr(dm, t).fillna(-1)):
#         setattr(db,t, getattr(dm, t))        

if 0:
    for t in ['covariate',
              'data',
              'fit_var',
              'prior',
              'scale_var',
              'start_var',

              # not required
              # 'option',


              # identical?
              # 'smooth',
              # 'data_subset',
              # 'truth_var', 
              'age', 'avgint', 'rate', 'density', 'integrand', 'mulcov', 'node', 'nslist', 'nslist_pair', 
              'smooth_grid', 'subgroup', 'time', 'var', 'weight', 'weight_grid']:

        try:
            tst = not np.all(getattr(db,t).fillna(-1) == getattr(dm, t).fillna(-1))
        except:
            tst = True
        if not tst:
            print (f'Table db.{t} and dm.{t} are equal')
        else:
            print (f'setting db.{t} = dm.{t}')
            setattr(db,t, getattr(dm, t))


if 1:
    db.data = dm.data
    print ('Are the two var tables alike?')
    print ((db.var.fillna(-1) == dm.var.fillna(-1)).all())
    print ('Are the two data tables alike?')
    print ((db.data.fillna(-1) == dm.data.fillna(-1)).all())

system(f'dismod_at {temp_file} set start_var fit_var')

print ("""BRADS RESULT:
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  1.1042442e+03 9.30e-04 2.07e+03  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
Warning: Cutting back alpha due to evaluation error
   1  2.8986326e+02 7.93e-04 1.06e+03  -1.0 4.05e+00    -  4.98e-01 4.89e-01f  2""")



system(f'dismod_at {temp_file} fit both')
system(f'dismod_at {temp_file} sample asymptotic both 10')

check_last_command(db)


'''


data['density_id'] = 3
data['nu'] = 5
db.data = data[db.data.columns]
system(f'dismod_at {temp_file} set start_var fit_var')
system(f'dismod_at {temp_file} fit both')
check_last_command(db)


os.system(f'DB_plot.py {temp_file} -v 475882')
'''


