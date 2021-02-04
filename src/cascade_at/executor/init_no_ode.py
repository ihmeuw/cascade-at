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

def compress_age_time_intervals(data, age_size = 10.0, time_size = 10.0):
    mask = (data.age_upper - data.age_lower) < age_size
    mean = data[['age_lower', 'age_upper']].mean(axis=1)
    data.loc[mask, 'age_lower'] = data.loc[mask, 'age_upper'] = mean[mask]
    mask = (data.time_upper - data.time_lower) < time_size
    mean = data[['time_lower', 'time_upper']].mean(axis=1)
    data.loc[mask, 'time_lower'] = data.loc[mask, 'time_upper'] = mean[mask]
    return data

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

def hold_out_ODE_integrands(db):
    data = db.data.merge(db.integrand, how='left')
    keep_mask = (data.integrand_name.isin(yes_ode_integrands) & (data.hold_out == 0)).values
    data.loc[~keep_mask, 'hold_out'] = 1
    return data

def drop_ODE_integrands(db):
    data = db.data.merge(db.integrand, how='left')
    keep_mask = (data.integrand_name.isin(yes_ode_integrands) & (data.hold_out == 0)).values
    data = data[~keep_mask].reset_index(drop=True)
    data.data_id = data.index
    return data

def useless_covariate(db, covariate_id):
    cov = db.covariate[db.covariate.covariate_id == covariate_id]
    covariate_name = cov.covariate_name.squeeze()
    reference = cov.reference.squeeze()
    return (db.data[covariate_name].unique() == reference).all()

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

def check_last_fit(which_fit) :
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

    covariate_id = int(db.covariate.loc[db.covariate.c_covariate_name == 's_one', 'covariate_id'])
    integrand_id = int(db.integrand.loc[db.integrand.integrand_name == integrand_name, 'integrand_id'])

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

original_file =  '/Users/gma/ihme/epi/at_cascade/data/475645/dbs/100/2/dismod.db'
original_file =  '/Users/gma/ihme/epi/at_cascade/data/475588/dbs/100/3/dismod.db'
original_file =  '/Users/gma/ihme/epi/at_cascade/data/475647/dbs/100/2/dismod.db'
original_file =  '/Users/gma/ihme/epi/at_cascade/data/475882/dbs/100/2/dismod.db'

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
data_original = db.data
mask = db.integrand.integrand_id.isin(db.data.integrand_id.unique())
data_integrands = sorted(set(db.integrand.loc[mask, 'integrand_name']) - set(['mtall', 'mtother']))
no_ode_integrands = sorted(set(['Sincidence', 'mtexcess', 'mtother', 'remission']).intersection(data_integrands))
yes_ode_integrands = sorted(set(data_integrands) - set(no_ode_integrands))
integrands = yes_ode_integrands + no_ode_integrands

check_ones_covariate(db)
db.covariate = fix_ones_covariate_reference(db.covariate)
db.data = compress_age_time_intervals(db.data)

# add measurement noise covariates
group_id = 0
factor   = { 'lower':1e-1, 'mean':1e-1, 'upper':1e-1 }
for integrand in integrands:
    add_meas_noise_mulcov(db, integrand, group_id, factor)

set_covariate_reference (db, 0, reference_name = 'median')

if 1:
    for integrand in data_integrands:
        data = random_subsample_data(db, integrand, max_sample = 1000)

# set bounds for all the covariates
for covariate_id in sorted(db.covariate.covariate_id.unique()):
    if useless_covariate(db, covariate_id ) :
        set_mulcov_zero( covariate_id )
    else :
        # Fixme specific.max_covariate_effect
        # set_mulcov_bound(db, , covariate_id)
        set_mulcov_bound(db, covariate_id, max_covariate_effect = 2)

dm = DismodIO('/Users/gma/ihme/epi/at_cascade/t1_diabetes/no_ode/no_ode.db')

system(f'dismod_at {temp_file} set option meas_noise_effect add_var_scale_none')
db.data = drop_ODE_integrands(db)[db.data.columns]

system(f'dismod_at {temp_file} init')
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

system(f'dismod_at {temp_file} set option zero_sum_child_rate "iota rho chi"')
system(f'dismod_at {temp_file} set option tolerance_fixed 1e-8')
system(f'dismod_at {temp_file} set option bound_random 3')


if 0:
    print ('Are the two var tables alike?')
    print ((db.var.fillna(-1) == dm.var.fillna(-1)).all())

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

system(f'dismod_at {temp_file} init')
system(f'dismod_at {temp_file} fit both')



"""

check_last_fit(db)

# Consider dropping mtexcess if there is sufficient mtspecific
data = data_original.copy().merge(db.integrand, how='left')
count = sum(data.integrand_name == 'mtspecific')
if count > enough_mtspecific:
    data.loc[data.integrand_name == 'mtexcess', 'hold_out'] = 1
    print (f'Holding out mtexcess because there are sufficient mtspecific data ({count} rows).')

data = compress_age_time_intervals(data)
db.data = data[db.data.columns]
system(f'dismod_at {temp_file} set start_var fit_var')
system(f'dismod_at {temp_file} fit both')
check_last_fit(db)

"""
