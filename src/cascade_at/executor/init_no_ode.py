import sys
import shutil
import subprocess
import copy
import numpy as np
import random
import time
from pathlib import Path

"""
If there is mtspecific, hold out mtexcess
Set the mulcov bounds
Check convergence
Check prediction
"""


sys.path.append('/Users/gma/Projects/IHME/GIT/cascade-at/src')
from cascade_at.dismod.api.dismod_io import DismodIO

def system (command) :
    # flush python's pending standard output in case this command generates more standard output
    sys.stdout.flush()
    print (command)
    if isinstance(command, str):
        command = command.split()
    #
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
    no_ode_integrands = ['Sincidence', 'remission', 'mtexcess', 'mtother']
    ode_integrands = set(db.integrand.integrand_name) - set(no_ode_integrands)
    data = db.data.merge(db.integrand, how='left')
    keep_mask = (data.integrand_name.isin(no_ode_integrands) & (data.hold_out == 0)).values
    data.loc[~keep_mask, 'hold_out'] = 1
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
    smooth_name    = f'{integrand_name}_noise_{new_smooth_id}'
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

original_file =  '/Users/gma/ihme/epi/at_cascade/data/475588/dbs/100/3/dismod.db'
original_file =  '/Users/gma/ihme/epi/at_cascade/data/475645/dbs/100/2/dismod.db'
temp_file = '/tmp/temp.db'
shutil.copy2(original_file, temp_file)

# seed used to randomly subsample data
random_seed = 0
if random_seed == 0 :
    random_seed = int( time.time() )
random.seed(random_seed)
msg = '\nrandom_seed  = ' + str( random_seed )
print(msg)

db = DismodIO(Path(temp_file))
data_original = db.data
mask = db.integrand.integrand_id.isin(db.data.integrand_id.unique())
data_integrands = sorted(set(db.integrand.loc[mask, 'integrand_name']) - set(['mtall', 'mtother']))

check_ones_covariate(db)
db.covariate = fix_ones_covariate_reference(db.covariate)
db.data = hold_out_ODE_integrands(db)[db.data.columns]

db.data = compress_age_time_intervals(db.data)

# add measurement noise covariates
group_id = 0
factor   = { 'lower':1e-1, 'mean':1e-1, 'upper':1e-1 }
for integrand in data_integrands:
    add_meas_noise_mulcov(db, integrand, group_id, factor)

system(f'dismod_at {temp_file} set option meas_noise_effect add_var_scale_none')
system(f'dismod_at {temp_file} init')
system(f'dismod_at {temp_file} fit both')
data = data_original
data = compress_age_time_intervals(data)
data = data.merge(db.integrand, how='left')
integrand_id = int(db.integrand.loc[db.integrand.integrand_name == 'mtexcess', 'integrand_id'])
data.loc[data.integrand_id == integrand_id, 'hold_out'] = 1
if 0:
    for integrand in data_integrands:
        data = random_subsample_data(db, integrand, max_sample = 500)
db.data = data[db.data.columns]
system(f'dismod_at {temp_file} set start_var fit_var')
system(f'dismod_at {temp_file} fit both')

