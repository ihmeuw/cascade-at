import sys
import shutil
import subprocess
import copy
import numpy as np
import pandas as pd
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

# ============================================================================
# Utilities that use global data tables but do not modify them
# ============================================================================

# def random_subsample_data_new(db, integrand_name, max_sample) :
#     # for a specified integrand, sample at most max_sample entries.
#     # This does random sampling that can be seeded by calling random.seed.
#     # The origianl order of the data is preserved (in index plots)
#     # by sorting the subsample.
#     data = db.data
#     integrand_id = int(db.integrand.loc[db.integrand.integrand_name == integrand_name, 'integrand_id'])
#     mask = data.integrand_id == integrand_id
#     index = list(range(len(mask)))
#     if sum(mask) > max_sample:
#         keep_ids = sorted(random.sample(index, max_sample))
#         keep = mask.index.isin(keep_ids)
#         data = data[~mask | keep]
#         print (f'Subsampled {integrand_name} to {sum(keep)} samples.')
#     return data

# -----------------------------------------------------------------------------

class FitNoODE(DismodIO):
    def __init__(db, *args, **kwds):
        super().__init__(*args, **kwds)

    def relative_covariate(db, covariate_id) :
        column_name = 'x_{}'.format(covariate_id)
        # sex is an absolute covariate and has 3 values, -0.5, 0.0, +0.5
        return len(set(db.data[column_name])) > 3

    def get_integrand_list (db, ode) :
        # If ode is true (false) get list of integrands that require
        # (do not require) the ode to model.
        integrand_model_uses_ode = {'prevalence', 'Tincidence', 'mtspecific', 'mtall', 'mtstandard'}
        data = db.data.merge(db.integrand, how='left')
        if ode:
            integrand_list = sorted(set(data.integrand_name.unique()).intersection(integrand_model_uses_ode))
        else:
            integrand_list = sorted(set(data.integrand_name.unique()) - integrand_model_uses_ode)
        return integrand_list

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
        db.data = data

    def random_subsample_data(db, integrand_name, max_sample) :
        # for a specified integrand, sample at most max_sample entries.
        # This does random sampling that can be seeded by calling random.seed.
        # The origianl order of the data is preserved (in index plots)
        # by sorting the subsample.
        #
        data = db.data.merge(db.integrand, how='left')
        table_in = data
        #
        # indices for this integrand
        count = 0
        count_list = []
        for i,row in table_in.iterrows() :
            if row['integrand_name'] == integrand_name :
                count_list.append(count)
                count += 1
        n_sample_in = count
        #
        # subsample of indices for this integrand
        n_sample_out = min(max_sample, n_sample_in)
        if n_sample_out < n_sample_in :
            count_list = random.sample(count_list,  n_sample_out)
            count_list = sorted( count_list )
        #
        # subsample the integrand
        index  = 0
        count  = 0
        table_out = []
        for i,row in table_in.iterrows() :
            if row['integrand_name'] != integrand_name :
                table_out.append(dict(row))
            else :
                if index < n_sample_out :
                    if count_list[index] == count :
                        table_out.append(dict(row))
                        index += 1
                count += 1
        assert index == n_sample_out
        assert count == n_sample_in

        msg  = '\nrandom_subsample_data\n'
        msg += 'number of {} samples: in = {} out = {}'
        print( msg.format(integrand_name, n_sample_in, n_sample_out) )
        data = pd.DataFrame(table_out)[db.data.columns]
        data['data_id'] = data.index
        db.data = data

    def hold_out_data (db, integrand_names=(), node_names=(), hold_out=False) :
        if isinstance(integrand_names, str):
            integrand_names = [integrand_names]
        if isinstance(node_names, str):
            node_names = [node_names]
        data = db.data.merge(db.integrand).merge(db.node)
        mask = [True]*len(data)
        if integrand_names:
            mask &= data.integrand_name.isin(integrand_names)
        if node_names:
            mask &= data.node_name.isin(node_names)
        print (f"Setting hold_out = {hold_out} for integrand {integrand_names}, node {node_names}")
        data.loc[mask, 'hold_out'] = hold_out
        db.data = data[db.data.columns]

    def set_data_likelihood (db, integrand_name, density_name, factor_eta=None, nu=None):
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
            print( msg )
        
        data = db.data.merge(db.integrand, how='left')
        density_id = int(db.density.loc[db.density.density_name == density_name, 'density_id'])
        #
        mask = data.integrand_name == integrand_name
        if factor_eta is None :
            eta = None
        else :
            median = data[mask].median
            eta = factor_eta * median
        data.loc[mask, 'density_id'] = density_id
        data.loc[mask, 'eta'] = eta
        data.loc[mask, 'nu'] = nu
        db.data = data[db.data.columns]

    def compress_age_time_intervals(db, age_size = 10.0, time_size = 10.0):
        data = db.data
        mask = (data.age_upper - data.age_lower) <= age_size
        mean = data[['age_lower', 'age_upper']].mean(axis=1)
        data.loc[mask, 'age_lower'] = data.loc[mask, 'age_upper'] = mean[mask]
        mask = (data.time_upper - data.time_lower) <= time_size
        mean = data[['time_lower', 'time_upper']].mean(axis=1)
        data.loc[mask, 'time_lower'] = data.loc[mask, 'time_upper'] = mean[mask]
        print ('compress_age_time_intervals -- all integrands')
        print ('Use midpoint for intervals less than or equal specified size')
        db.data = data[db.data.columns]

    # ============================================================================
    # Routines that Change Other Tables
    # ============================================================================

    def new_smoothing(db, integrand_name, age_grid, time_grid, value_prior, dage_prior, dtime_prior):
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
        new_row = copy.copy( smooth[smooth.smooth_id == smooth_id] )
        new_row['smooth_name'] = f'zero_smoothing #{new_smooth_id}'
        smooth = smooth.append(new_row).reset_index(drop=True)
        smooth['smooth_id'] = smooth.index
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
                smooth_grid = smooth_grid.append( new_row ).reset_index(drop=True)
        smooth_grid['smooth_grid_id'] = smooth_grid.index
        db.smooth = smooth
        db.smooth_grid = smooth_grid

    def new_bounded_smooth_id (db, smooth_id, lower, upper, density_name = 'uniform', smooth_name = '') :
        # add a new smoothing that has the same grid as smooth_id smoothing
        # and that constrains value to be within the specified lower and upper
        # bounds.The prior, smooth and smooth_grid tables are modified but
        # they are not written out. The lower and upper bounds can be None.

        def mean_from_limits(lower, upper):
            if lower is None and upper is None :
                mean = 0.0
            elif lower is not None and upper is not None :
                mean = (lower + upper) / 2.0
            elif lower is None:
                mean = 0.0 if upper >= 0.0 else upper
            elif upper is None:
                mean = 0.0 if lower <= 0.0 else lower
            else:
                raise Exception ('Tests failed')
            return mean

        smooth_table = db.smooth
        smooth_grid_table = db.smooth_grid
        prior_table = db.prior
        if smooth_id is None :
            return None
        #
        mean = mean_from_limits(lower, upper)
        #
        # smooth_table
        new_smooth_id = len(smooth_table)
        new_row = copy.copy( smooth_table.loc[smooth_id] )
        new_row['smooth_id'] = new_smooth_id
        new_row['smooth_name'] = f'{smooth_name}bound_smoothing_' + str( new_smooth_id )
        smooth_table = smooth_table.append( new_row )
        #
        new_prior_id  = len(prior_table)
        density_id    = int(db.density.loc[db.density.density_name == density_name, 'density_id'])
        std = np.nan if density_name == 'uniform' else np.sqrt(1/12)*(upper-lower)
        value_prior  = {
            'prior_name' : 'smoothing_{}_centered_prior'.format(new_smooth_id),
            'prior_id'   : new_prior_id,
            'density_id' : density_id,
            'lower'      : lower,
            'upper'      : upper,
            'mean'       : mean,
            'std'        : std,
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
        # corresponding absolute effect is bounded by
        # disease_specific_max_covariate_effect.
        # Noise covariate multipliers are not included.
        # The bounds for an integerand are set to zero if the covariate
        # is identically equalt the reference for that integrand.
        assert max_covariate_effect >= 0.0, 'disease specific max_covariate_effect is negative'

        data = db.data
        covariate = db.covariate
        mulcov = db.mulcov
        #
        # difference_dict = covariate minus reference
        covariate_name = f'x_{covariate_id}'
        reference = float(covariate.loc[covariate_id, 'reference'])
        difference_dict = {integrand_id: (data.loc[(data.integrand_id == integrand_id) & ~data[covariate_name].isna(), covariate_name] - reference).values
                           for integrand_id in data.integrand_id.unique()}
        if data[covariate_name].notna().any():
            difference_dict[covariate_name] = data[covariate_name] - reference
        #
        # lower_dict and  upper_dict
        lower_dict = dict()
        upper_dict = dict()
        for integrand_id in difference_dict :
            #
            # maximum and minimum difference
            min_difference = min(difference_dict[integrand_id])
            max_difference = max(difference_dict[integrand_id])
            #
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
            lower_dict[integrand_id] = lower
            upper_dict[integrand_id] = upper
        #

        for i,row in mulcov.iterrows() :
            if row['covariate_id'] == covariate_id :
                integrand_id = row['integrand_id']
                if integrand_id in difference_dict :
                    lower = lower_dict[integrand_id]
                    upper = upper_dict[integrand_id]
                    assert row['mulcov_type'] != 'rate_value'
                elif integrand_id is not None and np.isfinite(integrand_id):
                    lower = 0.0
                    upper = 0.0
                    assert row['mulcov_type'] != 'rate_value'
                else :
                    assert row['mulcov_type'] == 'rate_value'
                    column_name = 'x_{}'.format(covariate_id)
                    if column_name in difference_dict :
                        lower = lower_dict[column_name]
                        upper = upper_dict[column_name]
                    else :
                        lower = 0.0
                        upper = 0.0
                if row['mulcov_type'] != 'meas_noise' :
                    group_smooth_id = row['group_smooth_id']
                    group_smooth_id = db.new_bounded_smooth_id(
                        group_smooth_id, lower, upper, 'uniform'
                    )
                    row['group_smooth_id'] = group_smooth_id
                    #
                    subgroup_smooth_id = row['subgroup_smooth_id']
                    subgroup_smooth_id = db.new_bounded_smooth_id(
                        subgroup_smooth_id, lower, upper, 'uniform'
                    )
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
        density = db.density

        tst = (0.0 <= factor['lower'] <= factor['mean'] <= factor['upper'])
        assert tst, 'Factor is not monotonically increasing.'

        mask = (subgroup.group_id == group_id).values
        group_name = subgroup.loc[mask, 'group_name'].squeeze()
        assert group_name or not group_name.empty, 'Group name error for group_id = {group_id}' 

        # This covariate_id is the identically_one_covariate
        covariate_id = int(covariate.loc[covariate.c_covariate_name == 's_one', 'covariate_id'])
        integrand_id = int(integrand.loc[integrand.integrand_name == integrand_name, 'integrand_id'])
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
        density_id = int(density.loc[density.density_name == 'uniform', 'density_id'])
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

        smooth_id, smooth, smooth_grid, prior = db.new_smoothing(integrand_name,
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
                           
    def check_last_command(db, which_fit) :
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

    def set_avgint(db,
            covariate_integrand_list, predict_integrand_list, directory, which_fit
        ) :
        # -----------------------------------------------------------------------
        # create avgint table
        # For each covariate_integrand
        #    For data row corresponding to this covariate_integrand
        #        For each predict_intgrand
        #            write a row with specified covariates for predict_integrand
        #-----------------------------------------------------------------------
        #
        covariate_id_list = db.integrand.loc[db.integrand.integrand_name.isin(covariate_integrand_list),
                                             'integrand_id'].tolist()
        predict_id_list = db.integrand.loc[db.integrand.integrand_name.isin(predict_integrand_list),
                                             'integrand_id'].tolist()

        db.avgint = pd.DataFrame()
        avgint_cols = db.avgint.columns.tolist() + db.covariate.covariate_name.tolist()
        data = db.data[db.data.data_id.isin(db.data_subset.data_id)]
        data = data[data.integrand_id.isin(covariate_id_list)]
        data['avgint_id'] = data['data_id']

        covariate_data = data[data.integrand_id.isin(covariate_id_list)]
        avgint = pd.DataFrame()
        for integrand_id in predict_id_list:
            tmp = data.copy()
            tmp['integrand_id'] = integrand_id
            avgint = avgint.append(tmp)
        avgint = avgint.sort_values(by=['integrand_id', 'data_id'])[avgint_cols]
        avgint = avgint.reset_index(drop=True)
        avgint['avgint_id'] = avgint.index
        db.avgint = avgint
                           
    def setup_for_no_ode(db, max_covariate_effect):

        # seed used to randomly subsample data
        random_seed = 123
        if random_seed == 0 :
            random_seed = int( time.time() )
        random.seed(random_seed)
        msg = '\nrandom_seed  = ' + str( random_seed )
        print(msg)

        db.subset_data()

        # subsetting the data can remove some integrands
        no_ode_integrands = db.get_integrand_list(False)
        yes_ode_integrands = db.get_integrand_list(True)
        integrands = yes_ode_integrands + no_ode_integrands
        predict_integrands   = [ 'susceptible', 'withC' ]
        data_integrands = sorted(set(integrands) - set(['mtall', 'mtother']))
        msg = '\nintegrands   = ' + str( integrands )
        print(msg)

        reference_name  = 'median'
        for covariate_id in range( len(db.covariate) ) :
            if db.relative_covariate(covariate_id):
                db.set_covariate_reference(covariate_id)

        for integrand in integrands:
            db.random_subsample_data(integrand, max_sample = 1000)

        iota_zero = not np.isfinite(db.rate.loc[db.rate.rate_name == 'iota', 'parent_smooth_id']).squeeze()
        rho_zero = not np.isfinite(db.rate.loc[db.rate.rate_name == 'rho', 'parent_smooth_id']).squeeze()
        chi_zero = not np.isfinite(db.rate.loc[db.rate.rate_name == 'chi', 'parent_smooth_id']).squeeze()
        rate_case = ('iota_zero' if iota_zero else 'iota_pos') + '_' + ('rho_zero' if rho_zero else 'rho_pos')

        db.set_option('tolerance_fixed', '1e-8')
        db.set_option('max_num_iter_fixed', max_num_iter_fixed)
        db.set_option('quasi_fixed', 'false')
        db.set_option('zero_sum_child_rate', '"iota rho chi"')
        db.set_option('bound_random', '3')
        db.set_option('meas_noise_effect', 'add_var_scale_none')
        db.set_option('rate_case', rate_case)

        # add measurement noise covariates
        group_id = 0
        factor   = { 'lower':1e-1, 'mean':1e-1, 'upper':1e-1 }
        for integrand in integrands:
            db.add_meas_noise_mulcov(integrand, group_id, factor)

        db.compress_age_time_intervals()

        # set bounds for all the covariates
        for covariate_id in db.covariate.covariate_id.values:
            db.set_mulcov_bound(covariate_id, max_covariate_effect = max_covariate_effect)

        db.hold_out_data(integrand_names = yes_ode_integrands, hold_out=1)

        if 0:
            dmv = pd.read_csv('/Users/gma/ihme/epi/at_cascade/t1_diabetes/no_ode/variable.csv')
            os.system(f'dismodat.py {db.path} db2csv')
            dbv = pd.read_csv('/tmp/variable.csv')

        db.set_avgint(yes_ode_integrands, predict_integrands, db.path.parent, 'no_ode')


if __name__ == '__main__':

    def check_ones_covariate(db):
        data = db.data
        integrand = db.integrand
        covariate = db.covariate
        mask = covariate.c_covariate_name == 's_one'
        one_covariate_id, one_covariate_name  = covariate.loc[mask, ['covariate_id', 'covariate_name']].squeeze()
        assert (data[one_covariate_name] == 1).all(), f"Error in data table values for covariate:\n {covariate[mask]}"

    def fix_ones_covariate_reference(db):
        # Make sure the ones covariate reference is correct
        covariate = db.covariate
        mask = covariate.c_covariate_name == 's_one'
        if covariate.loc[mask, 'reference'].values == 1:
            covariate.loc[mask, 'reference'] = 0
            print (f"Fixed the 'one' covariate reference:\n{covariate[mask]}")
        db.covariate = covariate

    def check_data(db0, db1):
        db0_data = db0.data # .merge(db0.integrand, how='left')
        db1_data = db1.data # .merge(db1.integrand, how='left')
        if 0:
            for i in sorted(db0_data.integrand_name.unique()):
                print ('db0', i, len(db0_data[(db0_data.integrand_name == i) & (db0_data.hold_out == 0)]))
                print ('db1', i, len(db1_data[(db1_data.integrand_name == i) & (db1_data.hold_out == 0)]))
                print ((db0_data[(db0_data.integrand_name == i) & (db0_data.hold_out == 0)].fillna(-1) ==
                        db1_data[(db1_data.integrand_name == i) & (db1_data.hold_out == 0)].fillna(-1)).all())
            print ((db0.data.fillna(-1) == db1.data.fillna(-1)).all())
        tol = {'atol': 1e-8, 'rtol': 1e-10}

        mask0 = (db0.data.fillna(-1) != db1.data.fillna(-1)).any(1).values
        mask1 = (db0.data.fillna(-1) != db1.data.fillna(-1)).any(0).values
        assert np.allclose(db0.data.loc[mask0, mask1], db1.data.loc[mask0, mask1], **tol), 'ERROR: Data does not match'
        print ('Data tables agree')

    def check_var(db0, db1):
        def var_values(db):
            drop = ['smooth_id', 'parent_smooth_id', 'child_smooth_id', 'smooth_grid_id', 
                    'value_prior_id', 'dage_prior_id', 'dtime_prior_id', 'child_nslist_id',
                    'rate_id', 'integrand_id', 'age_id', 'time_id', 'group_smooth_id']
            v = (db.var.fillna(-1)
                 .merge(db.smooth_grid, how='left')
                 .merge(db.mulcov.fillna(-1), how='left')
                 .merge(db.covariate, how='left')
                 .merge(db.age, how='left')
                 .merge(db.time, how='left')
                 .merge(db.rate, how='left')
                 .merge(db.integrand, how='left')
                 .merge(db.node, how='left')
                 .merge(db.subgroup, on='group_id', how='left')
                 .merge(db.start_var, how='left', left_on = 'var_id', right_on = 'start_var_id')
                 .merge(db.scale_var, how='left', left_on = 'var_id', right_on = 'scale_var_id')
                 .merge(db.prior, left_on = 'value_prior_id', right_on = 'prior_id')
                 .fillna(-1).drop(columns=drop))
            return v
        d0 = var_values(db0)
        d1 = var_values(db1)
        vd0 = d0.drop(columns = ['prior_id', 'prior_name'])
        vd1 = d1.drop(columns = ['prior_id', 'prior_name'])
        mask0 = (vd0 != vd1).any(axis=1).values
        mask1 = (vd0 != vd1).any(axis=0).values
        if not mask1.any():
            print ('Var tables agree')
        else:
            print ("fit_ihme (Brad's):")
            print (vd1.loc[mask0, mask1].merge(d1[['var_id', 'prior_id', 'prior_name']], left_index = True, right_index = True))
            print ('init_no_ode:')
            print (vd0.loc[mask0, mask1].merge(d0[['var_id', 'prior_id', 'prior_name']], left_index = True, right_index = True))
            print ('ERROR -- var tables do not agree')
    max_num_iter_fixed = 50

    def main(case):
        crohns = '/Users/gma/ihme/epi/at_cascade/data/475533/dbs/1/2/dismod.db'
        dialysis = '/Users/gma/ihme/epi/at_cascade/data/475527/dbs/96/2/dismod.db'
        kidney = '/Users/gma/ihme/epi/at_cascade/data/475718/dbs/70/1/dismod.db'
        osteo_hip =  '/Users/gma/ihme/epi/at_cascade/data/475526/dbs/1/2/dismod.db'
        osteo_knee = '/Users/gma/ihme/epi/at_cascade/data/475746/dbs/64/2/dismod.db'
        t1_diabetes =  '/Users/gma/ihme/epi/at_cascade/data/475882/dbs/100/2/dismod.db'
        # t1_diabetes =  '/Users/gma/ihme/epi/at_cascade/data/475588/dbs/100/3/dismod.db'

        if case == 't1_diabetes':
            original_file = t1_diabetes
            max_covariate_effect = 2
        elif case == 'crohns':
            original_file = crohns
            max_covariate_effect = 4
        elif case == 'dialysis':
            original_file = dialysis
            max_covariate_effect = 4
        elif case == 'kidney':
            original_file = kidney
            max_covariate_effect = 4
        elif case == 'osteo_hip':
            original_file = osteo_hip
            max_covariate_effect = 4
        elif case == 'osteo_knee':
            original_file = osteo_knee
            max_covariate_effect = 4
        else:
            raise Exception(f'Disease {case} not found')

        dm_no_ode = DismodIO(f'/Users/gma/ihme/epi/at_cascade/{case}/no_ode/no_ode.db')
        dm_yes_ode = DismodIO(f'/Users/gma/ihme/epi/at_cascade/{case}/yes_ode/yes_ode.db')
                           
        dm = dm_no_ode

        assert os.path.exists(original_file)
        temp_file = '/tmp/temp.db'
        shutil.copy2(original_file, temp_file)
        db = FitNoODE(Path(temp_file))
        
        check_ones_covariate(db)
        fix_ones_covariate_reference(db)

        if 0:
            def rate_mulcov_priors(db):
                prior_ids = db.smooth_grid.loc[db.smooth_grid.smooth_id.isin(db.var[-2:].smooth_id),
                                               'value_prior_id']
                return db.prior[db.prior.prior_id.isin(prior_ids)]

            print (rate_mulcov_priors(DismodIO(original_file)))
            print (rate_mulcov_priors(DismodIO('/Users/gma/ihme/epi/at_cascade/t1_diabetes/no_ode/no_ode.db')))



        db.setup_for_no_ode(max_covariate_effect)

        system(f'dismod_at {temp_file} init')
        check_data(db, dm)
        check_var(db, dm)
        system(f'dismod_at {temp_file} fit both')
        system(f'dismod_at {temp_file} predict fit_var')
        db.check_last_command('no_ode')


for d in ['t1_diabetes','crohns','dialysis','kidney','osteo_hip','osteo_knee']:
    print ('>>>', d)
    main(d)

if 0:

    if 1:
        print ('Are the two var tables alike?')
        print ((db.var.fillna(-1) == dm.var.fillna(-1)).all())
        print ((db.var.fillna(-1)[:-3] == dm.var.fillna(-1)[:-3]).all())
        print ('Are the two data tables alike?')
        print ((db.data.fillna(-1) == dm.data.fillna(-1)).all())
        print ('Are the two mulcov tables alike?')
        print ((db.mulcov.fillna(-1) == dm.mulcov.fillna(-1)).all())


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
    db.hold_out_data(integrand_names = hold_outs, hold_out=1)

    db.data = compress_age_time_intervals(db)

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



    print ("""BRADS RESULT:
    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
       0  1.1042442e+03 9.30e-04 2.07e+03  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
    Warning: Cutting back alpha due to evaluation error
       1  2.8986326e+02 7.93e-04 1.06e+03  -1.0 4.05e+00    -  4.98e-01 4.89e-01f  2""")



    system(f'dismod_at {temp_file} set start_var fit_var')
    if 1:
        dm = dmy
        system(f'dismod_at {temp_file} fit both')
        system(f'dismod_at {temp_file} sample asymptotic both 10')

        check_last_command(db)


        if 0:
            print ('Are the two var tables alike?')
            print ((db.var.fillna(-1) == dm.var.fillna(-1)).all())
            assert np.all((db.var.fillna(-1) == dm.var.fillna(-1)))
            print ('Are the two data tables alike?')
            print ((db.data.fillna(-1) == dm.data.fillna(-1)).all())
            assert np.all((db.data.fillna(-1) == dm.data.fillna(-1)))
            print ('Are the two mulcov tables alike?')
            print ((db.mulcov.fillna(-1) == dm.mulcov.fillna(-1)).all())
            assert np.all((db.mulcov.fillna(-1) == dm.mulcov.fillna(-1)))

    data['density_id'] = 3
    data['nu'] = 5
    db.data = data[db.data.columns]
    system(f'dismod_at {temp_file} set start_var fit_var')
    system(f'dismod_at {temp_file} fit both')
    check_last_command(db)


    os.system(f'DB_plot.py {temp_file} -v 475882')

"""
sys.path.append('/opt/prefix/dismod_at/lib/python3.8/site-packages')
from dismod_at.ihme.t1_diabetes import relative_path
brad = DismodIO(Path('/Users/gma/ihme/epi/at_cascade') / relative_path)
brad2 = DismodIO('/Users/gma/ihme/epi/at_cascade/t1_diabetes/no_ode/no_ode.db')
for iid in brad.data.integrand_id.unique():
    print (brad.path, iid, len(brad.data[brad.data.integrand_id == iid]))
    print (brad2.path, iid, len(brad2.data[brad2.data.integrand_id == iid]))

    print (original_file, iid, len(db.data[db.data.integrand_id == iid]))

"""
