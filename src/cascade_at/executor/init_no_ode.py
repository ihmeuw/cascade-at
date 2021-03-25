import sys
import os
import shutil
import subprocess
import copy
import numpy as np
import pandas as pd
import random
import time
import tempfile
from pathlib import Path

"""
If there is mtspecific, hold out mtexcess on the ode fit.
Set the mulcov bounds
Check convergence
Check prediction
"""


sys.path.append('/Users/gma/Projects/IHME/GIT/cascade-at/src')
from cascade_at.dismod.api.dismod_io import DismodIO

db = None
_dm_no_ode_ = None
_dm_yes_ode_ = None
_dm_students_ = None 

__run_fit_ihme__ = True
_use_single_database_ = True
_fit_ihme_py_ = 'fit_ihme.py'

def null(*args, **kwds): pass

def system (command) :
    # flush python's pending standard output in case this command generates more standard output
    sys.stdout.flush()
    print (command)
    if isinstance(command, str):
        command = command.split()
    run = subprocess.run(command)
    if run.returncode != 0 :
        raise Exception(f'"{command}" failed.')

class FitNoODE(DismodIO):
    def __init__(db, *args, ode_hold_out_list = (), **kwds):
        if 'dismod' in kwds:
            db.dismod = kwds.pop('dismod')
        else:
            db.dismod = 'dismod_at'
        super().__init__(*args, **kwds)
        db.predict_integrands   = [ 'susceptible', 'withC' ]
        db.enough_mtspecific = 100
        db.input_data = db.data

        db.ode_hold_out_list = ode_hold_out_list
        db.set_integrand_lists()
        msg = '\nInitial integrands   = ' + str( db.integrands )
        print(msg)


    # ============================================================================
    # Utilities that use database tables but do not modify them
    # ============================================================================

    def relative_covariate(db, covariate_id) :
        column_name = 'x_{}'.format(covariate_id)
        # sex is an absolute covariate and has 3 values, -0.5, 0.0, +0.5
        # one is an absolute covariate and has perhaps 2 values, 0.0, 1.0
        # it is reasonable to assume that a covariate with more than 3 covariate values is relative
        return len(set(db.data[column_name])) > 3

    def set_integrand_lists (db) :
        # If ode is true (false) get list of integrands that require
        # (do not require) the ode to model.
        integrand_model_uses_ode = {'prevalence', 'Tincidence', 'mtspecific', 'mtall', 'mtstandard'}
        data = db.data.merge(db.integrand, how='left')
        integrands = [n for n in data.integrand_name.unique()
                      if np.any(data.loc[data.integrand_name == n, 'hold_out'].values == 0).any()]
        db.yes_ode_integrands = sorted(set(integrands).intersection(integrand_model_uses_ode))
        db.no_ode_integrands = sorted(set(integrands) - integrand_model_uses_ode)
        db.integrands = db.yes_ode_integrands + db.no_ode_integrands

    def get_rate_case(db):
        iota_zero = not np.isfinite(db.rate.loc[db.rate.rate_name == 'iota', 'parent_smooth_id']).squeeze()
        rho_zero = not np.isfinite(db.rate.loc[db.rate.rate_name == 'rho', 'parent_smooth_id']).squeeze()
        chi_zero = not np.isfinite(db.rate.loc[db.rate.rate_name == 'chi', 'parent_smooth_id']).squeeze()
        rate_case = ('iota_zero' if iota_zero else 'iota_pos') + '_' + ('rho_zero' if rho_zero else 'rho_pos')
        return rate_case

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

        # subsetting the data can remove some integrands, so get integrands after
        db.set_integrand_lists()
        msg = '\nintegrands   = ' + str( db.integrands )
        print(msg)


    def random_subsample_data(db, integrand_name, max_sample) :
        # for a specified integrand, sample at most max_sample entries.
        # This does random sampling that can be seeded by calling random.seed.
        # The origianl order of the data is preserved (in index plots)
        # by sorting the subsample.
        #
        # This code may seem a little obtuse, but for comparison, it matches Brad's sampling method
        #
        data_in = db.data.merge(db.integrand, how='left')
        integrand = data_in[data_in.integrand_name == integrand_name]

        n_sample_in = len(integrand)
        n_sample_out = min(max_sample, len(integrand))
        print (f'random_subsample_data')
        print (f'number of {integrand_name} samples: in = {n_sample_in} out = {n_sample_out}')

        # Dataframe indices of integrands other than the one being sampled
        non_integrand_indices = data_in.index[data_in.integrand_name != integrand_name].tolist()

        # Note: A preferred, direct integrand sampling (e.g., integrand.sample(n_sample_out)) didn't match Brad's sampling
        # Sample the integrand dataframe row index
        row_index = list(range(len(integrand)))
        if n_sample_out < n_sample_in :
            row_index = sorted(random.sample(range(len(integrand)),  n_sample_out))
        # Sample the dataframe by row index, and return the dataframe index
        integrand_indices = integrand.iloc[row_index].index.tolist()
        
        # Sample the database data by filtering on dataframe index
        data = data_in.loc[sorted(integrand_indices + non_integrand_indices)].reset_index(drop=True)
        data['data_id'] = data.index
        db.data = data[db.data.columns]

    def hold_out_data (db, integrand_names=(), node_names=(), hold_out=False) :
        if isinstance(integrand_names, str):
            integrand_names = [integrand_names]
        if isinstance(node_names, str):
            node_names = [node_names]
        data = db.data.merge(db.integrand).merge(db.node)
        mask = [False]*len(data)
        if integrand_names:
            mask |= data.integrand_name.isin(integrand_names)
        if node_names:
            mask |= data.node_name.isin(node_names)
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
        data = db.data.merge(db.integrand, how='left')
        #
        density_id = int(db.density.loc[db.density.density_name == density_name, 'density_id'])
        mask = data.integrand_name == integrand_name
        if factor_eta is None :
            eta = None
        else :
            median = data.meas_value[mask].median()
            eta = factor_eta * median
            data.loc[mask, 'density_id'] = density_id
            data.loc[mask, 'eta'] = eta
            data.loc[mask, 'nu'] = nu
            db.data = data[db.data.columns]

            msg += f'\n            = {median:6.4f} where m is the median of the {integrand_name} data'
            print( msg )

    def set_student_likelihoods(db, factor_eta = 1e-2, nu = 5):
        integrand_list = db.integrand.loc[db.data.integrand_id.unique(), 'integrand_name'].tolist()
        density_name   = 'log_students'
        factor_eta     = 1e-2
        nu             = 5
        for integrand_name in integrand_list :
            db.set_data_likelihood(integrand_name, density_name, factor_eta, nu)

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

    def get_option (db, name) :
        # Set option specified by name to its value where name and value are
        # strings. The routine system_command to prints the processing message
        # for this operation.
        option = db.option
        value = option.loc[option.option_name == name, 'option_value']
        if value.empty:
            return(None)
        return value.squeeze()

    def set_option (db, name, value) :
        # Set option specified by name to its value where name and value are
        # strings. The routine system_command to prints the processing message
        # for this operation.
        system(f'{db.dismod} {db.path} set option {name} {value}')

    def new_zero_smooth_id (db, smooth_id) :
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
        old_reference  = float(covariate.loc[covariate_id, 'reference'])
        new_reference = getattr(np, reference_name)(covariate_value)
        #
        print (777, len(db.data[~db.data.x_0.isna()]), db.data.x_0.median())
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
        difference_dict = {integrand_id:
                           (data.loc[(data.integrand_id == integrand_id) &
                                     ~data[covariate_name].isna(), covariate_name]
                            - reference).values
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

        for i,row in mulcov.iterrows() :
            if row['covariate_id'] == covariate_id :
                lower = - float('inf')
                upper = + float('inf')
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
                    covariate_name = 'x_{}'.format(covariate_id)
                    if covariate_name in difference_dict :
                        lower = lower_dict[covariate_name]
                        upper = upper_dict[covariate_name]
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
                #
                integrand_name = db.integrand.loc[db.integrand.integrand_id == integrand_id, 'integrand_name'].squeeze()
                msg  = '\nset_mulcov_bound\n'
                msg += 'integrand = {}, covariate = x_{}, max_covariate_effect = {}, '
                msg += 'lower = {:.5g}, upper = {:.5g}'
                msg  = msg.format(integrand_name, covariate_id, max_covariate_effect, lower, upper)
                print( msg )
        db.mulcov = mulcov

    def set_mulcov_value(db, covariate_name, rate_or_integrand_name, mulcov_value) :
        # Set the value for a specific covariate multiplier.
        # The corresponding multiplier must be in the covariate table.
        # Noise covariate multipliers are not included.
        #
        mulcov = (db.mulcov.merge(db.covariate, how='left').merge(db.rate, how='left').merge(db.integrand, how='left'))
        mask = ((mulcov.covariate_name == covariate_name)
                & (mulcov.rate_name == rate_or_integrand_name)
                & mulcov.mulcov_type.isin(['rate_value', 'meas_value']))
        assert mask.any(), f'Failed to find {covariate_name} for {rate_or_integrand_name} in mulcov table.'
        matches = mulcov[mask]
        for i, row in matches.iterrows():
            lower = upper = mulcov_value
            group_smooth_id = db.new_bounded_smooth_id(row.group_smooth_id, lower, upper)
            mulcov.loc[mulcov.mulcov_id == row.mulcov_id, 'group_smooth_id'] = group_smooth_id
            #
            subgroup_smooth_id = db.new_bounded_smooth_id(row.subgroup_smooth_id, lower, upper)
            mulcov.loc[mulcov.mulcov_id == row.mulcov_id, 'subgroup_smooth_id'] = subgroup_smooth_id
            print (f'\nset_mulcov_value')
            print (f'covariate = {covariate_name}, {row.mulcov_type}  = {rate_or_integrand_name}, value = {mulcov_value:.5g}')
        #
        db.mulcov = mulcov[db.mulcov.columns]

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

        smooth_id, smooth, smooth_grid, prior = db.new_smoothing(
            integrand_name, age_grid, time_grid, value_prior, dage_prior, dtime_prior )

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

    def check_last_command(db, command):
        log = db.log
        last_begin = [l for i,l in log.iterrows()
                      if l.message_type == 'command'
                      and l.message.startswith('begin ')]
        rtn = True
        if not last_begin:
            print (f"ERROR: Failed to find a 'begin' command.")
            rtn = False
        else:
            last_begin = last_begin[-1]
        if rtn:
            start_cmd = [l for i,l in log[last_begin.log_id:].iterrows()
                         if l.message_type == 'command'
                         and l.message.startswith(f'begin {command}')]
            if not start_cmd:
                print (f"ERROR: Expected 'begin {command}' but found '{last_begin.message}'.")
                rtn = False
            else:
                start_cmd = start_cmd[-1]
        if rtn:
            end_cmd = [l for i,l in log[start_cmd.log_id:].iterrows()
                       if l.message_type == 'command'
                       and l.message.startswith(f'end {command}')]
            if not end_cmd:
                print (f"ERROR: Did not find end for this '{start_cmd.message}' command")
                rtn = False
            for i,l in log[start_cmd.log_id:].iterrows():
                if l.message_type in ['error', 'warning']:
                    print (f"DISMOD {l.message_type}: {l.message.rstrip()}")
                    rtn = False
        if rtn:
            print (f"{db.dismod} {command} OK")
        else:
            print (f"ERROR: {db.dismod} {command} had errors, warnings, or failed to complete.")
        return rtn

    def set_avgint(db, covariate_integrand_list) :
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
        predict_id_list = db.integrand.loc[db.integrand.integrand_name.isin(db.predict_integrands),
                                           'integrand_id'].tolist()

        db.avgint = pd.DataFrame()

        # First access of an empty db.avgint does not have covariate names. 
        # Once it is initialized, it does. This is DismodIO weirdness.
        avgint_cols = db.avgint.columns.tolist()
        avgint_cols += sorted(set(db.covariate.covariate_name) - set(avgint_cols))

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

    def simplify_data(db, random_seed = None, subsample=False):
        # seed used to randomly subsample data
        if random_seed in [0, None]:
            random_seed = int( time.time() )
        random.seed(random_seed)
        msg = '\nrandom_seed  = ' + str( random_seed )
        print(msg)
        if subsample:
            db.subset_data() 
        for integrand in db.integrands:
            db.random_subsample_data(integrand, max_sample = 1000)
        db.compress_age_time_intervals()
 
    def setup_ode_fit(db, max_covariate_effect = 2,
                         ode_hold_out_list = [], mulcov_values = []):


        db.ode_hold_out_list = ode_hold_out_list
        db.check_ones_covariate()
        db.fix_ones_covariate_reference()

        rate_case = db.get_rate_case()

        db.set_option('tolerance_fixed', '1e-8')
        db.set_option('quasi_fixed', 'false')
        db.set_option('zero_sum_child_rate', '"iota rho chi"')
        db.set_option('bound_random', '3')
        db.set_option('meas_noise_effect', 'add_var_scale_none')
        db.set_option('rate_case', rate_case)

        reference_name  = 'median'
        for covariate_id in range( len(db.covariate) ) :
            if db.relative_covariate(covariate_id):
                db.set_covariate_reference(covariate_id)

        group_id = 0
        factor   = { 'lower':1e-1, 'mean':1e-1, 'upper':1e-1 }
        for integrand in db.integrands:
            db.add_meas_noise_mulcov(integrand, group_id, factor)

        # set bounds for all the covariates
        for covariate_id in db.covariate.covariate_id.values:
            db.set_mulcov_bound(covariate_id, max_covariate_effect = max_covariate_effect)
	#
	# Covariate multipliers that we are setting a specific value for
	# (this must be done after set_mulcov_bound).
        for [covariate_name, rate_or_integrand_name, mulcov_value] in mulcov_values :
            db.set_mulcov_value(covariate_name, rate_or_integrand_name, mulcov_value)
        db.set_avgint(db.integrands)

    def fit(db, msg = ''):
        t0 = time.time()
        system(f'{db.dismod} {db.path} fit both')
        print(f'{msg} time = {str(round(time.time() - t0))} seconds.')
        assert db.check_last_command('fit'), 'Exiting due to problems with this fit command'

    def ode_init(db, max_covariate_effect = 2, mulcov_values = [], ode_hold_out_list = []):

        db.simplify_data(random_seed = 123, subsample = False)

        db.setup_ode_fit(max_covariate_effect = max_covariate_effect,
                         mulcov_values = mulcov_values,
                         ode_hold_out_list = ode_hold_out_list)

        # Fit only a subsample the non-ODE integrands
        db.hold_out_data(integrand_names = db.yes_ode_integrands, hold_out=1)
        system(f'{db.dismod} {db.path} init')
        db.fit(msg = 'fit_no_ode')

        # Restore the original data
        db.data = db.input_data
        return db

    def check_ones_covariate(db):
        data = db.data
        integrand = db.integrand
        covariate = db.covariate
        mask = covariate.c_covariate_name == 's_one'
        one_covariate_id, one_covariate_name, reference  = covariate.loc[mask, ['covariate_id', 'covariate_name', 'reference']].squeeze()
        assert reference == 0, f'Error -- reference is not 0\n{covariate[mask]}'
        assert (data[one_covariate_name] == 1).all(), f"Error in data table values for covariate:\n {covariate[mask]}"

    def fix_ones_covariate_reference(db):
        # Make sure the ones covariate reference is correct
        covariate = db.covariate
        mask = covariate.c_covariate_name == 's_one'
        if covariate.loc[mask, 'reference'].values == 1:
            covariate.loc[mask, 'reference'] = 0
            print (f"Fixed the 'one' covariate reference:\n{covariate[mask]}")
        db.covariate = covariate

def setup_db(original_file, dismod = 'dismod_at', ode_hold_out_list = ()):
    assert os.path.exists(original_file)
    fd, path = tempfile.mkstemp(prefix = 'cascade_at', suffix = original_file.replace('/', '_'))
    shutil.copy2(original_file, path)
    db = FitNoODE(Path(path), dismod = dismod, ode_hold_out_list = ode_hold_out_list)
    print (f'Running fit ode commands on database {db.path}')
    return db

def init_ode_command(args):
    """
    1) Initialize the database for the non-ODE/ODE fitting strategy
    2) Hold out the ODE integrands
    3) Fit both on a subset of the integrands corresponding directly to the rates
       (e.g. Sincidence, chi and rho). Omega is always constrained.
    4) Restore the data table to it's original state
    """

    dismod, path, cmd, option = args[:4]

    db = setup_db(path, dismod = dismod)

    FIXME_ode_kwds = dict(max_covariate_effect = 2,
                          mulcov_values = [],
                          ode_hold_out_list = [])
    db.ode_init(**FIXME_ode_kwds)
    if not _use_single_database_:
        shutil.copy2(db.path, path)

def fit_ode_command(args):
    """
    1) Initialize the database for the non-ODE/ODE fitting strategy
    2) Hold out the ODE integrands
    3) Fit both on a subset of the integrands corresponding directly to the rates
       (e.g. Sincidence, chi and rho). Omega is always constrained.
    4) Restore the data table to it's original state
    """

    dismod, path, cmd, option = args[:4]

    db = setup_db(path, dismod = dismod)

    system(f'{db.dismod} {db.path} set start_var fit_var')

    db.fit(msg = 'fit_no_ode')

    if not _use_single_database_:
        shutil.copy2(db.path, path)

    
def fit_students_command(args, nu = 5):
    dismod, path, cmd, option = args[:4]

    db = setup_db(path, dismod = dismod)

    system(f'{db.dismod} {db.path} set start_var fit_var')

    db.set_student_likelihoods(factor_eta = 1e-2, nu = nu)

    db.fit(msg = 'fit_students')

    if not _use_single_database_:
        shutil.copy2(db.path, path)

def test_command():
    osteo_knee = '/opt/local/bin/dmdismod /Users/gma/ihme/epi/at_cascade/data/475746/dbs/64/2/dismod.db fit ode'
    osteo_hip = '/opt/local/bin/dmdismod /Users/gma/ihme/epi/at_cascade/data/475526/dbs/1/2/dismod.db fit ode'

    cmd = osteo_knee

    args = cmd.split()
    path = args[1]
    os.system(f'chmod u+w {path}')

    init_ode_command(args)

    fit_ode_command(args)
    
    fit_students_command(args)

"""

test_command()


"""
if __name__ == '__main__':

    def compare_dataframes(df0, df1):
        tol = {'atol': 1e-8, 'rtol': 1e-10}

        tmp = (df0.fillna(-1) != df1.fillna(-1))
        mask0 = tmp.any(1).values
        mask1 = tmp.any(0).values
        diff0 = df0.loc[mask0, mask1]
        diff1 = df1.loc[mask0, mask1]
        error = np.max(np.abs(diff0 - diff1))
        if not np.allclose(diff0, diff1, **tol):
            print (diff0)
            print (diff1)
            raise Exception('ERROR: dataframes do not match')
        msg = '' if error.empty else f' within tolerance, max(abs(error)) = {error}'
        return f'dataframes are equal{msg}.'

    def check_data(data0, data1):
        return compare_dataframes(data0, data1)

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
            raise Exception ('ERROR -- var tables do not agree')

    def test_cases(case):
        crohns = '/Users/gma/ihme/epi/at_cascade/data/475533/dbs/1/2/dismod.db'
        dialysis = '/Users/gma/ihme/epi/at_cascade/data/475527/dbs/96/2/dismod.db'
        kidney = '/Users/gma/ihme/epi/at_cascade/data/475718/dbs/70/1/dismod.db'
        osteo_hip =  '/Users/gma/ihme/epi/at_cascade/data/475526/dbs/1/2/dismod.db'
        osteo_hip_world = '/Users/gma/ihme/epi/at_cascade/data/475745/dbs/1/2/dismod.db'
        osteo_knee = '/Users/gma/ihme/epi/at_cascade/data/475746/dbs/64/2/dismod.db'
        t1_diabetes =  '/Users/gma/ihme/epi/at_cascade/data/475882/dbs/100/2/dismod.db'
        #t1_diabetes = '/Users/gma/ihme/epi/at_cascade/data/475588/dbs/100/3/dismod.db'
        #t1_diabetes = '/Users/gma/ihme/epi/at_cascade/data/475861/inputs/25079_parameters.json' # gma ran 3/3/2020 from a backup. All I could get was the json file with Brad's settings
        if case == 't1_diabetes':
            file_in = t1_diabetes
            max_covariate_effect = 2
            ode_hold_out_list = ['mtexcess']
            mulcov_values = []
        elif case == 'crohns':
            file_in = crohns
            max_covariate_effect = 2
            ode_hold_out_list = []
            mulcov_values = [[ 'x_0', 'iota', 3.8661 ]]
        elif case == 'dialysis':
            file_in = dialysis
            max_covariate_effect = 4
            ode_hold_out_list = []
            mulcov_values = []
        elif case == 'kidney':
            file_in = kidney
            max_covariate_effect = 2
            ode_hold_out_list = []
            mulcov_values = []
        elif case == 'osteo_hip':
            file_in = osteo_hip
            max_covariate_effect = 2
            ode_hold_out_list = []
            mulcov_values = []
        elif case == 'osteo_hip_world':
            file_in = osteo_hip_world
            max_covariate_effect = 2
            ode_hold_out_list = []
            mulcov_values = []
        elif case == 'osteo_knee':
            file_in = osteo_knee
            max_covariate_effect = 2
            ode_hold_out_list = []
            mulcov_values = []
        else:
            raise Exception(f'Disease {case} not found')

        if _use_single_database_:
            # Keep the original database unmodified
            path, ext = os.path.splitext(file_in)
            original_file = f'{path}_fitODE{ext}'
            shutil.copy2(file_in, original_file)
        else:
            original_file = file_in

        return original_file, max_covariate_effect, ode_hold_out_list, mulcov_values

    def test(case, original_file, max_covariate_effect, ode_hold_out_list, mulcov_values): 

        def fix_data_table(db, dm, bypass_hold_out = False):
            # For some reason, the fit_ihme.py data table is sometimes slightly different than the original
            # This causes divergence in the fit results
            if bypass_hold_out:
                hold_out = db.data.hold_out
            cols = db.data.columns.drop('hold_out')
            mask = (dm.data.fillna(-1) != db.data.fillna(-1))
            if bypass_hold_out:
                mask['hold_out'] = False
            mask0 = mask.any(1)
            data = db.data
            diff = np.max(np.abs(dm.data.values[mask] - data.values[mask]))
            assert diff < 1e-10, 'Error was too large'
            if np.any(mask):
                print (f'WARNING -- fixed {np.sum(mask.values)} slight differences max ({diff}) between fit_ihme and this data table.')
            data[mask0] = dm.data[mask0]
            if bypass_hold_out:
                data['hold_out'] = hold_out
            db.data = data
            assert np.all(dm.data[cols].fillna(-1) == db.data[cols].fillna(-1)) , 'Assignment in fix_data_table  failed'

        def check_input_tables(db, dm=None, check_hold_out = True):
            print (f'+++ db.path {db.path}')
            print (f'+++ dm.path {dm.path}')
            try:
                if check_hold_out:
                    print ('db.data', check_data(db.data, dm.data))
                else:
                    print ('db.data', check_data(db.data.drop(columns = 'hold_out'), dm.data.drop(columns = 'hold_out')))
                check_var(db, dm)
                print ('Check input tables OK')
            except Exception as ex:
                print ('\n\nERROR in inputs\n\n', ex)
                raise

        def check_output_tables(db, dm=None):
            print (f'+++ db.path {db.path}')
            print (f'+++ dm.path {dm.path}')
            try:
                dmv = pd.read_csv(dm.path.parent / 'variable.csv')
                os.system(f'dismodat.py {db.path} db2csv')
                dbv = pd.read_csv(db.path.parent / 'variable.csv')
                print ('variable.csv', compare_dataframes(dmv, dbv))
                print ('Check output tables OK')
            except Exception as ex:
                print ('\n\nERROR in output\n\n', ex)
                raise

        fit_ihme_path = f'/Users/gma/ihme/epi/at_cascade/{case}'
        global _dm_no_ode_, _dm_yes_ode_
        _dm_no_ode_ = DismodIO(f'{fit_ihme_path}/no_ode/no_ode.db')
        _dm_yes_ode_ = DismodIO(f'{fit_ihme_path}/yes_ode/yes_ode.db')
        _dm_students_ = DismodIO(f'{fit_ihme_path}/students/students.db')
        cascade_path = f'/Users/gma/ihme/epi/at_cascade/{case}/cascade'
        if not os.path.isdir(cascade_path):
            os.makedirs(cascade_path, exist_ok=True)
        path_no_ode = f'{fit_ihme_path}/cascade/no_ode.db'
        path_yes_ode = f'{fit_ihme_path}/cascade/yes_ode.db'
        path_students = f'{fit_ihme_path}/cascade/students.db'
        
        if 1:
            no_yes_ode = True
            no_ode = False
            yes_ode = False
        else:
            no_yes_ode = False
            no_ode = True
            yes_ode = True

        students = True

        subsample = True
        random_seed = 123
        __check__ = True

        kwds = dict(mulcov_values = mulcov_values, ode_hold_out_list = ode_hold_out_list)
        global db
        if no_yes_ode:
            if __check__:
                if __run_fit_ihme__:
                    cmd = f'{_fit_ihme_py_} ~/ihme/epi/at_cascade {case} no_ode {random_seed}'
                    print (cmd); os.system(cmd)
                    cmd = f'{_fit_ihme_py_} ~/ihme/epi/at_cascade {case} yes_ode'
                    print (cmd); os.system(cmd)

            print ('--- no_yes_ode ---')
            if _use_single_database_:
                db = FitNoODE(Path(original_file), ode_hold_out_list = ode_hold_out_list)
            else:
                db = setup_db(original_file)

            system(f'{db.dismod} {db.path} init')
            db.simplify_data(random_seed = random_seed, subsample = subsample)
            db.setup_ode_fit(max_covariate_effect, **kwds)
            db.hold_out_data(integrand_names = db.yes_ode_integrands, hold_out=1)

            if __check__ and case == 'crohns':
                fix_data_table(db, _dm_no_ode_)

            system(f'{db.dismod} {db.path} init')
            if __check__:
                check_input_tables(db, dm = _dm_no_ode_, check_hold_out = True)
            db.set_option('max_num_iter_fixed', 500)
            db.fit(msg = 'fit_no_ode')
            if __check__:
                check_output_tables(db, dm = _dm_no_ode_)

            shutil.copy2(db.path, path_no_ode)

            if _use_single_database_:
                db.data = db.input_data
            else:
                fit_var = db.fit_var
                db = setup_db(original_file)

            db.simplify_data(random_seed = random_seed, subsample = subsample)
            if not _use_single_database_:
                db.setup_ode_fit(max_covariate_effect, **kwds)
            db.hold_out_data(integrand_names = db.ode_hold_out_list, hold_out=1)

            if __check__ and case == 'crohns':
                fix_data_table(db, _dm_yes_ode_)

            if not _use_single_database_:
                system(f'{db.dismod} {db.path} init')
                db.fit_var = fit_var

            # use previous fit as starting point
            system(f'{db.dismod} {db.path} set start_var fit_var')

            if __check__:
                check_input_tables(db, dm = _dm_yes_ode_, check_hold_out = True)
            db.fit(msg='fit_with_ode')
            if __check__:
                check_output_tables(db, dm = _dm_yes_ode_)

            shutil.copy2(db.path, path_yes_ode)

            if _use_single_database_:
                db.data = db.input_data

        if no_ode:
            if __check__:
                if __run_fit_ihme__:
                    cmd = f'{_fit_ihme_py_} ~/ihme/epi/at_cascade {case} no_ode {random_seed}'
                    print (cmd); os.system(cmd)

            print ('--- no_ode ---')
            if _use_single_database_:
                db = FitNoODE(Path(original_file), ode_hold_out_list = ode_hold_out_list)
            else:
                db = setup_db(original_file)

            system(f'{db.dismod} {db.path} init')
            db.simplify_data(random_seed = random_seed, subsample = subsample)
            db.setup_ode_fit(max_covariate_effect, **kwds)
            db.hold_out_data(integrand_names = db.yes_ode_integrands, hold_out=1)

            if __check__ and case == 'crohns':
                fix_data_table(db, _dm_no_ode_)

            system(f'{db.dismod} {db.path} init')
            if __check__:
                check_input_tables(db, dm = _dm_no_ode_, check_hold_out = True)
            db.set_option('max_num_iter_fixed', 500)
            db.fit(msg = 'fit_no_ode')
            if __check__:
                check_output_tables(db, dm = _dm_no_ode_)

            shutil.copy2(db.path, path_no_ode)

            if _use_single_database_:
                db.data = db.input_data


        if yes_ode:
            if __check__:
                if __run_fit_ihme__:
                    cmd = f'{_fit_ihme_py_} ~/ihme/epi/at_cascade {case} yes_ode'
                    print (cmd); os.system(cmd)

            print ('--- yes_ode ---')

            if _use_single_database_:
                db = FitNoODE(Path(original_file), ode_hold_out_list = ode_hold_out_list)
            else:
                db = setup_db(original_file)

            db.simplify_data(random_seed = random_seed, subsample = subsample)
            if not _use_single_database_:
                db.setup_ode_fit(max_covariate_effect, **kwds)
            db.hold_out_data(integrand_names = db.ode_hold_out_list, hold_out=1)

            if __check__ and case == 'crohns':
                fix_data_table(db, _dm_yes_ode_)

            if not _use_single_database_:
                system(f'{db.dismod} {db.path} init')

            # use previous fit as starting point
            if not _use_single_database_:
                db.fit_var = DismodIO(path_no_ode).fit_var
            system(f'{db.dismod} {db.path} set start_var fit_var')

            if __check__:
                check_input_tables(db, _dm_yes_ode_, check_hold_out = True)
            db.set_option('max_num_iter_fixed', 500)
            db.fit(msg='fit_with_ode')
            if __check__:
                check_output_tables(db, dm = _dm_yes_ode_)

            shutil.copy2(db.path, path_yes_ode)

            if _use_single_database_:
                db.data = db.input_data

        if students:
            if __check__:
                if __run_fit_ihme__:
                    cmd = f'{_fit_ihme_py_} ~/ihme/epi/at_cascade {case} students'
                    print (cmd); os.system(cmd)

            print ('--- students ---')

            if _use_single_database_:
                db = FitNoODE(Path(original_file), ode_hold_out_list = ode_hold_out_list)
            else:
                db = setup_db(original_file)
            db.simplify_data(random_seed = random_seed, subsample = subsample)
            if not _use_single_database_:
                db.setup_ode_fit(max_covariate_effect, **kwds)
            db.hold_out_data(integrand_names = db.ode_hold_out_list, hold_out=1)
            db.set_student_likelihoods(factor_eta = 1e-2, nu = 5)

            if __check__ and case == 'crohns':
                fix_data_table(db, _dm_students_)

            if not _use_single_database_:
                system(f'{db.dismod} {db.path} init')

            # use previous fit as starting point
            if not _use_single_database_:
                db.fit_var = DismodIO(path_yes_ode).fit_var
            system(f'{db.dismod} {db.path} set start_var fit_var')

            db.set_option('max_num_iter_fixed', 500)
            if __check__:
                check_input_tables(db, _dm_students_, check_hold_out = True)
            db.fit(msg = 'fit_students')
            if __check__:
                check_output_tables(db, dm = _dm_students_)

            shutil.copy2(db.path, path_students)

            if _use_single_database_:
                db.data = db.input_data

if __name__ == '__main__':
    # for case in ['osteo_hip']:
    # for case in ['osteo_knee']:
    # for case in ['crohns']:
    # for case in ['t1_diabetes']: # Fixed the json on my local machine, now it has  Brad's settings
    for case in ['dialysis']:
    # for case in ['osteo_hip','osteo_knee', 'kidney','crohns', 't1_diabetes', 'dialysis']:
        print ('>>>', case, '<<<')
        original_file, max_covariate_effect, ode_hold_out_list, mulcov_values = test_cases(case)  

        test(case, original_file, max_covariate_effect, ode_hold_out_list, mulcov_values)


if 0 and __name__ == '__main__':
    os.system("dismod_db --model-version-id 475746 --parent-location-id 64 --sex-id 2 --fill --dm-commands init fit-ode set-start_var-fit_var set-scale_var-fit_var fit-both predict-fit_var --save-fit")


