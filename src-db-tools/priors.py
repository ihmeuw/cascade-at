#!/usr/bin/env python

import sys
import os
import time
import numpy as np
import tempfile
import shutil
import stat
import time
import pdb; from pdb import set_trace
import collections
from collections import OrderedDict 
from functools import lru_cache
import pandas as pd; pd.set_option('expand_frame_repr', False)

from db_queries.get_envelope import get_envelope
from db_queries.get_population import get_population

from cascade_at_gma.drill_no_csv.importer2sqlite import asymptotic_statistics

from cascade_ode.importer import settings, execute_select, get_model_version

from cascade_at_gma.lib.dismod_db_api import DismodDbAPI 
from cascade_at_gma.lib.dismod_db_functions import get_density_id, get_integrand_id, get_rate_smooth_names, get_node_id, node_id2location_id, get_rate_name, node_id2name, copyDB_dest, db_info, cleanup_prior
from cascade_at_gma.lib.dismod_db_functions import set_node_info, set_max_iters, set_tolerance_fixed, set_cov_reference, tempfile_DB
from cascade_at_gma.lib.run_dismod_commands import run_dismod_commands as run_AT_commands
from cascade_at_gma.lib.table_description import TableDescriptions
from cascade_at_gma.lib.utilities import sex_id2sex_dict, ihme_id2sex_dict, ihme_id2sex_id, sex2ihme_id
from cascade_at_gma.lib.constants import _time_window_for_fit_, sex_name2covariate
from cascade_at_gma.lib.get_covariate_estimates import get_covariate_estimates


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('priors.py')
del logging


tol15 = dict(atol=1e-15, rtol=1e-15)
tol14 = dict(atol=1e-14, rtol=1e-14)

_print_p_ = False

_fix_bugs_ = False

_zero_iterations_ = False
_redo_posterior_ = True

_convert_uniform_priors_to_gaussian_ = True
_make_nonuniform_prior_stds_nonzero_ = True

__my_debug__ = True

if 1:
    def get_model_version(mvid):
        from cascade_ode.importer import execute_select
        query = """
            SELECT * FROM epi.model_version
            JOIN epi.modelable_entity USING(modelable_entity_id)
            WHERE model_version_id=%s """ % (mvid)
        df = execute_select(query)
        return df

def check_bounds(prior):
    mask = np.all(np.isfinite(prior[['lower', 'upper']]), axis=1) & ~ ((prior['lower'] <= prior['mean']) & (prior['mean'] <= prior['upper']))
    if mask.any():
        print (prior[mask])
        raise Exception("Prior means exceed their limits.")

def drop_extraneous_mtall(DB, child_node_id):
    data = DB.data
    mtall_id = get_integrand_id(DB, 'mtall')
    mask = ((data.integrand_id != mtall_id) |
            ((data.integrand_id == mtall_id) & (data.node_id == child_node_id)))
    data = data[mask].reset_index(drop=True)
    data['data_id'] = list(data.index)
    DB.data = data
    
def apply_random_effects(DB, random_effects, child_node_ids):
    rate_smooth_names = get_rate_smooth_names(DB, DB.smooth.smooth_name)
    rate_smooths = DB.smooth[DB.smooth.smooth_name.isin(rate_smooth_names)]
    omega_id = int(DB.rate.loc[DB.rate.rate_name == 'omega', 'rate_id'])
    cols = ['prior_id', 'prior_name', 'n_age', 'n_time', 'age_id', 'time_id', 'lower', 'upper', 'mean', 'std', 'eta', 'density_id', 'smooth_id', 'smooth_name', 'smooth_grid_id']
    if __my_debug__:
        for name in rate_smooth_names:
            rate_smooth = rate_smooths[[name in _ for _ in rate_smooths.smooth_name]].merge(DB.smooth_grid, how='left')
            prior = DB.prior[DB.prior.prior_id.isin(rate_smooth.value_prior_id.values)].merge(rate_smooth, how='left', left_on='prior_id', right_on='value_prior_id')[cols]
            N = prior.n_age.unique() * prior.n_time.unique()
            try:
                assert len(N) == 1 and N[0] == len(prior), "Priors collection for rate %s failed" % name
            except Exception as ex:
                logger.error('FIXME -- ' + str(ex))
            logger.info ("There are %d priors for rate %s" % (len(prior), get_rate_name(DB, name)))

    rates = DB.rate.merge(DB.smooth, how='left', left_on='parent_smooth_id', right_on='smooth_id').merge(DB.smooth_grid, how='left').merge(DB.prior, left_on='value_prior_id', right_on='prior_id')

    rtn = {}
    prior = DB.prior.merge(DB.smooth_grid, how='left', left_on='prior_id', right_on='value_prior_id').merge(DB.rate, how='left', left_on='smooth_id', right_on='child_smooth_id')
    parent_node_id = int(DB.options.parent_node_id)
    # random_effect_node_ids = DB.node.loc[DB.node.parent == parent_node_id, 'node_id']
    random_effect_node_ids = random_effects.node_id
    for node_id in random_effect_node_ids:
        if node_id not in child_node_ids: continue
        p = DB.prior.copy()
        p['rate_id'] = None
        for rate_id in random_effects.rate_id.unique():
            random_effect = (random_effects[(random_effects.node_id == node_id) & (random_effects.rate_id == rate_id)])
            if all(random_effect.merge(DB.smooth_grid).const_value.isna() == True):
                prior_ids = rates.loc[rates.rate_id == rate_id, 'prior_id']
                mask = p.prior_id.isin(prior_ids)
                p.loc[mask, 'rate_id'] = rate_id
                effect = 1 if random_effect.empty else np.exp(random_effect.fit_var_value).values
                p.loc[mask, 'mean'] *= effect

                # Make sure the scaled mean is inbounds
                p.loc[mask, 'mean'] = p.loc[mask, ['mean','lower']].apply(max, axis=1)
                p.loc[mask, 'mean'] = p.loc[mask, ['mean','upper']].apply(min, axis=1)
                
                logger.info ('Applying random effect -- node: %d, rate: %s, effect: %s, prior : %s' % (node_id, rate_id, effect, sorted(set([_.rstrip('_1234567890') for _ in p[mask].prior_name]))))
            else:
                logger.info ('Ignoring random effect random effect -- node: %d, rate: %s, because it is a child const value.' % (node_id, rate_id))
        rtn[node_id] = p
    return rtn

def apply_child_priors(DB, child_priors):
    rate_smooth_names = get_rate_smooth_names(DB, DB.smooth.smooth_name)
    rate_smooths = DB.smooth[DB.smooth.smooth_name.isin(rate_smooth_names)]
    omega_id = int(DB.rate.loc[DB.rate.rate_name == 'omega', 'rate_id'])
    cols = ['prior_id', 'prior_name', 'n_age', 'n_time', 'age_id', 'time_id', 'lower', 'upper', 'mean', 'std', 'eta', 'density_id', 'smooth_id', 'smooth_name', 'smooth_grid_id']
    if __my_debug__:
        for name in rate_smooth_names:
            rate_smooth = rate_smooths[[name in _ for _ in rate_smooths.smooth_name]].merge(DB.smooth_grid, how='left')
            prior = DB.prior[DB.prior.prior_id.isin(rate_smooth.value_prior_id.values)].merge(rate_smooth, how='left', left_on='prior_id', right_on='value_prior_id')[cols]
            N = prior.n_age.unique() * prior.n_time.unique()
            try:
                assert len(N) == 1 and N[0] == len(prior), "Priors collection for rate %s failed" % name
            except Exception as ex:
                logger.error('FIXME -- ' + str(ex))
            logger.info ("There are %d priors for rate %s" % (len(prior), get_rate_name(DB, name)))

    rates = DB.rate.merge(DB.smooth, how='left', left_on='parent_smooth_id', right_on='smooth_id').merge(DB.smooth_grid, how='left').merge(DB.prior, left_on='value_prior_id', right_on='prior_id')

    rtn = {}
    prior = DB.prior.merge(DB.smooth_grid, how='left', left_on='prior_id', right_on='value_prior_id').merge(DB.rate, how='left', left_on='smooth_id', right_on='child_smooth_id')
    parent_node_id = int(DB.options.parent_node_id)
    child_node_ids = sorted(child_priors.node_id.unique())

    cols = ['rate_id', 'rate_name','smooth_id', 'smooth_grid_id', 'age_id', 'time_id',
            'value_prior_id', 'const_value']
    rate_priors = (DB.rate
                   .merge(DB.smooth, how='left', left_on='parent_smooth_id', right_on='smooth_id')
                   .merge(DB.smooth_grid, how='left')
                   .merge(DB.prior, how='left', left_on = 'value_prior_id', right_on='prior_id'))[cols]
    p = DB.prior
    for node_id in child_node_ids:
        child_prior = child_priors[child_priors.node_id == node_id]
        for rate_id in child_prior.rate_id.unique():
            rate_prior = rate_priors[rate_priors.rate_id == rate_id]
            child_rate = child_prior[child_prior.rate_id == rate_id]
            merge = rate_prior.merge(child_rate, on = ['rate_id', 'age_id', 'time_id']) 
            prior_ids = merge.value_prior_id.astype(int).values
            mask = p.prior_id.isin(prior_ids)
            if __debug__:
                assert len(merge) == len(p[mask]), "Number of child %s rates do not match number of rate priors." % (rate_prior.rate_name.iloc[0])
            p.loc[mask, 'mean'] = merge.sample_mean.values
            p.loc[mask, 'std'] = merge.sample_std.values
            # Make sure the scaled mean is inbounds
            p.loc[mask, 'mean'] = p.loc[mask, ['mean','lower']].apply(max, axis=1)
            p.loc[mask, 'mean'] = p.loc[mask, ['mean','upper']].apply(min, axis=1)

            [rate] = rate_prior.rate_name.unique()
            logger.info ('Applying child prior prediction -- node: %d, rate: %s, prior : %s' % (node_id, rate, sorted(set([_.rstrip('_1234567890') for _ in p[mask].prior_name]))))

        rtn[node_id] = p
    return rtn

# @lru_cache(maxsize = 4)
def get_age_ranges(mortality_age_grid):
    query = """
    SELECT age_group_id, age_group_years_start, age_group_years_end
    FROM shared.age_group"""
    ages = execute_select(query, 'cod')
    mask = ages.age_group_id.isin(mortality_age_grid)
    return ages[mask]

def fix_arg(arg):
    if arg is None:
        return -1
    try:
        return [a for a in arg]
    except:
        return arg

try:
    Importer
except:
    @lru_cache(maxsize = 4)
    def Importer(model_version_id, root_dir = None):
        try: sys.modules.pop('cascade_ode.importer')
        except: pass
        from cascade_ode.importer import Importer
        return Importer(model_version_id, root_dir = None)

try: cached_mtall
except:
    @lru_cache(maxsize = 4)
    def cached_mtall(location_id = None, year_id = None, sex_id = None, 
                     mortality_age_grid = (2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,31,32,235)):
                
        location_id = fix_arg(location_id)
        year_id = fix_arg(year_id)
        sex_id = fix_arg(sex_id)

        ages = get_age_ranges(mortality_age_grid)
        env = get_envelope(location_id = location_id,
                           location_set_id = 9,
                           age_group_id = list(mortality_age_grid),
                           year_id = year_id,
                           sex_id = sex_id,
                           with_hiv = 1)
        pop = get_population(location_id = location_id,
                             location_set_id = 9,
                             age_group_id = list(mortality_age_grid),
                             year_id = year_id,
                             sex_id = sex_id)
        mtall = env.merge(pop)
        if mtall.empty:
            for col in ['age_group_id', 'location_id', 'year_id', 'sex_id', 'run_id']:
                if len(set(env[col].unique()).intersection(pop[col])) == 0:
                    logger.error("Mtall merge failed because the env and population query %s did not match" % col)
                    break
            if col == 'run_id':
                mtall = env.rename(columns={'run_id':'env_run_id'}).merge(pop.rename(columns={'run_id':'pop_run_id'}))
        mtall = mtall.merge(ages, how='left')
        mtall['mean'] /= mtall['population']
        mtall['lower'] /= mtall['population']
        mtall['upper'] /= mtall['population']
        mtall['stdev'] = ((mtall.upper-mtall.lower) / (2.0*1.96)).replace({0: 1e-9})
        return mtall
    
def query_mtall(DB, model_version_id, node_id, sex = None, times=None):
    mtall_id = DB.integrand[DB.integrand.integrand_name == 'mtall'].integrand_id.squeeze()
    mtall_cols = DB.data.columns.tolist()

    location_id = tuple(node_id2location_id(DB, node_id))
    node_loc_df = pd.DataFrame(list(zip(location_id, node_id)), columns = ['location_id', 'node_id'])

    if sex is None:
        sex_id = -1
    else:
        try:
            sex_id = tuple([sex2ihme_id(s) for s in sex])
        except:
            sex_id = sex2ihme_id(sex)

    if times is None:
        year_id = -1
    else:
        try:
            year_id = tuple([int(t) for t in times])
        except:
            year_id = int(times)

    # Copy required, otherwise the dataframe could be modified 
    mtall = cached_mtall(year_id=year_id, sex_id=sex_id, location_id=location_id).copy()
    mtall.rename(columns = {'mean': 'meas_value', 'population' : 'sample_size',
                            'age_group_years_start' : 'age_lower', 'age_group_years_end' : 'age_upper',
                            'year_id' : 'time_lower'}, inplace=True)
    # Attach country covariates
    covs = (get_covariate_estimates(covariate_names_short = DB.country_covariates.covariate_name_short.tolist(),
                                    location_id = location_id, year_id = times,
                                    model_version_id = model_version_id, gbd_round_id = None, sex = 'all'))
    # Change from IHME to dismod both sex id convention
    covs['sex_id'] = covs.sex_id.replace({0: 3}).values

    mtall = mtall.merge(covs.rename(columns={'year_id': 'time_lower'}).drop(columns='age_group_id'), how='left') # Because country covariates are age-standardized they apply to all age groups

    mtall = mtall.merge(node_loc_df, how='left')
    mtall['integrand_id'] = mtall_id
    # Set the mtall CV
    CV = 1
    mtall['meas_std'] = CV * mtall['meas_value']
    mtall['time_upper'] = mtall['time_lower'] + 1
    mtall.loc[100 < mtall.age_upper, 'age_upper'] = 100
    mtall['x_sex'] = mtall['sex_id'].map(lambda x: ihme_id2sex_dict[x])

    
    mtall['x_one'] = 1
    mtall['weight_id'] = 0
    mtall['hold_out'] = 0
    mtall['density_id'] = get_density_id(DB, 'log_gaussian')
    mtall['eta'] = 1e-6

    xcovs = dict([(v.replace('x_c_', '').replace('x_s_', ''), 'x_%d' % k) for k,v in DB.covariate[['covariate_id', 'covariate_name']].values])
    mtall = mtall.rename(columns = xcovs)

    missing = list(set(DB.data.columns) - set(mtall.columns))
    if missing:
        ccovs = DB.country_covariates.xcov_name.tolist()
        if set(ccovs).issubset(missing):
            logger.info("FIXME -- mtall is missing country covariates -- setting them to Null (the reference).")
    mtall = mtall.join(pd.DataFrame(columns=missing), how='left')[mtall_cols]

    return mtall

# def mixin_child_mtall(DB, model_version_id, node_id, sex_reference, times):
#     data = DB.data
#     mtall = query_mtall(DB, model_version_id = model_version_id, node_id = node_id, sex = sex_reference, times = times)
#     mtall_id = DB.integrand[DB.integrand.integrand_name == 'mtall'].integrand_id.squeeze()
#     data = data[data.data_id != mtall_id]
#     data = data.append(mtall).reset_index(drop=True)
#     data['data_id'] = data.index.tolist()
#     return data

def get_descendants(DB, model_version_id, node_id, root_dir = None):
    from hierarchies.dbtrees import loctree as lt
    from cascade_at_gma.lib.dismod_db_functions import get_location_id

    location_id2node_id = (dict([(get_location_id(node_name), nid) for nid, node_name in DB.node.loc[:, ['node_id', 'node_name']].values]))

    imp = Importer(model_version_id, root_dir = root_dir)
    lsvid = imp.model_version_meta.location_set_version_id.values[0]
    loctree = lt(None, location_set_version_id=lsvid)
    [parent_node_name] = DB.node.loc[DB.node.node_id == node_id, 'node_name']
    parent_node = loctree.get_node_by_id(get_location_id(parent_node_name))
    descendant_location_ids = parent_node.all_descendants()
    descendant_node_ids = sorted([location_id2node_id[descendant_id.id] for descendant_id in descendant_location_ids if descendant_id.id in location_id2node_id])
    if __debug__:
        tmp = sorted([get_node_id(DB, descendant_id.id) for descendant_id in descendant_location_ids if descendant_id.id in location_id2node_id])
        if tuple(tmp) != tuple(descendant_node_ids): set_trace()

    missing_descendant_node_ids = sorted([descendant_id for descendant_id in descendant_location_ids if descendant_id.id not in location_id2node_id], key=lambda x: x.id)
    msg = "The following loctree.node.all_descendants were missing from the dismod.node table."
    logger.warn("-"*len(msg))
    logger.warn("The following loctree.node.all_descendants were missing from the dismod.node table.")
    for node in missing_descendant_node_ids:
        logger.warn ("Node id: %6d -- %s (%s)" % (node.id, node.info['location_name'], node.info['location_type']))
    logger.warn("-"*len(msg))
    return descendant_node_ids

def get_descendant_data(DB, model_version_id, node_id, ages, times, root_dir = None):
    """
    Get the data for locations that are descendant of this node_id
    """
    descendants = [node_id] + get_descendants(DB, model_version_id, node_id = node_id, root_dir = root_dir)
    data = DB.data[DB.data.node_id.isin(descendants)]
    data.dropna(subset=['node_id'], inplace=True)

    if 0:
        logger.warn("FIXME -- I need mtall for each parent node, but the upper levels are dropping that data.")
        location_id = node_id2location_id(node_id)

        mtall_id = DB.integrand[DB.integrand.integrand_name == 'mtall'].integrand_id.squeeze()

        xcovs = dict([(v, 'x_%d' % k) for k,v in DB.covariate[['covariate_id', 'covariate_name']].values])
        imp = Importer(model_version_id)
        mtall = imp.data[(imp.data.integrand == 'mtall') & (imp.data.location_id == location_id)]
        mtall['node_id'] = node_id
        mtall['integrand_id'] = mtall_id
        mtall['x_one'] = 1
        mtall['weight_id'] = 0
        mtall['density_id'] = get_density_id(DB, 'log_gaussian')
        mtall['eta'] = 1e-6
        mtall.rename(columns = xcovs, inplace=True)
        mtall.rename(columns={'mean': 'meas_value', 'meas_stdev': 'meas_std', 'age_group_years_start' : 'age_lower', 'age_group_years_end' : 'age_upper'}, inplace=True)
        missing = list(set(data.columns) - set(mtall.columns))
        if missing:
            logger.error("FIXME -- hardcoding the mtall country covariates to the reference.")
        mtall = mtall.join(pd.DataFrame(columns=missing), how='left')
        data = data.append(mtall[data.columns])
        
    data = data.reset_index(drop=True)
    data['data_id'] = list(data.index)
    return data

def sigma(meas_value, meas_std, eta):
    """
    Log-transformed standard deviation.
    See http://moby.ihme.washington.edu/bradbell/dismod_at/statistic.htm#Log-Transformed%20Standard%20Deviation,%20sigma
    """
    return np.log(meas_value + eta + meas_std) - np.log(meas_value + eta)

def convert_uniform_priors_to_gaussian(DB, cv = 0.1):
    logger.warn("Converting all uniform priors to gaussian.")
    prior = DB.prior
    mask = prior.density_id == get_density_id(DB, 'uniform')
    prior.loc[mask, 'density_id'] = get_density_id(DB, 'gaussian')
    prior.loc[mask, 'std'] = cv * prior.loc[mask, 'mean'].abs()
    return prior

def make_nonuniform_prior_stds_nonzero(DB, cv = 0.1):
    prior = DB.prior
    mask = (prior.density_id != get_density_id(DB, 'uniform')) & (~np.isfinite(prior['std']) | (prior['std'] <= 0))
    if 0 < sum(mask):
        logger.warn("Fixing the standard deviation of %d non-uniform priors with std values less than 0 or NaN." % sum(mask))
        prior.loc[mask, 'std'] = [(1 if m == 0 else m) for m in cv * prior.loc[mask, 'mean'].abs()]
    return prior

def random_effects_at_solution(DB):
    parent_node_id = int(DB.options.parent_node_id)
    cols = ['var_id', 'var_type', 'smooth_id', 'age_id', 'time_id', 'node_id', 'node_name', 'rate_id', 'integrand_id', 'covariate_id', 'fit_var_id', 'fit_var_value', 'truth_var_value']
    df = (DB.var.merge(DB.fit_var, how='left', left_on='var_id', right_on='fit_var_id').merge(DB.truth_var, how='left', left_on='var_id', right_on='truth_var_id').merge(DB.node, how='left'))[cols]
    df = df[((df.node_id != parent_node_id) & (df.var_type == 'rate'))].dropna(subset=['node_id'])
    
    if not df.empty:
        assert len(df.age_id.unique()), "Random effect age grid not implemented."
        assert len(df.time_id.unique()), "Random effect time grid not implemented."

    return df

def mulcov_values(DB):
    var = DB.var
    mulcovs = var[var.var_type != 'rate'].merge(DB.smooth_grid, how='left').merge(DB.prior, how='left', left_on='value_prior_id', right_on='prior_id')
    mask = np.isfinite(mulcovs.const_value)
    mulcovs.loc[mask, ['lower', 'mean', 'upper']] = mulcovs.loc[mask, 'const_value']
    mulcovs = mulcovs[['var_id', 'var_type', 'smooth_id', 'age_id', 'time_id', 'node_id', 'rate_id', 'integrand_id', 'covariate_id', 'prior_id', 'prior_name', 'lower', 'upper', 'mean', 'std', 'eta', 'density_id']]
    return mulcovs

def plot_simulate_residuals(DB):
    # Import matplotlib locally for thread safety
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt

    import pandas as pd; pd.set_option('expand_frame_repr', False)

    from cascade_at_gma.lib.weighted_residuals import weighted_residuals

    if __debug__:
        # Check my weighted residuals calculation
        data = DB.data.merge(DB.data_subset, how='left').merge(DB.fit_data_subset, how='left')
        assert (weighted_residuals(DB, data) - data.weighted_residual).abs().max() < 1e-8, "Weighted residuals check failed"

    # Sample/simulated data weighted residuals
    DB.avgint = DB.data_subset.merge(DB.data, how = 'left')
    run_AT_commands(DB.filename, 'predict')
    simulate = (DB.data_subset
                .merge(DB.data.drop(['meas_value', 'meas_std'], axis=1), how='left')
                .merge(DB.data_sim, on='data_subset_id')
                .merge(DB.predict, left_on=['simulate_index', 'data_id'], right_on=['sample_index', 'avgint_id']))

    # Histogram the data meas_values
    plt.close('all')
    data = DB.data.merge(DB.integrand, how='left')
    for integrand_id in data.integrand_id.unique():
        plt.figure()
        integrand_name = DB.integrand.loc[DB.integrand.integrand_id == integrand_id, 'integrand_name'].squeeze()
        sim = simulate[simulate.integrand_id == integrand_id]
        dat = data[data.integrand_id == integrand_id]
        nbins0 = 10
        nbins1 = max(1, int(sim.meas_value.max()/dat.meas_value.max()))*nbins0
        n, bins, patches = plt.hist(sim.meas_value.values, normed=True, bins=nbins1, log=True, color='grey')
        n, bins, patches = plt.hist(dat.meas_value.values, normed=True, bins=nbins0, log=True, color='green')
        plt.xlabel('Data.meas_value vs. Simulated.meas_value')
        plt.title(integrand_name)


    simulate['weighted_residual'] = weighted_residuals(DB, simulate)
    plt.figure()
    # Simulate weighted residuals
    n, bins, patches = plt.hist(simulate.weighted_residual.values, normed=True, bins=500, log=True)
    plt.xlabel('Simulated data weighted residuals')
    plt.ylabel('Log probability density')
    # fit_data_subset weighted residuals
    plt.figure()
    data = DB.data_subset.merge(DB.data, how='left').merge(DB.fit_data_subset, how='left')
    n, bins, patches = plt.hist(data.weighted_residual, normed=True, bins=500, log=True)
    plt.xlabel('fit_data_subset.weighted_residuals')
    plt.ylabel('Log Probability Density')

def check_priors(DB, parent_fit, parent_node_id):

    parent_fit = parent_fit.copy() # Prevent side effects

    run_AT_commands(DB.filename, ['init'])
    start_at_prior_mean = DB.var.merge(DB.start_var, left_on='var_id', right_on='start_var_id').merge(DB.rate, how='left')

    start_at_fit_var = DB.var.merge(DB.start_var, left_on='var_id', right_on='start_var_id').merge(DB.rate, how='left')
    # Zero out the child mulcovs
    mask = (start_at_fit_var.node_id != parent_node_id) & (start_at_fit_var.var_type == 'rate')
    start_at_fit_var.loc[mask]

    mask = start_at_fit_var.start_var_value != start_at_prior_mean.start_var_value

    cols = ['var_id', 'prior_name', 'mean', 'start_var_value']
    xx = DB.var.merge(DB.smooth_grid, how='left').merge(DB.smooth, how='left').merge(DB.prior, left_on='value_prior_id', right_on='prior_id').merge(DB.start_var, left_on='var_id', right_on='start_var_id')[cols]


    logger.info ('prior stats')
    logger.info (prior_stats[mask])
    logger.info ('var from priors')
    logger.info (xx[mask])
    logger.info ((prior_stats.sample_mean - xx.start_var_value)[mask])

    logger.info ('Fit var start:')
    logger.info (start_at_fit_var[mask])
    logger.info ('Prior mean start:')
    logger.info (start_at_prior_mean[mask])

    assert np.alltrue(start_at_fit_var.start_var_value == start_at_prior_mean.start_var_value), 'Prior != fit_var'

    # Zero out the child mulcovs in the parent fit, and check the child priors.
    mask = (DB.var.node_id != parent_node_id) & (DB.var.var_type == 'rate')
    parent_fit.loc[mask, 'fit_var_value'] = 0
    assert (parent_fit.fit_var_value - start_at_prior_mean.start_var_value).abs().max() < 1e-10, "Priors are incorrect."

def build_splits_fn(model_version_id, location_id):
    print ("Importing hierarchies.py  (requires database access) ... ", end='', flush=True)
    from hierarchies.dbtrees import loctree as lt
    print ("Done", flush=True)
    lsvid = int(get_model_version(model_version_id).location_set_version_id)
    loctree = lt(None, location_set_version_id=lsvid)
    root_level = loctree.get_nodelvl_by_id(location_id)

    def splits_fn(location_id, loctree):
        level = loctree.get_nodelvl_by_id(location_id)
        if level in (None, root_level):
            return dict(x_sex = [('both', sex_name2covariate['both'])])
        else:
            return dict(x_sex = [('female', sex_name2covariate['female']), ('male', sex_name2covariate['male'])])
    return splits_fn

def initialize_children(parent_posterior_DB, parent_random_effects, model_version_id, include_locations, self_times, fit_p = False, split = {},
                        root_loc_id=None, sex=None, mixin_mtall = False):

    from hierarchies.dbtrees import loctree as lt
    lsvid = int(get_model_version(model_version_id).location_set_version_id)
    loctree = lt(None, location_set_version_id=lsvid)

    splits_fn = build_splits_fn(model_version_id, root_loc_id)

    def get_splits(split, location_splits):
        """
        Seems like any prior split (the split argument) will be a single item, not multiple items.
        If location_splits (the split for this location) is a multiple item, then each of those become a single split on down the tree.
        If the split argument is in the location_splits, then do not split based on the location splits.
        If the split argument is not in the location_splits, then we need to make a new split.

        # Case 1) current_split and location_splits are the same -- continue with this split
        split = {x_cov: [('both', 0)]}
        location_splits = {x_cov: [('both', 0)]}
        assert get_splits(split, location_splits) == split

        # Case 2) current split is not in the location_splits -- return the new splits
        split = {x_cov: [('both', 0)]}
        location_splits  = {'x_sex': [('male', +0.5), ('female', -0.5)]}
        assert get_splits(split, location_splits) == location_splits

        # Case 3) current split is in the location_splits -- continue with this split
        split = {x_cov: [('male', +0.5)]}
        location_splits = {'x_sex': [('male', +0.5), ('female', -0.5)]}
        assert get_splits(split, location_splits) == split

        # Case 4) current split is split again by location_splits -- return the new splits
        split = {x_cov: [('male', +0.5)]}
        location_splits = {'x_sex': [('bull', +0.5), ('cow', -0.5)]}
        assert get_splits(split, location_splits) == location_splits
        """
        rtn = {}
        for x_cov in location_splits.keys():
            splits_for_this_location = location_splits[x_cov]
            if split:
                current_split = split[x_cov]
                assert len(current_split) == 1, "Seems like the current split should always be a single item."
                current_split = current_split[0]
            else:
                current_split = {}
            if current_split in splits_for_this_location:
                rtn.update(split)
            else:
                rtn.update(location_splits)
        return rtn

    def mark_all_descendants_complete(loctree, model_version_id, loc_id, cov_name, self_times):
        for child_location_id in [c.id for c in loctree.get_node_by_id(loc_id).all_descendants()]:
            child_name = node_id2name(parent_posterior_DB, child_node_id).replace(' ', '_').replace('&','and')
            data_path = os.path.expanduser(settings['cascade_at_gma_out_dir'])
            # child_filename = os.path.join(data_path, '%d/full/%d/%s/%s/prior/%d_constrained.db' % (model_version_id, child_location_id, cov_name, '_'.join(map(str, self_times)), model_version_id))
            child_filename = os.path.join(data_path, '%d/full/%d/%s/%s/prior/%d.db' % (model_version_id, child_location_id, cov_name, '_'.join(map(str, self_times)), model_version_id))
            logger.info ('Marking child: %d, location_id: %d, sex: %s, child location name: %s, complete because the child had no data.' % (child_node_id, child_location_id, cov_name, child_name))
            touch(child_filename + '-DAG')

    parent_node_id = int(parent_posterior_DB.options.parent_node_id)
    parent_loc_id = node_id2location_id(parent_posterior_DB, parent_node_id)
    try:
        child_loc_ids = sorted(set([c.id for c in loctree.get_node_by_id(parent_loc_id).children]))
    except:
        child_loc_ids = parent_posterior_DB.node.loc[parent_posterior_DB.node.parent == parent_node_id, 'node_name'].tolist()
    if include_locations:
        child_loc_ids = sorted(set(child_loc_ids).intersection(include_locations))
    child_node_ids = [_ for _ in [get_node_id(parent_posterior_DB, loc_id) for loc_id in child_loc_ids] if _ is not None]

    # Apply random effects
    child_priors = {k:None for k in child_node_ids if k is not None}
    if 0:
        child_priors.update(apply_random_effects(parent_posterior_DB, parent_random_effects, child_node_ids))
    else:
        from cascade_at_gma.drill_no_csv.DB_posterior import Posterior
        p = Posterior(parent_posterior_DB.filename)
        child_priors.update(apply_child_priors(parent_posterior_DB, p.child_priors))
    

    # # This filter is already done above
    # if include_locations:
    #     child_priors = {k:v for k,v in child_priors.items() if k in [get_node_id(parent_posterior_DB, lid) for lid in include_locations]}

    child_node_ids = tuple(child_priors.keys())
    mtall_id = parent_posterior_DB.integrand[parent_posterior_DB.integrand.integrand_name == 'mtall'].integrand_id.squeeze()
    x_sex = str(parent_posterior_DB.sex_covariate.xcov_name.squeeze())
    times = parent_posterior_DB.data[parent_posterior_DB.data.integrand_id == mtall_id].time_lower.unique()
    
    if mixin_mtall:
        t0 = time.time()
        all_child_mtall = query_mtall(parent_posterior_DB, model_version_id = model_version_id, node_id = child_node_ids, times = times)
        logger.info("Mtall query took %.2f seconds." % (time.time() - t0))

    childDBs = []
    for child_node_id, child_prior in child_priors.items():
        child_location_id = node_id2location_id(parent_posterior_DB, child_node_id)
        location_splits = splits_fn(child_location_id, loctree)
        
        if not split:
            logger.warn ('#FIXME -- starting mid-hierarchy seems to be messing up current_split -- or it might just be all messed up.')
            current_split = {}
            for x_cov, v in location_splits.items():
                for cov_name, cov_reference in v:
                    ref = parent_posterior_DB.covariate.loc[parent_posterior_DB.covariate.covariate_name == x_cov, 'reference'].squeeze()
                    if cov_reference == ref:
                        current_split[x_cov] = [(cov_name, cov_reference)]
        else:
            current_split = split.copy()

        splits = get_splits(current_split, location_splits)

        assert len(location_splits.keys()) <= 1, "FIXME -- directory naming is only set up for splitting on one covariate -- going to need more comprehensive naming for multiple splits."

        # FIXME -- this a bad way to get the path to the data files
        # data_path = os.path.expanduser(settings['cascade_at_gma_out_dir'])
        filename = parent_posterior_DB.filename
        data_path = filename[:filename.index('%s%s%s' % (os.path.sep, model_version_id, os.path.sep))]

        for x_cov,v in splits.items():
            for cov_name, cov_reference in v:
                col_name = 'x_%d' % parent_posterior_DB.covariate.loc[parent_posterior_DB.covariate.covariate_name == x_cov, 'covariate_id'].squeeze()
                this_split = {x_cov: [(cov_name, cov_reference)], 'location_id' : child_location_id}
                if child_prior is None:
                    mark_all_descendants_complete(loctree, model_version_id, child_location_id, cov_name, self_times)
                else:
                    child_name = node_id2name(parent_posterior_DB, child_node_id).replace(' ', '_').replace('&','and')
                    # child_filename = os.path.join(data_path, '%d/full/%d/%s/%s/prior/%d_constrained.db' % (model_version_id, child_location_id, cov_name, '_'.join(map(str, self_times)), model_version_id))
                    child_filename = os.path.join(data_path, '%d/full/%d/%s/%s/prior/%d.db' % (model_version_id, child_location_id, cov_name, '_'.join(map(str, self_times)), model_version_id))
                    logger.info ('Initializing child: %d, location_id: %d, sex: %s, child location name: %s' % (child_node_id, child_location_id, cov_name, child_name))
                    path = os.path.dirname(child_filename)
                    copyDB_dest(parent_posterior_DB, child_filename, verbose=True)
                    DB = DismodDbAPI(child_filename)
                    set_tolerance_fixed(DB, 1e-4)
                    set_node_info(DB, child_node_id)
                    set_cov_reference(DB, x_cov, cov_reference)
                    if mixin_mtall:
                        mask = (all_child_mtall.node_id == child_node_id) & np.isclose(all_child_mtall[col_name], cov_reference, atol=1e-8, rtol=1e-8)
                        data = DB.data.append(all_child_mtall[mask]).reset_index(drop=True)
                        data['data_id'] = data.index.tolist()
                        DB.data = data
                    db_info(DB)

                    # Clear the posterior, if it exists
                    child_posterior = os.path.dirname(child_filename.replace('/prior/', '/posterior/'))
                    logger.info('Clearing posterior %s.' % child_posterior)

                    if '167' in child_posterior: set_trace()

                    shutil.rmtree(child_posterior, ignore_errors=True)

                    # ((child_prior[:1]['lower'] <= child_prior[:1]['mean']) & (child_prior[:1]['mean'] <= child_prior[:1]['upper'])) & np.all(np.isfinite(child_prior[:1][['lower', 'upper']]), axis=1)
                    
                    if not _fix_bugs_:
                        # This should not happen
                        mask = np.all(np.isfinite(child_prior[['lower', 'upper']]), axis=1) & ~ ((child_prior['lower'] <= child_prior['mean']) & (child_prior['mean'] <= child_prior['upper']))
                        assert sum(mask) == 0, "Mean is outside the limits\n%s." % child_prior[mask]
                        mask = (child_prior.density_id != get_density_id(DB, 'uniform')) & ~np.isfinite(child_prior['std'].values)
                        assert sum(mask) == 0, "A non-uniform density has std = NaN\n%s" % child_prior[mask]
                         
                    else:
                        # Fix any means that fall outside the lower/upper bounds
                        mask = np.all(np.isfinite(child_prior[['lower', 'upper']]), axis=1) & ~ ((child_prior['lower'] <= child_prior['mean']) & (child_prior['mean'] <= child_prior['upper']))
                        # This should not happen
                        if mask.any():

                            logger.error("FIXME -- %s" % child_filename)
                            logger.error("FIXME -- Fixing means that are outside the limits -- not sure how they became this way:\n%s" % child_prior[mask])
                            logger.error("FIXME -- The code in apply_random_effects should have kept the rates within the bounds.")
                            
                            # Brad thinks that rate prior mean set to a limit makes for a particularly difficult optimization problem, and recommends the following mean value setting instead
                            # This has problems if the multiplier is < 0
                            child_prior.loc[mask, 'mean'] = np.clip(np.exp(np.log(child_prior.loc[mask, ['lower', 'upper']])).mean(axis=1),
                                                                    child_prior.loc[mask, 'lower'],
                                                                    child_prior.loc[mask, 'upper'])
                        mask = (child_prior.density_id != get_density_id(DB, 'uniform')) & ~np.isfinite(child_prior['std'].values)
                        assert sum(mask) == 0, "A non-uniform density has std = NaN\n%s" % child_prior[mask]

                    # Zero the child_prior meas_std mulcovs
                    mulcov_meas_prior_ids = DB.var[DB.var.var_type == 'mulcov_meas_std'].merge(DB.smooth_grid, how='left').value_prior_id.unique()
                    child_prior.loc[child_prior.prior_id.isin(mulcov_meas_prior_ids), 'mean'] = 0

                    check_bounds(child_prior)
                    DB.prior = child_prior

                    if _convert_uniform_priors_to_gaussian_:
                        DB.prior = convert_uniform_priors_to_gaussian(DB)

                    if _make_nonuniform_prior_stds_nonzero_:
                        DB.prior = make_nonuniform_prior_stds_nonzero(DB)

                    cleanup_prior(DB)
                    childDBs.append((this_split, DB))
    return childDBs

class FitStatistics(object):

    def __init__(self, filename_in, model_version_id, location_id, times, sex='both', include_locations = None, 
                 use_fit_for_mean = True, archive = False):

        self.model_version_id = model_version_id
        self.location_id = location_id
        self.times = times
        self.ages = list(range(2)) + list(range(5,105,5))
        self.sex = sex
        self.use_fit_for_mean = use_fit_for_mean
        self.archive = archive

        data_path = os.path.expanduser(settings['cascade_at_gma_out_dir'])
        # input_filename = (os.path.join(data_path, '%d/full/%d/%s/%s/%d_constrained.db' %
        input_filename = (os.path.join(data_path, '%d/full/%d/%s/%s/%d.db' %
                                       (model_version_id, location_id, sex, '_'.join(map(str, times)), model_version_id)))
        assert input_filename == filename_in
        self.input_filename = filename_in
        # filename = (os.path.join(data_path, '%d/full/%d/%s/%s/posterior/%d_constrained.db' %
        filename = (os.path.join(data_path, '%d/full/%d/%s/%s/posterior/%d.db' %
                                 (model_version_id, location_id, sex, '_'.join(map(str, times)), model_version_id)))
        # Copy the baseline fit to the local directory
        logger.info ("Copying", self.input_filename)
        logger.info ("     to", filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        shutil.copy2(self.input_filename, filename)
        self.DB = DismodDbAPI(filename)

        self.parent_node_id = int(self.DB.options.parent_node_id)
        self.parent_name = node_id2name(self.DB, self.parent_node_id)

        self.include_locations = include_locations
        db_info(self.DB)

    def __call__(self, max_num_iter_fixed = 50,
                 redo_parent_fit = True,
                 redo_posterior = True,
                 n_samples = 5,
                 fit_parent_as_if_child = False,
                 fit_children = True,
                 plot_p = True,
                 plot_simulate_residuals = False):
        self.plot_p = plot_p
        self.n_samples = n_samples

        logger.info ("\nWhat is causing the mismatch between the plotted mtall adjusted data and model values?")
        logger.info ('omega value prior %s' % self.DB.prior[['omega' in p.prior_name for i,p in self.DB.prior.iterrows()]])

        self.checkpoint = None
        if self.archive:
            self.checkpoint['input'] = copyDB(self.DB, 'input')

        set_max_iters(self.DB, max_num_iter_fixed = max_num_iter_fixed)

        # Clear tables to recompute
        if redo_parent_fit:
            del self.DB.fit_var
            del self.DB.sample
        if redo_posterior:
            del self.DB.sample
            
        # Compute (if necessary) the initial fit
        if ('fit_var' not in self.DB.tables or self.DB.fit_var.empty):
            logger.info("Computing the initial fit for %s" % self.DB.filename)
            self.DB = self.fit_parent(self.DB)
        if self.checkpoint:
            self.checkpoint['parent_fit'] = copyDB(self.DB, 'parent_fit')
        if __my_debug__:
            DB = tempfile_DB(self.DB)
            run_AT_commands(DB.filename, ['set start_var fit_var'])
            if self.checkpoint:
                self.checkpoint['start_at_fit_var'] = DB.var.merge(DB.start_var, left_on='var_id', right_on='start_var_id').merge(DB.rate, how='left')

        # Compute the posterior
        # dirname = os.path.dirname(filename) 
        # os.makedirs(dirname, exist_ok=True)


        # FIXME -- Refactor the following -- START
        # This causes an import conflict -- DB_posterior.py imports priors.py
        logger.debug('# FIXME -- Refactor the following -- START')
        from DB_posterior import sample, posterior
        # FIXME -- Refactor the following -- END

        from cascade_at_gma.drill_no_csv.DB_posterior import posterior # FIXME -- circular import here
        self.DB.prior, self.DB.smooth_grid = posterior(DB, use_fit_for_mean = True)

        if __my_debug__:
            self.debug()

        if self.checkpoint:
            self.checkpoint['posterior'] = copyDB(self.DB, 'posterior')

        if plot_simulate_residuals:
            plot_simulate_residuals(self.DB)

        # Run rerun the parent node, except with rate priors provided by the posterior sample statistics
        if fit_parent_as_if_child:
            self.fit_parent_as_if_child(self.DB)

        # Run the child nodes, with rate priors (sample posteriors) adjusted by the random effects
        self.parent_random_effects = random_effects_at_solution(self.DB)
        self.children = initialize_children(self.DB, self.parent_random_effects, self.model_version_id, self.include_locations, self.times)
        if fit_children:
            for DB in self.children:
                self.fit_child(DB)

    def fit_parent(self, DB):
        fn = DB.filename
        run_AT_commands(fn, ['init', 'fit both'])
        run_AT_commands(fn, commands='depend')
        TableDescriptions(fn, symbolic_pdfs = False)
        if self.plot_p:
            from cascade_at_gma.lib.plot_fit_metrics import TestAndPlot
            tp = TestAndPlot(fn, surface_time = self.times, surface_age = self.ages, plot_data_extent=True, time_window_for_plots = 2.51)
            tp(pdf_p = True, adjust_data=True, logscale=True)
        return DB
        
    def fit_child(self, DB):
        node_id = int(DB.options.parent_node_id)
        fn = DB.filename
        DB.data = get_descendant_data(DB, self.model_version_id, node_id, self.ages, self.times, root_dir = None)
        run_AT_commands(DB.filename, ['init', 'fit fixed'])
        run_AT_commands(fn, commands='depend')
        TableDescriptions(DB.filename, symbolic_pdfs = False)
        if self.plot_p:
            from cascade_at_gma.lib.plot_fit_metrics import TestAndPlot
            tp = TestAndPlot(DB.filename, surface_time = self.times, surface_age = self.ages, plot_data_extent=False, time_window_for_plots = 2.51)
            tp(pdf_p = True, adjust_data=True, logscale=True)
        return DB

    def fit_parent_as_if_child(self, DB):
        #    Note -- This runs the parent as if it were a child in the cascade. 
        #            To run a child, apply the child random effect estimates to the parent posteriors,
        #            and run the child area rather than the parent.

        logger.info ('Mulcov prior for fitting parent as if it were a child:')
        logger.info (mulcov_values(DB))
        if self.checkpoint:
            logger.info ("Mulcov prior for initial fit:")
            logger.info (mulcov_values(self.checkpoint['parent_fit']))
            logger.info ("Mulcov posterior to initial fit:")
            logger.info (mulcov_values(self.checkpoint['posterior']))

        prior = DB.prior
        mask = ['pini_value_prior' in name for name in prior.prior_name]
        prior.loc[mask, 'density_id'] = 0
        DB.prior = prior

        if _make_nonuniform_prior_stds_nonzero_:
            DB.prior = make_nonuniform_prior_stds_nonzero(DB)

        run_AT_commands(DB.filename, ['init', 'set start_var prior_mean'])
        if __my_debug__ and self.checkpoint: 
            # Make sure the priors are correct
            check_priors(DB, self.checkpoint['parent_fit'].fit_var, parent_node_id) 
        run_AT_commands(DB.filename, ['fit both'])

        if self.plot_p:
            from cascade_at_gma.lib.plot_fit_metrics import TestAndPlot
            tp = TestAndPlot(DB.filename, surface_time = self.times, surface_age = self.ages, plot_data_extent=False, time_window_for_plots = 2.51)
            tp(pdf_p = True, adjust_data=True, logscale=True)
        run_AT_commands(fn, commands='depend')
        TableDescriptions(DB.filename, symbolic_pdfs = False)
        return DB


    def debug(self):
        if self.checkpoint:
            # Make sure fit_var hasn't been mucked up by the sampling process
            assert np.allclose(self.checkpoint['parent_fit'].fit_var.fit_var_value, self.DB.fit_var.fit_var_value, **tol15), "The sampling process changed the fit_var table."

            # Make sure the prior_stats are set to fit_var values 
            assert np.allclose(prior_stats.sample_mean,  self.checkpoint['start_at_fit_var'].start_var_value, **tol15), "The prior_stats agree with start_var when set to fit_var."

        # Make sure the fixed_effects posterior statistics were moved into the priors correctly
        cols = ['node_id', 'var_id', 'var_type', 'mean', 'fit_var_value']
        fit_and_prior = self.DB.var.merge(self.DB.smooth_grid).merge(self.DB.fit_var, left_on='var_id', right_on='fit_var_id').merge(sample_prior, left_on='value_prior_id', right_on='prior_id')[cols]
        fixed_effects_mask = ~ ((fit_and_prior.node_id != parent_node_id) & (fit_and_prior.var_type == 'rate'))
        assert np.allclose(fit_and_prior.loc[fixed_effects_mask, 'mean'], fit_and_prior.loc[fixed_effects_mask, 'fit_var_value'], **tol14), "Priors for the fixed effects do not match prior_stats."

        # For the current problem formulation approach, we can't rerun the parent starting from only priors because, at the top level, the random effects (e.g., child rates) age/time 
        # grid is usually sparse, whereas the top level fit creates a dense age/time grid of parent rates, which then become a dense age/time grid of child rates at the next level.
        # To rerun the parent from the parent solution, we can either initialize start_var from:
        #   A) the priors alone by making the random effects age/time grid complete (e.g. dense) at the world level, or
        #   B) a mixture of priors for everything except the random effects, and from fit_var for the random effects.
        # To simplify problem setup at the world level, I'm going with option B.

        # Confirm that the mixed prior/fit_var approach is the same as initializing everything from fit_var
        # (e.g., start_var[~ random_effects] = priors (see note below), and
        #        start_var[random_effects] = fit_var[random_effects])
        # Note, the Dismod_AT 'set start_var prior_mean' command controls the assignment start_var from priors.
        # start_var = priors

        tmpDB = tempfile_DB(self.DB)
        run_AT_commands(tmp.DB.filename, ['init', 'set start_var prior_mean'])
        # start_var[random_effects] = fit_var[random_effects]
        start_and_fit = tmpDB.start_var.merge(fit_var, left_on='start_var_id', right_on='fit_var_id')
        random_effects_mask = ((tmpDB.var.node_id != parent_node_id) & (tmpDB.var.var_type == 'rate'))
        start_and_fit.loc[random_effects_mask, 'start_var_value'] = start_and_fit.loc[random_effects_mask, 'fit_var_value']

        # Confirm that start_var == fit_var
        assert np.allclose(start_and_fit.start_var_value, fit_var.fit_var_value, **tol14)

if (__name__ == '__main__'):

    try: del DB
    except: pass

    try: del sample_prior
    except: pass

    model_version_id = 100667

    # loc_ids = 100               # NA
    loc_ids = 102               # USA
    # loc_ids = 1                 # Earth
    location_id = loc_ids

    times = (1970,1980,1990,1995,2000,2005,2010,2016)
    times = [2010]

    # include_locations = (1,64,100,101,102,523,527,544,572)
    include_locations = (1,64,100,102,523)
    # include_locations = None

    from cascade_at_gma.drill_no_csv.DB_import import import_DB
    from cascade_at_gma.drill_no_csv.DB_groom import groom_DB
    from cascade_at_gma.drill_no_csv.DB_fit import fit_DB

    for location_id, sex in ((100,'both'), (102,'female')):
        _archive_ = False
        fn, cascade = import_DB(model_version_id = model_version_id, loc_id = location_id, times = times, time_window_for_fit = _time_window_for_fit_, archive = _archive_,
                                include_locations = include_locations, 
                                include_covariates = ['x_sex','x_one','x_c_LDI_pc','x_c_mean_BMI','x_s_marketscan_all_2000','x_s_marketscan_all_2010','x_s_marketscan_all_2012'],
                                include_integrands = ['incidence','prevalence','remission','mtall','mtexcess','mtspecific', 'susceptible', 'withC'])
        groom_DB(fn, times = times, archive = _archive_, time_window_for_fit = _time_window_for_fit_)

        if _zero_iterations_:
            max_iters = -1
        else:
            max_iters = 21
        fit_DB(fn, max_iters=max_iters, archive = _archive_)
        stats = FitStatistics(fn, model_version_id, location_id, times, sex=sex, include_locations=include_locations, archive = False)
        stats(redo_parent_fit = not True,
              redo_posterior = _redo_posterior_,
              fit_parent_as_if_child = not True,
              fit_children = True,
              n_samples = 5,
              max_num_iter_fixed = max_iters,
              plot_p = False and (True if max_iters > 10 else False),
              plot_simulate_residuals = False)
