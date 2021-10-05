import os, sys
import shutil
import pandas as pd; pd.set_option('expand_frame_repr', False)
import numpy as np
import pdb; from pdb import set_trace
from scipy.interpolate import NearestNDInterpolator

import logging
__log_to_file__ = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('weighted_residuals')
del logging

from cascade_at_gma.drill_no_csv.priors import get_age_ranges, sex_id2sex_dict
from cascade_at_gma.lib import utilities
from cascade_at_gma.lib.cached_property import cached_property
from cascade_at_gma.lib.dismod_db_api import DismodDbAPI
from cascade_at_gma.lib.dismod_db_functions import node_id2location_id
from cascade_at_gma.lib.get_covariate_estimates import get_covariate_estimates
from cascade_at_gma.lib.table_description import TableDescriptions
from cascade_at_gma.lib.constants import sex_name2covariate, mortality_integrands
from cascade_at_gma.lib.utilities import int_or_float

def get_x_sex(DB):
    if not DB.sex_covariate.empty:
        if 'c_covariate_name' in DB.covariate:
            x_sex = f'x_{DB.sex_covariate.covariate_id.squeeze()}'
        else:
            x_sex = DB.sex_covariate.xcov_name.squeeze()
        return x_sex

def get_x_one(DB):
    if not DB.sex_covariate.empty:
        if 'c_covariate_name' in DB.covariate:
            x_one = DB.covariate.loc[['one' in n for n in DB.covariate.c_covariate_name], 'covariate_name'].squeeze()
        else:
            x_one = f"x_{DB.covariate.loc[['one' in n for n in DB.covariate.covariate_name], 'covariate_id'].squeeze()}"
        return x_one
    
class CovariateInterpolator(object):
    def __init__(self, DB, node_id = None):
        """
        Interpolates country covariate values to the requested times.
        1) Get the covariate values for the location.
        2) If both is not present in the covariates, average male and female into a both covariate
        3) Return a nearest neighbor interpolator that interpolates on sex and time
        """

        filter_years = True

        if node_id is None:
            node_id = int(DB.options.parent_node_id)

        if not DB.sex_covariate.empty:
            self.x_sex = x_sex = get_x_sex(DB)
        else:
            self.x_sex = x_sex = None
        covariate_name_short = () if DB.country_covariates.empty else tuple(DB.country_covariates.covariate_name_short)
        if not covariate_name_short:
            logger.warn('DB.country_covariates returned no covariate names.')
        args = (covariate_name_short,)
        if not DB.cascade_option.empty:
            model_version_id = int(DB.cascade_options.model_version_id)
            gbd_round_id = int(DB.cascade_options.gbd_round_id)
        else:
            model_version_id = gbd_round_id = None
        kwds = dict(location_id = node_id2location_id(DB, node_id),
                    model_version_id = model_version_id,
                    gbd_round_id = gbd_round_id)
        ccov_vals = get_covariate_estimates(*args, **kwds)
        if ccov_vals.empty:
            self.ccov_vals = ccov_vals
            self.interp = None
        else:
            if filter_years:
                # Filter years to reduce the chances we there are NaN covariate values
                time_grid = [int_or_float(x) for x in DB.cascade_options.time_grid.split()]
                time_min, time_max = min(time_grid), max(time_grid)
                year_id = np.asarray(sorted(ccov_vals.year_id.unique()))
                min_indx, max_indx = np.argmin(abs(year_id-time_min)), np.argmin(abs(year_id-time_max))
                year_id_min, year_id_max = year_id[max(0, min_indx-1)], year_id[min(len(year_id)-1, max_indx+1)]
                mask = (year_id_min <= ccov_vals.year_id) & (ccov_vals.year_id <= year_id_max)
                ccov_vals = ccov_vals[mask]

            ccov_vals[x_sex] = ccov_vals.sex_id.apply(lambda x: sex_id2sex_dict[x])
            ccov_dict = dict(DB.country_covariates[['covariate_name_short', 'xcov_name']].values)
            ccov_vals.rename(columns = ccov_dict, inplace=True)
            ccov_vals.rename(columns = {'year_id': 'time'}, inplace=True)
            ccov_vals['time_lower'] = ccov_vals['time_upper'] = ccov_vals['time']
            if sex_name2covariate['both'] not in ccov_vals[x_sex].unique():
                both = ccov_vals.groupby(['location_id', 'time', 'age_group_id'], as_index=0).mean()
                both['sex_id'] = 0
                ccov_vals = ccov_vals.append(both).reset_index(drop=True)
            ages = get_age_ranges(ccov_vals.age_group_id.unique()).rename(columns={'age_group_years_start': 'age_lower', 'age_group_years_end': 'age_upper'})
            ccov_vals = ccov_vals.merge(ages, how='left', on = 'age_group_id')

            self.interp = NearestNDInterpolator(ccov_vals[[x_sex, 'time']], ccov_vals.values)
            self.ccov_vals = ccov_vals

    def __call__(self, sex, time_lower, time_upper):
        rtn = self.ccov_vals[(self.ccov_vals[self.x_sex] == sex) &
                             (self.ccov_vals.time_lower == time_lower) &
                             (self.ccov_vals.time_upper == time_upper)]
        if not rtn.empty:
            rtn = pd.DataFrame(rtn.iloc[0]).T
            rtn['time'] = rtn[['time_lower', 'time_upper']].mean(axis=1)
        else:
            time = np.mean([time_lower, time_upper])
            if not self.ccov_vals.empty:
                rtn = pd.DataFrame([self.interp(sex, time)],
                                   columns = self.ccov_vals.columns)
                rtn['time_lower'] = time_lower
                rtn['time_upper'] = time_upper
                rtn['time'] = rtn[['time_lower', 'time_upper']].mean(axis=1)
        return rtn

class AdjustMeasurements(object):
    """
    (Note -- On 2017.11.30 Theo said data should be adjusted to reference study covariates, but not to the parent country covariates.)

    Let:
    A = model prediction at the time, age, location, country covariate, and study covariate values of the measured data
        (Note: A is simply the model prediction of the measured data)
    B = model prediction at the time, age, location, country covariate values of the measured data, and study covariates = referece

    adjusted_data = measured_data / A * B

    Note: The data_subset table identifies the set of rows in the data table where:
            a) the node is a descendant of the parent node
            b) all of the covariates satisfy the max_difference criteria
          The fit_data_subset is the data fit at data_subset
    """
    def __init__(self, DB, dismod_at = 'dismod_at'):
        if DB.data_subset.merge(DB.data, how='left').empty:
            self.adjusted_data = pd.DataFrame([], columns=DB.data_subset.merge(DB.data, how='left').columns)
        else:
            self.DB = DB
            self.dismod_at = dismod_at

            try:
                self.parent_node_id = int(DB.options.parent_node_id)
            except:
                parent_node_id = int(DB.node.loc[DB.node.node_name == DB.options.parent_node_name, 'node_id'])
            if not DB.sex_covariate.empty:
                self.x_sex = x_sex = get_x_sex(DB)
                self.sex_ref = DB.covariate.loc[DB.covariate.covariate_id == DB.sex_covariate.covariate_id.squeeze(), 'reference'].squeeze()
            else:
                self.x_sex = None
                self.sex_ref = 0
                
            self.ccovs = DB.country_covariates.xcov_name.tolist() if not DB.country_covariates.empty else []
            self.scovs = DB.study_covariates.xcov_name.tolist() if not DB.study_covariates.empty else []

            # Set the sample table for predicting the fit
            self.adjusted_data = self._adjusted_data.copy()
            self.adjusted_data['model_at_adjusted_covs'] = self.model_at_adjusted_covs.avg_integrand
            for cov in self.ccovs + self.scovs:
                self.adjusted_data[cov+'_adj'] = self.model_at_adjusted_covs[cov]
            self.adjusted_data['model_at_data_covs'] = self.model_at_data_covs.avg_integrand
            for cov in self.ccovs + self.scovs:
                self.adjusted_data[cov+'_data'] = self.model_at_data_covs[cov]

    @cached_property
    def avgint(self):
        # Predict for only the data rows identified in the data_subset table
        avgint = self.DB.data_subset.merge(self.DB.data, how='left')
        avgint['avgint_id'] = avgint['data_subset_id']
        return avgint
        
    @cached_property
    def model_at_data_covs(self): # Part A) in the comments
        "Compute model prediction at the data covariate and node_id's (should be the same as fit_data_subset)"
        avgint = self.avgint.copy()
        # Set the covariates = data covariates
        self.DB.avgint = avgint

        # Predict data at the data covariates
        utilities.system("%s %s predict fit_var" % (self.dismod_at, self.DB.filename))
        prediction = self.avgint.merge(self.DB.predict, on='avgint_id', how='left').rename(columns = {'avgint_id' : 'data_subset_id'})
        return prediction

    if 0:
        # On 2017.11.30 Theo said data should be adjusted to reference sex and study covariates, but not to the parent country covariates.
        # But Brad said on 5/21 that what he really wants is to have them adjusted to the current location.
        # So -- back to the version in this if else clause
        @cached_property
        def model_at_adjusted_covs(self): # Part B) in the comments
            "Compute model predictions at the reference study covariates and parent country covariates and node_id"

            logger.warn('# On 2017.11.30 Theo said data should be adjusted to reference study covariates, but not to the parent country covariates.')

            avgint = self.avgint.copy()
            # Set study and sex covariates = reference values
            scovs = [] if self.DB.study_covariates.empty else self.DB.study_covariates.xcov_name.tolist()
            sex_cov = self.DB.sex_covariate.xcov_name.tolist()
            avgint.loc[:, scovs + sex_cov] = None
            self.DB.avgint = avgint

            # Predict data with study covariates = reference, sex_covariate = reference, and country covariates = parent
            utilities.system("%s %s predict fit_var" % (self.dismod_at, self.DB.filename))
            prediction = avgint.merge(self.DB.predict, on='avgint_id', how='left').rename(columns = {'avgint_id' : 'data_subset_id'})
            return prediction

    else:
        @cached_property
        def model_at_adjusted_covs(self): # Part B) in the comments
            "Compute model predictions at the reference study covariates and the parent node id country covariates"

            avgint = self.avgint.copy()
            ci = CovariateInterpolator(self.DB, node_id = self.parent_node_id) # Call syntax: ci(sex, time_lower, time_upper)
            if ci.interp:
                # Get the country covariates for this node
                time_cols = ['time_lower', 'time_upper']
                ccov_cols = [self.x_sex] + time_cols
                interp_keys = avgint[ccov_cols].drop_duplicates().values
                node_ccovs = pd.DataFrame([ci(*k).squeeze() for k in interp_keys], columns = ci(0,0,0).columns)
                both_average = node_ccovs[node_ccovs[self.x_sex] != 0].groupby(time_cols, as_index=False).mean()
                # Add both if missing
                both_average = node_ccovs[node_ccovs[self.x_sex] != 0].groupby(time_cols, as_index=False).mean()
                node_ccovs = node_ccovs.append(both_average).drop_duplicates(subset=ccov_cols).reset_index(drop=True)

            # Set all covariates to their reference
            ref_covs = self.ccovs + self.scovs
            ref_covs.remove(self.x_sex)
            avgint[ref_covs] = None 

            # Set the country covariates to this node's covariate values
            avgint.drop(self.ccovs, axis=1, inplace=True)
            if ci.interp:
                avgint = avgint.merge(node_ccovs[self.ccovs + ccov_cols], on = ccov_cols, how='left')
            # For some reason, even thought the country covariates for the children are set to the parent values, node_id still affects the avg_integrand calculation.
            # It must be covariate multipliers on the country covariates. I think that setting the node_id to the parent_node_id here is incorrect.
            # # Set node_id = parent
            # avgint['node_id'] = self.parent_node_id

            # Predict with study covariates = reference, country covariates = parent
            self.DB.avgint = avgint
            utilities.system("%s %s predict fit_var" % (self.dismod_at, self.DB.filename))
            return avgint.merge(self.DB.predict, on='avgint_id', how='left').rename(columns = {'avgint_id' : 'data_subset_id'})

    @property
    def _adjusted_data(self):
        self.DB.avgint = self.avgint
        # FIXME -- I think this next line was incorrect. It builds a random sample from fit_var, but it isn't used
        # status = utilities.system("%s %s sample fit_var" % (self.dismod_at, self.DB.filename), break_on_error=True)
        adjusted = self.avgint
        adjusted['adjustment'] = self.model_at_adjusted_covs.avg_integrand/self.model_at_data_covs.avg_integrand
        adjusted['adj_value'] = adjusted['meas_value'] * adjusted['adjustment']
        adjusted['adj_std'] = adjusted['meas_std'] * adjusted['adjustment']
        adjusted.drop('avgint_id', axis=1)
        return adjusted

def adjust_measurements(DB, dismod_at = 'dismod_at'):
    am = AdjustMeasurements(DB, dismod_at)
    return am.adjusted_data

def reference_covariates(DB, node_id = None, sex = [-.5,0,+.5], times = None, method='nearest'):
    """
    Interpolate the country covariate values table to the requested node_id and times.
    Set study covariates to their reference values (e.g. None).
    From the DB.data table:
    1) select data rows for the node_id
    2) group the selected data by the mean value of time_lower, time_upper
    3) compute the mean covariate value for the groups
    4) if sex is both, for those years without both data, include the average of male and female
    5) interpolate the grouped results to the presented times using an interpolation scheme
    """
    from scipy.interpolate import griddata
    from scipy.interpolate import NearestNDInterpolator
  
    mask = [_.endswith('one') for _ in DB.covariate.covariate_name]
    x_one = get_x_one(DB)

    sex = utilities.force_tuple(sex)
    ci = CovariateInterpolator(DB, node_id = node_id)
    if ci.interp is None:
        covs = pd.DataFrame()
    else:
        args = [(s, t) for s in sex for t in times]
        x = [ci(s, t, t).squeeze() for s in sex for t in times]
        covs = pd.DataFrame(x, columns=ci(0,2000,2000).columns).reset_index(drop=True)
        covs['time_lower'] = covs['time_upper'] = covs['time']
    ccovs = DB.country_covariates.xcov_name.tolist() if not DB.country_covariates.empty else []
    for cov in ccovs:
        if not np.isfinite(covs[cov]).all():
            logger.error('Covariate %s had missing values -- using the reference value.' % cov)
    scovs = DB.study_covariates.xcov_name.tolist() if not DB.study_covariates.empty else []
    for cov in scovs + [x_one]:
        covs[cov] = None
    return covs

def model_surface_avgint(DB, node_id = None, integrand_ids = None,
                         ages = tuple([0,1] + list(range(10,101,10))), dage = 0,
                         times = (1990,1995,2000,2005,2010,2015), dtime = 0):
    """
    Create an avgint table for model prediction at age/time gridpoints.
    Note: dismod predict predicts for covariate values in the avgint table. If null (reference) values 
    are used, then the prediction uses the reference covariate values in the prediction, and the prediction 
    will fit the measured data iff the covariate references the covariate values in the data table.
    """
    mortality_ids = DB.integrand.loc[[n.startswith('mt') for n in DB.integrand.integrand_name], 'integrand_id'].values

    if integrand_ids is None:
        integrand_ids = DB.integrand.integrand_id.values

    if node_id is None:
        node_id = int(DB.options.parent_node_id)

    # Country covariates should be computed at node_id, and the appropriate time
    # Country covariates are age-standardized, so there is no age variation
    # Study covariates should be at their reference values

    try:
        sex = DB.covariate[DB.covariate.covariate_id == DB.sex_covariate.covariate_id.squeeze()].reference.squeeze()
        x_sex = x_sex = get_x_sex(DB)
        x_one = get_x_one(DB)
        covs = reference_covariates(DB, node_id = node_id, sex = sex, times = times)
    except:
        set_trace()
    if covs.empty:
        xcov_names = [x_sex, x_one]
    else:
        # gma 7/9/2020 assert all(covs[DB.sex_covariate.xcov_name.squeeze()] == sex), "Sex is wrong"
        covs['node_id'] = node_id
        covs.drop(columns=['age_lower', 'age_upper', 'time_lower', 'time_upper'], inplace=True)

        xcov_names = ['x_%d' % cov_id for cov_id in DB.covariate.covariate_id]
    
    uniform_id = int(DB.density.loc[DB.density.density_name == 'uniform', 'density_id'])

    avgint_list = []
    sys.modules.pop('cascade_at_gma.tests.test_omega.test_omega', None)
    from cascade_at_gma.tests.test_omega.test_omega import use_age_intervals
    if use_age_intervals:
        for integrand_id in integrand_ids:
            for t in times:
                for (al, au) in ages:
                    dct = {'integrand_id' : integrand_id,
                           'node_id' : node_id,
                           'weight_id' : 0,
                           'age_lower' : al, 'age_upper' : au,
                           'time_lower' : t-dtime, 'time' : t, 'time_upper' : t+dtime,
                           'density_id' : uniform_id, 'data_name' : '(%g,%g)' % (al,t)}
                    avgint_list.append(dct)
    else:
        for integrand_id in integrand_ids:
            for t in times:
                for a in ages:
                    dct = {'integrand_id' : integrand_id,
                           'node_id' : node_id,
                           'weight_id' : 0,
                           'age_lower' : a-dage, 'age_upper' : a+dage,
                           'time_lower' : t-dtime, 'time' : t, 'time_upper' : t+dtime,
                           'density_id' : uniform_id, 'data_name' : '(%g,%g)' % (a,t)}
                    avgint_list.append(dct)
        
    df = pd.DataFrame(avgint_list)
    if not covs.empty:
        df = df.merge(covs, how='left', on=['node_id', 'time'])
    else:
        df[x_sex] = DB.covariate.loc[[n.endswith('sex') for n in DB.covariate.covariate_name], 'reference'].squeeze()
        # Drew convinced me that the correct reference for x_one is zero, rather than 1, hence this change to a hardcoded 1.
        # df[x_one] = DB.covariate.loc[[n.endswith('one') for n in DB.covariate.covariate_name], 'reference'].squeeze()
        df[x_one] = 1
    df['avgint_id'] = list(df.index)

    if len(DB.subgroup) > 1:
        print ("WARNING -- Don't know which subgroup to predict for -- using subgroup_id 0")
    df['subgroup_id'] = 0
    cols = ['avgint_id', 'integrand_id', 'node_id', 'subgroup_id', 'age_lower', 'age_upper', 'time_lower', 'time', 'time_upper', 'weight_id'] + xcov_names
    avgint = df[cols]

    return avgint

def weighted_residuals(DB, df):
    std_mulcovs = get_std_mulcovs(DB)
    min_cv = float(getattr(DB.options, 'minimum_meas_cv', 0))
    df['delta'] = adj_std(df, std_mulcovs, min_cv)
    df['residual'] = None
    for i, (density_id, dname) in DB.density.iterrows():
        mask = df.density_id == density_id
        d = df[mask]
        if dname in ('uniform'):
            r = 0
        elif dname in ('gaussian', 'laplace', 'students'):
            r = (d.meas_value - d.avg_integrand) / d.meas_std
        elif dname in ('log_gaussian', 'log_laplace', 'log_students'):
            r = (np.log(d.meas_value + d.eta) - np.log(d.avg_integrand + d.eta)) / sigma(d.meas_value, d.delta, d.eta)
        df.loc[mask, 'residual'] = r
    return df.residual

def get_meas_mulcovs(DB):
    return (DB.var[DB.var.var_type == 'mulcov_meas_value'].merge(DB.integrand, how='left').merge(DB.covariate).merge(DB.fit_var, left_on='var_id', right_on='fit_var_id')
           .drop(labels=['residual_value', 'residual_dage', 'residual_dtime', 'lagrange_value', 'lagrange_dage', 'lagrange_dtime'], axis=1))
def get_std_mulcovs(DB):
    return (DB.var[DB.var.var_type == 'mulcov_meas_std'].merge(DB.integrand, how='left').merge(DB.covariate).merge(DB.fit_var, left_on='var_id', right_on='fit_var_id')
            .drop(labels=['residual_value', 'residual_dage', 'residual_dtime', 'lagrange_value', 'lagrange_dage', 'lagrange_dtime'], axis=1))
def get_rate_mulcovs(DB):
    return (DB.var[DB.var.var_type == 'mulcov_rate_value'].merge(DB.rate, how='left').merge(DB.covariate).merge(DB.fit_var, left_on='var_id', right_on='fit_var_id')
            .drop(labels=['residual_value', 'residual_dage', 'residual_dtime', 'lagrange_value', 'lagrange_dage', 'lagrange_dtime'], axis=1))

def sigma(meas_value, meas_std, eta):
    """
    Log-transformed standard deviation.
    See http://moby.ihme.washington.edu/bradbell/dismod_at/statistic.htm#Log-Transformed%20Standard%20Deviation,%20sigma
    """
    return np.log(meas_value + eta + meas_std) - np.log(meas_value + eta)

def adj_std(df, std_mulcovs, min_cv):
    """
    The adjusted standard deviation. See http://moby.ihme.washington.edu/bradbell/dismod_at/data_like.htm.
    Also see http://moby.ihme.washington.edu/bradbell/dismod_at/avg_integrand.htm
    which shows that the weighting function is included in the avg_integrand.
    This function is an approximation of the adjusted standard deviation, as the covariate multipliers 
    and weighting function should be applied inside the integral, rather than outside 
    (e.g., scaling avg_integrand) as they are here.
    """
    rtn = pd.DataFrame()
    rtn['Delta'] = rtn['adj_meas_std'] = np.max([df.meas_std.values, min_cv * df.meas_value.abs().values], axis=0)
    for integrand_id in df.integrand_id.unique():
        df_mask = df.integrand_id == integrand_id
        for covariate_id, mulcov, reference in std_mulcovs.loc[std_mulcovs.integrand_id == integrand_id, ['covariate_id', 'fit_var_value', 'reference']].values:
            x_cov = 'x_%d' % covariate_id
            rtn.loc[df_mask, 'adj_meas_std'] += df.loc[df_mask, x_cov].values * mulcov
    return rtn.adj_meas_std

if (__name__ == '__main__'):
    if len(sys.argv) == 2:
        sqlite_filename = sys.argv[1]
    else:
        sqlite_filename = '/Users/gma/Projects/IHME/cascade_at_gma.data-tmp/100667/full/1/both/1990_1995_2000_2005_2010_2015/fit/100667.db'
        sqlite_filename = '/Users/gma/Projects/IHME/cascade_at_gma.data-tmp/100667/full/100/female/1990_1995_2000_2005_2010_2015/fit/100667.db'

    dismod_at = 'dismod_at'
    test_db = '/tmp/test_weighted_residuals.db'
    shutil.copy2(sqlite_filename, test_db)

    DB = DismodDbAPI(test_db)

    # Setup the plotter -- Import matplotlib locally for thread safety
    from matplotlib import pyplot as plt
    plt.close('all'); plt.interactive(1)

    if not (('fit_var' in DB.tables) and ('fit_data_subset' in DB.tables)):
        # Run fit
        utilities.system("%s %s %s" % (dismod_at, test_db, 'init'))
        utilities.system("%s %s %s" % (dismod_at, test_db, 'fit fixed'))

    self = AdjustMeasurements(DB)
    self.adjusted_data

    def plot_data(ax, data, integrand_id, time_lwr, time_upr, column = 'meas_value', **kwds):
        data = data[(data.integrand_id == integrand_id) & (time_lwr<= data.time_lower) & (data.time_upper <= time_upr)]
        if not data.empty:
            ax.plot((data.age_lower + data.age_upper)/2, data[column], label = column, **kwds)

    def plot_residuals(ax, DB, integrand_id, time_lwr, time_upr, **kwds):
        data = DB.data_subset.merge(DB.data, how='left').merge(DB.fit_data_subset)
        data = data[(DB.data_subset.merge(DB.data, how='left').integrand_id == integrand_id) & (time_lwr<= DB.data_subset.merge(DB.data, how='left').time_lower) & (DB.data_subset.merge(DB.data, how='left').time_upper <= time_upr)]
        if not data.empty:
            ax.plot((data.age_lower + data.age_upper)/2, data.weighted_residual, **kwds)
            xlim = ax.get_xlim()
            ax.plot(xlim, [0,0], '-k')

    def plot_predict(ax, predict, integrand_id, time_lwr, time_upr, **kwds):
        predict = predict[(predict.integrand_id == integrand_id) & (time_lwr<= predict.time_lower) & (predict.time_upper <= time_upr)]
        if not predict.empty:
            plt.figure(integrand_id)
            ax.plot(predict.age_lower, predict.avg_integrand, **kwds)

    # Predict surfaces at the fitted solution, and the reference covariate values
    times = list(map(int_or_float, DB.cascade_options.time_grid.split()))
    covariate_values = reference_covariates(DB, times=times)
    avgint = model_surface_avgint(DB)
    DB.avgint = avgint
    utilities.system("%s %s %s" % (dismod_at, test_db, 'predict fit_var'))
    predict = DB.avgint.merge(DB.predict)
            
    # Plot the unadjusted measured data vs the model.
    xlim = (0,110)

    x_sex = x_sex = get_x_sex(DB)

    data = DB.data
    adj = adjust_measurements(DB)
    data['adj_value'] = adj.adj_value
    data_integrand_ids = sorted(data.integrand_id.unique())

    iplt = 0
    for itime, t in enumerate(times):
        time_lwr, time_upr = t - 2.5, t + 2.5
        for integrand_id, integrand_name in DB.integrand[['integrand_id', 'integrand_name']].values:
            if integrand_id in data_integrand_ids:
                f = plt.figure(itime*10 + integrand_id)
                gs = plt.GridSpec(2,1,height_ratios=[3,1])
                upr, lwr = plt.subplot(gs[0]), plt.subplot(gs[1])
                upr.set_xlim(xlim), lwr.set_xlim(xlim)
                upr.set_yscale('log', nonposy='clip')
                f.show();
                plot_data(upr, data, integrand_id, time_lwr, time_upr, marker='x', ls='', c='grey', column='meas_value')
                plot_data(upr, data, integrand_id, time_lwr, time_upr, marker='+', ls='', c='k', column='adj_value')
                
                for x_s,sex in sex_name2covariate.items():
                    plot_predict(upr, predict[predict[x_sex] == x_s], integrand_id, time_lwr, time_upr, label=sex)
                plot_residuals(lwr, DB, integrand_id, time_lwr, time_upr, marker='+', ls='', c='k', label=None)
                handles, labels = upr.get_legend_handles_labels()
                upr.legend(handles, labels, loc='best')
                upr.set_title(integrand_name + ' ' + str(t))
    
