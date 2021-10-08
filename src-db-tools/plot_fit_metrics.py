#! /usr/bin/env python

import sys
import tempfile
import shutil
import warnings
from pdb import set_trace
from constants import sex_name2dismod_id

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('plot_fit_metrics')
del logging

_dismod_ = 'dmdismod'

_eps_ = 1e-10
_Z_95_ = 1.96                   # Z value for 95% gaussian confidence interval

__all_node_db__ = True

"""
Note: matplotlib pdf plotting seems to require the following to be installed on the Mac:
  sudo port install texlive-latex-extra
  sudo port install texlive-fonts-recommended

This module constructs an omega optimization constraint from mtall or mtother data collected from the mortality database.
The mortality data includes mtall age limits, time limits, and measured values.
We approximate an omega constraint from the mtall measured value, located at the point (mean(age limits) and mean(time limits)).

To impose the constraint on the dismod_at optimizer, we need to do the following:
1) Add a constraint prior.prior_name to the prior_table, with prior.lower = prior.upper = data.meas_value.
2) Add prior.prior_name to the smooth_table, with the appropriate n_age and n_time values required to define a gridded constraint.
3) Add n_age * n_time entries to the smooth_grid_table, which defines the age and time corresponding to each prior.
4) Correct the parent_smooth_id for the omega rate in rate_table.

The dismod_at fit commend produces the fit_data_subset_table, which contains integrand values corresponding to certain elements of the data table.
If the age and time limits in the data table are collapsed to 0 width, the mtall values in the fit_data_subset_table should exactly match the omega constraint.
(The routine test_omega_constraint checks to make sure this is true.)

We have 2 things to do -- add and remove.
1) Add rows to tables to implement the omega constraint.
2) Delete rows from tables that contain alternative omega fitting instructions.
Note: This gets a little complicated as the age and time table need to be sorted ascending. Adding to them requires mucking around with all the other tables
that use age_id and time_id.

See ~/Projects/IHME/install/dismod_ode-20140715/build/python/mtall2omega.py for the original dismod_ode version of this process.
"""

logger.info("""
Brad thinks:
1) On the prior plots, do not plot curves between the prior points
   Include the standard deviations on the points.
   Include the residuals
2) Add location names
3) Key for the +, x, O, and so on. Maybe a separate page.
Theo says:
1) Make all rates positive.
2) Make the y-axis scaling constant across years.
""")

_data_extent_plotting_limit_ = 5000  # If there are more than this number of data points, do not plot age/time/variation lines with the data.

if 1:
    try: del plotter
    except: pass

import sys
import os
from collections import OrderedDict
import numpy as np
import sqlite3
import itertools
import shutil

import matplotlib
if (__name__ != '__main__'):
    matplotlib.use('MacOSX')           # Agg seems to be thread-safe
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import cm 
from matplotlib.ft2font import FT2Font
from matplotlib.font_manager import FontProperties

import pandas as pd
pd.set_option('expand_frame_repr', False)

from pdb import set_trace

from functools import lru_cache as cached_property
from dismod_db_api import DismodDbAPI

import utilities
from utilities import startswith_filter, df_combinations
from utilities import sex2sex_name
from cached_property import cached_property
from weighted_residuals import model_surface_avgint, adjust_measurements

if 0:
    from cascade_at_gma.lib import utilities
    from cascade_at_gma.lib.cached_property import cached_property
    from cascade_at_gma.lib.constants import rate2integrand
    from cascade_at_gma.lib.dataframe_extensions import DataFrame
    from cascade_at_gma.lib.dismod_db_api import DismodDbAPI
    from cascade_at_gma.lib.utilities import startswith_filter, df_combinations
    from cascade_at_gma.lib.utilities import sex2sex_name
    sys.modules.pop('cascade_at_gma.lib.weighted_residuals', None)
    from cascade_at_gma.lib.weighted_residuals import model_surface_avgint, adjust_measurements

    import cascade_at_gma.drill_no_csv.DB_import

NaN = float('nan')
_framealpha_ = 0.2              # Legend transparency
_time_window_for_plots_ = 2.51

def set_ylim(ax, eps = _eps_):
    lwr, upr = ax.get_ylim()
    if np.ptp([lwr, upr]) == 0:
        upr = lwr + eps
    ax.set_ylim([lwr, upr])

def sort_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0], reverse=True))
    return handles, labels

def fill_0_na(obj, fill = 1e-8):
    shape = obj.shape
    df_p = type(obj) is pd.DataFrame
    if not df_p:
        obj = pd.DataFrame(obj)
    obj.fillna(fill)
    obj[abs(obj) < fill] = fill
    if not df_p:
        obj = obj.values.reshape(shape)
    return obj

def intersect(x,y):
    (x_lwr, x_upr), (y_lwr, y_upr) = x,y
    return not((x_lwr > y_upr) or (x_upr < y_lwr))

def fixup_covariate_names(df, xcov_mapping):
    x_covs = startswith_filter('x_', df.columns)
    rename = dict([(x_cov, xcov_mapping[x_cov]) for x_cov in x_covs if x_cov in xcov_mapping])
    df.rename(columns = rename, inplace=True)
    return df

def holdout_style(c): return dict(ms=8, markerfacecolor="None", markeredgecolor=c)

def plot_fit_var_vs_prior(DB, parent_smooth_ids, child_smooth_ids, effects='fixed', logscale=True, fontsize = 9, pdf=None, model_name = ''):
    assert effects in ('fixed', 'random')
    if effects == 'fixed':
        smooth_ids = parent_smooth_ids
    else:
        smooth_ids = child_smooth_ids
    try:
        parent_node_name = DB.options.parent_node_name
    except:
        [parent_node_name] = DB.node[DB.node.node_id == int(DB.options.parent_node_id)].node_name.values
    prior_and_fit = DB.fit[(DB.fit.var_type == 'rate') & (DB.fit.smooth_id.isin(smooth_ids))]
    mask = (prior_and_fit.node_id == int(DB.options.parent_node_id))
    prior_and_fit = prior_and_fit[mask if effects == 'fixed' else ~mask]
        
    rate_slices = df_combinations(prior_and_fit, 'rate_name')
    for rate_slice in rate_slices:
        rate = str(rate_slice.iloc[0].rate_name)
        slices = tuple(df_combinations(rate_slice, 'time'))

        fig = plt.figure()
        gs = plt.GridSpec(2,1,height_ratios=[3,1])
        upr, lwr = plt.subplot(gs[0]), plt.subplot(gs[1])
        colors = cm.jet(np.linspace(0,1,len(slices)))
        for color,slice in zip(colors, slices):
            time = slice.iloc[0].time
            x0,d0,y0,s0,l0,u0,y1,r = slice.loc[:,('age', 'density_name', 'mean', 'std', 'lower', 'upper', 'fit_var_value', 'residual_value')].T.values
            if logscale:
                y0 = fill_0_na(y0); y1 = fill_0_na(y1)
            # The prior
            try:
                upr.plot(x0,y0, 'o', ms=8, markerfacecolor="None", markeredgecolor=color, label=('prior %s' % time))
            except Exception as ex:
                print ("ERROR: plotting with marker edge color set to an array failed:", ex)
                upr.plot(x0,y0, 'o', ms=8, markerfacecolor="None", markeredgecolor='k', label=('prior %s' % time))
            # The fit
            upr.plot(x0,y1, '-x', c=color, label=('fit %s' % time))
            for x, d, y, s, l, u in zip(x0,d0,y0,s0,l0,u0):
                if __debug__ and d == 'uniform':
                    warnings.warn("FIXME -- When the prior distribution is uniform, should I be plotting the upper and lower bounds for the prior?")
                if rate == 'omega':
                    y = (y,y)
                elif d == 'uniform':
                    y = (l,u) 
                else:
                    try:
                        y = (y-_Z_95_*s, y+_Z_95_*s)
                    except:
                        set_trace()
                upr.plot((x,x),y, '-', c=color)
            if not np.alltrue(np.isfinite(r.astype(float))):
                warnings.warn("FIXME: %s residuals are all NaN -- is this correct?" % rate)
            else:
                lwr.plot(x0,r, '+', ms=5, mew=1, c=color, label=str(time))
        lwr.plot([0,100],[0,0],'--k')
        upr.set_title("%s %s: Prior vs. Fit" % (model_name, parent_node_name,))
        if logscale:
            upr.set_yscale('log', nonpositive='clip')
        set_ylim(upr)
        upr.set_ylabel(rate)
        # upr.legend(*sort_legend(upr), loc='best', fontsize=fontsize-2, framealpha=_framealpha_, frameon=True, shadow=True, facecolor='grey')
        upr.legend(*sort_legend(upr), loc='best', fontsize=fontsize-2, frameon=True, edgecolor='k')
        lwr.set_xlabel('age')
        lwr.set_ylabel('residual')
        if lwr.get_legend():
            # lwr.legend(*sort_legend(lwr), loc='best', fontsize=fontsize-2, framealpha=_framealpha_, frameon=True, shadow=True, facecolor='grey')
            lwr.legend(*sort_legend(lwr), loc='best', fontsize=fontsize-2, frameon=True, edgecolor='k')
        x_lwr, x_upr = upr.get_xlim()
        upr.set_xlim([-1,max(100, x_upr)+1])
        lwr.set_xlim([-1,max(100, x_upr)+1])
        y_lwr, y_upr = upr.get_ylim()
        upr.set_ylim([max(0, y_lwr), max(y_lwr*1.01, y_upr)])
        upr.grid(True)
        if pdf: 
            pdf.savefig(fig)
            plt.close(fig.number)
            
def logscale_plot(ax,x,y,z,*args,**kwds):
    """
    This routine is primarily here because matplotlib does not handle 3D z-axis log scaling correctly -- see the matplotlib documentation.
    """
    logscale = kwds.pop('logscale', None)
    if hasattr(ax, 'plot3D'):
        if logscale:
            z = fill_0_na(np.log(np.asarray(z, dtype=float)))
        try:
            ax.plot(x,y,z,*args,**kwds)
        except Exception as ex:
            logger.error(str(ex))
    else:
        ax.plot(x,z,*args,**kwds)

def plot_model_vs_data(DB,
                       predicted_surface,
                       data_fit_and_adjustments,
                       x_var_name = 'age',
                       projection = None,
                       adjust_data = None,
                       logscale = True, fontsize = 10, plot_data_extent = True,
                       surface_time_window = None,
                       title = "model vs. data",
                       pdf = None,
                       plot_integrands = None,
                       plot_years = None,
                       model_name = ""):
    """
    Plot the fit vs data for the model_variables estimated by the fit command.
    logscale -- plot axis logscale control
    plot_data_extent -- plot age and time extents
    """
    def intervals_from_midpoints(seq):
        if len(seq) == 1:
            return [seq*3]
        else:
            diff = np.diff(seq).tolist()
            diff = np.asarray([diff[0]] + diff + [diff[-1]])/2.0
            return [(mid-d0,mid,mid+d1) for mid,d0,d1 in zip(seq, diff, diff[1:])]

    def plot_age_time_epochs(ax, df, x_var_name = 'age', measurement_name = 'meas_value',
                             color='gray', linestyle='None', label=None, plot_data_extent=False, logscale=logscale,
                             adjust_data=None):
        """
        Plot either the center location (e.g. average age and time) or the age/time/standard deviation extent of a measurement.
        ax: a matplotlib axis
        df: a pandas dataframe containing the data
        """
        model_avgint = 'model_at_adjusted_covs' if adjust_data else 'avg_integrand'
        if x_var_name == 'age':
            columns = ['age_lower','age','age_upper','time_lower','time','time_upper', measurement_name,'meas_std',model_avgint,'hold_out']
        else:
            columns = ['time_lower','time','time_upper', 'age_lower','age','age_upper',measurement_name,'meas_std',model_avgint,'hold_out']

        Xl,X,Xu,Yl,Y,Yu,mv,md,fv,h = df.loc[:, columns].T.values

        mask = np.isfinite(mv.astype(float))
        if not np.all(mask):
            logger.warning('# FIXME -- data_subset causes model values to be NaN -- I am detecting this using np.nan, but really should use data_subset')

        Xl,X,Xu,Yl,Y,Yu,mv,md,fv,h = Xl[mask],X[mask],Xu[mask],Yl[mask],Y[mask],Yu[mask],mv[mask],md[mask],fv[mask],h[mask]
        if logscale:
            Yl = fill_0_na(Yl)
            Y = fill_0_na(Y)
            Yu = fill_0_na(Yu)

        # Plot the measurements
        points_p = ((Xl == Xu) & (Yl == Yu)).all()
        if plot_data_extent and not points_p:
            # Plot the age, time, and standard deviation variations
            for _xl,_x,_xu,_yl,_y,_yu,_mv,_md,_h in zip(Xl,X,Xu,Yl,Y,Yu,mv,md,h):
                kwds = dict(marker='+', color='black') if _xl == _xu else dict(ls='-',color=color)
                logscale_plot(ax,[_xl,_xu],[_y,_y],[_mv,_mv], **kwds, logscale=logscale)
                logscale_plot(ax,[_x,_x],[_yl,_yu],[_mv,_mv], c=color, ls='-', logscale=logscale)
                logscale_plot(ax,[_x,_x],[_y,_y],[_mv-_Z_95_*_md,_mv+_Z_95_*_md], c=color, ls='-', lw=1, logscale=logscale)
        else:
            logscale_plot(ax,X,Y,mv,marker='+', ms=8, mew=1, ls=linestyle, c=color, logscale=logscale)
            for _xl,_x,_xu,_yl,_y,_yu,_mv,_md,_h in zip(Xl,X,Xu,Yl,Y,Yu,mv,md,h):
                logscale_plot(ax,[_x,_x],[_y,_y],[_mv-_Z_95_*_md,_mv+_Z_95_*_md], c=color, ls='-', lw=1, logscale=logscale)
        
        # Tag the holdouts
        logscale_plot(ax,X[h==1],Y[h==1],mv[h==1], 'o', logscale=logscale, **holdout_style(color))

        # Plot the model prediction
        logscale_plot(ax,X,Y,fv, ls=linestyle, marker='x', ms=6, c='k', label=label, logscale=logscale)

        if len(Xl):
            xlim = min(Xl), max(Xu)
            return xlim
        return None

    def plot2D(integrand, df, pred_df, upr, lwr, x_var_name, ylim=None, clipy=None, adjust_data=None):

        def plot_residuals(ax, df, xlim):
            """
            Plot residuals and residual mean
            Note: The residuals are always (plotted) in linear space, for both linear and log scale variables
            Note: if density is log_gaussian, log_laplace, or log_students:
             residual = (log(mean + eta)-log(model_adjusted + eta))/(log(mean + std + eta) - log(mean + eta))
            else:
             residual = (mean - model_adjusted)/std
            ax: a matplotlib axis
            df: a pandas dataframe
            """
            a,h,r = df.loc[:,[x_var_name, 'hold_out', 'weighted_residual']].T.values
            r = np.asarray(r)
            color = 'k'
            if clipy is None:
                ax.plot(a,r, 'x', mew=1, ms=5, c=color)
            else:
                r[r>clipy] = clipy
                r[r<-clipy] = -clipy
                mask = (-clipy < r) & (r < clipy)
                ax.plot(a[mask],r[mask], 'x', mew=1, ms=5, c=color)
                mask = (-clipy == r)
                ax.plot(a[mask],r[mask], 'v', mew=1, ms=5, c=color)
                mask = (r == clipy)
                ax.plot(a[mask],r[mask], '^', mew=1, ms=5, c=color)

            ax.plot(a[(h == 1)],r[(h == 1)], 'o', **holdout_style('k')) # Tag holdouts
            ax.plot(xlim,[0,0], '--k')
            tmp = df[df.hold_out==0]
            if not tmp.empty:
                # Group by the independant variable and plot the residuals mean 
                groups = tmp.groupby(x_var_name)
                x, res = np.asarray([(group[x_var_name].mean(), group.weighted_residual.mean()) for x,group in groups]).T
                ax.plot(x,res,'-r')
            ax.set_xlim(xlim)
            if clipy is not None:
                ax.set_ylim([(-clipy * 1.5),(clipy * 1.5)])
            ax.set_xlabel(x_var_name)
            ax.set_ylabel('residual')

        plot_data_extent_p = plot_data_extent and (len(df) < _data_extent_plotting_limit_) and integrand != 'mtall'
        measurement_name = ('adj_value' if adjust_data else 'meas_value')
        if not adjust_data:
            assert 'adj_value' in df, "Data has column named adj_value."
        measurement_name = ('adj_value' if adjust_data else 'meas_value')
        xlim = plot_age_time_epochs(upr, df, x_var_name=x_var_name, measurement_name = measurement_name,
                                    linestyle='None', label=None, plot_data_extent=plot_data_extent_p, logscale=logscale,
                                    adjust_data = adjust_data)
        if x_var_name == 'age':
            xlim = (0, max(100, (xlim[1] if xlim is not None else 100))) # Theo want's full age range

        # Plot avgint fit vs independant variable
        if adjust_data:
            label = 'Model (scovs=ref,ccovs=parent)'
        else:
            label = 'Model (covs=mean(data))'
        curve_lbl = mlines.Line2D([], [], color='k', ls='-', label=label)
        if pred_df.empty:
            labels = df.time.median() if x_var_name == 'age' else df.age.median()
        else:
            labels = pred_df.time.unique() if x_var_name == 'age' else pred_df.time.unique()
            x,z = pred_df.loc[:, [x_var_name,'avg_integrand']].T.values
            shape = (x.tolist().count(x[0]), -1)
            for _x,_z,_l in zip(x.reshape(shape), z.reshape(shape), labels):
                upr.plot(_x,_z,'-',lw=1, label=_l, color='k') #color=color,

        if logscale:
            upr.set_yscale('log', nonpositive='clip')
        set_ylim(upr)

        plot_residuals(lwr, df, xlim)
        upr.set_title("%s %s %s %s\n%s" % (model_name, parent_node_name, integrand, title, labels))
        if xlim: upr.set_xlim(xlim)
        if ylim:
            if not logscale:
                upr.set_ylim(ylim)
            else:
                upr.set_ylim(ylim)
        upr.set_ylabel(str(integrand))

        # upr.legend(*sort_legend(upr), loc='upper left', fontsize=fontsize)
        lbl = 'Adjusted Data' if adjust_data else 'Unadjusted Data'
        data_lbl = mlines.Line2D([], [], color='grey', marker='+', markersize=10, label=lbl)
        if adjust_data:
            point_lbl = mlines.Line2D([], [], color='k', ls='', marker='x', markersize=8, label='Model (scovs=ref,ccovs=data)')
        else:
            point_lbl = mlines.Line2D([], [], color='k', ls='', marker='x', markersize=8, label='Model (covs=data)')
        if not pred_df.empty:
            # lines, ignore = sort_legend(upr)
            # upr.legend(lines, handles = handles, loc='best', fontsize=fontsize-3, framealpha=_framealpha_, frameon=True, shadow=True, facecolor='grey')
            handles = [data_lbl, point_lbl, curve_lbl]
            upr.legend(handles = handles, loc='best', fontsize=fontsize-3, framealpha=_framealpha_, frameon=True, shadow=True, facecolor='grey')
        upr.grid(True)
        if pdf: 
            pdf.savefig(fig)
            plt.close(fig.number)

    def plot2D_model(integrand, integrand_srf, adjustment_title, model_name=''):

        fig = plt.figure()
        plt.title("%s %s %s fit\n%s" % (model_name, parent_node_name, integrand, adjustment_title))
        if logscale:
            integrand_srf.avg_integrand = fill_0_na(integrand_srf.avg_integrand)
        grps = integrand_srf.groupby('time')
        y = None
        for year, grp in grps:
            x,y = grp.age, grp.avg_integrand
            plt.plot(x,y, '+-', lw=2, label=year)
            # plt.legend(*sort_legend(plt.gca()), loc='best', fontsize=fontsize-2, framealpha=_framealpha_, frameon=True, shadow=True, facecolor='grey')
            plt.legend(*sort_legend(plt.gca()), loc='best', fontsize=fontsize-2, frameon=True, edgecolor='k')
            plt.xlabel('age')
            plt.ylabel(integrand)
        ax = fig.gca()
        if y is not None:
            if logscale:
                ax.set_yscale('log', nonpositive='clip')
        set_ylim(ax)
        ax.grid(True)
        if pdf: 
            pdf.savefig(fig)
            plt.close(fig.number)

    def plot3D(integrand, integrand_fit, integrand_srf, adjustment_title, model_name=''):

        fig = plt.figure()
        upr, lwr = fig.add_subplot(111, projection=projection), None

        plot_data_extent_p = plot_data_extent and (len(integrand_fit) < _data_extent_plotting_limit_)  and integrand != 'mtall'
        plot_age_time_epochs(upr, integrand_fit, linestyle='None', label=None, plot_data_extent=plot_data_extent_p, logscale=logscale)

        # Plot avgint fit vs age wireframe
        ages = sorted(set(integrand_srf.age))
        times = sorted(set(integrand_srf.time))
        ti,ai = np.meshgrid(times, ages)
        z = integrand_srf.sort_values(by=['age', 'time']).loc[:, 'avg_integrand'].values.reshape(ai.shape)
        if logscale:
            z = np.log(fill_0_na(z))
        upr.plot_wireframe(ai, ti, z, rstride=1, cstride=1, color='k', lw=2)
        if logscale:
            upr.set_zticklabels([r'$10^{%d}$' % _ for _ in upr.get_zticks()])

        upr.set_title("%s %s %s %s\n%s" % (model_name, parent_node_name, integrand, title, adjustment_title))
        x_lwr, x_upr = upr.get_xlim()
        upr.set_xlim([-1,max(100, x_upr)+1])
        if times:
            upr.set_ylim(min(times), max(times))
        try:
            upr.set_zlim(z[np.isfinite(z)].min(), z[np.isfinite(z)].max())
        except:
            upr.set_zlim(_eps_,1)
        upr.set_xlabel('age')
        upr.set_ylabel('time')
        upr.set_zlabel(str(integrand))
        upr.grid(True)
        if pdf: 
            pdf.savefig(fig)
            plt.close(fig.number)

    if adjust_data:
        adjustment_title = "Model covariates = study_covariate reference"
    else:
        adjustment_title = "Model covariates = mean(data covariates)"
        
    parent_node_id = int(DB.options.parent_node_id)
    parent_node_name, = DB.node[DB.node.node_id == parent_node_id].node_name.values

    fit = data_fit_and_adjustments
    fit.loc[:, 'time'] = fit.loc[:, ['time_lower', 'time_upper']].mean(axis=1)
    fit.loc[:, 'age'] = fit.loc[:, ['age_lower', 'age_upper']].mean(axis=1)

    if 0:
        mask = (fit.integrand_name == 'mtspecific') & (fit.node_id == int(DB.options.parent_node_id)) & (fit.age > 60) & (fit.time == 2005)
        xx = fit[mask]


    x_cols = [_ for _ in fit.columns if _.startswith('x_')]
    indep_vars = ['time'] + x_cols

    for integrand in DB.integrand.integrand_name:

        if plot_integrands and (integrand not in plot_integrands): continue
        integrand_fit = fit[fit.integrand_name == integrand]
        integrand_srf = (pd.DataFrame([], columns=integrand_fit.columns) if predicted_surface is None
                         else predicted_surface[predicted_surface.integrand_name == integrand])

        if 0 and integrand == 'mtspecific':
            print (DB.data[(DB.data.age_lower >= .5) & (DB.data.age_lower < 20)].merge(DB.integrand[DB.integrand.integrand_name == 'mtspecific']))

            save = DB.avgint.copy()

            plt.figure()

            a = integrand_fit[(integrand_fit.time_lower == year)]
            b = integrand_srf[(integrand_srf.time_lower == year)]

            plt.plot(a.age_lower, a.model_at_data_covs)
            plt.plot(b.age_lower, b.avg_integrand)
            plt.plot(a.age_lower, a.avg_integrand)
            plt.plot(b.age_lower, b.avg_integrand)
            print (a.node_id.unique())
            print (b.node_id.unique())
            cols = ['node_id', 'integrand_name', 'node_id', 'age_lower', 'age_upper', 'time_lower', 'time_upper', 'x_0', 'x_1', 'x_2', 'avg_integrand']
            print (a.loc[a.age_lower == 10, cols])
            print (b.loc[b.age_lower == 10, cols])
            avgint = a.append(b).reset_index(drop=True)
            avgint['avgint_id'] = avgint.index.tolist()
            DB.avgint = avgint
            os.system(f"{dismod} {DB.filename} predict fit_var")
            pred = DB.avgint.merge(DB.predict)
            mask = pred.x_0.isna()
            plt.plot(*pred.loc[mask, ['age_lower', 'avg_integrand']].values.T, '+--')
            plt.plot(*pred.loc[~mask, ['age_lower', 'avg_integrand']].values.T, '+--')
            x_0 = pred.loc[~mask, 'x_0'].unique()[0]
            avgint.loc[mask, 'x_0'] = x_0
            avgint.loc[:, 'time_upper'] = 2011
            DB.avgint = avgint
            os.system(f"{dismod} {DB.filename} predict fit_var")
            pred = DB.avgint.merge(DB.predict)
            plt.plot(*pred.loc[mask, ['age_lower', 'avg_integrand']].values.T, '*--')
            plt.plot(*pred.loc[~mask, ['age_lower', 'avg_integrand']].values.T, '*--')

            for iid in DB.data.integrand_id.unique():
                print (DB.data[(DB.data.integrand_id == iid)].x_0.unique())
                

            # a = integrand_fit[(integrand_fit
        if projection == '3d':
            if not integrand_srf.empty and not pdf:
                plot3D(integrand, integrand_fit, integrand_srf, adjustment_title, model_name)
        else:
            # If the independant variable is age, group the data by time, and vice versa
            intervals = []
            if not integrand_srf.empty:
                if x_var_name == 'age':
                    vars = ['time_lower', 'time_upper']
                    if surface_time_window is not None:
                        intervals = [(t-surface_time_window, t, t+surface_time_window) for t in sorted(set(integrand_srf.time))]
                    else:
                        intervals = intervals_from_midpoints(sorted(set(integrand_srf.time)))
                elif x_var_name == 'time':
                    vars = ['age_lower', 'age_upper']
                    intervals = intervals_from_midpoints(sorted(set(integrand_srf.age)))
                else:
                    raise Exception("X_var_name must be 'age' or 'time'")
                if x_var_name == 'age':
                    # Plot the integrand model fit, across years, without the data
                    plot2D_model(integrand, integrand_srf, adjustment_title, model_name)
            else:
                if x_var_name == 'age':
                    surface_times = sorted(map(float, DB.cascade_options.time_grid.split()))
                    if len(surface_times) > 1:
                        surface_time_window = np.median(np.diff(surface_times))/2+.01
                    intervals = [(t-surface_time_window, t, t+surface_time_window) for t in surface_times]
                elif x_var_name == 'time':
                    surface_ages = sorted(map(int, DB.cascade_options.age_grid.split()))
                    if len(surface_ages) > 1:
                        surface_age_window = np.median(np.diff(surface_ages))/2+.01
                    intervals = [(t-surface_age_window, t, t+surface_age_window) for t in surface_ages]
                else:
                    raise Exception("X_var_name must be 'age' or 'time'")

            def get_ylim():
                # Y limits so plot scaling is consistent
                y_huge = None
                if ((integrand == 'mtall') or (integrand == 'mtother')):
                    if 1 < integrand_srf.avg_integrand.min():
                        logger.error("Integrand %s integrand_srf seems incorrect -- min value is %f." % (integrand, integrand_srf.avg_integrand.min()))
                    y = integrand_fit.adj_value[np.isfinite(integrand_fit.adj_value)].values
                else:
                    # std_dev's mess up the scaling 
                    y_plus_sigma = (integrand_fit.adj_value + _Z_95_*integrand_fit.meas_std).tolist() + integrand_srf.avg_integrand.tolist()
                    y = integrand_fit.adj_value.tolist() + integrand_fit.model_at_data_covs.tolist() + integrand_srf.avg_integrand.tolist()

                if logscale:
                    lwr = np.log10(np.nanmax([_eps_/10, np.nanmin(integrand_fit.meas_value.values)]))
                    upr = np.log10(np.nanmax([_eps_*10, np.nanmax(integrand_fit.meas_value.values)]))
                    ylim = (10**np.floor(lwr), 10**np.ceil(upr)) if len(integrand_fit) else None
                else:
                    # ylim = (min(0.0, np.nanmin(integrand_fit.meas_value.values)), np.nanmax(integrand_fit.meas_value.values)) if len(integrand_fit) else None
                    ylim = (min(0.0, np.nanmin(y)), np.nanmax(y)) if len(y) else None
                if pdf and len(integrand_fit) == 1:
                    # For some reason, writing pdfs sometimes fails on divide by zero. This upper ylim scaling fixes it (in some cases at least.)
                    ylim = (ylim[0], ylim[1]*1.1)

                return ylim

            if 0:
                if not integrand_fit.empty:
                    ylim = get_ylim()
                    if ylim is not None and (min(ylim) == max(ylim)): ylim = None
            ylim = None
            
            # Plot the model fit and measured data within time windows
            for lb,mid,ub in intervals:
                if x_var_name == 'age':
                    this_fit = integrand_fit[~((integrand_fit.time_upper <= lb) | (integrand_fit.time_lower >= ub))]
                    this_srf = integrand_srf[(integrand_srf.time_lower == mid)]
                elif x_var_name == 'time':
                    this_fit = integrand_fit[~((integrand_fit.age_upper <= lb) | (integrand_fit.age_lower >= ub))]
                    this_srf = integrand_srf[(integrand_srf.age_lower == mid)]
                if not this_fit.empty:
                    fig = plt.figure()
                    gs = plt.GridSpec(2,1,height_ratios=[3,1])
                    uprs, lwrs = plt.subplot(gs[0]), plt.subplot(gs[1])
                    # # Make the residual range .1 for the omega constraint so one can see if it actually fits.
                    # clipy = .1 if integrand == 'mtall' else 3
                    # plot2D(integrand, this_fit, this_srf, uprs, lwrs, x_var_name = x_var_name,
                    #        ylim=ylim, adjust_data=adjust_data, clipy = clipy)
                    plot2D(integrand, this_fit, this_srf, uprs, lwrs, x_var_name = x_var_name,
                           ylim=ylim, adjust_data=adjust_data)

class TestAndPlot(object):
    def __init__(self, sqlite_filename,
                 sex = None,
                 surface_time = None, surface_age = None,
                 plot_data_extent = True,
                 dismod_AT = _dismod_,
                 time_window_for_plots = _time_window_for_plots_,
                 predict_using_COD_covariate_values = True,
                 model_version_id = None,
                 model_name = ''):
        global _self; _self=self
        
        if not os.path.isfile(sqlite_filename):
            raise Exception("File %s does not exist." % sqlite_filename)
        if os.path.getsize(sqlite_filename) == 0:
            raise Exception("No data in file %s." % sqlite_filename)

        self.model_version_id = model_version_id
        self.model_name = ' '.join(model_name)
        self.sqlite_filename = sqlite_filename
        del sqlite_filename

        tempfile_name = tempfile.NamedTemporaryFile(mode='w', prefix='dismod_tempfile_', suffix='.db', delete=False).name
        logger.info("Plot working from database: %s" % tempfile_name)
        shutil.copy2(self.sqlite_filename, tempfile_name)
        self.DB = DB = DismodDbAPI(tempfile_name)
        if __all_node_db__:
            from utilities import sex2ihme_id
            self.sex_ref = DB.covariate.loc[DB.covariate.covariate_id == DB.sex_covariate.covariate_id.squeeze(), 'reference'].squeeze()
            self.sex_id = sex2ihme_id(self.sex_ref)
            path = [os.path.sep] + self.sqlite_filename.split(os.path.sep)[:-3]
            self.globalDB_path = os.path.join(*path + ['1', str(self.sex_id), 'dismod.db'])
            self.allDB_path = os.path.join(*([os.path.sep] + path + ['all_node.db']))
            self.globalDB = globalDB = DismodDbAPI((os.path.join(*path + ['1', str(self.sex_id), 'dismod.db'])))
            import sqlalchemy

            def dataframe_decompress_index(integrand, index):
                for i in index: pass

                index[f'all_{name}_id'] = index.index
                index.reset_index(inplace=True, drop=True)
                index[f'{name}_index_id'] = index.index
                index = index[[f'{name}_index_id', 'node_id', f'all_{name}_id']]
                return index



            with sqlalchemy.create_engine(f"sqlite:///{self.allDB_path}").connect() as conn:
                self.n_sex = 3
                self.omega_age = pd.read_sql_table('omega_age_grid', conn).merge(globalDB.age, how='left')
                self.omega_time = pd.read_sql_table('omega_time_grid', conn).merge(globalDB.time, how='left')
                self.all_option = pd.read_sql_table('all_option', conn)
                self.all_cov_reference = pd.read_sql_table('all_cov_reference', conn)
                self.fit_goal = pd.read_sql_table('fit_goal', conn)
                self.all_mtall = pd.read_sql_table('all_mtall', conn)
                self.mtall_index = pd.read_sql_table('mtall_index', conn)
                self.all_mtspecific = pd.read_sql_table('all_mtspecific', conn)
                self.mtspecific_index = pd.read_sql_table('mtspecific_index', conn)
                assert len(all_mtall) == len(all_mtspecific), "Mtall and mtspecific are not the same length."
                runLength = len(self.omega_age)*len(self.omega_time)*self.n_sex
                diff = set(np.diff(self.mtall_index.all_mtall_id))
                assert len(diff) == 1 and runLength == diff.pop()
                assert runLength * len(self.fit_goal) == len(self.all_mtall)
                index_cols = [[age, time, sex] for age in self.omega_age.age for time in self.omega_time.time for sex in (1,2,3)]

                mtall = self.all_mtall.copy()
                cols = ['node_id', 'age', 'time', 'sex_id']
                mtall[cols] = None
                for i, row in self.mtall_index.iterrows():
                    mtall.loc[row.all_mtall_id:row.all_mtall_id+len(index_cols)-1, ['age', 'time', 'sex_id']] = index_cols
                    mtall.loc[row.all_mtall_id:row.all_mtall_id+len(index_cols)-1, 'node_id'] = row.node_id

                mtspecific = self.all_mtspecific.copy()
                cols = ['node_id', 'age', 'time', 'sex_id']
                mtspecific[cols] = None
                for i, row in self.mtspecific_index.iterrows():
                    mtspecific.loc[row.all_mtspecific_id:row.all_mtspecific_id+len(index_cols)-1, ['age', 'time', 'sex_id']] = index_cols
                    mtspecific.loc[row.all_mtspecific_id:row.all_mtspecific_id+len(index_cols)-1, 'node_id'] = row.node_id

        try:
            self.sex = sex2sex_name(self.sex_ref)
            if 'c_covariate_name' in DB.covariate:
                self.sex_xcov = "x_%d" % int(DB.covariate.loc[['sex' in n for n in DB.covariate.c_covariate_name], 'covariate_id'].tolist()[0])
            else:
                self.sex_xcov = "x_%d" % int(DB.covariate.loc[['sex' in n for n in DB.covariate.covariate_name], 'covariate_id'].tolist()[0])
        except:
            self.sex = None
            self.sex_xcov = None

        self.c_xcovs = ['x_%s' % r.covariate_id for r in DB.covariate.itertuples() if r.covariate_name.startswith('x_c_')]
        self.s_xcovs = ['x_%s' % r.covariate_id for r in DB.covariate.itertuples() if r.covariate_name.startswith('x_s_')]

        self.dismod_AT = dismod_AT
        self.plot_data_extent = plot_data_extent

        self.surface_ages = surface_age
        self.surface_times = surface_time
        self.surface_time_window = time_window_for_plots

        self.predict_using_COD_covariate_values = predict_using_COD_covariate_values

        self.time_cols = ['time_lower', 'time_upper']
        self.age_cols = ['age_lower', 'age_upper']

    def __call__(self, adjust_data = True, pdf_p = False, logscale=False, plot_integrands = None, plot_years = None, plot3D = not True):
        self.adjust_data = adjust_data
        self.plot_integrands = plot_integrands
        
        if pdf_p:
            pdf_filename = self.sqlite_filename + '.pdf'
            print ("Plotting to %s" % pdf_filename)
            pdf = PdfPages(pdf_filename)
            plt.interactive(0)
        else:
            pdf = None
            plt.interactive(1)


        DB = self.DB

        if len(DB.tables) == 0:
            raise Exception("No tables in sqlite database %s." % DB.filename)

        try: 
            self.parent_node_name = DB.options.parent_node_name
            self.parent_node_id = int(DB.node.loc[DB.node.node_name == self.parent_node_name, 'node_id'])
        except:
            self.parent_node_id = int(float(DB.options.parent_node_id))

        self.parent_smooth_ids = tuple(filter(np.isfinite, DB.rate.parent_smooth_id.astype(float)))
        self.child_smooth_ids = tuple(filter(np.isfinite, DB.rate.child_smooth_id.astype(float)))

        if 'fit_var' in DB.tables:
            plot_fit_var_vs_prior(DB, self.parent_smooth_ids, self.child_smooth_ids, pdf=pdf, logscale=logscale, model_name = self.model_name)

        # plots = [(None,'3d'), ('age', None), ('time', None)]
        plots = [(None,'3d'), ('age', None)]

        try:
            self.plot_mulcovs(pdf=pdf)
        except Exception as ex:
            print ('Plot mulcovs failed', ex)

        print ("Plotting from temporary database", DB.filename)
        for var,proj in plots:
            plot_model_vs_data(DB,
                               self.predicted_surface if plot3D else None,
                               self.data_fit_and_adjustments,
                               x_var_name = var,
                               projection = proj,
                               logscale = logscale,
                               fontsize = 10,
                               plot_data_extent = self.plot_data_extent,
                               surface_time_window = self.surface_time_window,
                               pdf=pdf,
                               adjust_data = adjust_data,
                               plot_integrands = plot_integrands,
                               plot_years = plot_years,
                               model_name = self.model_name)
        if pdf: pdf.close()

        # @property
        # def mulcov(self):
        #     logger.info("Computing covariate multiplier sample.")
        #     # Covariate multiplier information
        #     DB = self.DB
        #     cols = ['var_id', 'var_type', 'xcov_name', 'covariate_id','covariate_name', 'integrand_id', 'integrand_name', 'rate_id', 'rate_name', 'smooth_name', 'start_var_value', 'lower', 'fit_var_value', 'upper', 'eta']
        #     df = DB.var.dropna(subset=['covariate_id'])
        #     df = df.merge(DB.var, how='left').merge(DB.smooth, how='left').merge(DB.smooth_grid, how='left').merge(DB.covariate, how='left').merge(DB.integrand, how='left').merge(DB.rate, how='left').merge(DB.prior, how='left', left_on='value_prior_id', right_on='prior_id')
        #     df['xcov_name'] = ['x_%d' % _ for _ in df.covariate_id]


        #     # xcov starting values
        #     try:
        #         start = DB.start_var
        #         utilities.system("%s %s sample fit_var 1" % (self.dismod_AT, DB.filename))
        #         df = df.merge(DB.start_var, left_on='var_id', right_on='start_var_id').rename(columns={'fit_var_value' : 'start_var_value'})
        #     except:
        #         raise
        #     finally:
        #         DB.start_var = start
        #     # xcov fit values
        #     df = df.merge(DB.fit_var, left_on='var_id', right_on='fit_var_id').rename(columns={'fit_var_value' : 'fit_var_value'})
        #     df = df.sort_values(by=['covariate_id', 'var_type', 'integrand_name', 'rate_name']).loc[:, cols].reset_index(drop=True)
        #     df.rename(columns={'integrand_name' : 'integrand', 'rate_name' : 'rate', 'start_var_value' : 'start', 'fit_var_value': 'fit'}, inplace=True)

        #     return df

    def plot_mulcovs(self, pdf):

        DB = self.DB

        mask = DB.var.var_type.isin(['mulcov_rate_value', 'mulcov_meas_value', 'mulcov_meas_std'])
        fit_var = (DB.var
                   .merge(DB.fit_var, left_on='var_id', right_on='fit_var_id')
                   .merge(DB.smooth, how='left', on='smooth_id')
                   .merge(DB.integrand, how='left')
                   .merge(DB.rate, how='left')
                   .merge(DB.smooth_grid, how='left')
                   .merge(DB.prior, how='left', left_on='value_prior_id', right_on='prior_id'))
        mulcov = (DB.mulcov
                  .merge(DB.covariate, how='left')
                  .merge(fit_var, how='left'))

        cols = ['mulcov_id', 'covariate_name', 'mulcov_type', 'rate_name', 'integrand_name', 'reference', 'max_difference', 'lower', 'fit_var_value', 'upper']
        mulcov = mulcov[cols].rename(columns={'fit_var_value' : 'fit'})

        col_names = [_.replace('_name', '') for _ in mulcov.columns if _ not in ('mulcov_id')]
        labelc = col_names
        labelr = mulcov.covariate_name.tolist()
        dpi=240

        if not mulcov.empty:
            def format(c):
                try: c = float(c)
                except: return c
                return '%.3g' % c if np.isfinite(c) else ''

            chars = [[format(mulcov.iloc[r,c]) for c in range(1,1+len(labelc))] for r in range(len(labelr))]
            fig = plt.figure() #dpi=dpi)
            lightgrn = (0.5, 0.8, 0.5)
            plt.title(self.model_name + '\n' + 'Covariate Multipliers')
            tab = plt.table(cellText=chars,
                            colLabels=labelc,
                            colWidths = [.2] + [0.1]*len(labelc),
                            colColours=[lightgrn]*len(labelc),
                            cellLoc='center',
                            loc='upper center')
            if pdf:
                tab.scale(2,2)
            else:
                tab.scale(1.2,2)

            plt.axis('off')
            if pdf: 
                pdf.savefig(fig, dpi=dpi, orientation='landscape', pad_inches=0, bbox_inches='tight')
                plt.close(fig.number)

    @cached_property
    def data_fit_and_adjustments(self):
        """
        The fit_data_subset table compares the model and data for the model_variables corresponding to a fit_command.
        A new fit_data_subset table is created each time the fit_command is executed.
        """
        # Mix data adjustments and some names into the data
        DB = self.DB
        if DB.data.empty:
            return pd.DataFrame([], columns = DB.data.columns.tolist() + ['adj_value']).merge(DB.integrand)
        data = adjust_measurements(DB)
        if 'fit_data_subset' in DB.tables:
            data = (DB.data_subset.merge(data, how='left')
                    .merge(DB.fit_data_subset, how='left', left_on = 'data_subset_id', right_on = 'fit_data_subset_id'))
            data = data.merge(DB.integrand, how='left').merge(DB.density, how='left')
        return data

    def debug_covariates(self):
        def format_locs(locs):
            indices = []
            from itertools import groupby
            from operator import itemgetter
            diff = np.asarray(np.diff(locs).tolist() + [1])
            for k,v in groupby(enumerate(diff),key=itemgetter(1)):
                v = list(v)
                indices += [(v[0][0],v[-1][0])]
            rtn = []
            for i,j in indices:
                if i == j: rtn.append(str(locs[i]))
                else: rtn.append('(%d - %d)' % (locs[i], locs[j]))
            return ', '.join(rtn)

        _plot_time_ = False

        DB=self.DB
        fignum = plt.gcf().number
        age_lower = 50
        time_lower = 2000
        if _plot_time_:
            xx = DB.avgint[DB.avgint.age_lower == age_lower]
        else:
            xx = DB.avgint[DB.avgint.time_lower == time_lower]
        data = DB.data_subset.merge(DB.data, how='left').merge(DB.predict, how='left', left_on='data_id', right_on='avgint_id')
        for integrand_id in sorted(xx.integrand_id.unique()):
            yy = xx[(xx.integrand_id == integrand_id)]
            dd = data[(data.integrand_id == integrand_id)]
            if yy.empty or dd.empty: continue
            locs = np.asarray(sorted(dd.node_id.unique()))
            locstr = format_locs(locs)
            name = DB.integrand.loc[DB.integrand.integrand_id == integrand_id, 'integrand_name'].squeeze()
            plt.figure()
            if _plot_time_:
                plt.plot(dd.time_lower, dd.avg_integrand, '*', color='red', label = 'prediction')
                plt.plot(dd.time_lower, dd.meas_value, '+', color='grey', label = 'meas_value')
            else:
                plt.plot(dd.age_lower, dd.avg_integrand, '*', color='red', label = 'prediction')
                plt.plot(dd.age_lower, dd.meas_value, '+', color='grey', label = 'meas_value')
            plt.legend()
            plt.ylabel('meas vs pred')
            plt.title(self.model_name + '\n' + '%s locs: %s' % (name, locstr))

            if __debug__ and not DB.country_covariates.empty and 'mean_BMI' in DB.country_covariates.covariate_name_short:
                x_bmi = DB.country_covariates.loc[DB.country_covariates.covariate_name_short == 'mean_BMI', 'xcov_name'].squeeze()
                if x_bmi in yy.columns:
                    plt.figure()
                    if _plot_time_:
                        plt.plot(yy.time_lower, yy[x_bmi], '-*', label='get_covariate_estimates')
                        plt.plot(dd[['time_lower', 'time_upper']].mean(axis=1), dd[x_bmi], '+', label='measured data')
                    else:
                        plt.plot(yy.age_lower, yy[x_bmi], '-*', label='get_covariate_estimates')
                        plt.plot(dd[['age_lower', 'age_upper']].mean(axis=1), dd[x_bmi], '+', label='measured data')
                    plt.legend()
                    plt.ylabel('BMI')
                    plt.title(self.model_name + '\n' + '%s locs: %s' % (name, locstr))
                x_ldi = DB.country_covariates.loc[DB.country_covariates.covariate_name_short == 'LDI_pc', 'xcov_name'].squeeze()
                if x_ldi in yy.columns:
                    plt.figure()
                    if _plot_time_:
                        plt.plot(yy.time_lower, yy[x_ldi], '-*', label='get_covariate_estimates')
                        plt.plot(dd[['time_lower', 'time_upper']].mean(axis=1), dd[x_ldi], '+', label='measured data')
                    else:
                        plt.plot(yy.age_lower, yy[x_ldi], '-*', label='get_covariate_estimates')
                        plt.plot(dd[['age_lower', 'age_upper']].mean(axis=1), dd[x_ldi], '+', label='measured data')
                    plt.legend()
                    plt.ylabel('LDI')
                    plt.title(self.model_name + '\n' + '%s locs: %s' % (name, locstr))
        

    @cached_property
    def predicted_surface(self):
        """
        The predicted_surface is a gridded surface predicted at the nominal fit_var variable values. 
        1) When plotting adjusted data, the surface must be modeled with node's country covariate values for the time/location/...
           otherwise it will not show the time trends.
        2) When plotting unadjusted data, the surface must be modeled with node's country covariate values for the time/location/...
           and the measured study covariate values.
        """
        DB = self.DB
        if DB.data.empty: 
            return pd.DataFrame([], columns = DB.data.columns.tolist() + ['time', 'avg_integrand']).merge(DB.integrand)

        data = self.data_fit_and_adjustments

        ode_step_size = float(DB.options.ode_step_size)
        sys.modules.pop('cascade_at_gma.tests.test_omega.test_omega', None)
        try:
            logger.info('Using the mtall age grid for plotting.')
            mtall = DB.data[DB.data.integrand_id.isin(DB.integrand.loc[DB.integrand.integrand_name == 'mtall', 'integrand_id'])]
            surface_ages = sorted(set(map(tuple, mtall[['age_lower', 'age_upper']].values)))
        except Exception as ex:
            logger.error(ex)
            logger.warn('FIXME -- Probably should get the plotting age grid from the database.')
            surface_ages = [(0.0, 0.01918), 
                            (0.01918, 0.07671),
                            (0.07671, 1.0),
                            (1.0, 5.0),
                            (5.0, 10.0),
                            (10.0, 15.0),
                            (15.0, 20.0),
                            (20.0, 25.0),
                            (25.0, 30.0),
                            (30.0, 35.0),
                            (35.0, 40.0),
                            (40.0, 45.0),
                            (45.0, 50.0),
                            (50.0, 55.0),
                            (55.0, 60.0),
                            (60.0, 65.0),
                            (65.0, 70.0),
                            (70.0, 75.0),
                            (75.0, 80.0),
                            (80.0, 100.0)]
        surface_ages = sorted(set(np.asarray(surface_ages).flat))
        surface_times = self.surface_times or sorted(set(data.loc[:, self.time_cols].mean(axis=1)))

        def rounder(x): return '%g' % x
        logger.info("Predicting surfaces at times: %s, ages: %s" % (', '.join(map(rounder, surface_times)), ', '.join(map(rounder, surface_ages))))

        country_xcovs = [] if DB.country_covariates.empty else DB.country_covariates.xcov_name.tolist()
        study_xcovs = [] if DB.study_covariates.empty else DB.study_covariates.xcov_name.tolist()
        sc_xcovs = country_xcovs + study_xcovs

        if self.plot_integrands:
            integrand_ids = list(DB.integrand.loc[DB.integrand.integrand_name.isin(self.plot_integrands), 'integrand_id'].values)
        else: 
            integrand_ids = None

        try:
            DB.avgint = model_surface_avgint(DB, node_id = self.parent_node_id,
                                             ages=surface_ages, times=surface_times,
                                             integrand_ids = integrand_ids)
        except Exception as ex:
            logger.error("Function model_surface_avgint failed -- exception was %s" % ex)
            raise ex

        if __debug__:
            try:
                self.debug_covariates()
            except Exception as ex:
                logger.error(ex)

        if len(DB.avgint.dropna(subset=self.c_xcovs, how='any')) != len(DB.avgint):
            logger.error("Every avgint entry should have country covariates -- some were null.")

        utilities.system("%s %s predict fit_var" % (self.dismod_AT, DB.filename))
        surface = DB.predict.merge(DB.avgint).merge(DB.integrand).merge(DB.node)
        surface['age'] = surface.loc[:, self.age_cols].mean(axis=1)
        surface['time'] = surface.loc[:, self.time_cols].mean(axis=1)

        return surface

    # Tests
    def check_optimization_results(self):
        pass

    def self_test(self):
        print ("Testing ... ")
        try:
            print ("No tests yet.")
            print ("   Passed.")
        except:
            print ("   FAILED.")
            
    def test(self):
        for _id in [_ for _ in self.rate_df.loc[:, ('parent_smooth_id', 'child_smooth_id')].values.flatten() if _ is not None]:
            assert _id in self.smooth_grid_df.index, "Smooth id's in the rate table must be in the smooth_grid table."

        for _id in self.smooth_grid_df.smooth_id.values:
            assert _id in self.smooth_df.index, "Smooth id's in the rate table must be in the smooth table index."

        for _id in [_ for _ in self.smooth_grid_df.loc[:,('value_prior_id', 'dage_prior_id', 'dtime_prior_id')].values.flatten() if _ is not None]:
            assert _id in self.prior_df.index, "Smooth value_priors must be in the prior table index."

def get_years(filename):
    def is_year(arg):
        try: 
            if 1900 < float(arg) < 2020: return True
        except: return None
        return False
    return sorted(set([float(_) for _ in os.path.split(os.path.dirname(filename))[-1].split('_') if is_year(_)]))


if 0:
    data = DB.data.merge(DB.integrand, how='left')
    avgint = data.rename(columns={'data_id': 'avgint_id'})
    avgint[['x_2', 'x_3', 'x_4']] = None
    DB.avgint = avgint
    c_xcovs = ['x_%s' % r.covariate_id for r in DB.covariate.itertuples() if r.covariate_name.startswith('x_c_')]
    assert len(avgint.dropna(subset = c_xcovs, how='any')) == len(avgint), "Every avgint entry should have country covariates -- some were null."
    utilities.system(f'{dismod} %s predict fit_var' % DB.filename)
    pred = DB.avgint.merge(DB.predict, how='left').merge(DB.integrand, how='left')
    adj = adjust_measurements(DB)
    assert np.alltrue(np.abs(adj.meas_value * adj.adjustment - adj.adj_value).fillna(0) < _eps_)

    print ('--------- data')
    print (data[(data.integrand_id == 2) & (data.age_lower==60) & (data.x_0 == .5) & (data.data_name == '138769503.0')].mean())
    print ('--------- prediction')
    print (pred[(pred.integrand_id == 2) & (pred.age_lower==60) & (pred.x_0 == .5) & (data.data_name == '138769503.0')].mean())
    print ('--------- adjusted')
    print (adj[(adj.integrand_id == 2) & (adj.age_lower==60) & (adj.x_0 == .5) & (data.data_name == '138769503.0')].mean())

    print (data[(data.integrand_id == 2) & (data.age_lower==60) & (data.node_id == 22) & (data.x_0 == .5) & (data.data_name == '138769503.0')].T)
    print (adj[(adj.integrand_id == 2) & (adj.age_lower==60) & (adj.node_id == 22) & (adj.x_0 == .5) & (data.data_name == '138769503.0')].T)


if (__name__ == '__main__'):

    plt.close('all')

    pdf_p = False
    plot_extent = False
    logscale = False

    if len(sys.argv) < 2:
        logscale = False; plot_extent = False; pdf_p = not True; sqlite_filename = '/Users/gma/Projects/IHME/cascade_at_gma.data-tmp/100667/full/1/both/1990_1995_2000_2005_2010_2015/100667.db'
        logscale = True; plot_extent = False; pdf_p = True; sqlite_filename = '/Users/gma/Projects/IHME/cascade_at_gma.data-tmp/100667/full/523/female/1990_1995_2000_2005_2010_2015/100667.db'
        logscale = False; plot_extent = False; pdf_p = True; sqlite_filename = '/Users/gma/Projects/IHME/cascade_at_gma.data-tmp/100667/full/102/female/1990_1995_2000_2005_2010_2015/100667.db'
        logscale = not False; plot_extent = False; pdf_p = not True; sqlite_filename = '/Users/gma/Projects/IHME/cascade_at_gma.data-tmp/100667/full/544/female/1990_1995_2000_2005_2010_2015/100667.db'
        logscale = not False; plot_extent = False; pdf_p = not True; sqlite_filename = '/Users/gma/Projects/IHME/cascade_at_gma.data-tmp/100667/full/1/both/1990_1995_2000_2005_2010_2015/100667.db'
        logscale = False; plot_extent = False; pdf_p = True; sqlite_filename = '/Users/gma/Projects/IHME/cascade_at_gma.data-tmp/100667-2018.01.08-drill/full/527/male/1990_1995_2000_2005_2010_2015/prior/100667.db'
        logscale = False; plot_extent = False; pdf_p = True; sqlite_filename = '/Users/gma/Projects/IHME/cascade_at_gma.data-tmp/100667-2018.01.08-drill/full/1/both/1990_1995_2000_2005_2010_2015/tmp.db'
        logscale = False; plot_extent = False; pdf_p = True; sqlite_filename = '/Users/gma/Projects/IHME/cascade_at_gma.data-tmp/100667/full/572/male/1990_1995_2000_2005_2010_2015/100667.db'
        logscale = False; plot_extent = True; pdf_p = False; sqlite_filename = '/Users/gma/Projects/IHME/cascade_at_gma.data-tmp/100667/full/1/both/1990_1995_2000_2005_2010_2015/fit/100667.db'
        logscale = False; plot_extent = True; pdf_p = False; sqlite_filename = '/Users/gma/Projects/IHME/cascade_at_gma.data-tmp/100667/full/102/both/2005_2010/100667.db'
        logscale = False; plot_extent = True; pdf_p = False; sqlite_filename = '/Users/gma/Projects/IHME/cascade_at_gma.data-tmp/100667/full/102/both/1990_1995_2000_2005_2010_2015/100667.db'
        model_version_id = 343691; logscale = True; plot_extent = True; pdf_p = False; sqlite_filename = '/Users/gma/Projects/IHME/modelers/10427/10427.db'; 

    else:
        ignore, sqlite_filename = sys.argv[:2]
        rest = sys.argv[2:]
        if rest:
            pdf_p = any(['pdf' in _ for _ in rest])
            logscale = any (['log' in _ for _ in rest])

    DB = DismodDbAPI(sqlite_filename)

    co = DB.cascade_option
    if 'model_version_id' not in co.cascade_option_name.values:
        co = pd.DataFrame([dict(cascade_option_name = 'model_version_id', cascade_option_value = str(model_version_id)),
                           dict(cascade_option_name = 'gbd_round_id', cascade_option_value = '5')]).reset_index(drop=True)
    DB.cascade_option = co

    surf_age = [0,1] + list(range(2,101,2))
    surf_time = get_years(sqlite_filename)
    if not surf_time:
        DB = DismodDbAPI(sqlite_filename)
        surf_time = [x for x in [1990,1995,2000,2005,2010,2015] if (DB.time.time.min() <= x <= DB.time.time.max())]
    
    plotter = TestAndPlot(sqlite_filename, surface_time = surf_time, surface_age = surf_age,
                          plot_data_extent = plot_extent, time_window_for_plots = _time_window_for_plots_)
    plotter(pdf_p=False or pdf_p, adjust_data=True, logscale=logscale, plot3D = True) #, plot_integrands=['mtspecific'])#, plot_integrands=['Sincidence']) #, plot_integrands=['prevalence']) # plot_integrands=['mtspecific']) #, 

    if sys.argv[0] and not pdf_p:
        # Holdup Python exit so the user can view the plots
        input("Hit RETURN to exit.")

    if 0:
        os.system ("dismodat.py %s db2csv" % sqlite_filename)

    if 0 and __debug__:
        # Tests to compute prevalence data adjustments and overplot them onto the plotter prevalence output.
        sys.modules.pop('cascade_at_gma.lib.weighted_residuals', None)
        from cascade_at_gma.lib.weighted_residuals import do_adjustments, subset, plotB, time_lwr, time_upr
        DB = DismodDbAPI(sqlite_filename)
        # plotter.DB = DB
        xx = do_adjustments(DB)
        dd, xx = subset(DB, xx)
        plt.figure(11);
        plotB(dd, xx)


    if 0:
        DB = DismodDbAPI(sqlite_filename)
        DBp = DismodDbAPI('/Users/gma/Downloads/dismod_plot.db')
        cols = ['integrand_name', 'node_id', 'meas_value', 'meas_std', 'adj_value', 'adj_residual', 'avg_integrand', 'weighted_residual', 'age_lower', 'age_upper', 'time_lower', 'time_upper', 'x_0', 'x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6']
        data = DB.data_subset.merge(DB.data).merge(DB.integrand).merge(DB.density).merge(DB.fit_data_subset, how='left')
        adj = adjust_measurements(DB).merge(DB.integrand)
        adj_csmr = adj[adj.integrand_name == 'mtspecific']
        data['adjustment'], data['adj_value'] = adjust_measurements(DB) 
        data['adj_residual'] = (data.meas_value - data.adj_value)/data.meas_std
        chunk = data.loc[data.integrand_name == 'prevalence']
        print ('\n', chunk[cols])
        print 

        for integrand in data.integrand_name.unique():
            plt.figure()
            xy = data[data.integrand_name == integrand]
            xy['age'] = (xy.age_upper+xy.age_lower)/2
            xy = xy.sort_values(by='age')
            age,m,f,w,a,r = xy[['age', 'meas_value', 'avg_integrand', 'weighted_residual', 'adj_value', 'adj_residual']].values.T
            plt.plot(age, w, '*', labelsize=15, label='weighted')
            plt.plot(age, r, '+', label='adjusted')
            plt.title(self.model_name + '\n' + integrand + ' adjusted')
            # plt.legend(*sort_legend(plt.gca()), loc='best', fontsize=fontsize-2, framealpha=_framealpha_, frameon=True, shadow=True, facecolor='grey')
            plt.legend(*sort_legend(plt.gca()), loc='best', fontsize=fontsize-2, frameon=True, edgecolor='k')
            plt.figure()
            plt.plot(age, m, '*', label='measured')
            plt.plot(age, f, '+', label='fit')
            plt.plot(age, a, 'x', label='adjusted')
            plt.title(self.model_name + '\n' + integrand + ' adjusted')
            # plt.legend(*sort_legend(plt.gca()), loc='best', framealpha=_framealpha_, frameon=True, shadow=True, facecolor='grey')
            plt.legend(*sort_legend(plt.gca()), loc='best', frameon=True, edgecolor='k')

    if 0:
        grps = DB.data.groupby('time_lower')
        for a,b in grps:
            print (a, b.x_2)

        plt.figure()
        plt.plot(grps.time_lower.mean(), grps.x_2.median())
       
msg = """
The plotting problem is that the prediction for the avgint = data does not include the mulcov_meas_value effects.
I need to apply the mulcov_meas_value to get the model(at reference covariate values) to match the unadjusted data,
and to get the model (at the data covariate values) to match unadjusted data.
Set the mulcov_meas_value = 0 to confirm this -- the model should match the measured data better.
This does not affect the residuals -- they are correct in both situations."""

logger.warning(msg)


if 0:
    pred = DB.avgint.merge(DB.predict).merge(DB.integrand, how='left')
    C = pred[pred.integrand_name == 'withC'].sort_values('age_lower').reset_index(drop=True)
    S = pred[pred.integrand_name == 'susceptible'].sort_values('age_lower').reset_index(drop=True)
    X = pred[pred.integrand_name == 'mtexcess'].sort_values('age_lower').reset_index(drop=True)
    plt.figure()
    plt.plot(C[['age_lower', 'age_upper']].mean(axis=1), C.avg_integrand, label='C')
    plt.plot(S[['age_lower', 'age_upper']].mean(axis=1), S.avg_integrand, label='S')
    plt.plot(X[['age_lower', 'age_upper']].mean(axis=1), X.avg_integrand, label='X')
    spec = X.avg_integrand*C.avg_integrand/(S.avg_integrand+C.avg_integrand)
    plt.plot(X[['age_lower', 'age_upper']].mean(axis=1), spec, label='mtspecific')
    plt.legend()




if 0: #def demonstrate_plotting_age_points_vs_intervals(DB):
    from cascade_at_gma.lib.weighted_residuals import data_mean, adj_std

    dismod_AT = _dismod_
    data = DB.data.merge(DB.data_subset).merge(DB.fit_data_subset)
    integrand_id = 5        # mtspecific

    # times = [1990, 2000, 2015]
    times = [1990, 2015]
    times = [2000]
    tmp = data[(data.integrand_id == integrand_id) & (data.age_lower == 50) & (data.time_lower.isin(times)) & (data.x_0 > 0) & (data.node_id.isin([3,8,25])) ]

    c_xcovs = ['x_%s' % r.covariate_id for r in DB.covariate.itertuples() if r.covariate_name.startswith('x_c_')]

    avgint = []
    age_intervals = np.linspace(0,100,21)
    dt = max(set(np.diff(age_intervals)))
    for node in sorted(tmp.node_id.unique()):
        for t in sorted(tmp.time_lower.unique()):
            [[x2,x3]] = tmp.loc[(tmp.time_lower == t) & (tmp.node_id==node), c_xcovs].values
            avgint += [dict(integrand_id = integrand_id, node_id = node, weight_id=0, age_lower=age, age_upper = age, time_lower = t, time_upper = t,
                            x_0 = .5, x_1 = 1, x_2 = x2, x_3 = x3, x_4 = None, x_5 = None, x_6 = None) for age in np.linspace(0,100,101)]
            avgint += [dict(integrand_id = integrand_id, node_id = node, weight_id=0, age_lower=age, age_upper = age+dt, time_lower = t, time_upper = t,
                            x_0 = .5, x_1 = 1, x_2 = x2, x_3 = x3, x_4 = None, x_5 = None, x_6 = None) for age in age_intervals]
    n = len(avgint)
    for node in sorted(tmp.node_id.unique()):
        for t in sorted(tmp.time_lower.unique()):
            [[x2,x3]] = tmp.loc[(tmp.time_lower == t) & (tmp.node_id==3), c_xcovs].values
            for age in sorted(tmp.age_lower.unique()):
                avgint += [dict(integrand_id = integrand_id, node_id = node, weight_id=0, age_lower=age, age_upper = age+dt, time_lower = t, time_upper = t,
                                x_0 = .5, x_1 = 1, x_2 = x2, x_3 = x3, x_4 = None, x_5 = None, x_6 = None)]
    n = len(avgint) - n
    avgint = pd.DataFrame(avgint)
    avgint['avgint_id'] = range(len(avgint))
    assert len(avgint.dropna(subset=c_xcovs, how='any')) == len(avgint), "Every avgint entry should have country covariates -- some were null."
    utilities.system("%s %s predict fit_var" % (dismod_AT, DB.filename))
    surface = DB.predict.merge(DB.avgint).merge(DB.integrand).merge(DB.node)
    for t in sorted(surface.time_lower.unique()):
        for node in sorted(surface.node_id.unique()):
            plt.figure()
            for i,row in surface[(surface.node_id == node) & (surface.time_lower == t)].iterrows():
                lwr, upr = row.age_lower, row.age_upper
                if lwr == upr:
                    plt.plot(lwr, row.avg_integrand, '+g')
                else:
                    plt.plot([lwr, upr], [row.avg_integrand]*2, '-r')
            for i,row in tmp[(tmp.node_id == node) & (tmp.time_lower == t)].iterrows():
                lwr, upr = row.age_lower, row.age_upper
                plt.plot([lwr, upr], [row.meas_value]*2, '-k')
            plt.xlim([0,100])
            # plt.ylim([0,.02])
            plt.title(self.model_name + '\n' + 'Node %d %d mtspecific computed at single age points vs 10 year age ranges'% (node,t))
            plt.xlabel('age')
            plt.ylabel('mtspecific')
            plt.grid(1)

    d = tmp
    p = surface.iloc[-n:]
    fit_data_subset = d.merge(DB.integrand)

    rtn = pd.DataFrame()
    rtn['adj_avg_integrand'] = data_mean(DB, fit_data_subset)
    rtn['adj_meas_std'] = adj_std(DB, fit_data_subset)
    avg_integrand, delta = rtn[['adj_avg_integrand', 'adj_meas_std']].T.values


    eta = 0.000012
    sigma = np.log(d.meas_value + eta + d.meas_std) - np.log(d.meas_value + eta).values
    r = (np.log(d.meas_value.values + eta) - np.log(p.avg_integrand.values + eta)) / sigma
    print (p.T)
    print (d.T)
    print (r)

    # demonstrate_plotting_age_points_vs_intervals(DB)

if 0 and sqlite_filename == '/Users/gma/Projects/IHME/cascade_at_gma.data-tmp/100667/full/1/both/1990_1995_2000_2005_2010_2015/100667.db':
    self=plotter
    data = DB.data_subset.merge(DB.data, how='left').merge(DB.fit_data_subset, how='left')
    print ('------------------------------------')
    print ('---------- predicted surface')
    mask = (self.predicted_surface.integrand_name=='mtspecific') & (self.predicted_surface.time_lower == 1990) & (self.predicted_surface.age_lower.isin((50,)))
    surf = self.predicted_surface[mask]
    print(surf.T)
    node_ids = surf.node_id.unique().tolist()
    mask = (data.x_0 < 0) & (data.time_lower == 1990) & (data.age_lower == 50) & (data.node_id.isin(node_ids))
    print ('---------- data')
    mask_data = data[mask].reset_index(drop=True)
    mask_data.data_id = list(mask_data.index)
    avgint_save = DB.avgint
    DB.avgint = mask_data
    utilities.system("{_dismod_} %s predict fit_var" % DB.filename)
    assert np.max(np.abs(mask_data.avg_integrand - DB.predict.avg_integrand)) < 1e-8
    DB.avgint = avgint_save
    print (mask_data.T)
    print ('---------- data fit and adjustments')
    mask = (self.data_fit_and_adjustments.node_id.isin(node_ids)) & (self.data_fit_and_adjustments.x_0 < 0) & (self.data_fit_and_adjustments.time_lower == 1990) & (self.data_fit_and_adjustments.age_lower == 50)
    fit = self.data_fit_and_adjustments[mask]
    print (fit.T)
    assert np.allclose(fit.x_3.values, surf.x_3.values), "Surface country covariate values do not match the data"
    assert np.allclose(fit.x_4.values, surf.x_4.values, atol=.5, rtol=.001), "Surface country covariate values do not match the data"

    print (data[data.integrand_name == 'mtspecific'])

if 0:
    pred = DB.data.merge(DB.predict, left_on='data_id', right_on='avgint_id')
    plt.plot(pred[pred.time_lower==1990].avg_integrand)
    plt.plot(pred[pred.time_lower==1990].meas_value)


if 0:
    from importer import *
    query = """SELECT location_id, year_id, age_group_id, sex_id, mean_value, model_version_id
    FROM covariate.model
    JOIN shared.location USING(location_id)
    JOIN covariate.model_version USING(model_version_id)
    JOIN covariate.data_version USING(data_version_id)
    WHERE is_best=1 AND covariate_id='57' """
    db = 'cod'
    execute_select(query, 'cov')

    def execute_select(query, db='epi'):
        import mysql_server
        conn_string = mysql_server[db]
        print ("Executing %s query to %s ..." % (db, conn_string), end='', flush = True)
        print (query, flush = True)
        t0 = time.time()
        set_trace()
        engine = sqlalchemy.create_engine(conn_string)
        engine.dispose()
        if 0:
            conn = engine.connect()
            df = pd.read_sql(query, conn.connection)
            conn.close()
        else:
            df = pd.read_sql(query, engine)
        print ("    ... Done -- %f seconds." % (time.time() - t0), flush = True)
        return df
