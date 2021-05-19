#!/usr/bin/env python
import sys
import os
import math
import numpy as np
import pandas as pd
import subprocess
from functools import lru_cache
from pathlib import Path

from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf

_test_plot_prior_ = not False

interactive = not sys.argv[0]
plt.interactive(interactive)

def system (command) :
    # flush python's pending standard output in case this command generates more standard output
    sys.stdout.flush()
    print (command)
    if isinstance(command, str):
        command = command.split()
    run = subprocess.run(command)
    if run.returncode != 0 :
        raise Exception(f'"{command}" failed.')

def case_study_title(db, version = None, disease = 'TBD', which_fit = 'TBD') :
    # return the title for this study and fit
    covariate = db.covariate
    sex_ref = float(covariate.loc[['sex' in n for n in covariate.c_covariate_name], 'reference'])
    if sex_ref < 0: sex = 'female'
    elif sex_ref > 0: sex = 'male'
    else: sex = 'both'
    location = db.node.loc[db.node.node_id ==
                           int(db.option.loc[db.option.option_name == 'parent_node_id', 'option_value']),
                           'node_name'].squeeze()
    return f"{location}\n{which_fit}, {disease}, {sex}, version={version}"

def get_prior(db, rate_name, age = None, time = None, sex = None, option = 'value'):
    assert not (age and time), "Specify one of age and time."
    rate = db.rate[db.rate.rate_name == rate_name]
    prior = (db.prior
             .merge(db.smooth_grid
                    .merge(rate, how = 'right', left_on = 'smooth_id', right_on = 'parent_smooth_id'),
                    left_on = 'prior_id', right_on = f'{option}_prior_id')
             .merge(db.age, how='left')
             .merge(db.time, how='left'))
    if age is None and time is None:
        return prior
    else:
        if age is not None:
            k = 'time'
            p = prior[prior.age == age].sort_values(by = k)
        if time is not None:
            k = 'age'
            p = prior[prior.time == time].sort_values(by = k)
        x,y = p[[k, 'mean']].values.T
        return x,y

def plot_rate(db, rate_name, title = 'TBD') :
    color_style_list = [
        ('blue',       'dashed'),  ('lightblue',  'solid'),
        ('red',        'dashed'),  ('pink',       'solid'),
        ('green',      'dashed'),  ('lightgreen', 'solid'),
        ('black',      'dashed'),  ('gray',       'solid'),
        ('brown',      'dashed'),  ('sandybrown', 'solid'),
        ('darkorange', 'dashed'),  ('gold',       'solid'),
        ('purple',     'dashed'),  ('violet',     'solid'),
    ]
    n_color_style = len( color_style_list )
    #
    # plot the fit_var grid values for a specified rate.
    #
    var = (db.var.merge(db.fit_var, how='left', left_on = 'var_id', right_on = 'fit_var_id')
           .merge(db.rate, how='left').merge(db.age, how='left').merge(db.time, how='left'))
    parent_node_id = int(db.option.loc[db.option.option_name == 'parent_node_id', 'option_value'])
    parent_node_name = db.node.loc[db.node.node_id == parent_node_id, 'node_name'].values
    rate_id = int(db.rate.loc[db.rate.rate_name == rate_name, 'rate_id'])

    age_min = db.age.age.min()
    age_max = db.age.age.max()
    time_min = db.time.time.min()
    time_max = db.time.time.max()

    fit = var[(var.var_type == 'rate') & (var.rate_id == rate_id) & (var.node_id == parent_node_id)]
    try: sample = db.sample
    except: sample = pd.DataFrame()
    if not sample.empty:
        msg = 'Sample table is the wrong length'
        assert len(sample) == len(sample.sample_index.unique()) * len(var), msg
        grps = sample.groupby(['var_id'], as_index=False)
        fit = fit.merge(grps.var_value.std(ddof=1).rename(columns = {'var_value': 'var_std'}), how='left')
    fit = fit.sort_values(by=['age', 'time', 'node_id'])
    n_age = len(fit.age_id.unique())
    n_time = len(fit.time_id.unique())

    shape = (n_age, n_time)
    age = fit.age.values.reshape(shape)
    time = fit.time.values.reshape(shape)
    rate = fit.fit_var_value.values.reshape(shape)
    prior = get_prior(db, rate_name)['mean']
    if not sample.empty:
        std = fit.var_std.values.reshape(shape)
    else:
        std = None
    #
    rate_max  = max(prior.max(), np.max(rate)) * 1.05
    rate_min  = min(prior.min(), np.min(rate)) * 0.95
    rate_min  = max(rate_min , rate_max * 1e-6)
    ylim = (rate_min, rate_max)
    n_subplot = 1
    if std is not None:
        std_max   = np.max(std) * 1.05
        std_min   = np.min(std) * 0.95
        std_min   = max(std_min, std_max * 1e-5)
        # no standard error subplot when it is identically zero
        if std_min > 0 :
            n_subplot = 2

    file_name = db.path.parent / (rate_name + '.pdf')
    pdf = matplotlib.backends.backend_pdf.PdfPages(file_name)
    #
    # ------------------------------------------------------------------------
    # for each time, plot rate and possibly std as a function of age
    # ------------------------------------------------------------------------
    n_fig       = math.ceil( n_time / ( n_color_style - 1) )
    n_per_fig   = math.ceil( n_time / n_fig )
    assert n_per_fig < n_color_style
    #
    color_index = -1
    #
    for i_fig in range( n_fig ) :
        # save for possible re-use by second subplot
        save_color_index = color_index
        #
        fig    = plt.figure()
        fig.subplots_adjust( hspace = .01 )
        #
        # axis for subplot and title for figure
        axis   = plt.subplot(n_subplot, 1, 1)
        axis.set_title(title)
        #
        start  = i_fig * n_per_fig
        if i_fig > 0 :
            start        = start - 1
            color_index -= 1
        stop   = min(n_time, start + n_per_fig )
        for j in range(start, stop) :
            color_index    = (color_index + 1) % n_color_style
            (color, style,) = color_style_list[color_index]
            x     = age[:,j]
            y     = rate[:,j]
            # avoid values less than or equal zero
            y     = np.maximum(y, rate_min)
            # extend as constant to min and max age
            x     = [age_min] + x.tolist() + [age_max]
            y     = [y[0]]    + y.tolist() + [y[-1]]
            # label used by legend
            label = str( time[0,j] )
            #
            plt.plot(x, y, label=label, color=color, linestyle='solid')
            if rate_name != 'omega':
                px, py = get_prior(db, rate_name, time = time[0,j])
                plt.plot(px, py, label = 'prior', color=color, linestyle='dotted')
            #
            # axis labels
            plt.xlabel('age')
            dage_mean_std = get_prior(db, rate_name, option = 'dage')['std'].mean()
            dtime_mean_std = get_prior(db, rate_name, option = 'dtime')['std'].mean()
            plt.ylabel(f"{rate_name} (mean std -- dAge: {dage_mean_std:2g}, dTime: {dtime_mean_std:.2g})")
            plt.yscale('log')
            plt.ylim(*ylim)
        for i in range(n_age) :
            x = age[i, 0]
            plt.axvline(x, color='black', linestyle='dotted', alpha=0.3)
        # Shrink current axis by 15% and place legend to right
        box = axis.get_position()
        axis.set_position([
            box.x0 + box.width*.05 , box.y0, box.width*0.85, box.height
        ])
        axis.legend(
            title = 'time', loc='center left', bbox_to_anchor=(1, 0.5)
        )
        # --------------------------------------------------------------------
        if n_subplot == 2 :
            # restart colors so are the same as for first subplot
            # (only need one legend for both subplots)
            color_index = save_color_index
            #
            # ais for subplot (uses same title as figure)
            axis   = plt.subplot(n_subplot, 1, 2)
            #
            start  = i_fig * n_per_fig
            if i_fig > 0 :
                start        = start - 1
                color_index -= 1
            stop   = min(n_time, start + n_per_fig )
            for j in range(start, stop) :
                color_index    = (color_index + 1) % n_color_style
                (color, style,) = color_style_list[color_index]
                x     = age[:,j]
                y     = std[:,j]
                # avoid values less than or equal zero
                y     = np.maximum(y, std_min)
                # extend as constant to min and max age
                x     = [age_min] + x.tolist() + [age_max]
                y     = [y[0]]    + y.tolist() + [y[-1]]
                # label used by legend
                label = str( time[0,j] )
                #
                plt.plot(x, y, label=label, color=color, linestyle='solid')
                if rate_name != 'omega':
                    px, py = get_prior(db, rate_name, time = time[0,j])
                    plt.plot(px, py, label = 'prior', color=color, linestyle='dotted')
                #
                # axis labels
                plt.xlabel('age')
                plt.ylabel('std error')
                plt.yscale('log')
                plt.ylim(std_min, std_max)
            for i in range(n_age) :
                x = age[i, 0]
                plt.axvline(x, color='black', linestyle='dotted', alpha=0.3)
            # Shrink current axis by 15% but do not need legend this time
            box = axis.get_position()
            axis.set_position([
                box.x0 + box.width*.05 , box.y0, box.width*0.85, box.height
            ])
        # --------------------------------------------------------------------
        pdf.savefig( fig )
        plt.close( fig )
    # ------------------------------------------------------------------------
    # for each age, plot rate as a function of time
    # ------------------------------------------------------------------------
    n_fig       = math.ceil( n_age / (n_color_style - 1) )
    n_per_fig   = math.ceil( n_age / n_fig )
    assert n_per_fig < n_color_style
    #
    color_index = -1
    #
    for i_fig in range( n_fig ) :
        # save for possible re-use by second subplot
        save_color_index = color_index
        #
        # new figure
        fig    = plt.figure()
        fig.subplots_adjust( hspace = .01 )
        #
        # axis for subplot and title for figure
        axis   = plt.subplot(n_subplot, 1 ,1)
        axis.set_title( title )
        #
        start  = i_fig * n_per_fig
        if i_fig > 0 :
            start        = start - 1
            color_index -= 1
        stop   = min(n_age, start + n_per_fig )
        for i in range(start, stop) :
            color_index    = (color_index + 1) % n_color_style
            (color, style) = color_style_list[color_index]
            x     = time[i,:]
            y     = rate[i,:]
            # avoid values less than or equal zero
            y     = np.maximum(y, rate_min)
            # extend as constant to min and max time
            x     = [time_min] + x.tolist() + [time_max]
            y     = [y[0]]     + y.tolist() + [y[-1]]
            # label used by legend
            label = str( age[i,0] )
            plt.plot(x, y, label=label, color=color, linestyle='solid')
            if rate_name != 'omega':
                px, py = get_prior(db, rate_name, age = age[i,0])
                plt.plot(px, py, label = 'prior', color=color, linestyle='dotted')
            #
            # axis labels
            plt.xlabel('time')
            dage_mean_std = get_prior(db, rate_name, option = 'dage')['std'].mean()
            dtime_mean_std = get_prior(db, rate_name, option = 'dtime')['std'].mean()
            plt.ylabel(f"{rate_name} (mean std -- dAge: {dage_mean_std:2g}, dTime: {dtime_mean_std:.2g})")
            plt.yscale('log')
            plt.ylim(*ylim)
        for j in range(n_time) :
            x = time[0, j]
            plt.axvline(x, color='black', linestyle='dotted', alpha=0.3)
        # Shrink current axis by 15% and place legend to right
        box = axis.get_position()
        axis.set_position([
            box.x0 + box.width*.05 , box.y0, box.width*0.85, box.height
        ])
        axis.legend(
            title = 'age', loc='center left', bbox_to_anchor=(1, 0.5)
        )
        # --------------------------------------------------------------------
        if n_subplot == 2 :
            # restart colors so are the same as for first subplot
            # (only need one legend for both subplots)
            color_index = save_color_index
            #
            # axis for subplot (uses same title as figure)
            axis   = plt.subplot(n_subplot, 1, 2)
            #
            start  = i_fig * n_per_fig
            if i_fig > 0 :
                start        = start - 1
                color_index -= 1
            stop   = min(n_age, start + n_per_fig )
            for i in range(start, stop) :
                color_index    = (color_index + 1) % n_color_style
                (color, style) = color_style_list[color_index]
                x     = time[i,:]
                y     = std[i,:]
                # avoid values less than or equal zero
                y     = np.maximum(y, std_min)
                # extend as constant to min and max time
                x     = [time_min] + x.tolist() + [time_max]
                y     = [y[0]]     + y.tolist() + [y[-1]]
                # label used by legend
                label = str( age[i,0] )
                plt.plot(x, y, label=label, color=color, linestyle='solid')
                if rate_name != 'omega':
                    px, py = get_prior(db, rate_name, age = age[i,0])
                    plt.plot(px, py, label = 'prior', color=color, linestyle='dotted')
                #
                # axis labels
                plt.xlabel('time')
                plt.ylabel('std error')
                plt.yscale('log')
                plt.ylim(std_min, std_max)
            for j in range(n_time) :
                x = time[0, j]
                plt.axvline(x, color='black', linestyle='dotted', alpha=0.3)
            # Shrink current axis by 15% but do not ned legent this time
            box = axis.get_position()
            axis.set_position([
                box.x0 + box.width*.05 , box.y0, box.width*0.85, box.height
            ])
        # --------------------------------------------------------------------
        pdf.savefig( fig )
        plt.close( fig )
    #
    pdf.close()
# ----------------------------------------------------------------------------

def plot_integrand(db, data, integrand_name, title='TBD') :
    # Plot the data, model, and residual values for a specified integrand.
    # Covariate values used for each model point are determined by
    # correspondign data point.
    fit_var = db.fit_var

    this_integrand_id = int(db.integrand.loc[db.integrand.integrand_name == integrand_name, 'integrand_id'])
    data = data[data.integrand_id == this_integrand_id]
    parent_node_id = int(db.option.loc[db.option.option_name == 'parent_node_id', 'option_value'])
    parent_node_name = db.node.loc[db.node.node_id == parent_node_id, 'node_name'].values

    #
    meas_value  = data['meas_value'].values
    age  = (( data['age_lower'] + data['age_upper'] ) / 2.0).values
    time = (( data['time_lower'] + data['time_upper'] ) / 2.0).values
    node = data['node_id'].values
    avg_integrand = data['avg_integrand'].values
    weighted_residual = data['weighted_residual'].values

    n_list = len(data)
    index_list = list(range(n_list))
    if n_list < 2 :
        msg = 'plot_integrand: ' + integrand_name + ' has less than 2 points'
        print(msg)
        #### return
    #
    # map node id to index in set of node_id's
    node_set = sorted( set( node ) )
    for index in index_list :
        node_id = node[index]
        node[index] = node_set.index( node_id )
    #
    # add 1 to index so index zero not hidden by y-axis
    index       = np.array( index_list ) + 1
    #
    y_median    = np.median( meas_value)
    y_max       = y_median * 1e+3
    y_min       = y_median * 1e-3
    r_norm      = np.linalg.norm( weighted_residual )
    r_avg_sq    = r_norm * r_norm / len( weighted_residual )
    r_max       = 4.0 * np.sqrt( r_avg_sq )
    r_min       = - r_max
    #
    avg_integrand = np.maximum( avg_integrand, y_min )
    avg_integrand = np.minimum( avg_integrand, y_max )
    #
    meas_value = np.maximum( meas_value, y_min )
    meas_value = np.minimum( meas_value, y_max )
    #
    weighted_residual = np.maximum( weighted_residual, r_min )
    weighted_residual = np.minimum( weighted_residual, r_max )
    #
    y_limit = [ 0.9 * y_min, 1.1 * y_max ]
    r_limit = [ 1.1 * r_min, 1.1 * r_max ]
    #
    point_size  = 1 # gma np.array( n_list * [ 1 ] )
    marker_size = 10 # gma np.array( n_list * [ 10 ] )
    #
    file_name = db.path.parent / (integrand_name + '.pdf')
    pdf = matplotlib.backends.backend_pdf.PdfPages(file_name)
    #
    for x_name in [ 'index', 'node', 'age', 'time' ] :
        x = eval(x_name)
        #
        fig, axes = plt.subplots(3, 1, sharex=True)
        fig.subplots_adjust(hspace=0)
        #
        #
        sp = plt.subplot(3, 1, 1)
        sp.set_xticklabels( [] )
        y =  meas_value
        plt.scatter(x, y, marker='.', color='black', s = point_size)
        plt.ylabel(integrand_name)
        plt.yscale('log')
        mask = ~((y_min < y) & (y < y_max))
        plt.scatter(x[mask], y[mask], marker='+', color='red', s=marker_size )
        if np.isfinite(y_limit).all():
            plt.ylim(y_limit[0], y_limit[1])
        #
        plt.title( title )
        #
        sp = plt.subplot(3, 1, 2)
        sp.set_xticklabels( [] )
        y = avg_integrand
        plt.scatter(x, y, marker='.', color='black', s = point_size)
        plt.ylabel('model')
        plt.yscale('log')
        mask = ~((y_min < y) & (y < y_max))
        plt.scatter(x[mask], y[mask], marker='+', color='red', s=marker_size )
        if np.isfinite(y_limit).all():
            plt.ylim(y_limit[0], y_limit[1])
        #
        # this plot at the bottom of the figure has its x tick labels
        plt.subplot(3, 1, 3)
        y = weighted_residual
        plt.scatter(x, y, marker='.', color='black', s = point_size)
        plt.ylabel('residual')
        mask = ~((r_min < y) & (y < r_max))
        plt.scatter(x[mask], y[mask], marker='+', color='red', s=marker_size )
        if np.isfinite(r_limit).all():
            plt.ylim(r_limit[0], r_limit[1])
        y = 0.0
        plt.axhline(y, linestyle='solid', color='black', alpha=0.3 )
        #
        plt.xlabel(x_name)
        #
        pdf.savefig( fig )
        plt.close( fig )
    #
    fig = plt.figure()
    ax = plt.gca()
    plt.title(title)
    r = data['weighted_residual'].values
    plt.xlabel(f'Weighted Residual (range = [{np.round(r.min(), 1)}, {np.round(r.max(),1)}])')
    plt.ylabel(f'{integrand_name.capitalize()} Count')
    n_bins = max(100, int(len(r)/100))
    ax.hist(r, bins=n_bins)
    pdf.savefig( fig )
    plt.close( fig )
    pdf.close()

@lru_cache
def get_fitted_data(db):
    data = (db.data_subset.merge(db.data, how='left')
            .merge(db.fit_data_subset, how='left', left_on= 'data_subset_id', right_on = 'fit_data_subset_id')
            .merge(db.integrand, how='left'))
    return data

def plot_predict(db, covariate_integrand_list, predict_integrand_list, title='TBD') :
    # Plot the model predictions for each integrand in the predict integrand
    # list. The is one such plot for each integrand in the covariate integrand
    # list (which determines the covariate values used for the predictions).
    # The avgint and predict tables are overwritten by this procedure.
    dummy_variable_used_to_end_doc_string = None
    # -----------------------------------------------------------------------
    # create avgint table
    # For each covariate_integrand
    #    For data row corresponding to this covariate_integrand
    #        For each predict_intgrand
    #            write a row with specified covariates for predict_integrand
    #-----------------------------------------------------------------------
    #
    covariate_id_list = db.integrand.loc[db.integrand.integrand_name.isin(covariate_integrand_list), 'integrand_id'].values
    predict_id_list = db.integrand.loc[db.integrand.integrand_name.isin(predict_integrand_list), 'integrand_id'].values

    data = get_fitted_data(db)
    db.avgint = avgint = pd.DataFrame()
    cols = db.avgint.columns.drop('avgint_id').tolist() + db.covariate.covariate_name.tolist()
    for cov_id in covariate_id_list:
        for integrand_id in predict_id_list:
            cov_data = data.loc[data.integrand_id == cov_id, cols]
            cov_data.loc[cov_data.integrand_id == cov_id, 'integrand_id'] = integrand_id
            avgint = avgint.append(cov_data)
    avgint = avgint.reset_index(drop=True)
    avgint['avgint_id'] = avgint.index
    db.avgint = avgint

    # Predict for this avgint table
    system(f'dismod_at {db.path} predict fit_var')
    #
    predict = avgint.merge(db.predict)
    # ------------------------------------------------------------------------
    # initialize
    file_name = db.path.parent / 'predict.pdf'
    pdf = matplotlib.backends.backend_pdf.PdfPages(file_name)
    #
    predict_id = 0
    for covariate_integrand_id in covariate_id_list :
        point_size            =  1 # gma n_data_rows * [ 1 ]
        n_predict_integrand   = len(predict_integrand_list)
        #
        fig, axes = plt.subplots(n_predict_integrand, 1, sharex=True)
        fig.subplots_adjust(hspace=0)
        #
        plot_index = 0
        for integrand_name, integrand_id in zip(predict_integrand_list, predict_id_list) :
            #
            # Last plot at the bottom of the figure has its x tick labels
            plot_index += 1
            sp = plt.subplot(n_predict_integrand, 1, plot_index)
            if plot_index < n_predict_integrand :
                sp.set_xticklabels( [] )
            mask = predict.integrand_id == integrand_id
            x  = predict.loc[mask, ['age_lower', 'age_upper']].mean(axis=1)
            y  = predict.loc[mask, 'avg_integrand']
            plt.scatter(x, y, marker='.', color='black', s=point_size )
            plt.yscale('log')
            plt.ylabel( integrand_name )
        plt.xlabel('age')
        covariate_name = db.integrand.loc[db.integrand.integrand_id == covariate_integrand_id, 'integrand_name'].squeeze()
        ############## plt.suptitle(f'Covariate Integrand = {covariate_name}\n{case_study_title(db, "")[-1]}' )
        plt.suptitle(f'Covariate Integrand = {covariate_name}\n{title}' )
        #
        pdf.savefig( fig )
        plt.close( fig )
    pdf.close()

def parse_args():
    import argparse
    from distutils.util import strtobool as str2bool
    parser = argparse.ArgumentParser()
    name_string = "-filename" if sys.argv[0] == '' else "filename"
    parser.add_argument(name_string, type=str, help="Dismod_AT sqlite database filename")
    parser.add_argument("-d", "--disease", type = str, help="Disease name (for plot title)")
    parser.add_argument("-f", "--fit_type", type = str, help="Type of the fit (for plot title)")
    parser.add_argument("-v", "--model_version_id", type = int, default = None,
                        help = f"Model version id -- default = None")
    parser.add_argument("-c", "--covariate_integrands", type = str, nargs='+', default = ['mtspecific', 'prevalence'],
                        help = f"Integrands from which to derive covariates -- default = [mtspecific prevalence]")
    args = parser.parse_args()
    return args

def main():
    from cascade_at.dismod.api.dismod_io import DismodIO

    args = parse_args()
    path = Path(args.filename).expanduser()
    assert path.is_file(), f"The database path {path} does not exist."
    global db
    db = DismodIO(path)

    title = case_study_title(db, version = args.model_version_id, disease = args.disease, which_fit = args.fit_type)

    data = get_fitted_data(db)
    data_integrands = sorted(set(data.integrand_name.unique()) - set(['mtall', 'mtother']))
    no_ode_integrands = sorted(set(['Sincidence', 'mtexcess', 'mtother', 'remission']).intersection(data_integrands))
    yes_ode_integrands = sorted((set(data_integrands) - set(no_ode_integrands)).intersection(data_integrands))
    all_integrands = no_ode_integrands + yes_ode_integrands

    covariate_integrand_list = yes_ode_integrands
    predict_integrand_list = [ 'susceptible', 'withC' ]
    
    rate = db.rate
    integrand = db.integrand
    rate_names = rate.loc[~rate.parent_smooth_id.isna(), 'rate_name'].tolist()
    for rate_name in rate_names:
        plot_rate(db, rate_name, title = title)
    for integrand_name in all_integrands:
        plot_integrand(db, data, integrand_name, title = title)
    plot_predict(db, covariate_integrand_list, predict_integrand_list, title = title)


if __name__ == '__main__':
    if _test_plot_prior_:
        sys.argv = 'plot /Users/gma/ihme/epi/at_cascade/data/475588/dbs/1/3/dismod.db -d t1-diabetes -f ODE-fit -v 475588'.split()
    main()

