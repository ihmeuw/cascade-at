import sys
import numpy
import math
from pathlib import Path
from cascade_at.dismod.api.dismod_io import DismodIO

sys.path.append('/opt/prefix/dismod_at/bin/')
case_study_title = "TBD"

from fit_ihme import new_fit_directory, plot_rate, plot_integrand, plot_predict

disease_directory = Path('/Users/gma/ihme/epi/at_cascade/t1_diabetes_test')
which_fit = 'no_ode'

db = DismodIO(disease_directory / which_fit / 'no_ode.db')

def plot_rate(db, rate_name, directory, which_fit) :
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
    # node_table
    node_table = db.node
    #
    # age_table
    age_table = db.age
    #
    # age_table
    time_table = db.time
    #
    # var_table
    var_table = db.var
    #
    # fit_var
    fit_var_table = db.fit_var
    #
    # smooth
    smooth_table = db.smooth
    #
    # sample
    try:
        sample_table = db.sample
    except:
        sample_table = None
    #
    # rate_id
    rate_id = int(db.rate.loc[db.rate.rate_name == rate_name, 'rate_id'])
    #
    # parent_node_id
    parent_node_id   = int(db.option.loc[db.option.option_name == 'parent_node_id', 'option_value'])
    #
    # parent_node_name
    parent_node_name = str(node_table.loc[node_table.node_id == parent_node_id, 'node_name'])
    #
    # age and time limits in plots
    age_min = age_table.age.min()
    age_max = age_table.age.max()
    #
    time_min = time_table.time.min()
    time_max = time_table.time.max()
    #
    # class for compariing an (age_id, time_id) pairs
    class pair:
        def __init__(self, age_id, time_id) :
            self.age_id  = age_id
            self.time_id = time_id
        def __lt__(self, other) :
            if self.age_id != other.age_id :
                self_age   = age_table.loc[ self.age_id, 'age']
                other_age  = age_table.loc[ other.age_id, 'age']
                return self_age < other_age
            self_time  = time_table.loc[ self.time_id, 'time']
            other_time = time_table.loc[ other.time_id, 'time']
            return self_time < other_time
        def _eq_(self, other) :
            equal = self.age_id == other.age_id
            equal = equal and (self.time_id == other.time_id)
            return equla
        def __gt__(self, other) :
            return __lt__(other, self)
        def __le__(self, other) :
            return __lt__(self, other) or __eq__(self, other)
        def __ge__(self, other) :
            return __lt__(other, self) or __eq__(other, self)
        def __ne__(self, other) :
            return not __eq__(self, other)
    #
    # triple_list
    triple_list  = list()
    smooth_id    = None
    for (var_id, row) in var_table.iterrows() :
        if row['var_type'] == 'rate' :
            if row['rate_id'] == rate_id :
                if  row['node_id'] == parent_node_id :
                    age_id  = row['age_id']
                    time_id = row['time_id']
                    triple_list.append( (age_id, time_id, var_id)  )
                    if smooth_id is None :
                        smooth_id = row['smooth_id']
                    else :
                        assert smooth_id == row['smooth_id']
    if smooth_id == None :
        print ('plot_rate: ' + rate_name + ' is identically zero')
        return
    #
    # n_age, n_time, n_var
    n_age  = smooth_table.loc[smooth_id, 'n_age']
    n_time = smooth_table.loc[smooth_id, 'n_time']
    n_var  = len(var_table)
    #
    # sort triple_list first by age and then by time
    key = lambda triple : pair( triple[0], triple[1] )
    triple_list = sorted(triple_list, key = key )
    #
    # creaate the mesghgird (age, time, rate)
    age  = numpy.zeros( (n_age, n_time), dtype = float)
    time = numpy.zeros( (n_age, n_time), dtype = float)
    rate = numpy.zeros( (n_age, n_time), dtype = float)
    # gma if sample_command :
    if 0:
        n_sample = int( number_sample_arg )
        assert len(sample_table) == n_sample * n_var
        std  = numpy.zeros( (n_age, n_time), dtype = float)
    #
    for i in range(n_age) :
        for j in range(n_time) :
            k       = i * n_time + j
            triple  = triple_list[k]
            #
            age_id  = triple[0]
            time_id = triple[1]
            var_id  = triple[2]
            #
            age[i, j]  = age_table.loc[age_id, 'age']
            time[i, j] = time_table.loc[time_id, 'time']
            rate[i, j] = fit_var_table.loc[var_id, 'fit_var_value']
            # gma if sample_command and rate_name != 'omega' :
            if 0:
                sumsq = 0.0
                for k in range(n_sample) :
                    sample_id = k * n_var + var_id
                    var_value = sample_table[sample_id]['var_value']
                    sumsq    += (var_value - rate[i, j])**2
                std[i, j] = numpy.sqrt(sumsq / n_sample)
    #
    rate_max  = numpy.max(rate) * 1.05
    rate_min  = numpy.min(rate) * 0.95
    rate_min  = max(rate_min , rate_max * 1e-6)
    n_subplot = 1
    # gma if sample_command and rate_name != 'omega' :
    if 0:
        std_max   = numpy.max(std) * 1.05
        std_min   = numpy.min(std) * 0.95
        std_min   = max(std_min, std_max * 1e-5)
        # no standard error subplot when it is identically zero
        if std_min > 0 :
            n_subplot = 2
    #
    from matplotlib import pyplot
    #
    import matplotlib.backends.backend_pdf
    file_name = directory / which_fit / (rate_name + '.pdf')
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
        fig    = pyplot.figure()
        fig.subplots_adjust( hspace = .01 )
        #
        # axis for subplot and title for figure
        axis   = pyplot.subplot(n_subplot, 1, 1)
        # gma axis.set_title( case_study_title(parent_node_name, which_fit) )
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
            y     = numpy.maximum(y, rate_min)
            # extend as constant to min and max age
            x     = [age_min] + x.tolist() + [age_max]
            y     = [y[0]]    + y.tolist() + [y[-1]]
            # label used by legend
            label = str( time[0,j] )
            #
            pyplot.plot(x, y, label=label, color=color, linestyle=style)
            #
            # axis labels
            pyplot.xlabel('age')
            pyplot.ylabel(rate_name)
            pyplot.yscale('log')
            pyplot.ylim(rate_min, rate_max)
        for i in range(n_age) :
            x = age[i, 0]
            pyplot.axvline(x, color='black', linestyle='dotted', alpha=0.3)
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
            axis   = pyplot.subplot(n_subplot, 1, 2)
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
                y     = numpy.maximum(y, std_min)
                # extend as constant to min and max age
                x     = [age_min] + x.tolist() + [age_max]
                y     = [y[0]]    + y.tolist() + [y[-1]]
                # label used by legend
                label = str( time[0,j] )
                #
                pyplot.plot(x, y, label=label, color=color, linestyle=style)
                #
                # axis labels
                pyplot.xlabel('age')
                pyplot.ylabel('std error')
                pyplot.yscale('log')
                pyplot.ylim(std_min, std_max)
            for i in range(n_age) :
                x = age[i, 0]
                pyplot.axvline(x, color='black', linestyle='dotted', alpha=0.3)
            # Shrink current axis by 15% but do not need legend this time
            box = axis.get_position()
            axis.set_position([
                box.x0 + box.width*.05 , box.y0, box.width*0.85, box.height
            ])
        # --------------------------------------------------------------------
        pdf.savefig( fig )
        pyplot.close( fig )
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
        fig    = pyplot.figure()
        fig.subplots_adjust( hspace = .01 )
        #
        # axis for subplot and title for figure
        axis   = pyplot.subplot(n_subplot, 1 ,1)
        # gma axis.set_title( case_study_title(parent_node_name, which_fit) )
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
            y     = numpy.maximum(y, rate_min)
            # extend as constant to min and max time
            x     = [time_min] + x.tolist() + [time_max]
            y     = [y[0]]     + y.tolist() + [y[-1]]
            # label used by legend
            label = str( age[i,0] )
            pyplot.plot(x, y, label=label, color=color, linestyle=style)
            #
            # axis labels
            pyplot.xlabel('time')
            pyplot.ylabel(rate_name)
            pyplot.yscale('log')
            pyplot.ylim(rate_min, rate_max)
        for j in range(n_time) :
            x = time[0, j]
            pyplot.axvline(x, color='black', linestyle='dotted', alpha=0.3)
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
            axis   = pyplot.subplot(n_subplot, 1, 2)
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
                y     = numpy.maximum(y, std_min)
                # extend as constant to min and max time
                x     = [time_min] + x.tolist() + [time_max]
                y     = [y[0]]     + y.tolist() + [y[-1]]
                # label used by legend
                label = str( age[i,0] )
                pyplot.plot(x, y, label=label, color=color, linestyle=style)
                #
                # axis labels
                pyplot.xlabel('time')
                pyplot.ylabel('std error')
                pyplot.yscale('log')
                pyplot.ylim(std_min, std_max)
            for j in range(n_time) :
                x = time[0, j]
                pyplot.axvline(x, color='black', linestyle='dotted', alpha=0.3)
            # Shrink current axis by 15% but do not ned legent this time
            box = axis.get_position()
            axis.set_position([
                box.x0 + box.width*.05 , box.y0, box.width*0.85, box.height
            ])
        # --------------------------------------------------------------------
        pdf.savefig( fig )
        pyplot.close( fig )
    #
    pdf.close()
# ----------------------------------------------------------------------------
def plot_integrand(integrand_name, directory, which_fit) :
    # Plot the data, model, and residual values for a specified integrand.
    # Covariate values used for each model point are determined by
    # correspondign data point.
    table_name = 'data_subset'
    (data_subset_table, col_name, col_type) = get_table(table_name)
    #
    table_name = 'fit_var'
    (fit_var_table, col_name, col_type) = get_table(table_name)
    #
    table_name = 'fit_data_subset'
    (fit_data_subset_table, col_name, col_type) = get_table(table_name)
    #
    # this_integrand_id
    this_integrand_id = integrand_name2id[integrand_name]
    #
    # parent_node_name
    parent_node_id   = get_parent_node_id()
    parent_node_name = node_table[parent_node_id]['node_name']
    #
    n_list                  = 0
    avg_integrand_list      = list()
    weighted_residual_list  = list()
    meas_value_list         = list()
    age_list                = list()
    time_list               = list()
    node_list               = list()
    for data_subset_id in range( len(data_subset_table) ) :
        data_id        = data_subset_table[data_subset_id]['data_id']
        row            = data_table[data_id]
        integrand_id   = row['integrand_id']
        #
        if integrand_id == this_integrand_id :
            n_list += 1
            #
            meas_value  = row['meas_value']
            meas_value_list.append( meas_value )
            #
            age  = ( row['age_lower'] + row['age_upper'] ) / 2.0
            age_list.append( age )
            #
            time = ( row['time_lower'] + row['time_upper'] ) / 2.0
            time_list.append(time)
            #
            node_id    = row['node_id']
            node_list.append( node_id )
            #
            row  = fit_data_subset_table[data_subset_id]
            #
            avg_integrand = row['avg_integrand']
            avg_integrand_list.append( avg_integrand )
            #
            weighted_residual = row['weighted_residual']
            weighted_residual_list.append( weighted_residual )
    index_list = range(n_list)
    if n_list < 2 :
        msg = 'plot_integrand: ' + integrand_name + ' has less than 2 points'
        print(msg)
        return
    #
    # map node id to index in set of node_id's
    node_set   = list( set( node_list ) )
    for index in index_list :
        node_id = node_list[index]
        node_list[index] = node_set.index( node_id )
    #
    avg_integrand     = numpy.array( avg_integrand_list )
    meas_value        = numpy.array( meas_value_list )
    weighted_residual = numpy.array( weighted_residual_list )
    age               = numpy.array( age_list )
    time              = numpy.array( time_list )
    node              = numpy.array( node_list )
    # add 1 to index so index zero not hidden by y-axis
    index             = numpy.array( index_list ) + 1
    #
    y_median    = numpy.median( meas_value)
    y_max       = y_median * 1e+3
    y_min       = y_median * 1e-3
    r_norm      = numpy.linalg.norm( weighted_residual )
    r_avg_sq    = r_norm * r_norm / len( weighted_residual )
    r_max       = 4.0 * numpy.sqrt( r_avg_sq )
    r_min       = - r_max
    #
    avg_integrand = numpy.maximum( avg_integrand, y_min )
    avg_integrand = numpy.minimum( avg_integrand, y_max )
    #
    meas_value = numpy.maximum( meas_value, y_min )
    meas_value = numpy.minimum( meas_value, y_max )
    #
    weighted_residual = numpy.maximum( weighted_residual, r_min )
    weighted_residual = numpy.minimum( weighted_residual, r_max )
    #
    y_limit = [ 0.9 * y_min, 1.1 * y_max ]
    r_limit = [ 1.1 * r_min, 1.1 * r_max ]
    #
    point_size  = numpy.array( n_list * [ 1 ] )
    marker_size = numpy.array( n_list * [ 10 ] )
    #
    from matplotlib import pyplot
    import matplotlib.backends.backend_pdf
    file_name = directory / which_fit / integrand_name + '.pdf'
    print (file_name)
    pdf = matplotlib.backends.backend_pdf.PdfPages(file_name)
    #
    for x_name in [ 'index', 'node', 'age', 'time' ] :
        x          = eval( x_name )
        #
        fig, axes = pyplot.subplots(3, 1, sharex=True)
        fig.subplots_adjust(hspace=0)
        #
        #
        sp = pyplot.subplot(3, 1, 1)
        sp.set_xticklabels( [] )
        y =  meas_value
        pyplot.scatter(x, y, marker='.', color='black', s = point_size)
        pyplot.ylabel(integrand_name)
        pyplot.yscale('log')
        for limit in [ y_max, y_min ] :
            flag = y == limit
            size = marker_size[flag]
            pyplot.scatter(x[flag], y[flag], marker='+', color='red', s=size )
        pyplot.ylim(y_limit[0], y_limit[1])
        #
        if x_name == 'index' :
            pyplot.title( case_study_title(parent_node_name, which_fit) )
        #
        sp = pyplot.subplot(3, 1, 2)
        sp.set_xticklabels( [] )
        y = avg_integrand
        pyplot.scatter(x, y, marker='.', color='black', s = point_size)
        pyplot.ylabel('model')
        pyplot.yscale('log')
        for limit in [ y_max, y_min ] :
            flag = y == limit
            size = marker_size[flag]
            pyplot.scatter(x[flag], y[flag], marker='+', color='red', s=size )
        pyplot.ylim(y_limit[0], y_limit[1])
        #
        # this plot at the bottom of the figure has its x tick labels
        pyplot.subplot(3, 1, 3)
        y = weighted_residual
        pyplot.scatter(x, y, marker='.', color='black', s = point_size)
        pyplot.ylabel('residual')
        for limit in [ r_max, r_min ] :
            flag = y == limit
            size = marker_size[flag]
            pyplot.scatter(x[flag], y[flag], marker='+', color='red', s=size )
        pyplot.ylim(r_limit[0], r_limit[1])
        y = 0.0
        pyplot.axhline(y, linestyle='solid', color='black', alpha=0.3 )
        #
        pyplot.xlabel(x_name)
        #
        pdf.savefig( fig )
        pyplot.close( fig )
    #
    pdf.close()

if 1:
    # plot rate
    rate_table = db.rate
    for i,row in rate_table.iterrows() :
        if row['parent_smooth_id'] is not None :
            rate_name = row['rate_name']
            print (rate_name)
            plot_rate(db, row['rate_name'], disease_directory, which_fit)
    #
if 0:
    # plot data
    for integrand_name in integrand_list_all :
        plot_integrand(db, integrand_name, fit_directory, which_fit)
    #
    # plot predictions
    predict_integrand_list   = [ 'susceptible', 'withC' ]
    covariate_integrand_list = integrand_list_yes_ode
    plot_predict(
        covariate_integrand_list,
        predict_integrand_list,
        fit_directory,
        which_fit
    )


# new_fit_directory(which_fit, disease_directory='/Users/gma/ihme/epi/at_cascade/t1_diabetes_test')
