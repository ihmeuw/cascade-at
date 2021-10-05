#!/usr/bin/env python

import sys
sys.path.append('/Users/gma/Projects/IHME/GIT/DB_tools')

from dismod_db_api import DismodDbAPI
import pandas as pd
from plot_fit_metrics import TestAndPlot

import matplotlib

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('plot_DB.py')

def int_or_float(_str):
    """
    int_or_float(2) # => 2
    int_or_float('2') # => 2
    int_or_float('2.0') # => 2
    int_or_float(2.0) # => 2
    int_or_float(2.1) # => 2.1
    int_or_float('2.1') # => 2.1
    """
    _str = float(_str)
    if _str % 1 == 0:
        return int(_str)
    return _str

def plot_DB(sqlite_filename, pdf_p = True, logscale = True, plot_extent = None, plot3D = True, plot_integrands = None, ages = None, times = None,
            model_version_id = None, model_name='', gbd_round_id = None):
    DB = DismodDbAPI(sqlite_filename)
    if 'cascade_option' in DB.tables and 'time_grid' in DB.cascade_option.cascade_option_name.values:
        surf_time = times if times is not None else tuple(map(int_or_float, DB.cascade_options.time_grid.split()))
        surf_age = ages if ages is not None else tuple(map(int_or_float, DB.cascade_options.age_grid.split()))
    else:
        surf_age = ages if ages is not None else ([0,1] + list(range(5, 100, 5)))
        surf_time = times if times is not None else [1990,1995,2000,2005,2010,2015]
        DB.cascade_option = DB.cascade_option.append(pd.DataFrame([dict(cascade_option_name = 'time_grid', cascade_option_value = ' '.join(map(str, surf_time))),
                                                                   dict(cascade_option_name = 'age_grid', cascade_option_value = ' '.join(map(str, surf_age)))]))

    if model_version_id is not None:
        if 'cascade_option' in DB.tables and 'model_version_id' in DB.cascade_option.cascade_option_name.values:
            cascade_option = DB.cascade_option
            cascade_option.loc[cascade_option.cascade_option_name == 'model_version_id', 'cascade_option_value'] = model_version_id
            DB.cascade_option = cascade_option
        else:
            DB.cascade_option = DB.cascade_option.append(pd.DataFrame([dict(cascade_option_name = 'model_version_id', cascade_option_value = model_version_id)]))

    if gbd_round_id is not None:
        if 'cascade_option' in DB.tables and 'gbd_round_id' in DB.cascade_option.cascade_option_name.values:
            cascade_option = DB.cascade_option
            cascade_option.loc[cascade_option.cascade_option_name == 'gbd_round_id', 'cascade_option_value'] = gbd_round_id
            DB.cascade_option = cascade_option
        else:
            DB.cascade_option = DB.cascade_option.append(pd.DataFrame([dict(cascade_option_name = 'gbd_round_id', cascade_option_value = gbd_round_id)]))
        
    assert DB.cascade_options.model_version_id is not None, 'Model version id must be specified'

    plot_extent = plot_extent
    plotter = TestAndPlot(sqlite_filename, surface_time = surf_time, surface_age = surf_age,
                          plot_data_extent = plot_extent, time_window_for_plots = 2.51,
                          model_version_id = model_version_id, model_name = model_name)
    plotter(pdf_p = pdf_p, adjust_data = True, logscale = logscale, plot3D = plot3D, plot_integrands = plot_integrands, plot_years = times)
    # os.system ("dismodat.py %s db2csv" % sqlite_filename)
    return plotter

if (__name__ == '__main__'):
    def parse_args(fn=None, pdf = False, logscale = True, plot_integrands = None, model_name = "Heading TBD"):
        import argparse
        from distutils.util import strtobool as str2bool
        parser = argparse.ArgumentParser()
        name_string = "-filename" if sys.argv[0] == '' else "filename"
        parser.add_argument(name_string, type=str, help="Dismod_AT sqlite database filename", default = fn)
        parser.add_argument("-a", "--ages", type = float, nargs='+', default = None,
                            help = "Ages to plot -- default %s" % None)
        parser.add_argument("-m", "--model_name", type = str, nargs='+', default = model_name,
                            help = f"Model name -- default = {model_name}")
        parser.add_argument("-v", "--model_version_id", type = int, default = None,
                            help = f"Model version id -- default = None")
        parser.add_argument("-g", "--gbd_round_id", type = int, default = 5,
                            help = f"GBD round id -- default = 5")
        parser.add_argument("-t", "--times", type = float, nargs='+', default = None,
                            help = "Times to plot -- default %s" % None)
        parser.add_argument("-p", "--pdf", nargs='?', type=str2bool, default = pdf, const = True,
                            help = f"Produce pdf plots -- default: {pdf}") 
        parser.add_argument("-l", "--logscale", nargs='?', type=str2bool, default = logscale, const = True,
                            help = f"Produce logscale plot axes -- default: {logscale}") 
        parser.add_argument("-e", "--extent", nargs='?', type=str2bool, default = False, const = True,
                            help = "Plot measured data age and time extents -- default: False") 
        parser.add_argument("-3D", "--plot3D", nargs='?', type=str2bool, default = True, const = True,
                            help = "Produce 3D plots -- default: True") 
        parser.add_argument("-i", "--plot_integrands", type = str, nargs='+', default = plot_integrands,
                            help = f"Integrands to plot -- default = {None if plot_integrands is None else list(plot_integrands)}")
        args = parser.parse_args()
        return args

    pdf_p = True
    logscale = True

    sqlite_filename = None
    mvid = None
    if sys.argv[0] == '':
        sqlite_filename = '/Users/gma/Projects/IHME/cascade_at.data-tmp/100667/full/102/female/1990_1995_2000_2005_2010_2015/100667.db'
        sqlite_filename = '/Users/gma/Projects/IHME/cascade_at.data-tmp/100667/full/64/female/1990_1995_2000_2005_2010_2015/100667.db'
        sqlite_filename = '/Users/gma/Projects/IHME/cascade_at.data-tmp/100667-2018.01.08-drill/full/523/female/1990_1995_2000_2005_2010_2015/100667.db'
        sqlite_filename = '/Users/gma/Projects/IHME/cascade_at.data-tmp/100667/full/102/both/1990_1995_2000_2005_2010_2015/100667.db'
        sqlite_filename = '/Users/gma/Projects/IHME/cascade_at.data-tmp/100667/full/102/female/1990_1995_2000_2005_2010_2015/100667.db'
        sqlite_filename = '/Users/gma/Projects/IHME/cascade_at.data-tmp/100667/full/100/both/1990_1995_2000_2005_2010_2015/fit/100667.db'
        sqlite_filename = '/Users/gma/Projects/IHME/cascade_at.data-tmp/327650/full/1/both/1990_1995_2000_2005_2010_2015/327650.db'
        sqlite_filename = '/Users/gma/Projects/IHME/cascade_at.data-tmp/343691/full/102/both/1990_1995_2000_2005_2010_2015/343691.db'
        sqlite_filename = '/Users/gma/Projects/IHME/modelers/10427/10427.db'
        mvid = 266798; sqlite_filename = '/Users/gma/Projects/IHME/modelers/1976-katie/VIS-run-2010.db'
        sqlite_filename = '/Users/gma/Projects/IHME/GIT/cascade_at.git/cascade_at/dmcascade/simulation/test.db'
        mvid = 268544; sqlite_filename = '/Users/gma/cluster/fit.db'
        mvid = 473171; sqlite_filename = '/Users/gma/Projects/IHME/marlena/473171-end-stage-renal-disease-after-transplant/fit_both.db'
        mvid = 472346; sqlite_filename = '/Users/gma/tmp/dismod_at_472346.db'
        mvid = 474019; sqlite_filename = '/Users/gma/Projects/IHME/DISMOD_AT/asymptotic_mean_error/473953/474019.db'
        mvid = 474101; sqlite_filename = '/Users/gma/Projects/IHME/DISMOD_AT/asymptotic_mean_error/473953/474101.db'
        mvid = 475648; sqlite_filename = '/Users/gma/ihme/ihme_db/temp.db'
        mvid = 475877; sqlite_filename = '/Users/gma/ihme/epi/at_cascade/data/475877/dbs/1/2/dismod.db'

        pdf_p = True
        logscale = True
        DB = DismodDbAPI(sqlite_filename)

        data = DB.data
        data['x_0'] = None
        DB.data = data

    args = parse_args(sqlite_filename, pdf=pdf_p, logscale=logscale)
    if mvid and args.model_version_id is None:
        args.model_version_id = mvid
    args_in = [args.filename]
    kwds_in = dict(pdf_p = args.pdf, logscale = args.logscale, plot_extent = args.extent, plot3D = args.plot3D, plot_integrands = args.plot_integrands, ages = args.ages, times = args.times,
                   model_name = args.model_name, model_version_id = args.model_version_id, gbd_round_id = args.gbd_round_id)
    
    # plot_integrands = ['Sincidence', 'remission', 'mtexcess', 'mtother', 'mtwith', 'susceptible', 'withC', 'prevalence', 'Tincidence', 'mtspecific', 'mtall', 'mtstandard', 'relrisk']
    # plot_integrands = ['Sincidence', 'remission', 'mtexcess', 'mtother', 'mtwith', 'susceptible', 'withC', 'prevalence', 'Tincidence', 'mtspecific', 'mtall', 'mtstandard']
    # logger.error("FIXME -- Relrisk prediction seems to be broken -- removing it from the plots.")
    # kwds_in.update(dict(plot_integrands = plot_integrands))

    print ("plot_DB(%s, %s)" % (', '.join([str(a) for a in args_in]), ', '.join(["%s = %s" % (k,v) for k,v in kwds_in.items()])))
    plot_DB(*args_in, **kwds_in)

    if not args.pdf:
        # Wait for the user to view the plots
        input('\nHit return to close the plots and exit.\n\n')
    else:
        os.system(f'open {args.filename}.pdf')
