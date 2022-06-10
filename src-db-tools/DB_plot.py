#!/usr/bin/env python

import sys
import os

from dismod_db_api import DismodDbAPI
from constants import sex_name2dismod_id
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
            sex = 'female', model_version_id = None, model_name='', gbd_round_id = None):
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
    plotter = TestAndPlot(sqlite_filename, surface_time = surf_time, surface_age = surf_age, sex = sex, 
                          plot_data_extent = plot_extent, time_window_for_plots = 2.51,
                          model_version_id = model_version_id, model_name = model_name)
    plotter(pdf_p = pdf_p, adjust_data = True, logscale = logscale, plot3D = plot3D, plot_integrands = plot_integrands, plot_years = times)
    # os.system ("dismodat.py %s db2csv" % sqlite_filename)
    return plotter

if (__name__ == '__main__'):
    def parse_args(fn=None, pdf = False, logscale = True, plot_integrands = None, model_name = "Heading TBD", sex = 'female'):
        import argparse
        from distutils.util import strtobool as str2bool
        parser = argparse.ArgumentParser()
        name_string = "-filename" if sys.argv[0] == '' else "filename"
        parser.add_argument(name_string, type=str, help="Dismod_AT sqlite database filename", default = fn)
        parser.add_argument("-a", "--ages", type = float, nargs='+', default = None,
                            help = "Ages to plot -- default %s" % None)
        parser.add_argument("-m", "--model_name", type = str, nargs='+', default = model_name,
                            help = f"Model name -- default = {model_name}")
        parser.add_argument("-s", "--sex_name", type = str, default = sex,
                            help = f"Sex to plot -- one of male, female or both, default = {sex}")
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
        parent_location_id = 1
        sex = 'female'
        sex_id = sex_name2dismod_id[sex]
        mvid = 475876; sqlite_filename = f'/Users/gma/ihme/epi/at_cascade/data/{mvid}/dbs/{parent_location_id}/{sex_id}/dismod.db'

        pdf_p = True
        logscale = True
        DB = DismodDbAPI(sqlite_filename)

        data = DB.data
        data['x_0'] = None
        DB.data = data

    args = parse_args(sqlite_filename, pdf=pdf_p, logscale=logscale, sex = sex)
    if mvid and args.model_version_id is None:
        args.model_version_id = mvid
    args_in = [args.filename]
    kwds_in = dict(pdf_p = args.pdf, logscale = args.logscale, plot_extent = args.extent, plot3D = args.plot3D, plot_integrands = args.plot_integrands, ages = args.ages, times = args.times,
                   model_name = args.model_name, model_version_id = args.model_version_id, gbd_round_id = args.gbd_round_id, sex = args.sex_name)
    
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