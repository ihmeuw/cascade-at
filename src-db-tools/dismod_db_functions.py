#!/usr/bin/env python

import sys
import os
import time
import numpy as np
import tempfile
import shutil
from pdb import set_trace

from cascade_at_gma.lib.dismod_db_api import DismodDbAPI

sys.path.append('/opt/prefix/dismod_at/lib/python3.9/site-packages')
from dismod_at.db2csv_command import db2csv_command

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('dismod_db_functions.py')

def copyDB(DB, addname, verbose = True):
    d,e = os.path.splitext(DB.filename)
    copyname = '%s_%s%s' % (d, addname, e)
    if verbose:
        logger.info("Copying %s" % DB.filename)
        logger.info("     to %s" % copyname)
    shutil.copy2(DB.filename, copyname)
    return DismodDbAPI(copyname)

def copyDB_dest(DB, dest, verbose = False):
    if verbose:
        logger.info("Copying %s" % DB.filename)
        logger.info("     to %s" % dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy2(DB.filename, dest)
    return DismodDbAPI(dest)

def db2csv(fn, dirname = ''):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    files = []
    db2csv_command(fn)
    for name in ['option', 'log', 'data', 'variable', 'predict']:
        file = name+'.csv'
        if os.path.isfile(os.path.join(dirname, file)):
            new_file = os.path.join(dirname, subdir, file)
            os.rename(os.path.join(dirname, file), os.path.join(dirname, new_file))
            files.append(os.path.split(new_file)[-1])
    logger.info('Files produced by db2csv: %s.' % files)

def archive_DB(DB, dirname, db2csv = True):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    fn_out = os.path.join(dirname, os.path.basename(DB.filename))
    logger.info ("Saving the Dismod_AT database to %s." % fn_out)
    if os.path.isfile(fn_out): os.unlink(fn_out)
    shutil.copy2(DB.filename, fn_out)
    if db2csv:
        logger.info("Writing db2csv files.")
        db2csv_command(fn_out)

def db_info(DB):
    data = DB.data.merge(DB.integrand, how='left')
    node_ids = data.node_id.unique().tolist()
    if None in node_ids:
        node_ids.remove(None)
    logger.info('DB filename : %s' % DB.filename)
    logger.info('  id : %d, len(var) : %s, len(data) : %d' % (id(DB), (len(DB.var) if 'var' in DB.tables else None), len(data)))
    logger.info('  nodes : ' + str(sorted(node_ids)))
    logger.info('  data  : ' + ', '.join(['%s : %d' % (integrand, len(g)) for integrand, g in data.groupby('integrand_name')]))

def get_density_id(DB, density_name):
    return DB.density.loc[DB.density.density_name == density_name, 'density_id'].squeeze()

def get_integrand_id(DB, integrand_name):
    return DB.integrand.loc[DB.integrand.integrand_name == integrand_name, 'integrand_id'].squeeze()

def get_rate_name(DB, name):
    rates = tuple(DB.rate.rate_name.values)
    for rate in rates:
        if rate in name.split('_'):
            return rate

def node_id2name(DB, node_id):
    tmp = DB.node.loc[DB.node.node_id == node_id, 'node_name'].squeeze().split()
    if len(tmp) > 1:
        return ' '.join(tmp[1:])
    return tmp[0]

def node2location_id(DB, node): 
    return node_id2location_id(DB, node.node_id.values)

def node_id2location_id(DB, node_id):
    def get_loc(loc):
        try:
            if '(' in loc:
                return int(loc[loc.index('(')+1 : loc.index(')')])
            elif 'c_location_id' in DB.node.columns:
                return int(DB.node.loc[DB.node.node_name == loc, 'c_location_id'].squeeze())
            return loc
        except Exception as ex:
            print ('ERROR:', ex)
            return None

    if 'c_location_id' in DB.node.columns:
        location = int(DB.node.loc[DB.node.node_id == node_id, 'c_location_id'])
    else:
        if not hasattr(node_id, '__iter__'):
            node_id = [node_id]
        location = DB.node.loc[DB.node.node_id.isin(node_id), 'node_name']
        if location.empty:
            logger.error("Could not find a location_id for node_id %s" % node_id)
            return None
        location = [get_loc(l) for l in location.values]
        if len(location) == 1:
            return location[0]
    return location

node_id2loc_id = node_id2location_id

def get_node_id(DB, location_id):
    if location_id in DB.node.node_name.tolist():
        rtn = DB.node.loc[DB.node.node_name == location_id, 'node_id'].squeeze()
    else:
        mask = [('%d' % location_id == str(_)) | ('(%d)' % location_id in str(_)) | (int(location_id) == _) for _ in DB.node.get('c_location_id', DB.node.node_name)]
        rtn = DB.node.loc[mask, 'node_id'].squeeze()
    try:
        if rtn.empty:
            rtn = None
    except: pass
    return rtn

def get_location_id(node_name):
    return int(node_name[node_name.index('(')+1 : node_name.index(')')])

def get_rate_smooth_names(DB, smooth_names):
    logger.warn("So far as I can tell, there is no way to connect a smooth table entry to a rate except via a naming convention. Is this correct?")
    rates = tuple(DB.rate.rate_name.values)
    covariates = tuple(DB.covariate.covariate_name)
    rtn = []
    for name in smooth_names:
        if (('smooth' in name
             or 'constrain' in name
             or name in ('rho', 'iota', 'chi', 'omega'))
            and 'child' not in name
            and 'dA' not in name
            and 'dT' not in name
            and name and not any([_ in name for _ in covariates])):
            for rate in rates:
                if rate in name:
                    rtn.append(name)
    return rtn

def tempfile_DB(DB_or_name):
    tempfile_name = tempfile.NamedTemporaryFile(mode='w', prefix='dismod_tempfile_', suffix='.db').name
    if isinstance(DB_or_name, str):
        shutil.copy2(DB_or_name, tempfile_name)
    else:
        shutil.copy2(DB_or_name.filename, tempfile_name)
    return DismodDbAPI(tempfile_name)

def cleanup_prior(DB):
    del DB.constraint
    del DB.fit_var
    del DB.fit_data_subset
    del DB.predict
    del DB.sample
    del DB.data_sim
    del DB.prior_sim
    del DB.start_var
    del DB.truth_var
    del DB.avgint
    del DB.depend_var


def set_max_iters(DB, max_num_iter_fixed=50, max_num_iter_random=None):
    if max_num_iter_fixed <= 0:
        max_num_iter_random = 0
    option = DB.option
    option.loc[option.option_name == 'max_num_iter_fixed', 'option_value'] = max_num_iter_fixed
    if max_num_iter_random is not None:
        option.loc[option.option_name == 'max_num_iter_random', 'option_value'] = max_num_iter_random
    DB.option = option

# def set_parent_node_id(DB, node_id):
#     option = DB.option
#     option.loc[DB.option.option_name == 'parent_node_id', 'option_value'] = node_id
#     DB.option = option

def set_node_info(DB, node_id):
    '''
    Dismod_AT allows both parent_node_id and parent_node_name, in the option table, so set them both. This is useful elsewhere.
    '''
    option = DB.option
    option.loc[option.option_name == 'parent_node_id', 'option_value'] = node_id
    child_node_name = DB.node.loc[DB.node.node_id == node_id, 'node_name'].squeeze()
    if 'parent_node_name' in option.option_name.values:
        option.loc[option.option_name == 'parent_node_name', 'option_value'] = child_node_name
    else:
        option = option.append([dict(option_id = len(option), option_name = 'parent_node_name', option_value = child_node_name)])
    DB.option = option


def set_tolerance_fixed(DB, tol = 1e-5):
    option = DB.option
    option.loc[option.option_name == 'tolerance_fixed', 'option_value'] = tol
    DB.option = option

def set_cov_reference(DB, name, reference):
    covariate = DB.covariate
    covariate.loc[covariate.covariate_name == name, 'reference'] = reference
    DB.covariate = covariate

def system(cmd, DB):
    logger.info (cmd)
    os.system(cmd)
    if 'end' not in DB.log.iloc[-1].message:
        raise Exception("Command %s failed." % cmd)

    
