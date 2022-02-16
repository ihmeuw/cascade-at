#!/usr/bin/env python

import sys
import os

from functools import reduce
import numpy as np
import pandas as pd
from copy import copy
import sqlite3
import shutil
import multiprocessing

from typing import List, Optional, Dict, Union

# from cascade_at.dismod.api.dismod_sqlite import get_engine

from cascade_at.core.db import decomp_step as ds

from cascade_at.dismod.api.dismod_io import DismodIO
from pathlib import Path

from cascade_at.settings.settings import SettingsConfig
from cascade_at.core.log import get_loggers
from cascade_at.inputs.base_input import BaseInput
from cascade_at.inputs.asdr import ASDR
from cascade_at.inputs.csmr import CSMR
from cascade_at.dismod.constants import IntegrandEnum
from cascade_at.inputs.covariate_data import CovariateData
from cascade_at.inputs.covariate_specs import CovariateSpecs
from cascade_at.inputs.data import CrosswalkVersion
from cascade_at.inputs.demographics import Demographics
from cascade_at.inputs.locations import LocationDAG, locations_by_drill
from cascade_at.inputs.population import Population
from cascade_at.inputs.utilities.covariate_weighting import (
    get_interpolated_covariate_values
)
from cascade_at.inputs.utilities.gbd_ids import get_location_set_version_id
from cascade_at.inputs.utilities.transformations import COVARIATE_TRANSFORMS
from cascade_at.inputs.utilities.gbd_ids import SEX_ID_TO_NAME
from cascade_at.inputs.utilities.reduce_data_volume import decimate_years
from cascade_at.model.utilities.grid_helpers import expand_grid
from cascade_at.inputs.utilities.data import calculate_omega, format_age_time, midpoint_age_time
from cascade_at.inputs.utilities.gbd_ids import (
    CascadeConstants, StudyCovConstants
)
from cascade_at.settings.convert import (
    measures_to_exclude_from_settings, data_eta_from_settings,
    nu_from_settings, density_from_settings,
    midpoint_list_from_settings
)

import sqlalchemy as sql
import db_tools

LOG = get_loggers(__name__)


__quick_test__ = False
__to_do__ = True

class Inputs:
    def __init__(self, demographics, population, covariate_specs):
        self.demographics = demographics
        self.population = population
        self.covariate_specs = covariate_specs
    def transform_country_covariates(self, df):
        """
        Transforms the covariate data with the transformation ID.
        :param df: (pd.DataFrame)
        :return: self
        """
        for c in self.covariate_specs.covariate_specs:
            if c.study_country == 'country':
                LOG.info(f"Transforming the data for country covariate "
                         f"{c.covariate_id}.")
                df[c.name] = df[c.name].apply(
                    lambda x: COVARIATE_TRANSFORMS[c.transformation_id](x)
                )
        return df

class AllNodeDatabase:
    def dataframe_compression_index(self, name = '', df = None):
        index = df[(df.index == 0) | (df.node_id.diff() > 0) | df.sex_id.diff() > 0]
        index[f'all_{name}_id'] = index.index
        index.reset_index(inplace=True, drop=True)
        index[f'{name}_index_id'] = index.index
        index = index[[f'{name}_index_id', 'node_id', 'sex_id', f'all_{name}_id']]
        return index

    def get_asdr(self, demographics=None, gbd_round_id=None, decomp_step=None):
        print ('*** Get ASDR ...', end=' ')
        from cascade_at.inputs.asdr import ASDR
        asdr = ASDR(demographics = demographics,
                    gbd_round_id = gbd_round_id,
                    decomp_step = decomp_step)
        asdr.get_raw()
        asdr = asdr.configure_for_dismod()
        asdr['age'] = asdr[['age_lower', 'age_upper']].mean(axis=1)
        asdr['time'] = asdr[['time_lower', 'time_upper']].mean(axis=1)
        asdr = asdr.sort_values(by = ['location_id', 'age', 'time'])

        print (f"{len(asdr)} rows. ***")
        return asdr

    def get_csmr(self, demographics=None, gbd_round_id=None, decomp_step=None, cause_id=None):
        print ('*** Get CSMR ...', end=' ')
        from cascade_at.inputs.csmr import CSMR
        csmr = CSMR(cause_id = cause_id,
                    demographics = demographics,
                    gbd_round_id = gbd_round_id,
                    decomp_step = decomp_step)
        csmr.get_raw()
        csmr = csmr.configure_for_dismod()
        csmr['age'] = csmr[['age_lower', 'age_upper']].mean(axis=1)
        csmr['time'] = csmr[['time_lower', 'time_upper']].mean(axis=1)
        csmr = csmr.sort_values(by = ['location_id', 'age', 'time'])

        print (f"{len(csmr)} rows. ***")
        return csmr

    def update_root_node_time_age(self, time_df, omega_time_grid, age_df, omega_age_grid):
        add_time = sorted(set(omega_time_grid) - set(time_df.time))
        time = time_df.append(pd.DataFrame(add_time, columns=['time'])).reset_index(drop=True)
        mask = time.time_id.isna()
        time.loc[mask,'time_id'] = time[mask].index

        age = sorted(set(omega_age_grid) - set(age_df.age))
        age = age_df.append(pd.DataFrame(age, columns=['age'])).reset_index(drop=True)
        mask = age.age_id.isna()
        age.loc[mask,'age_id'] = age[mask].index
        return time, age
    
    def cov_weighted_average(self, inputs, node):
        cols = ['location_id', 'sex_id', 'year_id', 'age_group_id']

        cov_dict = { c.name: inputs.covariate_data[c.covariate_id] for c in inputs.covariate_specs.covariate_specs if c.study_country == 'country' }
        covs = [c.configure_for_dismod(pop_df = inputs.population.raw, loc_df = node)[cols + ['mean_value']].rename(columns = {'mean_value': n})
                for n,c in cov_dict.items()]
        cov_names = list(cov_dict.keys())
        outer = reduce(lambda x, y: pd.merge(x, y, how='outer', on = cols ), covs)
        covs = outer.merge(inputs.population.raw)
        for name in cov_dict.keys():
            covs[name] = covs[name] * covs.population
        grps = covs.groupby(['location_id', 'sex_id'], as_index=False)
        weighted_avg = grps[['population'] + cov_names].sum(min_count=1)
        for name in cov_names:
            weighted_avg[name] /= weighted_avg.population
        weighted_avg.drop(columns = ['population'], inplace=True)
        return weighted_avg

    def cov_unweighted(self, inputs, node):
        cols = ['location_id', 'sex_id', 'year_id', 'age_group_id']

        cov_dict = { c.name: inputs.covariate_data[c.covariate_id] for c in inputs.covariate_specs.covariate_specs if c.study_country == 'country' }
        covs = [c.configure_for_dismod(pop_df = inputs.population.raw, loc_df = node)[cols + ['mean_value']].rename(columns = {'mean_value': n})
                for n,c in cov_dict.items()]

        cov_names = list(cov_dict.keys())
        outer = reduce(lambda x, y: pd.merge(x, y, how='outer', on = cols ), covs)
        covs = outer.merge(inputs.population.raw)

        return covs

    def write_table_sql(self, conn, table_name, dtypes):
        df = getattr(self, table_name)
        id_column = f"{table_name}_id"
        if id_column not in df:
            df[id_column] = df.reset_index(drop=True).index
        keys = ', '.join([f'{k} {v}' for k,v in dtypes.items() if k != id_column])
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.execute(f"CREATE TABLE {table_name} ({id_column} integer primary key, {keys})")
        cols = [k for k in dtypes if k != id_column]
        df[cols].to_sql(table_name, conn, index_label = id_column, if_exists="append")

    # def correct_all_node_database(self):
    #     # --------------------------------------------------------------------------
    #     # Corrections to all_node_database
    #     #
    #     # all_cov_reference
    #     # Change sex_id -> split_reference_id and map its values
    #     # 2 -> 0 (female), 3 -> 1 (both), 1 -> 2 (male)

    #     print (f"*** Correcting {self.all_node_db} ***")
    #     self.conn = sqlite3.connect(self.all_node_db)
    
    #     command  = 'ALTER TABLE all_cov_reference '
    #     command += 'RENAME COLUMN sex_id TO split_reference_id'
    
    #     # dismod_at.sql_command(connect_all, command)
    #     all_cov_reference_table = \
    #         dismod_at.get_table_dict(connect_all, 'all_cov_reference')
    #     split_map = { 1:2, 2:0, 3:1}
    #     for row in all_cov_reference_table :
    #         split_reference_id = split_map[ row['split_reference_id'] ]
    #         row['split_reference_id'] = split_reference_id
    #     dismod_at.replace_table(
    #         connect_all, 'all_cov_reference', all_cov_reference_table
    #     )

    def correct_root_node_database(self):
        # ---------------------------------------------------------------------------
        # Corrections to root_node_database
        #
        # integrand_table
        # All the covariate multipliers must be in integrand table
        conn = sqlite3.connect(self.root_node_db.path)

        self.covariate = self.root_node_db.covariate
        self.covariate['covariate_name'] = [(n[2:] if n[:2] in ['c_', 's_'] else n) for n in self.covariate.c_covariate_name]
        self.write_table_sql(conn, 'covariate', {'covariate_id': 'integer', 'covariate_name': 'text',
                                                 'reference': 'real', 'max_difference': 'real', 'c_covariate_name': 'text'})

        integrand_table = self.root_node_db.integrand
        mulcov_table    = self.root_node_db.mulcov
        mulcov_table['integrand_name'] = [f"mulcov_{name}" for name in mulcov_table.mulcov_id]
        mulcov_table['minimum_meas_cv'] = 0
        mask = mulcov_table.integrand_name.isin(integrand_table.integrand_name)
        integrand_table = integrand_table.append(mulcov_table.loc[~mask, ['integrand_name', 'minimum_meas_cv']]).reset_index(drop=True)
        integrand_table['integrand_id'] = integrand_table.index
        self.root_node_db.integrand = integrand_table
        #
        # option table, parent_node_id
        # at_cascade requires one to use parent_node_name (not parent_node_id)
        # (turn on ipopt_trace)
        self.option = self.root_node_db.option
        self.node = self.root_node_db.node
        if 'parent_node_name' not in self.option.option_name.values:
            parent_node_id = int(self.option.loc[self.option.option_name == 'parent_node_id', 'option_value'])
            parent_node_name, parent_loc_id = self.node.loc[self.node.node_id == parent_node_id, ['node_name', 'c_location_id']].squeeze()
            parent_node_name = (parent_node_name if parent_node_name.startswith(str(parent_loc_id)) else f'{parent_loc_id}_{parent_node_name}')
            mask = self.option.option_name == 'parent_node_id'
            self.option.loc[mask, 'option_name'] = 'parent_node_name'
            self.option.loc[mask, 'option_value'] = parent_node_name
        brads_options = {'data_extra_columns'          :'c_seq',
                         'meas_noise_effect'           :'add_std_scale_none',
                         'quasi_fixed'                 :'false' ,
                         'tolerance_fixed'             :'1e-8',
                         'max_num_iter_fixed'          :'40',
                         'print_level_fixed'           :'5',
                         'accept_after_max_steps_fixed':'10'}

        for k,v in brads_options.items():
            if k in self.option.option_name.values:
                self.option.loc[self.option.option_name == k, 'option_value'] = v
            else:
                self.option = self.option.append({'option_name': k, 'option_value': v}, ignore_index=True)
        self.option = self.option.reset_index(drop=True)
        self.option.option_id = self.option.index
        

        self.write_table_sql(conn, 'option', {'option_id': 'integer', 'option_name': 'text', 'option_value': 'text'})
        self.node['node_name'] = [n.node_name if n.node_name.startswith(str(n.c_location_id))
                                  else f'{n.c_location_id}_{n.node_name}' for i,n in self.node.iterrows()]
        self.write_table_sql(conn, 'node', {'node_id': 'integer', 'node_name': 'text', 'parent': 'integer', 'c_location_id': 'integer'})
        
        #
        # rate table
        # all omega rates must be null
        self.rate    = self.root_node_db.rate
        omega_rate_id = self.rate.loc[self.rate.rate_name == 'omega', 'rate_id']
        self.rate.loc[omega_rate_id, ['parent_smooth_id', 'child_smooth_id', 'child_nslist_id']] = None, None, None
        self.root_node_db.rate = self.rate
        self.write_table_sql(conn, 'rate', {'rate_id': 'integer', 'rate_name': 'text',
                                            'parent_smooth_id': 'integer', 'child_smooth_id': 'integer', 'child_nslist_id': 'integer'})

        #
        # nslist and nslist_pair tables
        self.root_node_db.nslist = pd.DataFrame()
        self.root_node_db.nslist_pair = pd.DataFrame()
        self.nslist = self.root_node_db.nslist
        self.nslist_pair = self.root_node_db.nslist_pair
        self.write_table_sql(conn, 'nslist', {'nslist_id': 'integer', 'nslist_name': 'text'})
        self.write_table_sql(conn, 'nslist_pair', {'nslist_pair_id': 'integer', 'nslist_id': 'integer', 'node_id': 'integer', 'smooth_id': 'integer'})
        print (f'*** Modified {self.root_node_db.path}')
        #

    def save_to_sql(self):
        print (f"*** Writing {self.all_node_db} ***")
        conn = sqlite3.connect(self.all_node_db)

        self.write_table_sql(conn, 'all_option', {'all_option_id': 'integer', 'option_name': 'text', 'option_value': 'text'})
        # self.write_table_sql(conn, 'fit_goal', {'fit_goal_id': 'integer', 'node_id': 'integer'})
        # self.write_table_sql(conn, 'all_cov_reference', {'all_cov_reference_id': 'integer', 'node_id': 'integer', 'sex_id':'integer', 'covariate_id':'integer', 'reference': 'real'})
        self.write_table_sql(conn, 'all_cov_reference', {'all_cov_reference_id': 'integer', 'node_id': 'integer', 'split_reference_id':'integer', 'covariate_id':'integer', 'reference': 'real'})
        self.write_table_sql(conn, 'omega_age_grid', {'omega_age_grid_id': 'integer', 'age_id': 'integer'})
        self.write_table_sql(conn, 'omega_time_grid', {'omega_time_grid_id': 'integer', 'time_id': 'integer'})
        
        self.write_table_sql(conn, 'node_split', {'node_split_id': 'integer', 'node_id': 'integer'})

        # Brad insists on calling sex_id split_reference_id
        brads_name_for_sex_id = 'split_reference_id'

        self.mtall_index.rename(columns={'sex_id': brads_name_for_sex_id}, inplace=True)
        self.write_table_sql(conn, 'mtall_index', {'mtall_index_id': 'integer', 'node_id': 'integer', brads_name_for_sex_id: 'integer', 'all_mtall_id': 'integer'})
        self.write_table_sql(conn, 'all_mtall', {'all_mtall_id': 'integer', 'all_mtall_value': 'real'})

        self.mtspecific_index.rename(columns={'sex_id': brads_name_for_sex_id}, inplace=True)
        self.write_table_sql(conn, 'mtspecific_index', {'mtspecific_index_id': 'integer', 'node_id': 'integer', brads_name_for_sex_id: 'integer', 'all_mtspecific_id': 'integer'})
        self.write_table_sql(conn, 'all_mtspecific', {'all_mtspecific_id': 'integer', 'all_mtspecific_value': 'real'})

        self.write_table_sql(conn, 'mulcov_freeze', {'mulcov_freeze_id': 'integer', 'fit_node_id': 'integer',
                                               'split_reference_id': 'integer', 'mulcov_id': 'integer'})

        self.write_table_sql(conn, 'split_reference', {'split_reference_id': 'integer', 'split_reference_name': 'text', 'split_reference_value': 'real'})

    def __init__(self,

                 mvid = None,
                 conn_def = 'dismod-at-dev',
                 location_set_version_id = None,
                 # ERROR ValueError: No active age group set version ID found for age group set ID 12, release ID 10
                 # gbd_round_id = 7,
                 # decomp_step = 'iterative',
                 # age_group_set_id = 19,

                 gbd_round_id = 6,
                 decomp_step = 'iterative',
                 age_group_set_id = None,

                 root_node_path = '/Users/gma/ihme/epi/at_cascade_brad/data/{mvid}/dismod.db',
                 in_parallel = False,
                 max_fit = 250,
                 cause_id = None,
                 ):

        print (root_node_path)

        self.conn_def = conn_def

        self.mvid = mvid
        self.decomp_step = decomp_step

        self.gbd_round_id = gbd_round_id
        gbd_round = ds.gbd_round_from_gbd_round_id(gbd_round_id)

        self.in_parallel = in_parallel
        self.max_fit = max_fit
        
            
        print ('*** Get parameter json and load_settings. ***')
        from cascade_at.settings.settings import settings_json_from_model_version_id, load_settings
        parameter_json = settings_json_from_model_version_id(
            model_version_id = self.mvid,
            conn_def = self.conn_def)
        settings = load_settings(settings_json=parameter_json)
        global settings_dict
        settings_dict = settings._to_dict_value()

        if settings.location_set_version_id:
            self.location_set_version_id = settings.location_set_version_id
        else:
            self.location_set_version_id = get_location_set_version_id(gbd_round_id = self.gbd_round_id)

        print ('*** Get demographics. ***')
        self.demographics = demographics = Demographics(gbd_round_id=self.gbd_round_id)

        # Subsample demographic years
        demographics.year_id = list(range(min(demographics.year_id), max(demographics.year_id)+5, 5))

        print ('*** Get population. ***')
        population = Population( demographics = demographics,
                                 decomp_step = self.decomp_step,
                                 gbd_round_id=self.gbd_round_id).get_population()

        covariate_specs = CovariateSpecs(
            country_covariates=settings.country_covariate,
            study_covariates=settings.study_covariate)

        self.inputs = inputs = Inputs(demographics, population, covariate_specs)

        print ('*** Get locations. ***')
        inputs.location_dag = LocationDAG(location_set_version_id = self.location_set_version_id,
                                          gbd_round_id = self.gbd_round_id)
        if settings.model.drill:
            drill_location_start = settings.model.drill_location_start
            drill_location_end = settings.model.drill_location_end
        else:
            drill_location_start = None
            drill_location_end = None

        # Need to subset the locations to only those needed for
        # the drill. drill_locations_all is the set of locations
        # to pull data for, including all descendants. drill_locations
        # is the set of locations just parent-children in the drill.
        drill_locations_all, drill_locations = locations_by_drill(
            drill_location_start=drill_location_start,
            drill_location_end=drill_location_end,
            dag=inputs.location_dag)

        if drill_locations_all:
            demographics.location_id = drill_locations_all
            demographics.drill_locations = drill_locations

        self.sex_id = settings.model.drill_sex if settings.model.drill_sex else 3

        root_node_path = Path(root_node_path.format(mvid=self.mvid, location_id=drill_location_start, sex_id=self.sex_id))
        self.root_node_db = DismodIO(root_node_path)

        self.all_node_db = (Path(os.path.join(*root_node_path.parts[:2 + root_node_path.parts.index(str(self.mvid))]))
                            .parent / 'all_node.db')

        self.age = self.root_node_db.age
        self.time = self.root_node_db.time
        self.node = self.root_node_db.node

        print ('*** Get options. ***')
        [[root_node_loc, root_node_name]] = self.root_node_db.node.loc[
            self.root_node_db.node.c_location_id == drill_location_start, ['c_location_id', 'node_name']].values
        if not root_node_name.startswith(str(root_node_loc)):
            root_node_name = f'{root_node_loc}_{root_node_name}'
   
        if settings.model.split_sex:
            root_split_reference_name = 'Both'
            split_covariate_name = 'sex'
            split_list =  f'-1 s_{split_covariate_name} -0.5 0.0 +0.5'
            split_node_id = self.node.loc[self.node.c_location_id == int(settings.model.split_sex), 'node_id'].squeeze()
            self.node_split = pd.DataFrame([{'node_split_id': 0, 'node_id': split_node_id}])
        else:
            root_split_reference_name = ''
            split_covariate_name = ''
            split_list =  ''
            self.node_split = pd.DataFrame([], columns = ['node_split_id', 'node_id'])

        # The values values in this section have direct JSON file settings and/or are hardcoded 
        if 1:
            absolute_covariates = ['one']
            max_number_cpu = max(1, multiprocessing.cpu_count() - 1)
            max_abs_effect = 2

            result_dir = root_node_path.parent
            shift_prior_std_factor = 2
            perturb_optimization_scaling = 0.2
            
            print ('TO DO -- set up the mulcov_freeze table values')

            self.mulcov_freeze = pd.DataFrame([], columns = ['mulcov_freeze_id', 'fit_node_id', 'split_reference_id', 'mulcov_id'])

            self.split_reference = pd.DataFrame([{'split_reference_id': 0, 'split_reference_name': 'Female', 'split_reference_value': -0.5},
                                                 {'split_reference_id': 1, 'split_reference_name': 'Both', 'split_reference_value': 0.0},
                                                 {'split_reference_id': 0, 'split_reference_name': 'Male', 'split_reference_value': 0.5}])


        self.all_option = pd.DataFrame([('absolute_covariates', ' '.join(absolute_covariates)),
                                        ('split_covariate_name', split_covariate_name),
                                        ('root_split_reference_name', root_split_reference_name),
                                        ('result_dir', str(result_dir)),
                                        ('root_node_name', root_node_name),
                                        ('max_abs_effect', str(max_abs_effect)),
                                        ('max_fit', str(self.max_fit)),
                                        ('max_number_cpu', str(max_number_cpu)),
                                        # ('split_list', split_list),
                                        ('shift_prior_std_factor', str(shift_prior_std_factor)),
                                        ('perturb_optimization_scaling', str(perturb_optimization_scaling)),
                                        ],
                                       columns = ['option_name', 'option_value'])

        self.all_option['option_id'] = self.all_option.index

        print ("*** Drill information. ***")
        print (f"    Drill locations: {demographics.drill_locations}")
        print (f"    All locations: {demographics.location_id}")

        self.fit_goal = pd.DataFrame(demographics.location_id, columns = ['c_location_id']).merge(self.root_node_db.node, how='left')
        self.fit_goal['fit_goal_id'] = self.fit_goal.index

        self.age_group_id = demographics.age_group_id
        self.sex_id = demographics.sex_id
        self.drill_locations = demographics.drill_locations


        self.country_covariate_ids = list(covariate_specs.country_covariate_ids)

        from cascade_at.dismod.api.fill_extract_helpers import reference_tables

        self.age = reference_tables.construct_age_time_table(
            variable_name='age', variable=demographics.age_group_id,
            data_min=0, data_max=100)

        self.time = reference_tables.construct_age_time_table(
            variable_name='time', variable=demographics.year_id,
            data_min=0, data_max=2020)

        print ('*** Get covariate_reference. ***')
        inputs.covariate_data = {c: CovariateData(
            covariate_id=c,
            demographics=self.demographics,
            decomp_step=self.decomp_step,
            gbd_round_id=self.gbd_round_id
        ).get_raw() for c in self.country_covariate_ids}

        from cascade_at.dismod.api.fill_extract_helpers.reference_tables import construct_node_table
        node = construct_node_table(inputs.location_dag).rename(columns = {'c_location_id': 'location_id'})
        level = [inputs.location_dag.depth(node) for node in node.location_id]
        node['level'] = level
        node['parent_id'] = node.parent

        cov_df = self.cov_unweighted(inputs, node)
        self.covariates_raw = cov_df.merge(self.node, left_on = 'location_id', right_on = 'c_location_id').copy()

        cov_df = inputs.transform_country_covariates(cov_df)
        self.covariates_transformed = cov_df.merge(self.node, how='left', left_on = 'location_id', right_on = 'c_location_id').copy()

        cov_location = self.covariates_raw.merge(self.node, how='right',  left_on = 'location_id', right_on = 'c_location_id')

        cov_names = [c.name for c in inputs.covariate_specs.covariate_specs if c.study_country == 'country']
        cols = list(set(cov_df.columns) - set(cov_names))
        
        covariate = pd.DataFrame()
        for name in cov_names:
            covariate_id = int(self.root_node_db.covariate.loc[self.root_node_db.covariate.c_covariate_name == name, 'covariate_id'])
            c = cov_df[cols + [name]].rename(columns={name: 'reference'})
            c['covariate_id'] = covariate_id
            covariate = covariate.append(c)

        covariate = self.root_node_db.node.merge(covariate, how='left', left_on = 'c_location_id', right_on='location_id')
        covariate['node_id'] = covariate['node_id'].astype(int)
        covariate['all_cov_reference_id'] = covariate.reset_index(drop=True).index
        split_map = { 1:2, 2:0, 3:1}
        covariate['split_reference_id'] = [split_map[x] for x in covariate.sex_id]
        self.all_cov_reference = covariate

        asdr = self.get_asdr(demographics=demographics, gbd_round_id=self.gbd_round_id, decomp_step=self.decomp_step)
        csmr = self.get_csmr(demographics=demographics, gbd_round_id=self.gbd_round_id, decomp_step=self.decomp_step, cause_id = cause_id)


        # Clear the nulls
        asdr = asdr[~asdr.meas_value.isnull()].reset_index(drop=True)
        csmr = csmr[~csmr.meas_value.isnull()].reset_index(drop=True)

        if __debug__:
            missing_asdr = set(demographics.location_id) - set(asdr.location_id)
            if missing_asdr: print(f"Warning -- ASDR data is missing for locations: {sorted(missing_asdr)}")
            missing_csmr = set(demographics.location_id) - set(csmr.location_id)
            if missing_csmr: print(f"Warning -- CSMR data is missing for locations: {sorted(missing_csmr)}")

        print ("*** Omega age and time grids. ***")
        self.omega_age = sorted(set(asdr.age.unique()) & set(csmr.age.unique()))
        self.omega_age_grid = pd.DataFrame(self.omega_age, columns = ['age'])
        self.omega_age_grid['omega_age_grid_id'] = self.omega_age_grid.index

        self.omega_time = sorted(set(asdr.time.unique()) & set(csmr.time.unique()))
        self.omega_time_grid = pd.DataFrame(self.omega_time, columns = ['time'])
        self.omega_time_grid['omega_time_grid_id'] = self.omega_time_grid.index

        print (f"*** Updating {self.root_node_db.path} time and age tables with omega values. ***")
        self.root_node_db.time, self.root_node_db.age = self.update_root_node_time_age(self.root_node_db.time, self.omega_time, self.root_node_db.age, self.omega_age)

        self.omega_age_grid = self.omega_age_grid.merge(self.root_node_db.age, how='left')
        print (f"    Age_ids: {self.omega_age_grid.age_id.tolist()}")
        self.omega_time_grid = self.omega_time_grid.merge(self.root_node_db.time, how='left')
        print (f"    Time_ids: {self.omega_time_grid.time_id.tolist()}")

        self.asdr = (asdr
                     .merge(self.root_node_db.node, how='left', left_on = 'location_id', right_on = 'c_location_id')
                     .merge(self.root_node_db.time, how='left')
                     .merge(self.root_node_db.age, how='left')
                     .sort_values(by=['node_id', 'sex_id']))

        self.csmr = (csmr
                     .merge(self.root_node_db.node, how='left', left_on = 'location_id', right_on = 'c_location_id')
                     .merge(self.root_node_db.time, how='left')
                     .merge(self.root_node_db.age, how='left')
                     .sort_values(by=['node_id', 'sex_id']))

        all_mtall = (self.asdr[['node_id', 'time_id', 'age_id', 'sex_id', 'meas_value']]
                     .rename(columns={'meas_value': 'all_mtall_value'}).reset_index(drop=True))
        all_mtall['all_mtall_id'] = all_mtall.index
        self.all_mtall = all_mtall
        self.mtall_index = self.dataframe_compression_index('mtall', self.all_mtall)

        all_mtspecific = (self.csmr[['node_id', 'time_id', 'age_id', 'sex_id', 'meas_value']]
                          .rename(columns={'meas_value': 'all_mtspecific_value'}).reset_index(drop=True))
        all_mtspecific['all_mtspecific_id'] = all_mtspecific.index
        self.all_mtspecific = all_mtspecific
        self.mtspecific_index = self.dataframe_compression_index('mtspecific', self.all_mtspecific)


        print ('*** Get age metadata. ***')
        import cascade_at.core.db
        self.age_groups = cascade_at.core.db.db_queries.get_age_metadata(age_group_set_id=age_group_set_id, gbd_round_id=self.gbd_round_id)

def main(root_node_path = '', mvid = None, cause_id = None, age_group_set_id = None):

    global self
    self = AllNodeDatabase(root_node_path = root_node_path, mvid = mvid, cause_id = cause_id, age_group_set_id = age_group_set_id)

    self.correct_root_node_database()

    self.save_to_sql()

if __name__ == '__main__':

    if 0:
        def parse_args(mvid=None, cause_id=None, age_group_set_id = None):
            import argparse
            from distutils.util import strtobool as str2bool
            parser = argparse.ArgumentParser()
            name_string = "-filename" if sys.argv[0] == '' else "filename"
            parser.add_argument("-v", "--model_version_id", type = int, default = mvid,
                                help = f"Model Version ID -- default = {mvid}")
            parser.add_argument("-c", "--cause_id", type = int, default = cause_id,
                                help = f"Cause ID -- default = {cause_id}")
            parser.add_argument("-a", "--age_group_set_id", type = int, default = age_group_set_id,
                                help = "Age Group Set ID -- default {age_group_set_id}")
            args = parser.parse_args()
            return args

        defaults = {}
        if (len(sys.argv) == 1 and sys.argv[0] == ''):
            _mvid_ = 475877
            _mvid_ = 475876
            _mvid_ = 475879
            _mvid_ = 475873
            _cause_id_ = 975        # diabetes mellitus type 1
            _cause_id_ = 587        # diabetes mellitus
            _age_group_set_id_ = 12
            defaults = dict(mvid = _mvid_, cause_id = _cause_id_, age_group_set_id = _age_group_set_id_)
        args = parse_args(**defaults)
        print (1111111111111, args)
        main(mvid = args.model_version_id, cause_id = args.cause_id, age_group_set_id = args.age_group_set_id)

    else:
        def parse_args(root_node_path = '', age_group_set_id = 4, mvid = 3, cause_id = 2):
            import argparse
            from distutils.util import strtobool as str2bool

            parser = argparse.ArgumentParser()
            parser.add_argument("-r", "--root-node-path", type = str, default = root_node_path,
                                help = f"Age Group Set ID -- default ''")
            parser.add_argument("-m", "--model-version-id", type = int, default = mvid,
                                help = f"Model Version ID -- default = {mvid}")
            parser.add_argument("-c", "--cause-id", type = int, default = cause_id,
                                help = f"Cause ID -- default = {cause_id}")
            parser.add_argument("-a", "--age-group-set-id", type = int, default = age_group_set_id,
                                help = "Age Group Set ID -- default {age_group_set_id}")
            args = parser.parse_args()
            return args

        if (len(sys.argv) == 1 and sys.argv[0] == ''):
            _mvid_ = 475877
            _mvid_ = 475876
            _mvid_ = 475879
            _mvid_ = 475873
            _cause_id_ = 975        # diabetes mellitus type 1
            _cause_id_ = 587        # diabetes mellitus
            _age_group_set_id_ = 12
            root_node_path = f'/Users/gma/ihme/epi/at_cascade_brad/data/{_mvid_}/dismod.db'
            defaults = dict(root_node_path = root_node_path, mvid = _mvid_, cause_id = _cause_id_, age_group_set_id = _age_group_set_id_)
            args = parse_args(**defaults)
        else:
            args = parse_args()

        _mvid_ = 475873

        sys.argv = f'all_node_database.py -m {_mvid_} --cause-id 587 --age-group-set 12 --root-node-path /Users/gma/ihme/epi/at_cascade_brad/data/cascade_dir/data/{_mvid_}/root_node.db'.split()
        args = parse_args()
        main(root_node_path = args.root_node_path, mvid = args.model_version_id, cause_id = args.cause_id, age_group_set_id = args.age_group_set_id)

        """
        if not __debug__ and _mvid_ == _mvid_:
            _root_node_db_ = f'/Users/gma/ihme/epi/at_cascade_brad/data/cascade_dir/data/{_mvid_}/root_node.db'
            copy_files = [f'/Users/gma/ihme/epi/at_cascade_brad/data/cascade_dir/data/{_mvid_}/dbs/100/3/dismod_ODE_import.db',
                          _root_node_db_]
            print (f'FOR TESTING -- Copying {copy_files[0]} to {copy_files[0]}')
            shutil.copy(*copy_files)


        """
