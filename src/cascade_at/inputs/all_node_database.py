#!/usr/bin/env python

import sys
import os

from functools import reduce
import numpy as np
import pandas as pd
from copy import copy
import sqlite3
import shutil

from typing import List, Optional, Dict, Union

from cascade_at.dismod.api.dismod_sqlite import get_engine

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
        index = df[(df.index == 0) | (df.node_id.diff() > 0)]
        index[f'all_{name}_id'] = index.index
        index.reset_index(inplace=True, drop=True)
        index[f'{name}_index_id'] = index.index
        index = index[[f'{name}_index_id', 'node_id', f'all_{name}_id']]
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

    def write_table_sql(self, table_name, dtypes):
        df = getattr(self, table_name)
        id_column = f"{table_name}_id"
        if id_column not in df:
            df[id_column] = df.reset_index(drop=True).index
        keys = ', '.join([f'{k} {v}' for k,v in dtypes.items() if k != id_column])
        self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        self.conn.execute(f"CREATE TABLE {table_name} ({id_column} integer primary key, {keys})")
        cols = [k for k in dtypes if k != id_column]
        df[cols].to_sql(table_name, self.conn, index_label = id_column, if_exists="append")

    def save_to_sql(self):
        print (f"*** Writing {self.all_node_db} ***")
        self.conn = sqlite3.connect(self.all_node_db)

        self.all_cov_reference = self.covariate
 
        self.write_table_sql('all_option', {'all_option_id': 'integer', 'option_name': 'text', 'option_value': 'text'})
        self.write_table_sql('fit_goal', {'fit_goal_id': 'integer', 'node_id': 'integer'})
        self.write_table_sql('all_cov_reference', {'all_cov_reference_id': 'integer', 'node_id': 'integer', 'covariate_id':'integer', 'reference': 'real'})
        self.write_table_sql('omega_age_grid', {'omega_age_grid_id': 'integer', 'age_id': 'integer'})
        self.write_table_sql('omega_time_grid', {'omega_time_grid_id': 'integer', 'time_id': 'integer'})
        
        self.write_table_sql('mtall_index', {'mtall_index_id': 'integer', 'node_id': 'integer', 'all_mtall_id': 'integer'})
        self.write_table_sql('all_mtall', {'all_mtall_id': 'integer', 'all_mtall_value': 'real'})
        self.write_table_sql('mtspecific_index', {'mtspecific_index_id': 'integer', 'node_id': 'integer', 'all_mtspecific_id': 'integer'})
        self.write_table_sql('all_mtspecific', {'all_mtspecific_id': 'integer', 'all_mtspecific_value': 'integer'})


    def __init__(self,

                 mvid = None,
                 conn_def = 'dismod-at-dev',
                 location_set_version_id = None,
                 gbd_round_id = 6,
                 decomp_step = 'step4',
                 root_node_path = '/Users/gma/ihme/epi/at_cascade/data/{mvid}/dbs/{location_id}/{sex_id}/dismod.db',
                 in_parallel = False,
                 age_group_set_id = None,
                 cause_id = None,
                 ):

        self.in_parallel = in_parallel
        self.mvid = mvid
        
        self.decomp_step = decomp_step

        self.gbd_round_id = gbd_round_id
        gbd_round = ds.gbd_round_from_gbd_round_id(gbd_round_id)
        self.conn_def = conn_def
            
        print ('*** Get parameter json and load_settings. ***')
        from cascade_at.settings.settings import settings_json_from_model_version_id, load_settings
        parameter_json = settings_json_from_model_version_id(
            model_version_id = self.mvid,
            conn_def = self.conn_def)
        settings = load_settings(settings_json=parameter_json)

        if settings.location_set_version_id:
            self.location_set_version_id = settings.location_set_version_id
        else:
            self.location_set_version_id = get_location_set_version_id(gbd_round_id = self.gbd_round_id)

        self.parent_location_id = getattr(settings.model, 'drill_location_start', 0)
        self.sex_id = getattr(settings.model, 'drill_sex', 3)

        root_node_path = Path(root_node_path.format(mvid=self.mvid, location_id=self.parent_location_id, sex_id=self.sex_id))
        self.root_node_db = DismodIO(root_node_path)

        all_node_path = root_node_path.parts[:2 + root_node_path.parts.index(str(self.mvid))]
        self.all_node_db = Path(os.path.join(*all_node_path)) / 'all_node.db'

        self.age = self.root_node_db.age
        self.time = self.root_node_db.time
        self.node = self.root_node_db.node

        print ('*** Get options. ***')
        root_node_name = self.root_node_db.node.loc[self.root_node_db.node.c_location_id == self.parent_location_id, 'node_name'].squeeze()
        self.all_option = pd.DataFrame([('sex_level', settings.model.split_sex),
                                    ('in_parallel', self.in_parallel),
                                    ('root_node_name', root_node_name)],
                                   columns = ['option_name', 'option_value'])
        self.all_option['option_id'] = self.all_option.index
                       
        print ('*** Get demographics. ***')
        demographics = Demographics(gbd_round_id=self.gbd_round_id)

        # Subsample demographic years
        demographics.year_id = list(range(min(demographics.year_id), max(demographics.year_id)+5, 5))

        print ('*** Get population. ***')
        population = Population( demographics = demographics, # 
                                 decomp_step = self.decomp_step,
                                 gbd_round_id=self.gbd_round_id).get_population()
            

        covariate_specs = CovariateSpecs(
            country_covariates=settings.country_covariate,
            study_covariates=settings.study_covariate
        )

        inputs = Inputs(demographics, population, covariate_specs)

        print ('*** Get locations. ***')
        inputs.location_dag = LocationDAG(location_set_version_id = self.location_set_version_id,
                                   gbd_round_id = self.gbd_round_id)
        if settings.model.drill:
            self.drill_location_start = settings.model.drill_location_start
            self.drill_location_end = settings.model.drill_location_end
        else:
            self.drill_location_start = None
            self.drill_location_end = None

        # Need to subset the locations to only those needed for
        # the drill. drill_locations_all is the set of locations
        # to pull data for, including all descendants. drill_locations
        # is the set of locations just parent-children in the drill.
        drill_locations_all, drill_locations = locations_by_drill(
            drill_location_start=self.drill_location_start,
            drill_location_end=self.drill_location_end,
            dag=inputs.location_dag)

        if drill_locations_all:
            demographics.location_id = drill_locations_all
            demographics.drill_locations = drill_locations
        self.demographics = demographics

        print ("*** Drill information. ***")
        print (f"    Drill locations: {demographics.drill_locations}")
        print (f"    All locations: {demographics.location_id}")

        self.fit_goal = pd.DataFrame(demographics.location_id, columns = ['c_location_id']).merge(self.root_node_db.node, how='left')
        self.fit_goal['fit_goal_id'] = self.fit_goal.index

        asdr = self.get_asdr(demographics=demographics, gbd_round_id=self.gbd_round_id, decomp_step=self.decomp_step)
        csmr = self.get_csmr(demographics=demographics, gbd_round_id=self.gbd_round_id, decomp_step=self.decomp_step, cause_id = cause_id)

        if __debug__:
            missing_asdr = set(demographics.location_id) - set(asdr.location_id)
            assert not missing_asdr, f"ASDR data is missing for locations: {missing_asdr}"
            missing_csmr = set(demographics.location_id) - set(csmr.location_id)
            assert not missing_csmr, f"CSMR data is missing for locations: {missing_csmr}"

        print ("*** Omega age and time grids. ***")
        omega_age_grid = sorted(set(asdr.age.unique()) & set(csmr.age.unique()))
        self.omega_age_grid = pd.DataFrame(omega_age_grid, columns = ['age'])
        self.omega_age_grid['omega_age_grid_id'] = self.omega_age_grid.index
        
        omega_time_grid = sorted(set(asdr.time.unique()) & set(csmr.time.unique()))
        self.omega_time_grid = pd.DataFrame(omega_time_grid, columns = ['time'])
        self.omega_time_grid['omega_time_grid_id'] = self.omega_time_grid.index

        self.root_node_db.time, self.root_node_db.age = self.update_root_node_time_age(self.root_node_db.time, omega_time_grid, self.root_node_db.age, omega_age_grid)

        self.omega_age_grid = self.omega_age_grid.merge(self.root_node_db.age, how='left')
        print (f"    Age_ids: {self.omega_age_grid.age_id.tolist()}")
        self.omega_time_grid = self.omega_time_grid.merge(self.root_node_db.time, how='left')
        print (f"    Time_ids: {self.omega_time_grid.time_id.tolist()}")

        self.asdr = (asdr
                     .merge(self.root_node_db.node, how='left', left_on = 'location_id', right_on = 'c_location_id')
                     .merge(self.root_node_db.time, how='left')
                     .merge(self.root_node_db.age, how='left'))

        self.csmr = (csmr
                     .merge(self.root_node_db.node, how='left', left_on = 'location_id', right_on = 'c_location_id')
                     .merge(self.root_node_db.time, how='left')
                     .merge(self.root_node_db.age, how='left'))

        all_mtall = self.asdr[['node_id', 'time_id', 'age_id', 'sex_id', 'meas_value']].rename(columns={'meas_value': 'all_mtall_value'}).reset_index(drop=True)
        all_mtall['all_mtall_id'] = all_mtall.index
        self.all_mtall = all_mtall
        self.mtall_index = self.dataframe_compression_index('mtall', self.all_mtall)

        all_mtspecific = self.csmr[['node_id', 'time_id', 'age_id', 'sex_id', 'meas_value']].rename(columns={'meas_value': 'all_mtspecific_value'}).reset_index(drop=True)
        all_mtspecific['all_mtspecific_id'] = all_mtspecific.index
        self.all_mtspecific = all_mtspecific
        self.mtspecific_index = self.dataframe_compression_index('mtspecific', self.all_mtspecific)

        if 0:
            print ('*** Check the new Dismod-AT development database host. ***')
            HOST = 'epiat-unmanaged-db-p01.db.ihme.washington.edu'
            USER = 'dbview'
            PASSWORD = 'E3QNSLvQTRJm'
            s = (f'mysql+pymysql://{USER}:{PASSWORD}@{HOST}:3306'
                 '/information_schema?use_unicode=1&charset=utf8mb4')
            try:
                engine = sql.create_engine(s)
                conn = engine.connect()
                print (conn.engine)
                conn.close()
            except:
                print ("ERROR -- make sure you are connected to the VPN")
                exit()


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

        cov_df = self.cov_weighted_average(inputs, node)
        cov_df = inputs.transform_country_covariates(cov_df)
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
        self.covariate = covariate

        print ('*** Get age metadata. ***')
        import cascade_at.core.db
        self.age_groups = cascade_at.core.db.db_queries.get_age_metadata(age_group_set_id=age_group_set_id, gbd_round_id=self.gbd_round_id)


def main(mvid = None, cause_id = None, age_group_set_id = None):

    global self
    self = AllNodeDatabase(mvid = mvid, cause_id = cause_id, age_group_set_id = age_group_set_id)
    self.save_to_sql()

if __name__ == '__main__':

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
        _mvid_ = 475876
        _cause_id_ = 975        # diabetes mellitus type 1
        _cause_id_ = 587        # diabetes mellitus
        _age_group_set_id_ = 12
        defaults = dict(mvid = _mvid_, cause_id = _cause_id_, age_group_set_id = _age_group_set_id_)
    args = parse_args(**defaults)
    main(mvid = args.model_version_id, cause_id = args.cause_id, age_group_set_id = args.age_group_set_id)
