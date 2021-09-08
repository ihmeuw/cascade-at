if 1:
    if 1:
        _gbd_round_id_ = 6
        _decomp_step_ = 'step4'
        _conn_def_ = 'dismod-at-dev'
        _age_group_set_id_ = 12
        _year_id_ = [2000]
        _cause_id_ = 975
        mvid = 475871


import sys

from functools import reduce
import numpy as np
import pandas as pd
from copy import copy
import sqlite3
import shutil

from typing import List, Optional, Dict, Union

from cascade_at.dismod.api.dismod_sqlite import get_engine

from cascade_at.core.db import decomp_step as ds

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

LOG = get_loggers(__name__)


import sqlalchemy as sql
import db_tools
# from db_tools import ezfuncs

# from cascade_at.executor.dismod_ihme_input import CovariateReference
class CovariateReference:
    def __init__(self, inputs):
        self.inputs = inputs
        self.cov_ids = {c.covariate_id: c.name
                        for c in inputs.covariate_specs.covariate_specs
                        if c.study_country == 'country'}
        self.loc_df = inputs.location_dag.df
        self.pop_df = inputs.population.configure_for_dismod()
    def configure_for_dismod(self, covariate_data):
        from cascade_at.inputs.utilities.covariate_weighting import CovariateInterpolator
        def handle_exception(**kwds):
            try: return cov.interpolate(**kwds)
            except: return None

        cols = ['location_id', 'year_id', 'age_group_id', 'sex_id']
        cov_df = [v.configure_for_dismod(self.pop_df, self.loc_df)[cols + ['mean_value']].rename(columns={'mean_value': k}) for k,v in covariate_data.items()]
        cov_df = reduce(lambda x, y: pd.merge(x, y, how='inner', on = cols ), cov_df)

        self.inputs.transform_country_covariates(cov_df)
        return cov_df


# class CovariateReference:
#     def __init__(self, inputs):
#         self.inputs = inputs
#         self.cov_ids = {c.covariate_id: c.name
#                         for c in inputs.covariate_specs.covariate_specs
#                         if c.study_country == 'country'}
#         self.loc_df = inputs.location_dag.df
#         self.pop_df = inputs.population.configure_for_dismod()
#     def configure_for_dismod(self, covariate_data):
#         from cascade_at.inputs.utilities.covariate_weighting import CovariateInterpolator
#         def handle_exception(**kwds):
#             try: return cov.interpolate(**kwds)
#             except: return None
#         cov_df = covariate_data.configure_for_dismod(self.pop_df, self.loc_df)
#         cov = CovariateInterpolator(cov_df, self.pop_df)
#         cov_id = covariate_data.covariate_id
#         cov_name = self.cov_ids[cov_id]
#         cov_df = pd.DataFrame([{'location_id': loc_id, 'sex_id': sex_id,
#                                 cov_name: handle_exception(loc_id = loc_id,sex_id = sex_id, age_lower=0, age_upper=100, time_lower = 1970, time_upper = 2020)}
#                               for loc_id in sorted(cov_df.location_id.unique())
#                               for sex_id in (1,2,3)])
#         self.inputs.transform_country_covariates(cov_df)
#         return cov_df


# class foo:
#     def __init__(self): pass

#     def interpolate_country_covariate_values(self, df: pd.DataFrame, cov_dict: Dict[Union[float, str], pd.DataFrame]):
#         """
#         Interpolates the covariate values onto the data
#         so that the non-standard ages and years match up to meaningful
#         covariate values.
#         """
#         LOG.info(f"Interpolating and merging the country covariates.")
#         interp_df = get_interpolated_covariate_values(
#             data_df=df,
#             covariate_dict=cov_dict,
#             population_df=self.population.configure_for_dismod()
#         )
#         return interp_df

#     def add_covariates_to_data(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Add on covariates to a data frame that has age_group_id, year_id
#         or time-age upper / lower, and location_id and sex_id. Adds both
#         country-level and study-level covariates.
#         """
#         cov_dict_for_interpolation = {
#             c.name: self.country_covariate_data[c.covariate_id]
#             for c in self.covariate_specs.covariate_specs
#             if c.study_country == 'country'
#         }

#         df = self.interpolate_country_covariate_values(
#             df=df, cov_dict=cov_dict_for_interpolation)
#         df = self.transform_country_covariates(df=df)

#         df['s_sex'] = df.sex_id.map(
#             SEX_ID_TO_NAME).map(StudyCovConstants.SEX_COV_VALUE_MAP)
#         df['s_one'] = StudyCovConstants.ONE_COV_VALUE

#         return df

#     def transform_country_covariates(self, df):
#         """
#         Transforms the covariate data with the transformation ID.
#         :param df: (pd.DataFrame)
#         :return: self
#         """
#         for c in self.covariate_specs.covariate_specs:
#             if c.study_country == 'country':
#                 LOG.info(f"Transforming the data for country covariate "
#                          f"{c.covariate_id}.")
#                 df[c.name] = df[c.name].apply(
#                     lambda x: COVARIATE_TRANSFORMS[c.transformation_id](x)
#                 )
#         return df





def cov_weighted_average(inputs, node):
    cov_dict_for_interpolation = { c.name: inputs.covariate_data[c.covariate_id] for c in inputs.covariate_specs.covariate_specs if c.study_country == 'country' }

    cols = ['location_id', 'sex_id', 'year_id', 'age_group_id']
    covs = [c.configure_for_dismod(pop_df = inputs.population.raw, loc_df = node)[cols + ['mean_value']].rename(columns = {'mean_value': n})
            for n,c in cov_dict_for_interpolation.items()]
    cov_names = list(cov_dict_for_interpolation.keys())
    outer = reduce(lambda x, y: pd.merge(x, y, how='outer', on = cols ), covs)
    covs = outer.merge(inputs.population.raw)
    for name in cov_names:
        covs[name] = covs[name] * covs.population
    grps = covs.groupby(['location_id', 'sex_id'], as_index=False)
    weighted_avg = grps[['population'] + cov_names].sum(min_count=1)
    for name in cov_names:
        weighted_avg[name] /= weighted_avg.population
    weighted_avg.drop(columns = ['population'], inplace=True)
    return weighted_avg



class AllNodeDatabase:

    def get_asdr(self, demographics=None, gbd_round_id=None, decomp_step=None):
        print ('*** Get ASDR. ***')
        from cascade_at.inputs.asdr import ASDR
        asdr = ASDR(demographics = demographics,
                    gbd_round_id = gbd_round_id,
                    decomp_step = decomp_step)
        asdr.get_raw()
        asdr = asdr.configure_for_dismod()
        asdr['age'] = asdr[['age_lower', 'age_upper']].mean(axis=1)
        asdr['time'] = asdr[['time_lower', 'time_upper']].mean(axis=1)
        asdr = asdr.sort_values(by = ['location_id', 'age', 'time'])

        print (f"Got {len(asdr)} rows.")
        return asdr

    def get_csmr(self, demographics=None, gbd_round_id=None, decomp_step=None, cause_id=None):
        print ('*** Get CSMR. ***')
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

        print (f"Got {len(csmr)} rows.")
        return csmr

    def get_root_node_db(self, filename = None):
        from cascade_at.dismod.api.dismod_io import DismodIO
        from pathlib import Path
        root_db_database = DismodIO(Path(filename))
        return root_db_database

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

    def __init__(self, location_set_version_id = None):
        location_set_version_id = None
        _gbd_round_id_ = 6
        self.gbd_round_id = _gbd_round_id_
        gbd_round = ds.gbd_round_from_gbd_round_id(_gbd_round_id_)
        _decomp_step_ = 'step4'
        _conn_def_ = 'dismod-at-dev'
        _age_group_set_id_ = 12
        _year_id_ = [2000]
        _cause_id_ = 975
        mvid = 475871
        
        global obj
        obj=self

        root_node_db = self.get_root_node_db('~/ihme/epi/at_cascade/data/475871/dbs/100/2/dismod.db')
        self.db = root_node_db
        self.age = root_node_db.age
        self.time = root_node_db.time
        self.node = root_node_db.node

        print ('*** Get demographics. ***')
        demographics = Demographics(gbd_round_id=_gbd_round_id_)
        # demographics.location_id = [1]
        # demographics.drill_locations = [1]
        if 1:
            demographics.location_id = [1, 4, 31, 64, 103, 137, 158, 166]
        else:
            demographics.location_id = sorted(self.node.c_location_id.values)


        demographics.year_id = list(range(min(demographics.year_id), max(demographics.year_id)+2, 5))

        print ('*** Get population. ***')
        population = Population( demographics=demographics,
                                 decomp_step=_decomp_step_,
                                 gbd_round_id=_gbd_round_id_).get_population()
            
        if 0:
            asdr = self.get_asdr(demographics=demographics, gbd_round_id=_gbd_round_id_, decomp_step=_decomp_step_)
            csmr = self.get_csmr(demographics=demographics, gbd_round_id=_gbd_round_id_, decomp_step=_decomp_step_, cause_id = _cause_id_)

            omega_age_grid = sorted(set(asdr.age.unique()) & set(csmr.age.unique()))
            self.omega_age_grid = pd.DataFrame(omega_age_grid, columns = ['age_id'])
            self.omega_age_grid['omega_age_grid_index'] = self.omega_age_grid.index

            omega_time_grid = sorted(set(asdr.time.unique()) & set(csmr.time.unique()))
            self.omega_time_grid = pd.DataFrame(omega_time_grid, columns = ['time_id'])
            self.omega_time_grid['omega_time_grid_index'] = self.omega_time_grid.index

            self.db.time, self.db.age = self.update_root_node_time_age(self.db.time, omega_time_grid, self.db.age, omega_age_grid)

            self.asdr = (asdr
                         .merge(self.db.node, left_on = 'location_id', right_on = 'c_location_id')
                         .merge(self.db.time, how='left')
                         .merge(self.db.age, how='left'))

            self.csmr = (csmr
                         .merge(self.db.node, left_on = 'location_id', right_on = 'c_location_id')
                         .merge(self.db.time, how='left')
                         .merge(self.db.age, how='left'))

        def dataframe_compression_index(name = '', df = None):
            index = df[(df.index == 0) | (df.node_id.diff() > 0)]
            index[f'all_{name}_id'] = index.index
            index.reset_index(inplace=True, drop=True)
            index[f'{name}_index_id'] = index.index
            index = index[[f'{name}_index_id', 'node_id', f'all_{name}_id']]
            return index

            all_mtall = self.asdr[['node_id', 'time_id', 'age_id', 'meas_value']].rename(columns={'meas_value': 'all_mtall_value'}).reset_index(drop=True)
            all_mtall['all_mtall_id'] = all_mtall.index
            self.all_mtall = all_mtall
            self.mtall_index = dataframe_compression_index('mtall', self.all_mtall)

            all_mtspecific = self.asdr[['node_id', 'time_id', 'age_id', 'meas_value']].rename(columns={'meas_value': 'all_mtspecific_value'}).reset_index(drop=True)
            all_mtspecific['all_mtspecific_id'] = all_mtspecific.index
            self.all_mtspecific = all_mtspecific
            self.mtspecific_index = dataframe_compression_index('mtspecific', self.all_mtspecific)


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

        self.age_group_id = demographics.age_group_id
        self.year_id = demographics.year_id
        self.sex_id = demographics.sex_id
        self.drill_locations = demographics.drill_locations

        if location_set_version_id is None:
            self.location_set_version_id = get_location_set_version_id(gbd_round_id = _gbd_round_id_)
        else:
            self.location_set_version_id = location_set_version_id

        if 1:
            print ('*** Get parameter json and load_settings. ***')
            from cascade_at.settings.settings import settings_json_from_model_version_id, load_settings
            parameter_json = settings_json_from_model_version_id(
                model_version_id=mvid,
                conn_def = _conn_def_)
            settings = load_settings(settings_json=parameter_json)

            covariate_specs = CovariateSpecs(
                country_covariates=settings.country_covariate,
                study_covariates=settings.study_covariate
            )



            global inputs
            inputs = Inputs(demographics, population, covariate_specs)
            print ('*** Get locations. ***')
            from cascade_at.inputs.locations import LocationDAG, locations_by_drill
            inputs.location_dag = LocationDAG(location_set_version_id = self.location_set_version_id,
                                       gbd_round_id = _gbd_round_id_)

            covariate_data = {cov.name : CovariateData(
                covariate_id = cov.covariate_id,
                demographics = inputs.demographics,
                decomp_step = _decomp_step_,
                gbd_round_id = _gbd_round_id_
            ).get_raw()
                              for cov in inputs.covariate_specs.covariate_specs
                              if cov.study_country == 'country'}

            ref = CovariateReference(inputs)

            COV_REF = ref.configure_for_dismod(covariate_data)


        if settings.model.drill:
            self.drill_location_start = settings.model.drill_location_start
            self.drill_location_end = settings.model.drill_location_end
        else:
            self.drill_location_start = None
            self.drill_location_end = None
        self.country_covariate_ids = list(covariate_specs.country_covariate_ids)


        from cascade_at.dismod.api.fill_extract_helpers import reference_tables

        self.age = reference_tables.construct_age_time_table(
            variable_name='age', variable=demographics.age_group_id,
            data_min=0, data_max=100)

        self.time = reference_tables.construct_age_time_table(
            variable_name='time', variable=demographics.year_id,
            data_min=0, data_max=2020)


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

        print ('*** Get covariate_reference. ***')
        covariate_ids = [57, 68]
        inputs.covariate_data = {c: CovariateData(
            covariate_id=c,
            demographics=self.demographics,
            decomp_step=_decomp_step_,
            gbd_round_id=_gbd_round_id_
        ).get_raw() for c in covariate_ids}

        from cascade_at.dismod.api.fill_extract_helpers.reference_tables import construct_node_table
        node = construct_node_table(inputs.location_dag).rename(columns = {'c_location_id': 'location_id'})
        level = [inputs.location_dag.depth(node) for node in node.location_id]
        node['level'] = level
        node['parent_id'] = node.parent

        covariate_reference = cov_weighted_average(inputs, node)

        # This the lookup that interpolates ages and times

        self.location_id = sorted(inputs.location_dag.dag.nodes)

        # node = pd.DataFrame(self.location_id, columns=['c_location_id'])
        
        print ('*** Get age metadata. ***')
        import cascade_at.core.db
        self.age_groups = cascade_at.core.db.db_queries.get_age_metadata(age_group_set_id=_age_group_set_id_, gbd_round_id=_gbd_round_id_)


        
        
        print ('*** Get covariate_reference. ***')
        covariate_ids = [57, 68]
        def get_covariate_reference(covariate_ids):
            cols = ['covariate_id', 'location_id', 'year_id', 'age_group_id', 'sex_id', 'mean_value']
            covariates = pd.DataFrame()
            names = None
            for cov_id in covariate_ids:
                cov = CovariateData(
                    covariate_id=cov_id,
                    demographics=self.demographics,
                    decomp_step=_decomp_step_,
                    gbd_round_id=_gbd_round_id_
                ).get_raw()
                cov_df = cov.configure_for_dismod(pop_df = inputs.population.raw, loc_df = node)
                if not cov_df.empty:
                    cov_name = cov.raw.covariate_name_short[0]
                    cov_df['covariate_id'] = cov_id
                    cov_df['covariate_name'] = cov_name
                    covariates = covariates.append(cov_df)
                covariate = covariates.groupby(['covariate_id', 'location_id', 'sex_id'], as_index=False).median()
            covariate.reset_index(inplace=True, drop=True)
            covariate['all_covariate_reference_id'] = covariate.index
            covariate = covariate.merge(self.db.node, left_on = 'location_id', right_on = 'c_location_id')
            covariate.rename(columns={'mean_value': 'reference'}, inplace=True)
            return covariate[['all_covariate_reference_id', 'covariate_id', 'node_id', 'sex_id', 'reference']]

        # covariate = get_covariate_reference(covariate_ids)

        # self.covariate = covariate


if __name__ == '__main__':

    self = AllNodeDatabase()
    
    fn = '/tmp/all_node_datatabase.db'
    shutil.rmtree(fn, ignore_errors=True)
    conn = sqlite3.connect(fn)
    self.node.to_sql('node', conn, if_exists="replace")
    self.covariate.to_sql('covariate', conn, if_exists="replace")
    # self.asdr.to_sql('all_mtall', conn, if_exists="replace")
    # self.csmr.to_sql('all_mtspecific', conn, if_exists="replace")

    self.omega_age_grid.to_sql('omega_age_grid', conn, if_exists="replace")
    self.omega_time_grid.to_sql('omega_time_grid', conn, if_exists="replace")

    self.all_mtall[['all_mtall_id', 'all_mtall_value']].to_sql('all_mtall', conn, if_exists="replace")
    self.mtall_index.to_sql('mtall_index', conn, if_exists="replace")

    self.all_mtspecific[['all_mtspecific_id', 'all_mtspecific_value']].to_sql('all_mtspecific', conn, if_exists="replace")
    self.mtspecific_index.to_sql('mtspecific_index', conn, if_exists="replace")



if 0:

    from cascade_at.core.db import elmo
    try:
        elmo.get_crosswalk_version(crosswalk_version_id=31907)
    except:
        print ("Did you apply the elmo patch -- see ~/Projects/IHME/SHARED_FUNCTION-patches/")

    print ('*** Check an ezfuncs query for run_id. ***')
    run_query = f"""
            SELECT MAX(co.output_version_id) AS version
            FROM cod.output_version co
            JOIN shared.decomp_step ds USING (decomp_step_id)
            WHERE co.is_best = 1
            AND co.best_end IS NULL
            AND ds.gbd_round_id = {_gbd_round_id_}
            """
    run_id = db_tools.ezfuncs.query(run_query, conn_def='cod').version.astype(int).squeeze()




    if sys.platform.lower() != 'darwin':
        # Permissions are wrong on the Mac to do this
        run_query = '''LOAD DATA INFILE
        "/Users/gma/ihme/epi/at_cascade/data/475869/outputs/fits/100/100_2.0_summary.csv"
        REPLACE INTO TABLE
        epi.model_estimate_fit
        FIELDS
        TERMINATED BY ","
        OPTIONALLY ENCLOSED BY '"'
        LINES
        TERMINATED BY "\n"
        IGNORE 1 LINES
        (@dummy,measure_id,year_id,age_group_id,location_id,sex_id,model_version_id,mean,lower,upper)
        '''
        conn_def='epi-uploader'
        conn_def='dismod-at-dev'
        run_id = db_tools.ezfuncs.query(run_query, conn_def=conn_def).version.astype(int).squeeze()




    print ('*** Check COD. ***')
    from cascade_at.inputs.csmr import get_best_cod_correct
    cod = get_best_cod_correct(gbd_round_id = _gbd_round_id_)

    print ('OK')





'''

class AllNodeDatabase:
    def __init__(self, model_version_id: int,
                 gbd_round_id: int, decomp_step_id: int,
                 conn_def: str,
                 country_covariate_id: List[int],
                 csmr_cause_id: int, crosswalk_version_id: int,
                 location_set_version_id: Optional[int] = None,
                 drill_location_start: Optional[int] = None,
                 drill_location_end: Optional[List[int]] = None):
    
    def get_raw_inputs(self):
        """
        Get the raw inputs that need to be used
        in the modeling.
        """
        LOG.info("Getting all raw inputs.")
        LOG.warning("FIXME -- gma -- asdr.py and csmr.py were getting different locations -- not sure if they should use location_ids or drill_locations.")
        LOG.warning("FIXME -- gma -- suspect it should be drill_locations, but it seems Drill leaf node handling is not implemented properly.")
        self.asdr = ASDR(
            demographics=self.demographics,
            decomp_step=self.decomp_step,
            gbd_round_id=self.gbd_round_id
        ).get_raw()
        self.csmr = CSMR(
            cause_id=self.csmr_cause_id,
            demographics=self.demographics,
            decomp_step=self.decomp_step,
            gbd_round_id=self.gbd_round_id,
        ).get_raw()
        self.data = CrosswalkVersion(
            crosswalk_version_id=self.crosswalk_version_id,
            exclude_outliers=self.exclude_outliers,
            demographics=self.demographics,
            conn_def=self.conn_def,
            gbd_round_id=self.gbd_round_id
        ).get_raw()
        self.covariate_data = [CovariateData(
            covariate_id=c,
            demographics=self.demographics,
            decomp_step=self.decomp_step,
            gbd_round_id=self.gbd_round_id
        ).get_raw() for c in self.country_covariate_id]
        self.population = Population(
            demographics=self.demographics,
            decomp_step=self.decomp_step,
            gbd_round_id=self.gbd_round_id
        ).get_population()


class MeasurementInputs:

    def __init__(self, model_version_id: int,
                 gbd_round_id: int, decomp_step_id: int,
                 conn_def: str,
                 country_covariate_id: List[int],
                 csmr_cause_id: int, crosswalk_version_id: int,
                 location_set_version_id: Optional[int] = None,
                 drill_location_start: Optional[int] = None,
                 drill_location_end: Optional[List[int]] = None):
        """
        The class that constructs all of the measurement inputs. Pulls ASDR,
        CSMR, crosswalk versions, and country covariates, and puts them into
        one data frame that then formats itself for the dismod database.
        Performs covariate value interpolation if age and year ranges
        don't match up with GBD age and year ranges.

        Parameters
        ----------
        model_version_id
            the model version ID
        gbd_round_id
            the GBD round ID
        decomp_step_id
            the decomp step ID
        csmr_cause_id: (int) cause to pull CSMR from
        crosswalk_version_id
            crosswalk version to use
        country_covariate_id
            list of covariate IDs
        conn_def
            connection definition from .odbc file (e.g. 'epi') to connect to the IHME databases
        location_set_version_id
            can be None, if it's none, get the best location_set_version_id for estimation hierarchy of this GBD round
        drill_location_start
            which location ID to drill from as the parent
        drill_location_end
            which immediate children of the drill_location_start parent to include in the drill

        Attributes
        ----------
        self.decomp_step : str
            the decomp step in string form
        self.demographics : cascade_at.inputs.demographics.Demographics
            a demographics object that specifies the age group, sex,
            location, and year IDs to grab
        self.integrand_map : Dict[int, int]
            dictionary mapping from GBD measure IDs to DisMod IDs
        self.asdr : cascade_at.inputs.asdr.ASDR
            all-cause mortality input object
        self.csmr : cascade_at.inputs.csmr.CSMR
            cause-specific mortality input object from cause csmr_cause_id
        self.data : cascade_at.inputs.data.CrosswalkVersion
            crosswalk version data from IHME database
        self.covariate_data : List[cascade_at.inputs.covariate_data.CovariateData]
            list of covariate data objects that contains the raw covariate data mapped to IDs
        self.location_dag : cascade_at.inputs.locations.LocationDAG
            DAG of locations to be used
        self.population: (cascade_at.inputs.population.Population)
            population object that is used for covariate weighting
        self.data_eta: (Dict[str, float]): dictionary of eta value to be
            applied to each measure
        self.density: (Dict[str, str]): dictionary of density to be
            applied to each measure
        self.nu: (Dict[str, float]): dictionary of nu value to be applied
            to each measure
        self.dismod_data: (pd.DataFrame) resulting dismod data formatted
            to be used in the dismod database

        Examples
        --------
        >>> from cascade_at.settings.base_case import BASE_CASE
        >>> from cascade_at.settings.settings import load_settings
        >>>
        >>> settings = load_settings(BASE_CASE)
        >>> covariate_id = [i.country_covariate_id for i in settings.country_covariate]
        >>>
        >>> i = MeasurementInputs(
        >>>    model_version_id=settings.model.model_version_id,
        >>>    gbd_round_id=settings.gbd_round_id,
        >>>    decomp_step_id=settings.model.decomp_step_id,
        >>>    csmr_cause_id = settings.model.add_csmr_cause,
        >>>    crosswalk_version_id=settings.model.crosswalk_version_id,
        >>>    country_covariate_id=covariate_id,
        >>>    conn_def='epi',
        >>>    location_set_version_id=settings.location_set_version_id
        >>> )
        >>> i.get_raw_inputs()
        >>> i.configure_inputs_for_dismod(settings)
        """
        LOG.info(f"Initializing input object for model version ID {model_version_id}.")
        LOG.info(f"GBD Round ID {gbd_round_id}.")
        LOG.info(f"Pulling from connection {conn_def}.")

        self.model_version_id = model_version_id
        self.gbd_round_id = gbd_round_id
        self.decomp_step_id = decomp_step_id
        self.csmr_cause_id = csmr_cause_id
        self.crosswalk_version_id = crosswalk_version_id
        self.country_covariate_id = country_covariate_id
        self.conn_def = conn_def
        self.drill_location_start = drill_location_start
        self.drill_location_end = drill_location_end
        self.decomp_step = ds.decomp_step_from_decomp_step_id(self.decomp_step_id)
        if location_set_version_id is None:
            self.location_set_version_id = get_location_set_version_id(gbd_round_id=self.gbd_round_id)
        else:
            self.location_set_version_id = location_set_version_id

        self.demographics = Demographics(
            gbd_round_id=self.gbd_round_id,
            location_set_version_id=self.location_set_version_id)
        self.location_dag = LocationDAG(
            location_set_version_id=self.location_set_version_id,
            gbd_round_id=self.gbd_round_id
        )
        # Need to subset the locations to only those needed for
        # the drill. drill_locations_all is the set of locations
        # to pull data for, including all descendants. drill_locations
        # is the set of locations just parent-children in the drill.
        drill_locations_all, drill_locations = locations_by_drill(
            drill_location_start=self.drill_location_start,
            drill_location_end=self.drill_location_end,
            dag=self.location_dag
        )
        if drill_locations_all:
            self.demographics.location_id = drill_locations_all
            self.demographics.drill_locations = drill_locations

        self.exclude_outliers = True
        self.asdr = None
        self.csmr = None
        self.population = None
        self.data = None
        self.covariates = None
        self.age_groups = None

        self.data_eta = None
        self.density = None
        self.nu = None
        self.measures_to_exclude: Optional[List[str]] = None
        self.measures_midpoint: Optional[List[str]] = None

        self.dismod_data = None
        self.covariate_data = None
        self.country_covariate_data = None
        self.covariate_specs = None
        self.omega = None

    def get_raw_inputs(self):
        """
        Get the raw inputs that need to be used
        in the modeling.
        """
        LOG.info("Getting all raw inputs.")
        LOG.warning("FIXME -- gma -- asdr.py and csmr.py were getting different locations -- not sure if they should use location_ids or drill_locations.")
        LOG.warning("FIXME -- gma -- suspect it should be drill_locations, but it seems Drill leaf node handling is not implemented properly.")
        self.asdr = ASDR(
            demographics=self.demographics,
            decomp_step=self.decomp_step,
            gbd_round_id=self.gbd_round_id
        ).get_raw()
        self.csmr = CSMR(
            cause_id=self.csmr_cause_id,
            demographics=self.demographics,
            decomp_step=self.decomp_step,
            gbd_round_id=self.gbd_round_id,
        ).get_raw()
        self.data = CrosswalkVersion(
            crosswalk_version_id=self.crosswalk_version_id,
            exclude_outliers=self.exclude_outliers,
            demographics=self.demographics,
            conn_def=self.conn_def,
            gbd_round_id=self.gbd_round_id
        ).get_raw()
        self.covariate_data = [CovariateData(
            covariate_id=c,
            demographics=self.demographics,
            decomp_step=self.decomp_step,
            gbd_round_id=self.gbd_round_id
        ).get_raw() for c in self.country_covariate_id]
        self.population = Population(
            demographics=self.demographics,
            decomp_step=self.decomp_step,
            gbd_round_id=self.gbd_round_id
        ).get_population()

    def configure_inputs_for_dismod(self, settings: SettingsConfig,
                                    mortality_year_reduction: int = 5):
        """
        Modifies the inputs for DisMod based on model-specific settings.

        Arguments
        ---------
        settings
            Settings for the model
        mortality_year_reduction
            number of years to decimate csmr and asdr
        """
        self.data_eta = data_eta_from_settings(settings)
        self.density = density_from_settings(settings)
        self.nu = nu_from_settings(settings)
        self.measures_to_exclude = measures_to_exclude_from_settings(settings)
        self.measures_midpoint = midpoint_list_from_settings(settings)

        # If we are constraining omega, then we want to hold out the data
        # from the DisMod fit for ASDR (but never CSMR -- always want to fit
        # CSMR).
        data = self.data.configure_for_dismod(
            measures_to_exclude=self.measures_to_exclude,
            relabel_incidence=settings.model.relabel_incidence
        )
        asdr = self.asdr.configure_for_dismod(
            hold_out=settings.model.constrain_omega)
        csmr = self.csmr.configure_for_dismod(hold_out=0)

        if settings.model.constrain_omega:
            self.omega = calculate_omega(asdr=asdr, csmr=csmr)
        else:
            self.omega = None

        if not csmr.empty:
            csmr = decimate_years(
                data=csmr, num_years=mortality_year_reduction)
        if not asdr.empty:
            asdr = decimate_years(
                data=asdr, num_years=mortality_year_reduction)

        self.dismod_data = pd.concat([data, asdr, csmr], axis=0, sort=True)
        self.dismod_data.reset_index(drop=True, inplace=True)

        self.dismod_data["density"] = self.dismod_data.measure.apply(
            self.density.__getitem__)
        self.dismod_data["eta"] = self.dismod_data.measure.apply(
            self.data_eta.__getitem__)
        self.dismod_data["nu"] = self.dismod_data.measure.apply(
            self.nu.__getitem__)

        for measure in self.dismod_data.measure.unique():
            if measure in self.measures_midpoint:
                midpoint_age_time(df=self.dismod_data, measure=measure)
            else:
                format_age_time(df=self.dismod_data, measure=measure)

        # This makes the specs not just for the country covariate but adds on
        # the sex and one covariates.
        self.covariate_specs = CovariateSpecs(
            country_covariates=settings.country_covariate,
            study_covariates=settings.study_covariate
        )
        self.country_covariate_data = {c.covariate_id: c.configure_for_dismod(
            pop_df=self.population.configure_for_dismod(),
            loc_df=self.location_dag.df
        ) for c in self.covariate_data}

        self.dismod_data = self.add_covariates_to_data(df=self.dismod_data)
        self.dismod_data.loc[
            self.dismod_data.hold_out.isnull(), 'hold_out'] = 0.
        self.dismod_data.drop(['age_group_id'], inplace=True, axis=1)

        return self

    def prune_mortality_data(self, parent_location_id: int) -> pd.DataFrame:
        """
        Remove mortality data for descendants that are not children of parent_location_id
        from the configured dismod data before it gets filled into the dismod database.
        """
        df = self.dismod_data.copy()
        direct_children = self.location_dag.parent_children(parent_location_id)
        direct_children = df.location_id.isin(direct_children)
        mortality_measures = df.measure.isin([
            IntegrandEnum.mtall.name, IntegrandEnum.mtspecific.name
        ])
        remove_rows = ~direct_children & mortality_measures
        df = df.loc[~remove_rows].copy()
        return df

    def add_covariates_to_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add on covariates to a data frame that has age_group_id, year_id
        or time-age upper / lower, and location_id and sex_id. Adds both
        country-level and study-level covariates.
        """
        cov_dict_for_interpolation = {
            c.name: self.country_covariate_data[c.covariate_id]
            for c in self.covariate_specs.covariate_specs
            if c.study_country == 'country'
        }

        df = self.interpolate_country_covariate_values(
            df=df, cov_dict=cov_dict_for_interpolation)
        df = self.transform_country_covariates(df=df)

        df['s_sex'] = df.sex_id.map(
            SEX_ID_TO_NAME).map(StudyCovConstants.SEX_COV_VALUE_MAP)
        df['s_one'] = StudyCovConstants.ONE_COV_VALUE

        return df

    def to_gbd_avgint(self, parent_location_id: int, sex_id: int) -> pd.DataFrame:
        """
        Converts the demographics of the model to the avgint table.
        """
        LOG.info(f"Getting grid for the avgint table "
                 f"for parent location ID {parent_location_id} "
                 f"and sex_id {sex_id}.")
        if self.drill_location_start is not None:
            locations = self.demographics.drill_locations
        else:
            locations = self.location_dag.parent_children(parent_location_id)
        grid = expand_grid({
            'sex_id': [sex_id],
            'location_id': locations,
            'year_id': self.demographics.year_id,
            'age_group_id': self.demographics.age_group_id
        })
        grid['time_lower'] = grid['year_id'].astype(int)
        grid['time_upper'] = grid['year_id'] + 1.
        grid = BaseInput(
            gbd_round_id=self.gbd_round_id).convert_to_age_lower_upper(df=grid)
        LOG.info("Adding covariates to avgint grid.")
        grid = self.add_covariates_to_data(df=grid)
        return grid

    def interpolate_country_covariate_values(self, df: pd.DataFrame, cov_dict: Dict[Union[float, str], pd.DataFrame]):
        """
        Interpolates the covariate values onto the data
        so that the non-standard ages and years match up to meaningful
        covariate values.
        """
        LOG.info(f"Interpolating and merging the country covariates.")
        interp_df = get_interpolated_covariate_values(
            data_df=df,
            covariate_dict=cov_dict,
            population_df=self.population.configure_for_dismod()
        )
        return interp_df

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

    def calculate_country_covariate_reference_values(
            self, parent_location_id: int, sex_id: int) -> CovariateSpecs:
        """
        Gets the country covariate reference value for a covariate ID and a
        parent location ID. Also gets the maximum difference between the
        reference value and covariate values observed.

        Run this when you're going to make a DisMod AT database for a specific
        parent location and sex ID.

        :param: (int)
        :param parent_location_id: (int)
        :param sex_id: (int)
        :return: List[CovariateSpec] list of the covariate specs with the
            correct reference values and max diff.
        """
        covariate_specs = copy(self.covariate_specs)

        age_min = self.dismod_data.age_lower.min()
        age_max = self.dismod_data.age_upper.max()
        time_min = self.dismod_data.time_lower.min()
        time_max = self.dismod_data.time_upper.max()

        children = self.location_dag.children(parent_location_id)

        for c in covariate_specs.covariate_specs:
            transform = COVARIATE_TRANSFORMS[c.transformation_id]
            if c.study_country == 'study':
                if c.name == 's_sex':
                    c.reference = StudyCovConstants.SEX_COV_VALUE_MAP[
                        SEX_ID_TO_NAME[sex_id]]
                    c.max_difference = StudyCovConstants.MAX_DIFFERENCE_SEX_COV
                elif c.name == 's_one':
                    c.reference = StudyCovConstants.ONE_COV_REFERENCE
                    c.max_difference = StudyCovConstants.MAX_DIFFERENCE_ONE_COV
                else:
                    raise ValueError(f"The only two study covariates allowed are sex and one, you tried {c.name}.")
            elif c.study_country == 'country':
                LOG.info(f"Calculating the {transform.__name__} transformed reference and max difference for country covariate {c.covariate_id}.")

                cov_df = self.country_covariate_data[c.covariate_id]
                cov_df.loc[:, 'mean_value'] = transform(cov_df.loc[:, 'mean_value'])

                parent_df = (
                    cov_df.loc[cov_df.location_id == parent_location_id].copy()
                )
                child_df = cov_df.loc[cov_df.location_id.isin(children)].copy()
                all_loc_df = pd.concat([child_df, parent_df], axis=0)

                # if there is no data for the parent location at all (which
                # there should be provided by Central Comp)
                # then we are going to set the reference value to 0.
                if cov_df.empty:
                    reference_value = 0
                    max_difference = np.nan
                else:
                    pop_df = self.population.configure_for_dismod()
                    pop_df = (
                        pop_df.loc[pop_df.location_id == parent_location_id].copy()
                    )

                    df_to_interp = pd.DataFrame({
                        'location_id': parent_location_id,
                        'sex_id': [sex_id],
                        'age_lower': [age_min], 'age_upper': [age_max],
                        'time_lower': [time_min], 'time_upper': [time_max]
                    })
                    reference_value = get_interpolated_covariate_values(
                        data_df=df_to_interp,
                        covariate_dict={c.name: parent_df},
                        population_df=pop_df
                    )[c.name].iloc[0]
                    LOG.info(f"Setting covariate {c.name} max_difference = nan to disable data hold_out due to covariate value.")
                    max_difference = np.nan
                c.reference = reference_value
                c.max_difference = max_difference
        covariate_specs.create_covariate_list()
        return covariate_specs

    def reset_index(self, drop, inplace):
        pass


class MeasurementInputsFromSettings(MeasurementInputs):
    def __init__(self, settings: SettingsConfig):
        """
        Wrapper for MeasurementInputs that takes a settings object rather
        than the individual arguments. For convenience.

        Examples
        --------
        >>> from cascade_at.settings.base_case import BASE_CASE
        >>> from cascade_at.settings.settings import load_settings

        >>> settings = load_settings(BASE_CASE)
        >>> i = MeasurementInputs(settings)
        >>> i.get_raw_inputs()
        >>> i.configure_inputs_for_dismod()
        """
        covariate_ids = [i.country_covariate_id for i in
                         settings.country_covariate]

        if settings.model.drill:
            drill_location_start = settings.model.drill_location_start
            drill_location_end = settings.model.drill_location_end
        else:
            drill_location_start = None
            drill_location_end = None

        super().__init__(
            model_version_id=settings.model.model_version_id,
            gbd_round_id=settings.gbd_round_id,
            decomp_step_id=settings.model.decomp_step_id,
            csmr_cause_id=settings.model.add_csmr_cause,
            crosswalk_version_id=settings.model.crosswalk_version_id,
            country_covariate_id=covariate_ids,
            conn_def='epi',
            location_set_version_id=settings.location_set_version_id,
            drill_location_start=drill_location_start,
            drill_location_end=drill_location_end
        )
'''
