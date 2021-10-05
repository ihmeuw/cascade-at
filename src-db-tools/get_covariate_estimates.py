import sys, os
import numpy as np
import pandas as pd
import json
from functools import lru_cache

import db_queries

import utilities
from dismod_db_api import DismodDbAPI

if 0:
    from cascade_at_gma.lib import utilities
    from cascade_at_gma.lib.dismod_db_api import DismodDbAPI

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('get_covariate_estimates.py')

inf = np.float('inf')

"""
Interface for get_covariate_estimates
db_queries.get_covariate_estimates (covariate_id=57, location_id='all', location_set_id=22, location_set_version_id=0, year_id='all', sex_id='all', age_group_id='all', gbd_round_id=5, status='best', model_version_id=None)
"""

def nid(nodes): return sorted([n.id for n in nodes])

cascade_both, male, female, shared_db_both = 0,1,2,3
ihme_db_sex_dict = dict(all = (male,female,shared_db_both), both = (shared_db_both,), male = (male,), female = (female,))
cascade_sex_dict = dict(all = (male,female,cascade_both), both = (cascade_both,), male = (male,), female = (female,))

__use_shared_function__ = True

@lru_cache(1)
def ihme_location_hierarchy(model_version_id):
    # Get the location hierarchy
    from hierarchies.dbtrees import loctree as lt
    # gma 7/9/2020 from cascade_ode.importer import get_model_version
    def get_model_version (model_version_id):
        if 0: from cascade_ode.importer import execute_select # This import must be inside this function otherwise argparse help is incorrect
        from execute_select import execute_select # This import must be inside this function otherwise argparse help is incorrect
        import db_tools
        query = """
        SELECT * FROM epi.model_version
        WHERE model_version_id=%s """ % (int(model_version_id))
        
        df = execute_select(query, conn_def = 'epi')
        return df

    model_df = get_model_version(model_version_id)
    lsvid = int(model_df.location_set_version_id)
    gbd_round_id = int(model_df.gbd_round_id)
    loctree = lt(None, location_set_version_id = lsvid, gbd_round_id = gbd_round_id)
    loctree.location_set_version_id = lsvid
    loctree.all_nodes = [loctree.get_node_by_id(1)] + loctree.get_node_by_id(1).all_descendants()
    loctree.world = [n for n in loctree.all_nodes if 'global'in n.info['location_type']]
    loctree.superregion = [n for n in loctree.all_nodes if 'super' in n.info['location_type']]
    loctree.region = [n for n in loctree.all_nodes if 'region' == n.info['location_type']]
    leaves = loctree.world[0].leaves()
    loctree.midlevel = [n for n in loctree.all_nodes if n not in (loctree.world + loctree.superregion + loctree.region + loctree.leaves())]

    return loctree

@lru_cache(maxsize = 1)
def query_model_parameters(model_version_id):
    if 0: from cascade_ode.importer import execute_select # This import must be inside this function otherwise argparse help is incorrect
    from execute_select import execute_select # This import must be inside this function otherwise argparse help is incorrect

    # gma 7/9/2020
    # query = """
    #     SELECT * FROM at_model_parameter
    #     LEFT JOIN shared.measure USING(measure_id)
    #     LEFT JOIN epi.parameter_type USING(parameter_type_id)
    #     LEFT JOIN epi.study_covariate USING(study_covariate_id)
    #     WHERE model_version_id=%s """ % (model_version_id)
    # df = execute_select(query)
    # df.drop(['date_inserted', 'inserted_by', 'last_updated', 'last_updated_by', 'last_updated_action'], axis=1, inplace=True)
    # df['mean'] = df['mean'].fillna((df.upper+df.lower)/2.0)
    # df['std'] = df['std'].fillna(inf)

    query = ('SELECT * FROM model_version_at '
             f'WHERE model_version_id={model_version_id}')
    df = execute_select(query)
    df.drop(['date_inserted', 'inserted_by', 'last_updated', 'last_updated_by', 'last_updated_action'], axis=1, inplace=True)
    return df

@lru_cache(maxsize = 10)
def query_covariates(covariate_names_short):
    if 0: from cascade_ode.importer import execute_select # This import must be inside this function otherwise argparse help is incorrect
    from execute_select import execute_select # This import must be inside this function otherwise argparse help is incorrect

    covs = pd.DataFrame([], columns=['covariate_id', 'covariate_name_short'])
    all_cov_names = execute_select('SELECT covariate_id, covariate_name_short FROM shared.covariate')
    for cov_name in covariate_names_short:
        df = all_cov_names[[cov_name.startswith(n) for n in all_cov_names.covariate_name_short]]
        if df.empty:
            raise Exception(f"Failed to find covariate name '{cov_name}'")
        elif len(df) > 1:
            raise Exception(f"Found too many covariates {df.covariate_name_short.tolist()} matching name '{cov_name}'")
        else:
            covs = covs.append(df)
    return covs

@lru_cache(50)
def importer_style_query_covariate_estimates(covariate_id, covariate_name_short, location_id):
    if 0: from cascade_ode.importer import execute_select # This import must be inside this function otherwise argparse help is incorrect
    from execute_select import execute_select # This import must be inside this function otherwise argparse help is incorrect

    logging.warn("Getting country covariate {} via an importer.py-like query.".format(covariate_name_short))
    dataq = """
    SELECT location_id, year_id, age_group_id, sex_id, mean_value, model_version_id
    FROM covariate.model
    JOIN shared.location USING(location_id)
    JOIN covariate.model_version USING(model_version_id)
    JOIN covariate.data_version USING(data_version_id)
    WHERE is_best=1 AND covariate_id='%s' """ % (covariate_id)
    covs_df = execute_select(dataq, 'cov')

    model_version_ids = covs_df.model_version_id.unique().tolist()
    logging.warn("Used covariate model_version_id(s) of {}.".format(model_version_ids))
    
    covs_df['covariate_id'] = covariate_id
    colname = 'raw_c_%s' % covariate_name_short
    
    return covs_df

def apply_covariate_transform(model_version_id, covdata):
    # gma -- Get the transform type and transform the covariate data from the database units
    covariate_id = int(covdata.covariate_id.unique())
    model_params = query_model_parameters(model_version_id)
    model_json = json.loads(model_params.parameter_json.values[0])
    ccovs = model_json['country_covariate']
    ccov = [c for c in ccovs if c['country_covariate_id'] == covariate_id][0]
    transform_type_id = ccov['transformation']
    if transform_type_id == 1:
        covdata['mean_value'] = np.log(covdata.mean_value)
    elif transform_type_id == 2:
        covdata['mean_value'] = np.log(covdata.mean_value / (1-covdata.mean_value))
    elif transform_type_id == 3:
        covdata['mean_value'] = covdata.mean_value**2
    elif transform_type_id == 4:
        covdata['mean_value'] = np.sqrt(covdata.mean_value)
    elif transform_type_id == 5:
        covdata['mean_value'] = covdata.mean_value*1000
    return covdata

@lru_cache(50)
def _get_covariate_estimates_internal(covariate_name_short, location_id = 'all', year_id = 'all',
                                      model_version_id = None, gbd_round_id = None, sex = 'all'):
    """
    Run the queries to retrieve country covariate values.

    Note that some of the country covariate queries (e.g. LDI_pc) only return sex = both, and 
    some (e.g. mean_BMI) return sex = (male, female). This routine fills in missing the sex values,
    and returns only the requested sex(s).

    The shared function db_queries.get_covariate_estimates is called with sex_id = 'all' 
    because some covariates return male/female, some return both, who knows what will be returned.

    The both sex value = 3 returned from the database is changed to the both = 0 cascade/dismod convention.

    Returns:
    sex:    sex_ids returned
    both:   both=0
    male:   male=1
    female: female=2
    all:    both=0, male=1, and female=2
    """

    # db_queries.get_covariate_estimates does not handle tuples
    if isinstance(location_id, tuple):
        location_id = list(location_id)
    if isinstance(year_id, tuple):
        year_id = year_id[0] if len(year_id) == 1 else list(year_id)

    lh = ihme_location_hierarchy(model_version_id)
    covariates_df = query_covariates(covariate_name_short)
    group_by = ['location_id', 'year_id', 'age_group_id']
    merge = group_by + ['sex_id']
    cols = merge + ['mean_value']
    ccovs = None
    for cov_id,short_name in covariates_df[['covariate_id', 'covariate_name_short']].values:
        # Query the covariate estimates from the database
        if __use_shared_function__:
            kwds = dict(location_id = location_id,
                        year_id = year_id,
                        location_set_version_id = lh.location_set_version_id,
                        gbd_round_id = gbd_round_id,
                        decomp_step = 'iterative',
                        sex_id = 'all')
            if gbd_round_id is None: kwds.pop('gbd_round_id')
            df = db_queries.get_covariate_estimates(cov_id, **kwds)
        else:
            df = importer_style_query_covariate_estimates(cov_id, short_name, utilities.force_tuple(location_id))
            if year_id != 'all':
                df = df[df.year_id.isin(utilities.force_tuple(year_id))]
            df_loc_ids = df.location_id.unique()
            assert set(location_id).issubset(df_loc_ids), "Missing covariates in some locations"
            df = df[df.location_id.isin(location_id)]
                
        if not df.empty:
            df = apply_covariate_transform(model_version_id, df)[cols]
            df = df.rename(columns={'mean_value': short_name})
            mask = df.age_group_id == 27
            if not mask.empty:
                logger.error("Some 'age standardized' age groups have different age_group_id's. HARDCODED A HACK TO FIX THIS PROBLEM.")
                df.loc[mask, 'age_group_id'] = 22
            sex_ids = df.sex_id.unique().tolist()
            if sex_ids == [shared_db_both]:
                # If query returns both, then set male = female = both in the df
                cpy = df.copy()
                for sex_id in (male,female):
                    cpy['sex_id'] = sex_id
                    df = df.append(cpy)
            elif set(sex_ids) == set([male,female]):
                # If query returns male,female, compute both
                mean = df.groupby(group_by, as_index = False).mean()
                mean['sex_id'] = shared_db_both
                df = df.append(mean)
        ccovs = df if ccovs is None else ccovs.merge(df, on=merge)
    mask = ccovs.sex_id.astype(int).isin(ihme_db_sex_dict[sex]).values
    ccovs = ccovs[mask]
    ccovs.loc[ccovs.sex_id == shared_db_both, 'sex_id'] = cascade_both
    return ccovs.reset_index(drop=True)

def get_covariate_estimates(covariate_names_short, location_id = None, year_id = 'all',
                            model_version_id = None, gbd_round_id = None, sex = 'all'):
    """
    DataFrame returned has only age-standardized covariates.
    Makes the arguments to _get_covariate_estimates_internal pickleable so that function call can be cached.

    For some covariates, the shared function db_queries.get_covariate_estimates does not return values for
    all nodes, so this function searches for child nodes that do return covariate values, and computes the
    mean of the child nodes to get a value for the parent.

    Example:
    model_version_id = 100667
    covariate_names_short = ('LDI_pc', 'mean_BMI')
    location_id = 100
    year_id = (1990, 1995, 2000, 2005, 2010, 2015)
    year_id = (1990,2000)
    gbd_round_id = 4
    df = get_covariate_estimates(covariate_names_short, location_id = location_id, year_id = year_id, model_version_id = model_version_id, gbd_round_id = gbd_round_id)
    assert len(df.location_id.unique()) == 1 and df.location_id.unique()[0] == location_id
    """
    covariate_names_short = utilities.force_tuple(covariate_names_short)
    location_id = utilities.force_tuple(location_id)
    year_id = utilities.force_tuple(year_id)

    df = pd.DataFrame()
    if covariate_names_short:
        df = _get_covariate_estimates_internal(covariate_names_short, location_id = location_id, year_id = year_id,
                                               model_version_id = model_version_id, gbd_round_id = gbd_round_id,
                                               sex = sex)
        if not df.empty and set(location_id) == set(df.location_id):
            logger.info('Found %s country covariate value(s) for location(s): %s' % (covariate_names_short, str(sorted(df.location_id.unique()))))
            cov_source = location_id
        else:
            if df.empty:
                logger.warning(f'db_queries.get_covariate_estimates is returning no values for {covariate_names_short} at location(s): {sorted(location_id)}')
            # If the parent node returns no covariate values, search for them in the node descendanta
            lh = ihme_location_hierarchy(model_version_id)
            nodes = [lh.get_node_by_id(lid) for lid in location_id]
            leaves = tuple(set([leaf for node in nodes for leaf in node.leaves()]))
            one_level_above_leaves = nid(set([l.parent for l in leaves]))
            df = _get_covariate_estimates_internal(covariate_names_short, location_id = leaves, year_id = year_id,
                                                   model_version_id = model_version_id, gbd_round_id = gbd_round_id,
                                                   sex = sex)
            if not df.empty:
                logger.info('Found covariate info one level above the leaves: %s' % list(one_level_above_leaves))
                # This is no longer valid because of data summarization issues -- assert set(one_level_above_leaves) == set(df.location_id)
                cov_source = one_level_above_leaves
            else:
                df = _get_covariate_estimates_internal(covariate_names_short, location_id = leaves, year_id = year_id,
                                                       model_version_id = model_version_id, gbd_round_id = gbd_round_id,
                                                       sex = sex)
                if not df.empty:
                    logger.info('Found covariate info in the leaves: %s' % leaves)
                    cov_source = leaves
                else:
                    logger.error('Found no covariate values anywhere.')
            df = df.groupby(['year_id', 'sex_id', 'age_group_id'], as_index=False).mean()
            # df.location_id = node.id
            logger.info('Aggregating descendant nodes into covariate values for node(s) %s.' % list(nodes))
    return df

class CountryCovariateXcovs(object):
    def __init__(self, DB, location_ids = [], years = []):
        model_version_id = DB.cascade_options.model_version_id
        if not DB.country_covariates.empty:
            self.covariate_names = dict(DB.country_covariates[['covariate_name_short', 'xcov_name']].values) 
            self.ccovs = get_covariate_estimates(self.covariate_names.keys(), location_id = location_ids, year_id = years,
                                                 model_version_id = model_version_id, gbd_round_id = None, sex = 'all')
        else:
            self.covariate_names = []
            self.ccovs = []
    def __call__(self,  location_id = None, sex_id = None, year_id = None):
        if not self.ccovs:
            return []
        else:
            covs = (self.ccovs[(self.ccovs.year_id == year_id) & (self.ccovs.sex_id == sex_id) & (self.ccovs.location_id == location_id)]
                    .rename(columns = self.covariate_names)[list(self.covariate_names.values())])
            assert len(covs) == 1, "Country covariate query did not not return an unique set of covariate values:n %s" % covs
            return dict(covs.iloc[0])

if (__name__ == '__main__'):
    import unittest

    class TestGetCovariateEstimates(unittest.TestCase):
                             
        def test_get_covariate_estimates_internal(self):
            model_version_id = 100667
            covariate_names_short = ('LDI_pc', 'mean_BMI')
            year_id = (1990, 1995, 2000, 2005, 2010, 2015)
            location_id = (1, 64, 100, 102, 527, 572)
            gbd_round_id = 6

            for sex, sex_ids in cascade_sex_dict.items():
                args = covariate_names_short
                kwds = dict(location_id=location_id, year_id=year_id, model_version_id=model_version_id, gbd_round_id = gbd_round_id, sex = sex)
                df = _get_covariate_estimates_internal(args, **kwds)

                tst = set(utilities.force_tuple(covariate_names_short)).issubset(set(df.columns))
                self.assertTrue(tst, 'Covariate(s) %s returned did not match query.' % list(covariate_names_short))

                tst = set(df.location_id.unique()) == set(location_id)
                self.assertTrue(tst, 'Nodes with covariates are incorrect.')

                tst = set(df.sex_id.unique()) == set(sex_ids)
                self.assertTrue(tst, ("Returned the wrong sex_id's for case %s" % sex))

        def test_get_covariate_estimates(self):
            model_version_id = 100667
            covariate_names_shorts = ['LDI_pc',
                                      'mean_BMI',
                                      ['LDI_pc', 'mean_BMI']]
            year_ids = [1990,
                        [1990, 1995, 2000, 2005, 2010, 2015]]
            location_ids = [100,
                            [1, 64, 100, 102, 527, 572]]

            if 0:
                covariate_names_shorts = ['mean_BMI']
                location_ids = [100]
                year_ids = [1990]

            gbd_round_id = 6
            for covariate_names_short in covariate_names_shorts:
                for year_id in year_ids:
                    for location_id in location_ids:
                        for sex, sex_ids in cascade_sex_dict.items():
                            args = covariate_names_short
                            kwds = dict(location_id=location_id, year_id=year_id, model_version_id=model_version_id, gbd_round_id = gbd_round_id, sex = sex)
                            df = get_covariate_estimates(args, **kwds)

                            tst = set(utilities.force_tuple(covariate_names_short)).issubset(set(df.columns))
                            self.assertTrue(tst, 'Covariate(s) %s returned did not match query.' % covariate_names_short)

                            tst = set(df.year_id.unique()) == set(utilities.force_tuple(year_id))
                            self.assertTrue(tst, 'Years(s) %s returned did not match query.' % year_id)

                            tst = set(df.location_id.unique()) == set(utilities.force_tuple(location_id))
                            self.assertTrue(tst, 'Location(s) %s returned did not match query.' % location_id)

                            tst = set(df.sex_id.unique()) == set(utilities.force_tuple(sex_ids))
                            self.assertTrue(tst, 'Sex(s) %s returned did not match query sex %s.' % (list(sex_ids), df.sex_id.unique().tolist()))

    unittest.main(exit = False)
