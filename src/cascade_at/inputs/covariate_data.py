import pandas as pd

from cascade_at.core.db import db_queries
from cascade_at.core.log import get_loggers
from cascade_at.inputs.utilities.transformations import COVARIATE_TRANSFORMS
from cascade_at.inputs.utilities.gbd_ids import CascadeConstants
from cascade_at.inputs.base_input import BaseInput

LOG = get_loggers(__name__)


class CovariateData(BaseInput):
    def __init__(self, covariate_id, demographics, decomp_step, gbd_round_id):
        """
        Get covariate estimates, and map them to the necessary demographic
        ages and sexes. If only one age group is present in the covariate
        data then that means that it's not age-specific and we want to copy
        the values over to all the other age groups we're working with in
        demographics. Same with sex.

        :param covariate_id: (int)
        :param demographics: (cascade_at.inputs.demographics.Demographics)
        :param decomp_step: (str)
        :param gbd_round_id: (int)
        """
        super().__init__()
        self.covariate_id = covariate_id
        self.demographics = demographics
        self.decomp_step = decomp_step
        self.gbd_round_id = gbd_round_id

        self.raw = None

    def get_raw(self):
        """
        Pulls the raw covariate data from the database.
        :return:
        """
        self.raw = db_queries.get_covariate_estimates(
            covariate_id=self.covariate_id,
            location_id=self.demographics.location_id,
            year_id=self.demographics.year_id,
            gbd_round_id=self.gbd_round_id,
            decomp_step=self.decomp_step
        )
        return self

    def configure_for_dismod(self, pop_df, loc_df):
        """
        Configures covariates for DisMod.
        :return: self
        """
        df = self.raw[[
            'location_id', 'year_id', 'age_group_id', 'sex_id', 'mean_value'
        ]]
        df = self.convert_to_age_lower_upper(df)
        df = self.complete_covariate_sex(cov_df=df, pop_df=pop_df)
        df = self.complete_covariate_locations(cov_df=df, pop_df=pop_df, loc_df=loc_df)
        return df

    @staticmethod
    def complete_covariate_locations(cov_df, pop_df, loc_df):
        """
        Completes the covariate locations that aren't in the database as a population-weighted average.
        :param cov_df:
        :param pop_df:
        :param loc_df:
        :return:
        """
        parent_pop = pop_df[['location_id', 'age_group_id', 'sex_id', 'year_id', 'population']].copy()
        parent_pop.rename(columns={'location_id': 'parent_id', 'population': 'parent_population'}, inplace=True)

        df = pop_df.merge(cov_df, on=['location_id', 'age_group_id', 'sex_id', 'year_id'], how='left')
        dl = df.merge(loc_df[['location_id', 'parent_id', 'level']], on=['location_id'], how='outer')

        dp = dl.merge(parent_pop, on=['parent_id', 'age_group_id', 'sex_id', 'year_id'], how='left')
        dp['cov_weighted'] = dp.mean_value / dp.parent_population

        na_locs = dp.loc[dp.mean_value.isnull()].location_id
        good_covs = dp.loc[~dp.location_id.isin(na_locs)]
        good_covs = good_covs.groupby([
            'parent_id', 'year_id', 'age_group_id', 'sex_id'
        ])['cov_weighted'].sum().reset_index()
        good_covs.rename(columns={'parent_id': 'location_id'}, inplace=True)

        df = df.merge(good_covs, on=['location_id', 'age_group_id', 'year_id', 'sex_id'])
        df.loc[df.location_id.isin(na_locs), 'mean_value'] = df.loc[df.location_id.isin(na_locs), 'cov_weighted']
        df.drop(['cov_weighted'], inplace=True, axis=1)
        return df

    @staticmethod
    def complete_covariate_sex(cov_df, pop_df):
        """
        Fills in missing sex values so that both is propagated to male and female if missing,
        and both is created as a pop-weighted average between male and female if both missing.
        :param cov_df:
        :param pop_df:
        :return:
        """
        if set(cov_df.sex_id) == {1, 2, 3}:
            result_df = cov_df
        elif set(cov_df.sex_id) == {3}:
            cov_1 = cov_df.copy()
            cov_1['sex_id'] = 1
            cov_2 = cov_df.copy()
            cov_2['sex_id'] = 2
            result_df = cov_df.append([cov_1, cov_2])
        elif set(cov_df.sex_id) == {1, 2}:
            both = cov_df.merge(pop_df.loc[pop_df.sex_id == 3],
                                on=['location_id', 'year_id', 'age_group_id', 'sex_id'], how='left')
            both.rename(columns={'population': 'both_pop'}, inplace=True)
            both = both.merge(pop_df.loc[pop_df.sex_id.isin([1, 2])],
                              on=['location_id', 'year_id', 'age_group_id', 'sex_id'], how='left')
            both['cov_weighted'] = both.mean_value / both.both_pop
            both.groupby(['location_id', 'year_id', 'age_group_id'])['cov_weighted'].sum().reset_index()
            result_df = cov_df.append([both])
        else:
            raise RuntimeError(f"Unknown covariate sex IDs {set(cov_df.sex_id)}.")
        return result_df
