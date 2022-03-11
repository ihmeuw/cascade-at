import pandas as pd
from typing import List
from itertools import chain

from cascade_at.core.db import db_queries
from cascade_at.core.log import get_loggers
from cascade_at.inputs.base_input import BaseInput
from cascade_at.inputs.demographics import Demographics

LOG = get_loggers(__name__)


class CovariateData(BaseInput):
    def __init__(self, covariate_id: int, demographics: Demographics,
                 decomp_step: str, gbd_round_id: int):
        """
        Get covariate estimates, and map them to the necessary demographic
        ages and sexes. If only one age group is present in the covariate
        data then that means that it's not age-specific and we want to copy
        the values over to all the other age groups we're working with in
        demographics. Same with sex.
        """
        self.covariate_id = covariate_id
        self.demographics = demographics
        self.decomp_step = decomp_step
        self.gbd_round_id = gbd_round_id

        super().__init__(gbd_round_id=gbd_round_id)

        self.raw = None

    def get_raw(self):
        """
        Pulls the raw covariate data from the database.
        """
        self.raw = db_queries.get_covariate_estimates(
            covariate_id=self.covariate_id,
            year_id=self.demographics.year_id,
            gbd_round_id=self.gbd_round_id,
            decomp_step=self.decomp_step
        )
        return self

    def configure_for_dismod(self, pop_df: pd.DataFrame, loc_df: pd.DataFrame):
        """
        Configures covariates for DisMod. Completes covariate
        ages, sexes, and locations based on what covariate data is already
        available.

        To fill in ages, it copies over all age or age standardized
        covariates into each of the specific age groups.

        To fill in sexes, it copies over any both sex covariates to
        the sex specific groups.

        To fill in locations, it takes a population-weighted average of child
        locations for parent locations all the way up the location hierarchy.

        Parameters
        ----------
        pop_df
            A data frame with population info for all ages, sexes, locations, and years
        loc_df
            A data frame with location hierarchy information
        """
        df = self.raw[[
            'location_id', 'year_id', 'age_group_id', 'sex_id', 'mean_value'
        ]]
        df = self._complete_covariate_ages(cov_df=df)
        df = self._complete_covariate_locations(cov_df=df, pop_df=pop_df, loc_df=loc_df,
                                                locations=self.demographics.location_id)
        df = self._complete_covariate_sex(cov_df=df, pop_df=pop_df)
        df = self.convert_to_age_lower_upper(df)
        return df
    
    def _complete_covariate_ages(self, cov_df):
        """
        Adds on covariate ages for all age group IDs.
        """
        if (22 in cov_df.age_group_id.tolist()) or (27 in cov_df.age_group_id.tolist()):
            covs = pd.DataFrame()
            for age in self.demographics.age_group_id:
                df = cov_df.copy()
                df['age_group_id'] = age
                covs = covs.append(df)
        else:
            covs = cov_df.copy()
        return covs

    # The IHME databases are supposed to supply covariate values for all locations
    # In the meantime, Brad's code is now supplying the missing values
    @staticmethod
    def _complete_covariate_locations(cov_df: pd.DataFrame, pop_df: pd.DataFrame, loc_df: pd.DataFrame,
                                      locations: List[int]):
        """
        Completes the covariate locations that aren't in the database as a population-weighted average.
        """

        def ancestors(loc):
            ploc = loc_df.loc[loc_df.location_id == loc, 'path_to_top_parent'].squeeze()
            if ploc or not ploc.empty:
                return [int(l) for l in ploc.split(',') if int(l) != loc]
            return []

        cov_locations = set(cov_df.location_id.unique().tolist())
        ancestor_locations = set(chain(*[ancestors(loc) for loc in locations]))
        ancestors_without_covariates = ancestor_locations - cov_locations

        loc_subset_df = loc_df.loc[loc_df.location_id.isin(ancestors_without_covariates),
                                   ['location_id', 'parent_id', 'level', 'location_name']]
        all_levels = loc_subset_df.level.unique().tolist()

        df = cov_df.merge(pop_df, how='left')
        df['weighted_mean'] = df.population*df.mean_value

        add_cov = pd.DataFrame()
        for level in sorted(all_levels, reverse=True):
            for loc in ancestors_without_covariates:
                if level in loc_subset_df.loc[loc_subset_df.location_id == loc, 'level'].values:
                    LOG.info(f"Filling in covariate values for location hierarchy level {level}, location {loc}.")
                    children = loc_subset_df.loc[loc_subset_df.parent_id == loc, 'location_id'].values
                    child_covs = df[df.location_id.isin(children)]
                    grps = child_covs.groupby(['year_id', 'age_group_id', 'sex_id'], as_index=False)
                    x = grps['weighted_mean'].sum().merge(grps['population'].sum())
                    x['mean_value'] = x['weighted_mean'] / x['population']
                    x['location_id'] = loc
                    x = x.merge(pop_df[['location_id', 'age_group_id', 'year_id', 'sex_id', 'age_lower', 'age_upper']],
                                on=['location_id', 'year_id', 'age_group_id', 'sex_id'], how='left')
                    add_cov = add_cov.append(x)
        if not add_cov.empty:
            cov_df = cov_df.append(add_cov[cov_df.columns]).reset_index(drop=True)
        return cov_df

    @staticmethod
    def _complete_covariate_sex(cov_df: pd.DataFrame, pop_df: pd.DataFrame):
        """
        Fills in missing sex values so that both is propagated to male and female if missing,
        and both is created as a pop-weighted average between male and female if both missing.
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
            both_pop = pop_df.loc[pop_df.sex_id == 3][['location_id', 'year_id', 'age_group_id', 'population']].copy()
            both = cov_df.merge(both_pop,
                                on=['location_id', 'year_id', 'age_group_id'], how='left')
            both.rename(columns={'population': 'both_pop'}, inplace=True)
            both = both.merge(pop_df.loc[pop_df.sex_id.isin([1, 2])],
                              on=['location_id', 'year_id', 'age_group_id', 'sex_id'], how='left')
            both['cov_weighted'] = both.mean_value * both.population / both.both_pop
            both = both.groupby(['location_id', 'year_id', 'age_group_id'])['cov_weighted'].sum().reset_index()
            both['sex_id'] = 3
            both.rename(columns={'cov_weighted': 'mean_value'}, inplace=True)
            result_df = cov_df.append([both])
        else:
            raise RuntimeError(f"Unknown covariate sex IDs {set(cov_df.sex_id)}.")
        return result_df
