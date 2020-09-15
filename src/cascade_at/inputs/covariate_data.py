import pandas as pd
from typing import List

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
        df = self._complete_covariate_sex(cov_df=df, pop_df=pop_df)
        df = self._complete_covariate_locations(cov_df=df, pop_df=pop_df, loc_df=loc_df,
                                                locations=self.demographics.location_id)
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

    @staticmethod
    def _complete_covariate_locations(cov_df: pd.DataFrame, pop_df: pd.DataFrame, loc_df: pd.DataFrame,
                                      locations: List[int]):
        """
        Completes the covariate locations that aren't in the database as a population-weighted average.
        """
        parent_pop = pop_df[['location_id', 'age_group_id', 'sex_id', 'year_id', 'population']].copy()
        parent_pop.rename(columns={'location_id': 'parent_id', 'population': 'parent_population'}, inplace=True)

        loc_subset_df = loc_df.loc[loc_df.location_id.isin(locations)]
        all_levels = loc_subset_df.level.unique().tolist()
        cov_locations = cov_df.location_id.unique().tolist()
        cov_levels = loc_subset_df.loc[loc_subset_df.location_id.isin(cov_locations)].level.unique().tolist()
        missing_levels = [x for x in all_levels if x not in cov_levels]

        df = cov_df.copy()

        for level in sorted(missing_levels, reverse=True):
            LOG.info(f"Filling in covariate values at location hierarchy level {level}.")
            # Get one location below this level
            ldf = loc_subset_df.loc[loc_subset_df.level == level + 1].copy()

            # Merge on the population just for these locations (left) --
            # builds out the full age-sex-year data frame for populations
            lp = ldf.merge(pop_df, on=['location_id'], how='left')

            # Merge on the covariate data just for these location-populations
            clp = lp.merge(df, on=['location_id', 'age_group_id', 'sex_id', 'year_id'], how='left')

            # Get the parent population based on parent ID
            dp = clp.merge(parent_pop, on=['parent_id', 'age_group_id', 'sex_id', 'year_id'], how='left')
            dp.drop('location_id', inplace=True, axis=1)
            
            # Calculate the weighted value for each row
            dp['cov_weighted'] = dp.mean_value * dp.population / dp.parent_population

            # Group by parent ID and other demographics, over location IDs, summing
            # to get the final weighted covariate value
            dp = dp.groupby([
                'parent_id', 'year_id', 'age_group_id', 'sex_id'
            ])['cov_weighted'].sum().reset_index()

            # Set the new parent ID as location ID so that it can be used one level up the tree
            dp.rename(columns={'parent_id': 'location_id', 'cov_weighted': 'mean_value'}, inplace=True)
            df = df.append(dp)

        return df

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
