import pandas as pd

from cascade_at.core.db import db_queries
from cascade_at.core.log import get_loggers
from cascade_at.inputs.utilities.transformations import COVARIATE_TRANSFORMS
from cascade_at.inputs.utilities.ids import CascadeConstants
from cascade_at.inputs.base_input import BaseInput

LOG = get_loggers(__name__)


class Covariate(BaseInput):
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

    def configure_for_dismod(self):
        """
        Configures covariates for DisMod.
        :return: self
        """
        df = self.raw[[
            'location_id', 'year_id', 'age_group_id', 'sex_id', 'mean_value'
        ]]
        if len(df.age_group_id.unique()) == 1:
            if df.age_group_id.unique()[0] in CascadeConstants.NON_AGE_SPECIFIC_ID:
                new_age_dfs = []
                for age in self.demographics.age_group_id:
                    new_df = df.copy()
                    new_df['age_group_id'] = age
                    new_age_dfs.append(new_df)
                df = pd.concat(new_age_dfs)

        if len(df.sex_id.unique()) == 1:
            if df.sex_id.unique()[0] in CascadeConstants.NON_SEX_SPECIFIC_ID:
                new_sex_dfs = []
                for sex in self.demographics.sex_id:
                    new_df = df.copy()
                    new_df['sex_id'] = sex
                    new_sex_dfs.append(new_df)
                df = pd.concat(new_sex_dfs)

        df = self.convert_to_age_lower_upper(df)
        return df
