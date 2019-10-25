import pandas as pd

from cascade_at.core.db import db_queries
from cascade_at.core.log import get_loggers
from cascade_at.inputs.utilities.transformations import COVARIATE_TRANSFORMS
from cascade_at.inputs.utilities.ids import CascadeConstants

LOG = get_loggers(__name__)


def get_covariate(covariate_id, demographics, decomp_step, gbd_round_id):
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
    :return:
    """
    df = db_queries.get_covariate_estimates(
        covariate_id=covariate_id,
        location_id=demographics.location_id,
        year_id=demographics.year_id,
        gbd_round_id=gbd_round_id,
        decomp_step=decomp_step
    )[[
        'location_id', 'year_id', 'age_group_id', 'sex_id', 'mean_value'
    ]]

    if len(df.age_group_id.unique()) == 1:
        if df.age_group_id.unique()[0] in CascadeConstants.NON_AGE_SPECIFIC_ID:
            new_age_dfs = []
            for age in demographics.age_group_id:
                new_df = df.copy()
                new_df['age_group_id'] = age
                new_age_dfs.append(new_df)
            df = pd.concat(new_age_dfs)

    if len(df.sex_id.unique()) == 1:
        if df.sex_id.unique()[0] in CascadeConstants.NON_SEX_SPECIFIC_ID:
            new_sex_dfs = []
            for sex in demographics.sex_id:
                new_df = df.copy()
                new_df['sex_id'] = sex
                new_sex_dfs.append(new_df)
            df = pd.concat(new_sex_dfs)

    return df


def get_covariates(covariate_ids, demographics, decomp_step, gbd_round_id):
    """
    Get all covariates, just a wrapper for get_covariate.

    :param covariate_ids: (list of int)
    :param demographics: (cascade_at.inputs.demographics.Demographics)
    :param decomp_step: (str)
    :param gbd_round_id: (int)
    :return:
    """
    covs = []
    for covariate_id in covariate_ids:
        df = get_covariate(
            covariate_id=covariate_id,
            demographics=demographics,
            decomp_step=decomp_step,
            gbd_round_id=gbd_round_id
        )
        covs.append(df)
    return covs
