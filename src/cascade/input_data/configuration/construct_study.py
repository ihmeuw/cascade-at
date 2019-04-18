"""
Given data from the db, organize it into columns of data which then go
onto the model.
"""

import numpy as np
import pandas as pd

from cascade.core.log import getLoggers
from cascade.input_data import InputDataError

CODELOG, MATHLOG = getLoggers(__name__)


def add_study_covariate_to_observations(observations, study_covariates, id_to_name):
    """
    Given observations without study covariate columns, this puts those columns
    into the DataFrame and fills them, including the sex and one covariates.

    Args:
        observations (pd.DataFrame): Observations. Must include ``seq`` and ``sex_id``.
        study_covariates (pd.DataFrame): Of compressed study covariates, so
            these are the ``seq`` values at which covariates are nonzero.
        id_to_name (Dict[int,str]): There is one entry for each study covariate.

    Returns:
        pd.DataFrame: With the columns attached.
    """
    # Integer-typed column names cause trouble for Pandas.
    string_ids = study_covariates.assign(study_covariate_id=study_covariates.study_covariate_id.astype(str))
    str_id_to_name = {str(cid): cname for (cid, cname) in id_to_name.items()}
    with_ones = string_ids.assign(value=pd.Series(np.ones(len(study_covariates), dtype=np.double)))
    try:
        cov_columns = with_ones.pivot(index="seq", columns="study_covariate_id", values="value") \
            .fillna(0.0).rename(columns=str_id_to_name)
    except ValueError as ve:
        study_covariates.to_hdf("covariates", "table")
        raise InputDataError(f"Could not create covariate columns") from ve
    missing_columns = set(str_id_to_name.values()) - set(cov_columns.columns)
    if missing_columns:
        cov_columns = cov_columns.assign(**{
            miss_col: pd.Series(np.zeros(len(cov_columns), dtype=np.double))
            for miss_col in missing_columns
        })
    # else nothing to create.
    extra_data = set(cov_columns.columns) - set(str_id_to_name.values())
    if extra_data:
        cov_columns = cov_columns.drop(columns=extra_data)
    # else no extra columns to drop.
    try:
        obs_with_covs = observations.merge(cov_columns, left_on="seq", right_index=True, how="left")
    except KeyError as ke:
        raise InputDataError(f"These study covariate IDs have seq IDs that don't "
                             f"correspond to the bundle seq IDs") from ke
    # This sets NaNs in covariate columns to zeros.
    filled = obs_with_covs.fillna(value={fname: 0.0 for fname in str_id_to_name.values()})
    # Special study covariates won't have appeared in the database, so they
    # will be zeros. Fill them in.
    sex_assignment = {1: 0.5, 2: -0.5, 3: 0.0, 4: 0.0}
    return filled.assign(s_sex=filled.sex_id.apply(sex_assignment.get), s_one=1)


def normalize_covariate_data(observations, study_covariates, id_to_name):
    """
    The input is study covariates in a sparse-columnar format, so it's a list
    of which covariates are nonzero for which seq numbers, where a seq
    number identifies a row in the bundle index. If there are no covariates,
    the returned DataFrame is empty.

    Args:
        observations (pd.DataFrame): Observations including those in the bundle.
        study_covariates (pd.DataFrame): Contains seq numbers and covariate ids.
            Optionally contains the ``bundle_id``.
        id_to_name: Dictionary from ids to names

    Returns:
        pd.DataFrame: Each column is a full row of zeros and ones, and the row
            name is the name of the covariate, without the ``x_`` in front.
            Even if there are no covariates, this will have the
            ``covariate_sequence_number``.
    """
    full = add_study_covariate_to_observations(observations, study_covariates, id_to_name)
    # Now separate the covariates from the observations because they will
    # be transformed and added back.
    keep_cols = {"seq"} | set(id_to_name.values())
    # Rename seq to covariate_sequence_number because if there is ever a
    # ovariate with the same name, then it will collide, and seq is too short
    # not to collide.
    return full.drop(columns=[dc for dc in full.columns if dc not in keep_cols]).rename(
        columns={"seq": "covariate_sequence_number"})
