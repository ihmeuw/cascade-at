"""
Given data from the db, organize it into columns of data which then go
onto the model.
"""
from collections.__init__ import defaultdict

import numpy as np
import pandas as pd

from cascade.input_data import InputDataError
from cascade.input_data.configuration.covariate_records import CovariateRecords
from cascade.input_data.db.study_covariates import get_study_covariates, covariate_ids_to_names
from cascade.core.log import getLoggers

CODELOG, MATHLOG = getLoggers(__name__)


def unique_study_covariate_transform(configuration):
    """
    Iterates through all covariate IDs, including the list of ways to
    transform them, because each transformation is its own column for Dismod.
    """
    seen_covariate = defaultdict(set)
    if configuration.study_covariate:
        for covariate_configuration in configuration.study_covariate:
            seen_covariate[covariate_configuration.study_covariate_id].add(
                covariate_configuration.transformation)

    for always_include_special_covariate in [0, 1604]:
        seen_covariate[always_include_special_covariate].add(0)

    for cov_id, cov_transformations in seen_covariate.items():
        yield cov_id, list(sorted(cov_transformations))


def _normalize_covariate_data(observations, study_covariates, id_to_name):
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
    with_ones = study_covariates.assign(value=pd.Series(np.ones(len(study_covariates), dtype=np.double)))
    try:
        cov_columns = with_ones.pivot(index="seq", columns="study_covariate_id", values="value") \
            .fillna(0.0).rename(columns=id_to_name)
    except ValueError as ve:
        study_covariates.to_hdf("covariates", "table")
        raise InputDataError(f"Could not create covariate columns") from ve
    missing_columns = set(id_to_name.values()) - set(cov_columns.columns)
    if missing_columns:
        cov_columns = cov_columns.assign(**{
            miss_col: pd.Series(np.zeros(len(cov_columns), dtype=np.double))
            for miss_col in missing_columns
        })
    try:
        obs_with_covs = observations.merge(cov_columns, left_on="seq", right_index=True, how="left")
    except KeyError as ke:
        raise InputDataError(f"These study covariate IDs have seq IDs that don't "
                             f"correspond to the bundle seq IDs") from ke
    # This sets NaNs in covariate columns to zeros.
    full = obs_with_covs.fillna(value={fname: 0.0 for fname in id_to_name.values()})
    # Now separate the covariates from the observations because they will
    # be transformed and added back.
    keep_cols = {"seq"} | set(id_to_name.values())
    # Rename seq to covariate_sequence_number because if there is ever a
    # ovariate with the same name, then it will collide, and seq is too short
    # not to collide.
    return full.drop(columns=[dc for dc in full.columns if dc not in keep_cols]).rename(
        columns={"seq": "covariate_sequence_number"})


def add_special_study_covariates(covariate_records, model_context, sex_id):
    """
    Adds the following covariates to the covariate records: one, sex.
    These are special and have to happen after avgints are defined.

    Args:
        covariate_records (CovariateRecords): The study covariates.
        model_context (ModelContext): Uses observations and average integrand cases.
    """
    observations = model_context.input_data.observations
    if not observations.seq.equals(covariate_records.measurements.covariate_sequence_number):
        raise RuntimeError(f"The study covariates and the measurements aren't "
                           f"in the same order.")
    average_integrand_cases = model_context.average_integrand_cases
    sex_assignment = {1: 0.5, 2: -0.5, 3: 0.0, 4: 0.0}
    sex_col = observations.sex_id.apply(sex_assignment.get)
    covariate_records.measurements = covariate_records.measurements.assign(sex=sex_col)
    # covariate_records.average_integrand_cases. These are set when making avgints.
    covariate_records.average_integrand_cases = covariate_records.average_integrand_cases.assign(
        sex=average_integrand_cases.sex_id.apply(sex_assignment.get)
    )
    covariate_records.id_to_reference[0] = sex_assignment[sex_id]
    # This max difference means a sex of male gets male and both, a sex of
    # female gets female and both, and a sex of both gets all data.
    covariate_records.id_to_max_difference[0] = 0.75
    covariate_records.id_to_name[0] = "sex"

    covariate_records.measurements = covariate_records.measurements.assign(
        one=pd.Series(np.ones(observations.shape[0], dtype=np.double), index=observations.index))
    covariate_records.average_integrand_cases = covariate_records.average_integrand_cases.assign(
        one=pd.Series(np.ones(average_integrand_cases.shape[0], dtype=np.double),
                      index=average_integrand_cases.index))
    covariate_records.id_to_reference[1604] = 0.0
    covariate_records.id_to_name[1604] = "one"


def get_bundle_study_covariates(model_context, bundle_id, execution_context, tier):
    # Sparse data is specified as a list of seq IDs that have a particular study covariate.
    sparse_covariate_data = get_study_covariates(execution_context, bundle_id, tier=tier)
    unique_ids = list(sorted(sparse_covariate_data.study_covariate_id.unique()))
    records = CovariateRecords("study")
    id_to_name = covariate_ids_to_names(execution_context, unique_ids)
    observations = model_context.input_data.observations
    records.measurements = _normalize_covariate_data(observations, sparse_covariate_data, id_to_name)
    records.id_to_name = id_to_name
    records.id_to_reference = {rid: 0.0 for rid in records.id_to_name}
    columns_to_add = records.measurements.columns
    average_integrand_cases_index = model_context.average_integrand_cases.index
    records.average_integrand_cases = pd.DataFrame(
        data=np.zeros((len(average_integrand_cases_index), len(columns_to_add)), dtype=np.double),
        columns=columns_to_add,
        index=average_integrand_cases_index,
    )
    return records
