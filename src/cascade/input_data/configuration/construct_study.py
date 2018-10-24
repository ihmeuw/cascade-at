"""
Given data from the db, organize it into columns of data which then go
onto the model.
"""
from collections.__init__ import defaultdict

import numpy as np
import pandas as pd

from cascade.input_data import InputDataError
from cascade.input_data.configuration.covariate_records import CovariateRecords
from cascade.input_data.db.study_covariates import _get_study_covariates, \
    covariate_ids_to_names
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
            These have the "in_bundle" column.
        study_covariates (pd.DataFrame): Contains seq numbers and covariate ids.
            Optionally contains the ``bundle_id``.
        id_to_name: Dictionary from ids to names

    Returns:
        pd.DataFrame: Each column is a full row of zeros and ones, and the row
        name is the name of the covariate, without the ``x_`` in front.
    """
    study_ids = study_covariates.set_index("seq")
    study_covariate_columns = list()
    indices_not_found = list()
    bundle = observations[observations["in_bundle"] == 1]
    # Get rid of records, by seq number, which don't appear in both bundle and covariates.
    study_subset = study_ids.join(bundle["mean"]).dropna().drop(columns="mean").study_covariate_id
    MATHLOG.info(f"There are {study_subset.shape[0]} nonzero study covariates in this bundle.")
    for cov_id in sorted(id_to_name):  # Sort for stable behavior.
        cov_column =  pd.Series([0.0] * observations.shape[0], index=observations.index.values, name=id_to_name[cov_id])
        try:
            cov_column.loc[study_subset[study_subset == cov_id].index] = 1.0
        except KeyError:
            indices_not_found.append(cov_id)
        study_covariate_columns.append(cov_column)
    if indices_not_found:
        raise InputDataError(f"These study covariate IDs have seq IDs that don't "
                             f"correspond to the bundle seq IDs: {indices_not_found}.")

    if study_covariate_columns:
        return pd.concat(study_covariate_columns, axis=1)
    else:
        return pd.DataFrame(index=observations.index)


def add_special_study_covariates(covariate_records, model_context):
    """
    Adds the following covariates to the covariate records: one, sex.
    These are special and have to happen after avgints are defined.

    Args:
        covariate_records (CovariateRecords): The study covariates.
        model_context (ModelContext): Uses observations and average integrand cases.
    """
    observations = model_context.input_data.observations
    average_integrand_cases = model_context.average_integrand_cases
    sex_assignment = {1: 0.5, 2: -0.5, 3: 0.0, 4: 0.0}
    sex_col = observations.sex_id.apply(sex_assignment.get)
    covariate_records.measurements = covariate_records.measurements.assign(sex=sex_col)
    covariate_records.average_integrand_cases = covariate_records.average_integrand_cases.assign(
        sex=average_integrand_cases.sex_id.apply(sex_assignment.get)
    )
    # covariate_records.average_integrand_cases. These are set when making avgints.
    covariate_records.id_to_reference[0] = 0.0
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
    sparse_covariate_data = _get_study_covariates(execution_context, bundle_id, tier=tier)
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
