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


def unique_study_covariate_transform(configuration):
    """
    Iterates through all covariate IDs, including the list of ways to
    transform them, because each transformation is its own column for Dismod.
    """
    seen_covariate = defaultdict(set)
    if configuration.study_covariate:
        for covariate_configuration in configuration.study_covariate:
            seen_covariate[covariate_configuration.study_covariate].add(
                covariate_configuration.transformation)

    for cov_id, cov_transformations in seen_covariate.items():
        yield cov_id, list(sorted(cov_transformations))


def _normalize_covariate_data(bundle_index, study_covariates, covariate_ids):
    """
    The input is study covariates in a sparse-columnar format, so it's a list
    of which covariates are nonzero for which seq numbers, where a seq
    number identifies a row in the bundle index. If there are no covariates,
    the returned DataFrame is empty.

    Args:
        bundle_index (pd.Index): The index of seq numbers for the bundle.
        study_covariates (pd.DataFrame): Contains seq numbers and covariate ids.
            Optionally contains the ``bundle_id``.
        covariate_ids (List[int]): List of covariate ids to read.

    Returns:
        pd.DataFrame: Each column is a full row of zeros and ones, and the row
        name is the name of the covariate, without the ``x_`` in front.
    """
    study_ids = study_covariates.set_index("seq").study_covariate_id
    study_covariate_columns = list()
    indices_not_found = list()
    for cov_id in sorted(covariate_ids):  # Sort for stable behavior.
        cov_column =  pd.Series([0.0] * len(bundle_index), index=bundle_index.values, name=cov_id)
        try:
            cov_column.loc[study_ids[study_ids == cov_id].index] = 1.0
        except KeyError:
            indices_not_found.append(cov_id)
        study_covariate_columns.append(cov_column)
    if indices_not_found:
        raise InputDataError(f"Study covariates list ids not found in the bundle for "
                             f"covariates: {indices_not_found}.")

    if study_covariate_columns:
        return pd.concat(study_covariate_columns, axis=1)
    else:
        return pd.DataFrame(index=bundle_index)


def add_avgint_records_to_study_covariates(average_integrand_cases_index, covariate_records):
    """All study covariates get added to the average integrands with level=0."""
    columns_to_add = covariate_records.measurements.columns
    covariate_records.average_integrand_cases = pd.DataFrame(
        data=np.zeros((len(average_integrand_cases_index), len(columns_to_add)), dtype=np.double),
        columns=columns_to_add,
        index=average_integrand_cases_index,
    )


def get_bundle_study_covariates(bundle_index, bundle_id, execution_context, tier):
    covariate_data = _get_study_covariates(execution_context, bundle_id, tier=tier)
    unique_ids = list(sorted(covariate_data.study_covariate_id.unique()))
    records = CovariateRecords("study")
    records.measurements = _normalize_covariate_data(bundle_index, covariate_data, unique_ids)
    records.id_to_name = covariate_ids_to_names(execution_context, unique_ids)
    records.id_to_reference = {rid: 0.0 for rid in records.id_to_name}
    # Cannot fill out study covariates for average integrand cases until they are made.
    return records
