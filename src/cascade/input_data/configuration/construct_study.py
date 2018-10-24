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
            seen_covariate[covariate_configuration.study_covariate].add(
                covariate_configuration.transformation)

    for always_include_special_covariate in [0, 1604]:
        seen_covariate[always_include_special_covariate].add(0)

    for cov_id, cov_transformations in seen_covariate.items():
        yield cov_id, list(sorted(cov_transformations))


def _normalize_covariate_data(bundle, study_covariates, id_to_name):
    """
    The input is study covariates in a sparse-columnar format, so it's a list
    of which covariates are nonzero for which seq numbers, where a seq
    number identifies a row in the bundle index. If there are no covariates,
    the returned DataFrame is empty.

    Args:
        bundle (pd.DataFrame): Bundle with its seq column as index
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
    # Get rid of records, by seq number, which don't appear in both bundle and covariates.
    study_subset = study_ids.join(bundle["mean"]).dropna().drop(columns="mean").study_covariate_id
    MATHLOG.info(f"There are {study_subset.shape[0]} nonzero study covariates in this bundle.")
    for cov_id in sorted(id_to_name):  # Sort for stable behavior.
        cov_column =  pd.Series([0.0] * bundle.shape[0], index=bundle.index.values, name=id_to_name[cov_id])
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
        return pd.DataFrame(index=bundle.index)


def add_special_study_covariates(covariate_records, bundle, average_integrand_cases):
    """
    Adds the following covariates to the covariate records: one, sex.
    These are special and have to happen after avgints are defined.

    Args:
        covariate_records (CovariateRecords): The study covariates.
        bundle (pd.DataFrame): The input data bundle, which still has a sex id.
        average_integrand_cases (pd.DataFrame): Desired output records.
    """
    sex_col = bundle.sex_id.apply({1: 0.5, 2: -0.5, 3: 0.0, 4: 0.0}.get)
    covariate_records.measurements = covariate_records.measurements.assign(sex=sex_col)
    # covariate_records.average_integrand_cases. These are set when making avgints.
    covariate_records.id_to_reference[0] = 0.0
    covariate_records.id_to_name[0] = "sex"

    covariate_records.measurements = covariate_records.measurements.assign(
        one=pd.Series(np.ones(bundle.shape[0], dtype=np.double), index=bundle.index))
    covariate_records.average_integrand_cases = covariate_records.average_integrand_cases.assign(
        one=pd.Series(np.ones(average_integrand_cases.shape[0], dtype=np.double),
                      index=average_integrand_cases.index))
    covariate_records.id_to_reference[1604] = 0.0
    covariate_records.id_to_name[1604] = "one"


def add_avgint_records_to_study_covariates(average_integrand_cases_index, covariate_records):
    """All study covariates get added to the average integrands with level=0."""
    columns_to_add = covariate_records.measurements.columns
    covariate_records.average_integrand_cases = pd.DataFrame(
        data=np.zeros((len(average_integrand_cases_index), len(columns_to_add)), dtype=np.double),
        columns=columns_to_add,
        index=average_integrand_cases_index,
    )


def get_bundle_study_covariates(bundle, bundle_id, execution_context, tier):
    covariate_data = _get_study_covariates(execution_context, bundle_id, tier=tier)
    unique_ids = list(sorted(covariate_data.study_covariate_id.unique()))
    records = CovariateRecords("study")
    id_to_name = covariate_ids_to_names(execution_context, unique_ids)
    records.measurements = _normalize_covariate_data(bundle, covariate_data, id_to_name)
    records.id_to_name = id_to_name
    records.id_to_reference = {rid: 0.0 for rid in records.id_to_name}
    # Cannot fill out study covariates for average integrand cases until they are made.
    return records
