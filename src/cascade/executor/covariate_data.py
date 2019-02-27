
import numpy as np
import pandas as pd

from cascade.core.log import getLoggers
from cascade.input_data import InputDataError
from cascade.input_data.configuration.builder import COVARIATE_TRANSFORMS
from cascade.input_data.db.country_covariates import country_covariate_names
from cascade.input_data.db.study_covariates import covariate_ids_to_names

CODELOG, MATHLOG = getLoggers(__name__)


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
    return obs_with_covs.fillna(value={fname: 0.0 for fname in id_to_name.values()})


def transformed_name(covariate_name, study_country, transform_id):
    settings_transform = COVARIATE_TRANSFORMS[transform_id]
    transform_name = settings_transform.__name__
    MATHLOG.info(f"Transforming {covariate_name} with {transform_name}")
    if study_country == "study":
        assert transform_id == 0
        return f"s_{covariate_name}"
    else:
        return f"c_{covariate_name}_{transform_name}"


def add_covariate_data_to_observations_and_avgints(execution_context, data, local_settings, epiviz_covariates):
    """
    Add study and country covariates, properly transformed, to observations
    and average integrand cases.

    Args:
        execution_context: Execution context for db and file access.
        data: The object that contains observations, average integrands
            and raw covariate data.
        local_settings: A settings object for this estimation location.
        epiviz_covariates List[EpiVizCovariate]: The specification for the
            covariates.

    Returns:
        None: Everything is added to observations and avgints.
    """
    # Assign all of the names. Study covariates aren't ever transformed.
    study_covariate_ids = set([evc.covariate_id for evc in epiviz_covariates if evc.study_country == "study"])
    study_id_to_name = covariate_ids_to_names(execution_context, study_covariate_ids)
    study_id_to_name = {si: f"s_{sn}" for (si, sn) in study_id_to_name.items()}
    country_id_to_name = country_covariate_names()
    country_id_to_name = {ci: f"c_{cn}" for (ci, cn) in country_id_to_name.items()}

    for name_covariate in epiviz_covariates:
        if name_covariate.study_country == "study":
            short = study_id_to_name[name_covariate.covariate_id]
            name_covariate.name = short  # The column won't be transformed.
        else:
            short = country_id_to_name[name_covariate.covariate_id]
        name_covariate.untransformed_covariate_name = short

    # Add untransformed study covariates to observations.
    data.observations = normalize_covariate_data(
        data.observations, data.sparse_covariate_data, study_id_to_name)

    # Create untransformed study covariates on avgints.
    study_columns = sorted(study_id_to_name.keys())
    average_integrand_cases_index = data.average_integrand_cases.index
    data.average_integrand_cases = pd.DataFrame(
        # They are all zero, which is the correct, final, value.
        data=np.zeros((len(average_integrand_cases_index), len(study_columns)), dtype=np.double),
        columns=study_columns,
        index=average_integrand_cases_index,
    )
