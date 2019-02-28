
import numpy as np
import pandas as pd

from cascade.core.log import getLoggers
from cascade.input_data.configuration.construct_study import (
    add_study_covariate_to_observations
)
from cascade.input_data.db.country_covariates import country_covariate_names
from cascade.input_data.db.study_covariates import covariate_ids_to_names

CODELOG, MATHLOG = getLoggers(__name__)


def find_covariate_names(execution_context, epiviz_covariates):
    study_covariate_ids = {evc.covariate_id for evc in epiviz_covariates if evc.study_country == "study"}
    sex_and_one_covariates = {0, 1604}
    study_covariate_ids |= sex_and_one_covariates
    study_id_to_name = covariate_ids_to_names(execution_context, study_covariate_ids)
    CODELOG.debug(f"Study covariates for this model {study_id_to_name}")
    study_id_to_name = {si: f"s_{sn}" for (si, sn) in study_id_to_name.items()}
    country_id_to_name = country_covariate_names()
    country_id_to_name = {ci: f"c_{cn}" for (ci, cn) in country_id_to_name.items()}
    return study_id_to_name, country_id_to_name


def add_covariate_data_to_observations_and_avgints(data, local_settings, epiviz_covariates):
    """
    Add study and country covariates, properly transformed, to observations
    and average integrand cases. The strategy is to add them directly to the
    DataFrame for observations and average integrand cases.

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
    for name_covariate in epiviz_covariates:
        if name_covariate.study_country == "study":
            short = data.study_id_to_name[name_covariate.covariate_id]
        else:
            short = data.country_id_to_name[name_covariate.covariate_id]
        name_covariate.untransformed_covariate_name = short

    # Add untransformed study covariates to observations.
    data.observations = add_study_covariate_to_observations(
        data.observations, data.sparse_covariate_data, data.study_id_to_name)

    # Create untransformed study covariates on avgints.
    study_columns = sorted(data.study_id_to_name.keys())
    average_integrand_cases_index = data.average_integrand_cases.index
    data.average_integrand_cases = pd.DataFrame(
        # They are all zero, which is the correct, final, value.
        data=np.zeros((len(average_integrand_cases_index), len(study_columns)), dtype=np.double),
        columns=study_columns,
        index=average_integrand_cases_index,
    )
    MATHLOG.info(f"Study covariates added: {study_columns}")
