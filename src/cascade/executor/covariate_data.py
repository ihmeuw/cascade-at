
import numpy as np
import pandas as pd

from cascade.core import getLoggers
from cascade.input_data.configuration.construct_study import (
    add_study_covariate_to_observations
)
from cascade.input_data.db.country_covariates import country_covariate_names
from cascade.input_data.db.study_covariates import covariate_ids_to_names
from cascade.input_data.configuration.construct_country import (
    assign_interpolated_covariate_values,
    reference_value_for_covariate_mean_all_values
)
from cascade.input_data.configuration.builder import COVARIATE_TRANSFORMS

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
    add_study_covariate_to_observations_and_avgints(data)
    add_country_covariate_to_observations_and_avgints(data, local_settings, epiviz_covariates)


def assign_epiviz_covariate_names(study_id_to_name, country_id_to_name, epiviz_covariates):
    # Assign all of the names. Study covariates aren't ever transformed.
    for name_covariate in epiviz_covariates:
        if name_covariate.study_country == "study":
            short = study_id_to_name.get(name_covariate.covariate_id, None)
        else:
            short = country_id_to_name.get(name_covariate.covariate_id, None)
        if short is None:
            raise RuntimeError(
                f"Covariate {name_covariate} is not found in id-to-name mapping."
            )
        name_covariate.untransformed_covariate_name = short


def add_country_covariate_to_observations_and_avgints(data, local_settings, epiviz_covariates):
    """Adds country covariates to observation and average integrand cases."""
    country_specs = {ccov for ccov in epiviz_covariates if ccov.study_country == "country"}
    for covariate_id in {evc.covariate_id for evc in country_specs}:
        ccov_ranges_df = data.country_covariates[covariate_id]
        reference = reference_value_for_covariate_mean_all_values(ccov_ranges_df)
        for df_name in ["observations", "average_integrand_cases"]:
            measurement = getattr(data, df_name)
            if measurement is not None:
                observations_column = assign_interpolated_covariate_values(
                    measurement, ccov_ranges_df, data.country_covariates_binary[covariate_id])
                ccov_transforms = [ccov for ccov in country_specs if ccov.covariate_id == covariate_id]
                for transformed in ccov_transforms:
                    settings_transform = COVARIATE_TRANSFORMS[transformed.transformation_id]
                    transformed.reference = settings_transform(reference)
                    measurement = measurement.assign(
                        **{transformed.name: settings_transform(observations_column)})
                setattr(data, df_name, measurement)
            # else nothing to add to the data.


def add_study_covariate_to_observations_and_avgints(data):
    # Add untransformed study covariates to observations.
    data.observations = add_study_covariate_to_observations(
        data.observations, data.sparse_covariate_data, data.study_id_to_name)
    assert "age_lower" in data.observations.columns
    # Create untransformed study covariates on avgints.
    study_columns = sorted(data.study_id_to_name.values())
    average_integrand_cases_index = data.average_integrand_cases.index
    avgint_columns = pd.DataFrame(
        # They are all zero except sex, which is the correct, final, value.
        data=np.zeros((len(average_integrand_cases_index), len(study_columns)), dtype=np.double),
        columns=study_columns,
        index=average_integrand_cases_index,
    )
    data.average_integrand_cases = pd.concat([data.average_integrand_cases, avgint_columns], axis=1)
    avgints = data.average_integrand_cases
    if "s_sex" in avgints.columns and "sex_id" in avgints:
        avgints.loc[avgints.sex_id == 1, "s_sex"] = 0.5
        avgints.loc[avgints.sex_id == 2, "s_sex"] = -0.5
        data.average_integrand_cases = avgints
    else:
        MATHLOG.warning(f"Sex covariate missing when assigning integrands.")
    MATHLOG.info(f"Study covariates added: {study_columns}")
